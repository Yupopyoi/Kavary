# -*- coding: utf-8 -*-

# This software is based on original work provided under the MIT License:
# - Copyright (c) 2023 liujing04
# - Copyright (c) 2023 源文雨
# - Copyright (c) 2023 Ftps
#
# The current implementation includes modifications and extensions beyond the original code.
#
# Modified and maintained by:
#
# - Copyright (c) 2025 Yupopyoi
#
# See the LICENSE file for more details.

import os
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("fairseq").setLevel(logging.ERROR)

import faiss 
import numpy as np
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils
from typing import Optional

from infer_pack.models import VoiceSynthesizer, VoiceSynthesizerNoF0

class RVCModel:
    def __init__(self, key, hubert_path: str, pth_path: str, index_path: str, index_rate: float):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hubert_path = hubert_path
        self.pth_path = pth_path
        self.index_path = index_path
        self.index_rate = index_rate
        self.f0_up_key = key
        
        self.model: Optional[torch.nn.Module] = None
        self.voice_synthesizer: Optional[VoiceSynthesizer] = None
        self.index = None
        self.done_faiss_log_output = False
        
        self._init_constants()
        self._load_index()
        self._load_hubert_model()
        self._load_generator_model()

    def _init_constants(self):
        self.sampling_rate_input = 16000
        self.sampling_rate_output = 44100
        self.frame_size = 160
        self.frame_shift_ms = self.frame_size / self.sampling_rate_input * 1000
        
        self.f0_min = 50   # Lowest  fundamental frequency
        self.f0_max = 1100 # Highest fundamental frequency
        self.f0_mel_min = self._hz_to_mel(self.f0_min)
        self.f0_mel_max = self._hz_to_mel(self.f0_max)

    def _load_index(self):
        if self.index_rate != 0:
            try:
                self.index = faiss.read_index(self.index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                self._log("Index search enabled.")
            except Exception as e:
                self._log(f"Failed to load index: {e}")
                self.index = None

    def _load_hubert_model(self):
        try:
            self._log(f"Loading Hubert model from {self.hubert_path}")
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [self.hubert_path],
                suffix=""
            )
            self.model = models[0].to(self.device).half().eval()
        except Exception as e:
            self._log(f"Failed to load Hubert model: {e}")
            raise

    def _load_generator_model(self):
        try:
            self._log(f"Loading generator model from {self.pth_path}")
            
            # Load model checkpoint
            cpt = torch.load(self.pth_path, map_location="cpu", weights_only=False)

            # Correct the number of speakers (n_spk) based on weight
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

            # Determine if F0 information is used
            self.use_f0 = bool(cpt.get("f0", 1))

            # Instantiate voice synthesizer
            if self.use_f0 == True:
                self.voice_synthesizer = VoiceSynthesizer(*cpt["config"], is_half=True)
            else:
                self.voice_synthesizer = VoiceSynthesizerNoF0(*cpt["config"])

            # Remove unneeded encoder
            del self.voice_synthesizer.enc_q

            # Load model weights
            self.voice_synthesizer.load_state_dict(cpt["weight"], strict=False)

            # Prepare for inference
            self.voice_synthesizer = self.voice_synthesizer.to(self.device).half().eval()
        except Exception as e:
            self._log(f"Failed to load generator model: {e}")
            raise
    
    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert frequency (Hz) to Mel scale."""
        return 1127 * np.log(1 + hz / 700)
    
    def start_microphone_inference(self, device_idx=None, block_duration=0.1):
        import queue
        import threading
        import sounddevice as sd
        import librosa

        input_q = queue.Queue()
        output_q = queue.Queue()

        block_size_16k = int(self.sampling_rate_input * block_duration)
        block_size_44k = int(self.sampling_rate_output * block_duration)

        # 入力デバイスのサンプリングレートを確認
        input_sr = int(sd.query_devices(device_idx, 'input')['default_samplerate'])
        self._log(f"input_sr : {input_sr}")

        def callback(indata, frames, time, status):
            if status:
                print(f"[RVCModel] Input status: {status}")
            audio = indata[:, 0].copy()
            if input_sr != self.sampling_rate_input:
                audio = librosa.resample(audio, orig_sr=input_sr, target_sr=self.sampling_rate_input)
            audio = highpass_filter(audio, cutoff=360, sr=self.sampling_rate_input)
            input_q.put(audio)
            
        from scipy.signal import butter, lfilter

        def highpass_filter(waveform: np.ndarray, cutoff=120, sr=44100, order=4) -> np.ndarray:
            nyq = 0.5 * sr
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            filtered = lfilter(b, a, waveform)
            return filtered  
        
        def lowpass_filter(waveform: np.ndarray, cutoff=1200, sr=44100, order=4) -> np.ndarray:
            nyq = 0.5 * sr
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered = lfilter(b, a, waveform)
            return filtered  

        def processing_loop():
            while True:
                audio_block = input_q.get()
                audio_tensor = torch.from_numpy(audio_block).float()
                with torch.no_grad():
                    output_tensor = self.infer(audio_tensor)
                output_audio = output_tensor.cpu().numpy()
                output_filtered = highpass_filter(output_audio, cutoff=500, sr=self.sampling_rate_output)
                output_filtered = lowpass_filter(output_filtered, cutoff=4000, sr=self.sampling_rate_output)
                output_q.put(output_filtered)

        def output_loop(out_stream):
            while True:
                out_block = output_q.get()
                out_block = out_block[: block_size_44k]  # trim if needed
                out_stream.write(out_block.astype(np.float32).reshape(-1, 1))

        with sd.InputStream(
            samplerate=input_sr,
            channels=1,
            dtype='float32',
            callback=callback,
            blocksize=block_size_16k,
            device=device_idx
        ) as in_stream, sd.OutputStream(
            samplerate=self.sampling_rate_output,
            channels=1,
            dtype='float32',
            blocksize=block_size_44k, 
            device=None
        ) as out_stream:
            
            print(f"[RVCModel] Real-time voice conversion started (block = {block_duration:.3f}s)...")
            threading.Thread(target=processing_loop, daemon=True).start()
            output_loop(out_stream)
        
    def get_f0(self, audio: np.ndarray, f0_up_key: float, inp_f0: Optional[np.ndarray] = None) -> np.ndarray | np.ndarray:
        """
        Estimate the fundamental frequency (F0) and create a coarse F0 representation.
        
        Args:
            audio (np.ndarray): Input audio waveform (float32 expected).
            f0_up_key (float): Pitch shift value (in semitones).
            inp_f0 (Optional[np.ndarray]): Optional externally provided F0 curve to replace part of estimation.

        Returns:
            f0_coarse (np.ndarray): Coarsely quantized F0 values (for model input).
            f0 (np.ndarray): Raw continuous F0 contour.
        """
        
        # Small time padding when applying input F0 (to avoid replacing the unstable first frame)
        padding_frames = 1

        # --- Step 1: F0 estimation using pyworld ---
        f0, time_axis = pyworld.harvest(
            audio.astype(np.double),
            fs=self.sampling_rate_input,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=self.frame_shift_ms,
        )

        # Refine F0 estimation
        f0 = pyworld.stonemask(audio.astype(np.double), f0, time_axis, self.sampling_rate_input)

        # Median filtering to smooth F0 curve
        f0 = signal.medfilt(f0, kernel_size=3)

        # Pitch shifting by f0_up_key (in semitones)
        # In music theory, each semitone shift corresponds to multiplying F0 by 2^(1/12)
        f0 *= 2 ** (f0_up_key / 12)

        # --- Step 2: Replace part of F0 with input F0 if provided ---
        frames_per_second = self.sampling_rate_input // self.frame_size  # Frame rate (frames/sec)

        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * frames_per_second + 1
            ).astype(np.int16)
            
            # Interpolate input F0 to match frame rate
            replace_f0 = np.interp(
                np.arange(delta_t),
                inp_f0[:, 0] * 100,
                inp_f0[:, 1]
            )
            
            # Replace the corresponding part of f0
            start = padding_frames * frames_per_second
            end = start + len(replace_f0)
            f0[start:end] = replace_f0[:f0[start:end].shape[0]]

        # Backup continuous F0 curve
        f0_continuous = f0.copy()

        # --- Step 3: Quantize F0 to coarse representation (1-255) ---
        f0_mel = self._hz_to_mel(f0)

        # Normalize mel scale to [1, 255]
        valid_idx = f0_mel > 0
        f0_mel[valid_idx] = (f0_mel[valid_idx] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel = np.clip(f0_mel, 1, 255)

        # Round to integer values
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        
        print(f"[Debug] f0 mean: {np.mean(f0)} max: {np.max(f0)}")

        return f0_coarse, f0_continuous, np.mean(f0)

    def infer(self, waveform: torch.Tensor, speaker_id: int = 0) -> np.ndarray:
        """
        Run inference on input audio features to generate a converted waveform.

        Args:
            waveform (torch.Tensor): Input waveform tensor (float32, 1D).

        Returns:
            np.ndarray: Output waveform array (float32).
        """
        assert waveform.dim() == 1, "Input waveform must be 1D"

        # Prepare input features
        feats = waveform.view(1, -1).half().to(self.device)
        padding_mask = torch.zeros_like(feats, dtype=torch.bool)

        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9,  # Hardcoded: use layer 9 output (This is the best for voice changing)
        }

        # Extract Hubert features
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0])

        # --- Optional: Feature refinement using FAISS index ---
        if self.index is not None and self.big_npy is not None and self.index_rate > 0:
            feats_np = feats[0].cpu().numpy().astype(np.float32)

            # HACK: Properly rebuild the FAISS index for 256-dimensional features.
            # Currently, feature vectors are zero-padded from 256 to 768 dimensions
            # to forcibly match the existing FAISS index trained with 768-dim features.
            # This is a temporary hack and may degrade audio quality.
            if feats_np.shape[1] != self.index.d:
                padding_dim = self.index.d - feats_np.shape[1]
                feats_np = np.pad(feats_np, ((0, 0), (0, padding_dim)), mode='constant')
                self._log(f"Feature dimension padded from {feats_np.shape[1] - padding_dim} to {feats_np.shape[1]}")

            distances, neighbor_indices = self.index.search(feats_np, k=8)
            weights = np.square(1.0 / distances)
            weights /= np.sum(weights, axis=1, keepdims=True)
            weighted_neighbors = np.sum(self.big_npy[neighbor_indices] * weights[:, :, None], axis=1)

            # HACK: Truncates back to 256 dimensions
            weighted_neighbors = weighted_neighbors[:, :256]

            refined_feats = torch.from_numpy(weighted_neighbors.astype(np.float16)).unsqueeze(0).to(self.device)

            feats = feats * (1.0 - self.index_rate) + refined_feats * self.index_rate
            
        else:
            if self.done_faiss_log_output == False:
                self._log("FAISS index search disabled or unavailable.")
                self.done_faiss_log_output = True
            
        # --- Feature upsampling ---
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # --- Pitch extraction ---
        if self.use_f0 == True:
            audio_np = waveform.cpu().numpy()
            pitch_coarse, pitch_continuous, f0_mean = self.get_f0(audio_np, self.f0_up_key)
            
            max_len = min(feats.shape[1], 13000, pitch_coarse.shape[0])
            
            if f0_mean < 20:
                return torch.zeros(max_len * 2, dtype=torch.float32)
        else:
            pitch_coarse, pitch_continuous = None, None
            max_len = min(feats.shape[1], 13000)

        # --- Trimming to match length ---
        feats = feats[:, :max_len, :]
        if self.use_f0 == True:
            pitch_coarse = pitch_coarse[:max_len]
            pitch_continuous = pitch_continuous[:max_len]
            pitch_coarse = torch.LongTensor(pitch_coarse).unsqueeze(0).to(self.device)
            pitch_continuous = torch.FloatTensor(pitch_continuous).unsqueeze(0).to(self.device)

        feature_length = torch.LongTensor([max_len]).to(self.device)
        speaker_id_tensor = torch.LongTensor([speaker_id]).to(self.device) 

        # --- Run the synthesizer ---
        with torch.no_grad():
            if self.use_f0 == True:
                output = self.voice_synthesizer.infer(feats, feature_length, pitch_coarse, pitch_continuous, speaker_id_tensor)
            else:
                output = self.voice_synthesizer.infer(feats, feature_length, speaker_id_tensor)

        output_audio = output[0][0, 0].cpu().float()
        print(f"[Debug] feats.shape={feats.shape}, pitch_coarse.shape={pitch_coarse.shape}, max_len={max_len}")

        return output_audio
    

    def _log(self, message):
        print(f"[RVCModel] {message}")

class Config:
    def __init__(self) -> None:
        self.hubert_path: str = ""
        self.pth_path: str = ""
        self.index_path: str = ""
        self.npy_path: str = ""
        self.pitch: int = 12
        self.samplerate: int = 44100
        self.block_time: float = 1.0  # [s]
        self.buffer_num: int = 1
        self.threhold: int = -30
        self.crossfade_time: float = 0.08
        self.extra_time: float = 0.04
        self.I_noise_reduce = True
        self.O_noise_reduce = True
        self.index_rate = 0.0
        