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

import math
import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from infer_pack import attentions
from infer_pack import commons
from infer_pack import modules
from infer_pack.commons import init_weights
from infer_pack.modules import ResidualCouplingLayer, Flip

class TextEncoder256(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        use_f0: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.use_f0 = use_f0
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        # NOTE:
        # The following layer names must NOT be changed without updating the corresponding keys
        # in the pretrained model's state_dict:
        # - self.proj           → required for "proj.weight" and "proj.bias"
        # - self.emb_phone      → required for "emb_phone.weight" and "emb_phone.bias"
        # - self.emb_pitch      → required for "emb_pitch.weight"
        # - self.encoder        → required for keys like "encoder.layers.0.attn.q_proj.weight"
        #
        # If you change these names, pretrained weights will not load correctly unless you
        # rename the keys in the state_dict before loading.
        
        self.emb_phone = nn.Linear(256, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        
        if use_f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def forward(
        self,
        phone_features: torch.Tensor,   # [B, T, 256]
        pitch_indices: torch.Tensor,    # [B, T] or None
        lengths: torch.Tensor           # [B]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phone_features: phoneme embeddings, shape [B, T, 256]
            pitch_indices: pitch indices (coarse), shape [B, T], or None
            lengths: input lengths, shape [B]

        Returns:
            mean, log_scale, attention_mask: tensors for flow input
        """
        
        # Embed phonemes and pitch if applicable
        if pitch_indices == None:
            features = self.emb_phone(phone_features)
        else:
            features = self.emb_phone(phone_features) + self.emb_pitch(pitch_indices)
            
        features = features * math.sqrt(self.hidden_channels)  # [b, t, h]
        features = self.lrelu(features)
        features = torch.transpose(features, 1, -1)  # [b, h, t]
        
        # Create attention mask
        attention_mask = torch.unsqueeze(commons.sequence_mask(lengths, features.size(2)), 1).to(
            features.dtype
        )
        features = self.encoder(features * attention_mask, attention_mask)
        
        # Project to mean & log-scale
        stats = self.proj(features) * attention_mask

        mean, logs = torch.split(stats, self.out_channels, dim=1)
        return mean, logs, attention_mask


class ResidualCouplingBlock(nn.Module):
    """
    A stack of invertible residual coupling layers and flip layers used in normalizing flows.
    Performs reversible transformations between latent variables and audio features.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0, # Global conditioning input
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        self.coupling_layers = [] # Coupling is only recorded explicitly
        
        for _ in range(n_flows):
            coupling = ResidualCouplingLayer(
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels=gin_channels,
                mean_only=True,
            )
            self.flows.append(coupling)
            self.flows.append(Flip())
            self.coupling_layers.append(coupling)

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
            x: input tensor [B, C, T]
            x_mask: mask tensor [B, 1, T]
            g: optional global conditioning tensor
            reverse: whether to apply the flow in reverse

        Returns:
            transformed tensor (same shape as input)
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x # transformed through all flows

    def remove_weight_norm(self):
        for coupling in self.coupling_layers:
            coupling.remove_weight_norm()


class PosteriorEncoder(nn.Module):
    """
    Posterior encoder for mapping acoustic features (e.g., mel) into latent z.
    Implements a VAE-style encoder that outputs mean and log-variance,
    then samples z = m + N(0, 1) * exp(logs).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        
        # NOTE: Do not rename `self.proj` — this name is tied to pretrained weight keys (e.g., "proj.weight")
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,           # [B, in_channels, T]
        x_lengths: torch.Tensor,   # [B]
        g: torch.Tensor = None     # optional global condition
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """
        Args:
            x: input acoustic features (e.g. mel)
            x_lengths: valid lengths for each sequence
            g: optional global conditioning

        Returns:
            z: sampled latent representation [B, C, T]
            m: mean of posterior [B, C, T]
            logs: log std dev [B, C, T]
            x_mask: binary mask [B, 1, T]
        """
        
        # Compute mask [B, 1, T]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        # Pre-network and encoding
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        
        # Compute posterior stats
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1) # mean & log-variance
        
        # Sample z using reparameterization trick
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        """Remove weight norm from the internal encoder layers."""
        self.enc.remove_weight_norm()


class Generator(torch.nn.Module):
    """
    HiFi-GAN style Generator used in VoiceSynthesizerNoF0.
    Converts latent representation z into waveform audio using 
    upsampling and residual blocks.
    """
    # NOTE:
    # This Generator class (HiFi-GAN style) is currently only used by VoiceSynthesizerNoF0.
    # Since it is not expected to be used actively in this project, it has not been refactored.
    # If reused in the future, refactoring for clarity, naming, and documentation is recommended.
    
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SineGenerator(torch.nn.Module):
    """
    Generate sine wave signals and unvoiced (UV) masks based on input F0.
    
    This module is commonly used in neural vocoders to synthesize source excitation
    from F0 and harmonic structure, optionally adding noise for unvoiced segments.

    Args:
        samp_rate (int): Sampling rate [Hz]
        harmonic_num (int): Number of harmonic overtones to add
        sine_amp (float): Amplitude of sine signal
        noise_std (float): Standard deviation of noise
        voiced_threshold (float): F0 threshold to classify voiced/unvoiced
        flag_for_pulse (bool): (Unused) Intended for compatibility with PulseGen
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,  # Currently unused
    ):
        super(SineGenerator, self).__init__()
        self.sampling_rate = samp_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.dim = self.harmonic_num + 1 # fundamental + overtones
        self.voiced_threshold = voiced_threshold
        self.unvoiced_noise_ratio = 1 / 3  

    def _f0_to_uv(self, f0):
        """Generate a voiced/unvoiced mask from F0"""
        return (f0 > self.voiced_threshold).float().unsqueeze(-1)  # [B, T, 1]

    def _expand_f0_harmonics(self, f0: torch.Tensor) -> torch.Tensor:
        """Expand F0 into fundamental and harmonic components [B, T] → [B, T, dim]"""
        B, T = f0.shape
        f0 = f0[:, None].transpose(1, 2)  # [B, 1, T] → [B, T, 1]
        f0_harmonics = torch.zeros(B, T, self.dim, device=f0.device)
        f0_harmonics[:, :, 0] = f0[:, :, 0]
        
        # Fill in harmonic frequencies: 2nd, 3rd, ..., (harmonic_num+1)-th harmonics
        for harmonic_order in range(self.harmonic_num):
            f0_harmonics[:, :, harmonic_order + 1] = f0[:, :, 0] * (harmonic_order + 2)
            
        return f0_harmonics

    def _compute_phase(self, f0_buf: torch.Tensor, upsample_factor: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized phase increment and interpolated cumulative phase [B, T, dim] → [B, T', dim]"""
        rad = (f0_buf / self.sampling_rate) % 1  # [B, T, dim]

        rand_phase = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
        rand_phase[:, 0] = 0  # Fundamental wave is phase-fixed
        rad[:, 0, :] += rand_phase

        cum_phase = torch.cumsum(rad, dim=1) * upsample_factor
        phase_interp = F.interpolate(cum_phase.transpose(2, 1), scale_factor=upsample_factor, mode="linear", align_corners=True).transpose(2, 1)
        rad_interp = F.interpolate(rad.transpose(2, 1), scale_factor=upsample_factor, mode="nearest").transpose(2, 1)
        return rad_interp, phase_interp

    def _generate_sine(self, rad_interp: torch.Tensor, phase_interp: torch.Tensor) -> torch.Tensor:
        """Generate sine wave from interpolated phase"""
        
        # Normalize phase into [0, 1) range to simulate periodic waveform behavior.
        phase_wrapped = phase_interp % 1
        
        # Detect where phase wraps from ~1.0 to ~0.0 (phase reset points)
        reset_idx = (phase_wrapped[:, 1:, :] - phase_wrapped[:, :-1, :]) < 0
        
        cumsum_shift = torch.zeros_like(rad_interp)
        cumsum_shift[:, 1:, :] = reset_idx * -1.0
        full_phase = torch.cumsum(rad_interp + cumsum_shift, dim=1)
        
        sine = torch.sin(full_phase * 2 * np.pi) * self.sine_amp
        return sine

    def forward(self, f0: torch.Tensor, upsample_factor: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            f0_buf = self._expand_f0_harmonics(f0)  # [B, T, dim]
            rad_interp, phase_interp = self._compute_phase(f0_buf, upsample_factor)
            sine = self._generate_sine(rad_interp, phase_interp)

            # Voiced/unvoiced mask
            uv = self._f0_to_uv(f0)
            uv = F.interpolate(uv.transpose(2, 1), scale_factor=upsample_factor, mode="nearest").transpose(2, 1)

            # Noise
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp * self.unvoiced_noise_ratio
            noise = noise_amp * torch.randn_like(sine)

            sine_out = sine * uv + noise

        return sine_out, uv, noise


class HarmonicExcitationSource(torch.nn.Module):
    """
    Generate excitation signal from F0 using harmonics + sine + tanh shaping.
    This is used as the source module for HN-NSF-style vocoders.
    The class name in the original program was "SourceModuleHnNSF".

    Args:
        sampling_rate (int): Audio sampling rate in Hz
        harmonic_num (int): Number of harmonic components above F0
        sine_amp (float): Amplitude of the sine wave component
        add_noise_std (float): Standard deviation of additive Gaussian noise
        voiced_threshold (float): F0 threshold to detect voiced/unvoiced
        use_half_precision (bool): Whether to cast output to float16
    """
    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
        use_half_precision=True,
    ):
        super(HarmonicExcitationSource, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.use_half_precision = use_half_precision
        # to produce sine waveforms
        self.sine_generator = SineGenerator(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        # to merge source harmonics into a single excitation
        # NOTE:
        # Do not rename `l_linear` or `l_tanh`.
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        sine_wavs, uv, _ = self.sine_generator(x, upp)
        
        if self.use_half_precision:
            sine_wavs = sine_wavs.half()
            
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


class NSFGenerator(torch.nn.Module):
    """
    NSF-based waveform generator.
    Combines harmonic excitation + noise + upsampling + residual blocks
    to synthesize waveform from input acoustic features and F0.
    """
    # NOTE:
    # Due to the complexity of this class, only variable renaming was applied in this refactor.
    # Full structural cleanup (e.g., method decomposition) is deferred for future work.
    
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sampling_rate,
        use_half_precision=False,
    ):
        super(NSFGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = HarmonicExcitationSource(
            sampling_rate=sampling_rate, harmonic_num=0, use_half_precision=use_half_precision
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
        )
        
        resblock_module = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        # NOTE:
        # Do not rename `self.ups` 
        self.ups = nn.ModuleList()
        
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_module(ch, k, d))

        self.conv_post = Conv1d(ch, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upsample_factor = np.prod(upsample_rates)

    def forward(self, x, f0, g=None):
        har_source, noise_source, uv = self.m_source(f0, self.upsample_factor)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    
            x = xs / self.num_kernels
            
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()


class VoiceSynthesizer(nn.Module):
    
    """
    Full voice synthesis module with encoder-flow-decoder architecture.
    
    Converts phonetic inputs and pitch into waveform using:
        - Pitch-conditioned text encoder (enc_p)
        - Posterior encoder from spectrogram (enc_q)
        - Residual flow module (flow)
        - NSF-based generator (dec)
    """
    
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        **kwargs
    ):
        super().__init__()
        if type(sr) == type("strr"):
            sr2sr = {
                "32k": 32000,
                "40k": 40000,
                "48k": 48000,
            }
            sr = sr2sr[sr]
            
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        
        # NOTE:
        # Do not rename the following module attributes (enc_p, enc_q, flow, dec, emb_g),
        # as their names must match keys in pretrained state_dict for model loading.
        
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        
        self.dec = NSFGenerator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sampling_rate=sr,
            use_half_precision=kwargs["is_half"],
        )
        
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, 
            hidden_channels, 
            kernel_size=5, 
            dilation_rate=1, 
            n_layers=3, 
            gin_channels=gin_channels
        )
        
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def infer(self, phone, phone_lengths, pitch, nsff0, speaker_id, max_len=None):
        noise_scale = 0.66666  # Controls randomness during inference
        g = self.emb_g(speaker_id).unsqueeze(-1)
        
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        output_waveform = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return output_waveform, x_mask, (z, z_p, m_p, logs_p)


class VoiceSynthesizerNoF0(VoiceSynthesizer):
    
    # NOTE:
    # This class has not been tested yet.
    # Verify functionality before use.
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc_p = TextEncoder256(
            self.inter_channels,
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
            use_f0=False
        )

        # Replace NSF-based decoder as it is not needed (former Generator)
        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )

    # Overrides infer() from VoiceSynthesizer to remove F0 conditioning
    def infer(self, phone, phone_lengths, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        noise_scale = 0.66666
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        output_waveform = self.dec((z * x_mask)[:, :, :max_len], g=g)
        return output_waveform, x_mask, (z, z_p, m_p, logs_p)
