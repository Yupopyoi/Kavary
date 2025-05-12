# MIT License
#
# Copyright (c) 2025 Yupopyoi
#
# This test script is also released under the MIT License.
# See the LICENSE file for more details.

"""
python src/cui_wav_test.py --input your_voice.wav --pth_name your_rvc_model.pth
"""

import argparse
import librosa
import os
import pathlib
import soundfile
import torch

from rvc_executor import RVCModel

def main():
    work_directory = pathlib.Path(__file__).parent.parent # work directory of "Kavary"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', type=str, required=True)
    parser.add_argument('--output_wav', type=str, default= "rvc_output.wav")
    parser.add_argument('--hubert_name', type=str, default= "hubert_base.pt")
    parser.add_argument('--pth_name', type=str, required=True)
    parser.add_argument('--index_name', type=str, default="None")
    parser.add_argument('--pitch', type=int, default=12)
    parser.add_argument('--index_rate', type=float, default=0.0)
    args = parser.parse_args()
    
    print(f"\n[CUI_WAVE_TEST] Welcome to CUI_WAVE_TEST! : {args.pth_name}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load RVC Model
    rvc = RVCModel(args.pitch, 
                   os.path.join(work_directory, "model", args.hubert_name), 
                   os.path.join(work_directory, "model", args.pth_name), 
                   os.path.join(work_directory, "model", args.index_name), 
                   args.index_rate)

    # Load wav File
    wav, _ = librosa.load(args.input_wav, sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().to(device)
    
    # Inference with RVC
    infered_audio = rvc.infer(wav_tensor)

    # Resample 16000 → 44100 and save it as wav flie
    soundfile.write(args.output_wav, infered_audio.cpu().numpy(), 44100)
    
    print(f"\n\033[32m[CUI_WAVE_TEST] The wav file was output successfully! : {args.output_wav}\n\033[0m")

if __name__ == '__main__':
    main()
