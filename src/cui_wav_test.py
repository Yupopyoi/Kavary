# MIT License
#
# Copyright (c) 2025 Yupopyoi
#
# This test script is also released under the MIT License.
# See the LICENSE file for more details.

import argparse
import librosa
import soundfile
import torch

from rvc_executor import RVCModel

"""

python cui_wav_test.py --input test_input.wav --output test_output.wav --hubert_path ../model/hubert_base.pt --pth_path ../model/Gura.pth --index_path ../model/Gura.index --pitch 12 --index_rate 0.0
python cui_wav_test.py --input test_input.wav --output test_output.wav --hubert_path ../model/hubert_base.pt --pth_path ../model/Miko.pth --index_path ../model/Miko.index --pitch 12 --index_rate 0.0
python cui_wav_test.py --input test_input.wav --output test_output.wav --hubert_path ../model/hubert_base.pt --pth_path ../model/Pekora.pth --index_path ../model/Pekora.index --pitch 12 --index_rate 0.0

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--hubert_path', type=str, required=True)
    parser.add_argument('--pth_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--pitch', type=int, default=12)
    parser.add_argument('--index_rate', type=float, default=0.0)
    args = parser.parse_args()
    
    print(f"\n[CUI_WAVE_TEST] Welcome to CUI_WAVE_TEST! : {args.pth_path}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load RVC Model
    rvc = RVCModel(args.pitch, args.hubert_path, args.pth_path, args.index_path, args.index_rate)

    # Load wav File
    wav, _ = librosa.load(args.input, sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().to(device)
    
    # Inference with RVC
    infered_audio = rvc.infer(wav_tensor)

    # Resample 16000→44100 and save it as wav flie
    soundfile.write(args.output, infered_audio.cpu().numpy(), 44100)
    
    print(f"\n\033[32m[CUI_WAVE_TEST] The wav file was output successfully! : {args.output}\n\033[0m")

if __name__ == '__main__':
    main()
