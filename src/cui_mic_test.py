import sounddevice as sd
from rvc_executor import RVCModel
import argparse

def list_input_devices():
    print("=== Available Input Devices ===")
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    for idx, d in enumerate(input_devices):
        print(f"[{idx}] {d['name']}")
    return input_devices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--default_pitch", type=int, default=12, help="Input device index")
    args = parser.parse_args()

    if args.device is None:
        input_devices = list_input_devices()
        index = int(input("Select input device index: "))
        device_info = input_devices[index]
        device_idx = sd.query_devices().index(device_info)
    else:
        device_idx = args.device
        print(f"Selected device index : {device_idx}")

    # 必要なモデル設定を指定
    model = RVCModel(
        key=args.default_pitch,  # ピッチ補正
        hubert_path="../model/hubert_base.pt",
        pth_path="../model/Miko.pth",
        index_path="../model/Miko.index",
        index_rate=0.0,
    )

    # マイク入力→変換→再生
    model.start_microphone_inference(device_idx=device_idx, block_duration=2)

if __name__ == "__main__":
    main()
