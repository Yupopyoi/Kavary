# Kavary

[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)を用いた、リアルタイム音声変換の開発を目的としたリポジトリです。  
最終的には、Unity(C#)から、HTTP通信を用いてこれらのスクリプトを実行することになります。  

## How to Use

- Hugging Face で [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) を入手します
- hubert_base.pt は ```model``` ディレクトリの中に保存してください  
- 任意の RVC モデル (.pth) と、対応する index ファイル（あれば）も、同様に ```model``` ディレクトリの中に保存してください  

## wav ファイル変換

wavファイルの変換には、```src/cui_wav_test.py``` を用います。  

### wav_test 実行例

#### index を使用しない場合

```bash RunWavTest_without_index
python src/cui_wav_test.py --input_wav your_voice.wav --pth_name your_rvc_model.pth
```

#### index を使用する場合

```bash RunWavTest_with_index
python src/cui_wav_test.py --input_wav your_voice.wav --pth_name your_rvc_model.pth --index_name your_rvc_model.index --index_rate 0.5
```

### cui_wav_test.py 引数

| 引数名     | 省略可否 |  型  |   デフォルト値   |                          説明                        |
|:---------:|:-------:|:-----:|:--------------:|:----------------------------------------------------:|
|input_wav  | 不可　　 |  str  |                | 変換に使用する wav ファイルへの相対パス                 |
|output_wav | 可能 　　|  str  | rvc_output_wav | 変換後の wav ファイルの相対パス                        |
|hubert_name| 可能 　　|  str  | hubert_base.pt | hubertモデルの名称                                    |
|pth_name   | 不可 　　|  str  |                | 変換に用いる RVC モデルの名称（.pth）                   |
|index_name | 可能 　　|  str  | None           | indexファイルの名称 (省略時はindexを使用しない)         |
|pitch      | 可能 　　|  int  | 12             | ピッチの変換量（半音単位）, 男性->女性では 12           |
|index_rate | 可能 　　| float | 0              | index の反映割合 [0,1]                                |

> [!IMPORTANT]
> _name で終わる引数名については、```model``` ディレクトリより下の名称のみを記載してください。

## リアルタイム音声変換

リアルタイム音声変換のテストには、```src/cui_mic_test.py``` を用います。

### mic_test 実行例

#### 初回実行

```bash first_time
python src/cui_mic_test.py --pth_name your_rvc_model.pth --index_name your_rvc_model.index --index_rate 0.5
```

実行すると
> === Available Input Devices ===  
>[0] Microsoft サウンド マッパー - Input  
>[1] マイク (Realtek(R) Audio)  

のように、使用できる音声入力デバイスが出力されるので、整数をターミナルに入力して、デバイスを選択してください

#### ２回目以降

```bash second_time
python src/cui_mic_test.py --pth_name your_rvc_model.pth --index_name your_rvc_model.index --index_rate 0.5 --device 1
```

> [!TIP]
> ```--device``` により、デバイス選択をスキップすることができます

### cui_mic_test.py 引数

| 引数名     　| 省略可否 |  型  |   デフォルト値   |                          説明                        |
|:-----------:|:-------:|:-----:|:--------------:|:----------------------------------------------------:|
|device       | 可能 　　|  int  |                | 使用デバイス（省略時は後に選択を求められます）           |
|hubert_name  | 可能 　　|  str  | hubert_base.pt | hubertモデルの名称                                    |
|pth_name     | 不可 　　|  str  |                | 変換に用いる RVC モデルの名称（.pth）                   |
|index_name   | 可能 　　|  str  | None           | indexファイルの名称 (省略時はindexを使用しない)         |
|default_pitch| 可能 　　|  int  | 12             | ピッチの変換量（半音単位）, 男性->女性では 12           |
|index_rate   | 可能 　　| float | 0              | index の反映割合 [0,1]                                |

## Issues

- リアルタイム音声変換時に音が途切れる
  - バッファの量と音声処理の同期が取れていない？
- 話している時にノイズが発生する
- 遅延が生じる（```block_duration```の半分だけ遅延する）
  - 極力短くしたい

## ToDo

- 上記 Issues の改善
- rvc_executor.py のリファクタリング
