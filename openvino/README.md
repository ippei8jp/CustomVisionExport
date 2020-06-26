# CustopmVisionでExportしたモデルをopenVINO用に変換するサンプル

## ディレクトリ構成

| ディレクトリ         | 内容                                              |
|----------------------|---------------------------------------------------|
| convert_model        | モデル変換処理                                    |  
| ov_classification.py |画像分類処理プログラム                             |    

## 画像分類処理プログラム実行方法

オプション指定時、``ModuleNotFoundError:``でエラー終了する場合がある。  
この場合、オプションとオプションパラメータの間をスペースでなく、``=``にするとうまくいくようだ。  
例： ``--input fuga.jpg`` → ``--input=fuga.jpg``

### オプション
```
usage: ov_classification.py [-h] [-m MODEL] [-i INPUT] [--labels LABELS]
                            [-d DEVICE] [-l CPU_EXTENSION] [--disp]

optional arguments:
  -h, --help            Show this help message and exit.

Input Options:
  -m MODEL, --model MODEL
                        Optional.
                        Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Optional.
                        Path to a image file.
  --labels LABELS       Optional.
                        Labels mapping file.Default is to change the extension of the modelfile
                        to '.labels'.
  -d DEVICE, --device DEVICE
                        Optional
                        Specify the target device to infer on; 
                        CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.
                        The demo will look for a suitable plugin 
                        for device specified.
                        Default value is CPU
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional.
                        Required for CPU custom layers. 
                        Absolute path to a shared library
                        with the kernels implementations.
                        以前はlibcpu_extension_avx2.so 指定が必須だったけど、
                        2020.1から不要になった

Output Options:
  --disp                Optional.
                        image display
```


| オプション           | 内容                                              |
|----------------------|---------------------------------------------------|
| -h<br>--help | ヘルプ表示        |
| -m MODEL<br>--model MODEL | モデルファイル(.xmlファイルを指定する)<br> 省略時は``convert_model/converted_model.xml`` |
| -i INPUT<br>--input INPUT | 入力画像ファイル <br> 省略時は``../jpeg/face0.jpg`` |
| --labels LABELS  | ラベルファイル<br>省略時はモデルファイルの拡張子を``.labels``に変更したもの  |
| -d DEVICE<br>--device DEVICE | デバイス<br> CPU MYRIAD(またはNCS)のいずれかを指定<b>    省略時は``CPU``   |
| -l CPU_EXTENSION<br>--cpu_extension CPU_EXTENSION | CPU拡張ライブラリ<br>以前は``libcpu_extension_avx2.so`` 指定が必須だったけど<br>2020.1から不要になった |
| --disp                   | 確認用に入力画像を表示する<br>省略時は表示しない |

### 注意

openVINOが使用できるpythonの実行環境で実行すること。


## 参考
openVINOのインストールについては↓こちら  
[openVINO フルパッケージをubuntuにインストール(改訂版)](https://ippei8jp.github.io/memoBlog/2020/06/16/openVINO_ubuntu_2.html)
