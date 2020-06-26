# CustopmVisionでExportしたモデルをTensorflowLite用に変換するサンプル

変換スクリプトはCPU用モデルとEdge-TPU用モデルを出力する。  
ファイル名本体末尾に``_edgetpu``付加されたファイルがEdge-TPU用モデルファイル  

## ディレクトリ構成

| ディレクトリ         | 内容                                              |
|----------------------|---------------------------------------------------|
| convert_model        | モデル変換処理                                    |  
| tflite_classification.py |画像分類処理プログラム                         |    

## 画像分類処理プログラム実行方法

オプション指定時、``ModuleNotFoundError:``でエラー終了する場合がある。  
この場合、オプションとオプションパラメータの間をスペースでなく、``=``にするとうまくいくようだ。  
例： ``--input fuga.jpg`` → ``--input=fuga.jpg``

### オプション
```
usage: tflite_classification.py [-h] [-m MODEL] [-i INPUT] [--labels LABELS]
                                [--disp]

optional arguments:
  -h, --help            Show this help message and exit.

Input Options:
  -m MODEL, --model MODEL
                        Optional.
                        Path to an .tflite file with a trained model.
  -i INPUT, --input INPUT
                        Optional.
                        Path to a image file. 
  --labels LABELS       Optional.
                        Labels mapping file

Output Options:
  --disp                Optional.
                        image preview
```


| オプション           | 内容                                              |
|----------------------|---------------------------------------------------|
| -h<br>--help | ヘルプ表示        |
| -m MODEL<br>--model MODEL | モデルファイル(.tfliteファイルを指定する)<br> 省略時は``convert_model/converted_model.tflite`` |
| -i INPUT<br>--input INPUT | 入力画像ファイル <br> 省略時は``../jpeg/face0.jpg`` |
| --labels LABELS  | ラベルファイル<br>省略時はモデルファイルの拡張子を``.labels``に変更したもの  |
| --disp                   | 確認用に入力画像を表示する<br>省略時は表示しない |

### 注意

TensorflowLite(2.1.0で確認)が使用できるpythonの実行環境で実行すること。  
Tensorflowでも大丈夫(1.15で確認)なように修正入れといた。

## 参考
TensorflowLiteのインストールについては↓こちら  
[Google Coral USB Accelerator を使う その1](https://ippei8jp.github.io/memoBlog/2020/05/15/coral_1.html)


Tensorflow1.15のインストールについては↓こちら  
[Google Coral USB Accelerator を使う その5](https://ippei8jp.github.io/memoBlog/2020/05/27/coral_5.html)



