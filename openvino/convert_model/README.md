# CustopmVisionでExportしたモデルをopenVINO用に変換するサンプル

変換処理本体 ＆ 変換モデルデータ格納
変換したモデルはCPU/NCS共通。

## ディレクトリ構成

| ディレクトリ          | 内容                                              |
|-----------------------|---------------------------------------------------|
| run.sh                | 変換処理実行スクリプト                            |  

## 変換処理プログラム実行方法

run.shの以下のパラメータを変更して、run.shを実行する

| 変数          | 内容                                              |
|---------------|---------------------------------------------------|
| INPUT_DIR=    | 入力モデルファイルがあるディレクトリ              |
| INPUT_FILE    | 入力モデルファイル **拡張子を付ける**             | 
| INPUT_SHAPE   | 入力ノードのサイズ  おそらく変更不要              |
| OUTPUT_DIR    | 変換したモデルファイルを出力するディレクトリ      | 
| OUTPUT_FILE   | 変換したモデルファイルの名前 **拡張子は付けない** |


openVINOが使用できるpythonの実行環境で実行すること。

