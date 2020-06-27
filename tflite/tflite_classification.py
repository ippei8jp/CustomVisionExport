#!/usr/bin/env python
import sys
import os
import time
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import platform
import math
import cv2
import numpy as np

try :
    # tensorflow_lite からロードしてみる
    import tflite_runtime.interpreter as tflite
    tflite_Interpreter = tflite.Interpreter
    tflite_load_delegate = tflite.load_delegate
except ModuleNotFoundError :
    # だめだったら tensorflow からロードしてみる
    print('tflite_runtime.interpreter がインポートできませんでした.')
    print('代わりに tensorflow.lite のインポートを試みます.')
    import tensorflow.lite as tflite
    # ビミョーに階層違うので、ここで差分吸収
    tflite_Interpreter = tflite.Interpreter
    tflite_load_delegate = tflite.experimental.load_delegate

# shared library
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

# コマンドラインパーサの構築 =====================================================
def build_argparser():
    parser = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    input_args = parser.add_argument_group('Input Options')
    output_args = parser.add_argument_group('Output Options')
    exec_args = parser.add_argument_group('Execution Options')
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                        help='Show this help message and exit.')
    input_args.add_argument("-m", "--model", type=str, default="convert_model/frozen_model/converted_model.tflite",
                        help="Optional.\n"
                             "Path to an .tflite file with a trained model.")
    input_args.add_argument("-i", "--input", type=str, default="../jpeg/face0.jpg",
                        help="Optional.\n"
                             "Path to a image file. \n")
    input_args.add_argument("--labels", type=str, default=None, 
                        help="Optional.\n"
                             "Labels mapping file."
                             "Default is to change the extension of the modelfile\n"
                             "to '.labels'.")
    output_args.add_argument("--disp", action='store_true', 
                        help="Optional.\n"
                             "image preview")
    return parser


# interpreter の生成
def make_interpreter(model_file):
    # CPU/TPU使用の識別
    # 「ファイル名に"_edgetpu"が含まれていたら」の識別方法もアリかもしれない
    with open(model_file, "rb") as f:
        # モデルデータを読み込む
        tfdata = f.read()
        # モデルファイル中に"edgetpu-custom-op"が含まれていたらTPU使用モデル
        cpu = not b"edgetpu-custom-op" in tfdata
    
    if cpu :
        print('**** USE CPU ONLY!! ****')
    else :
        print('**** USE WITH TPU ****')

    if cpu :
        return tflite_Interpreter(model_path=model_file)
    else :
        return tflite_Interpreter(
                model_path = model_file,
                experimental_delegates = [
                    tflite_load_delegate(EDGETPU_SHARED_LIB)
                ])

def main():
    args = build_argparser().parse_args()
    
    # model = args.model
    
    # 入力ファイル
    input_file = os.path.abspath(args.input)
    assert os.path.isfile(input_file), "Specified input file doesn't exist"
    
    '''
    # ラベルファイル
    labels_file = None
    if args.labels:
        if os.path.isfile(args.labels) :
            labels_file = args.labels
    
    labels_map = None
    if labels_file:
        # ラベルファイルの読み込み
        with open(labels_file, 'r') as f:
            labels_map = [x.strip() for x in f]
    
    # print(labels_map)
    '''
        # ラベルファイル
    labels_file = None
    if args.labels :
        labels_file = args.labels
    else :
        labels_file = os.path.splitext(args.model)[0] + ".labels"

    if not os.path.isfile(labels_file) :
        print(f"[WARNING] label file is not exist : {labels_file}")
        labels_file = None
    
    labels_map = None
    if labels_file:
        # ラベルファイルの読み込み
        with open(labels_file, 'r') as f:
            print(f"[INFO] Loading label file: {labels_file}")
            labels_map = [x.strip() for x in f]

    
    # interpreterの構築
    print("[INFO] Creating interpreter...")
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    
    # 入力レイヤの情報
    input_details = interpreter.get_input_details()
    # モデルの入力サイズ
    _, input_height, input_width, _ = input_details[0]['shape']
    
    # モデルの入力データ型
    input_dtype = input_details[0]['dtype']
    # print(f'input_dtype = {input_dtype}')
    
    # 推論開始
    print("[INFO] Starting inference...")

    for loop_cnt in range(10) :         # 2回目以降速いかな？そんなことはなかった... でも1割くらいは違ってる...
        print(f'==== LOOP COOUNT {loop_cnt} ====')
        # 画像の読み込み
        frame = cv2.imread(input_file)
        # 幅と高さを取得
        img_height, img_width, _ = frame.shape
        if args.disp :
            cv2.imshow("INPUT IMAGE", frame)
        
        # モデル入力用にリサイズ
        in_frame = cv2.resize(frame, (input_width, input_height))   # モデルの入力サイズに画像をリサイズ
        in_frame = in_frame[:, :, [2,1,0]]                          # BGR -> RGB
        in_frame = np.expand_dims(in_frame, axis=0)                 # 3D -> 4D
        if input_dtype != np.uint8 :
            in_frame = in_frame.astype(input_dtype)                 # モデルの入力型がuint8以外だったら型変換
    
        # 推論実行 =============================================================================
        inf_start = time.time()                                 # 推論処理開始時刻          --------------------------------
        # 推論本体
        interpreter.set_tensor(input_details[0]['index'], in_frame)
        interpreter.invoke()
        inf_end = time.time()                               # 推論処理終了時刻          --------------------------------
        inf_time = inf_end - inf_start                      # 推論処理時間
        
        # 検出結果の解析 =============================================================================
        output_details = interpreter.get_output_details()
        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
        
        tflite_results1 = interpreter.get_tensor(output_details[0]['index'])  # Locations (Top, Left, Bottom, Right)
        tflite_result = tflite_results1[0]

        result_dtype = type(tflite_result[0])
        if tflite_result.dtype == np.uint8 :
            # QINT8モデルなら
            tflite_result = tflite_result.astype(np.float32)                # 入力型がuint8だったらfloat32に型変換
            tflite_result = tflite_result / 256                             # -1～1の値に正規化
        
        max_index = tflite_result.argmax()
        if labels_map :
            max_label = labels_map[max_index]
        else :
            max_label = f'Index={max_index}'
        max_score = tflite_result[max_index] * 100
        print(f'****RESULT****')
        # 結果
        print(f'     {max_label} : {max_score:.2f}%     ({tflite_results1})')
        # 測定データの表示
        print(f'     Inference time : {(inf_time * 1000):.3f} ms')

    if args.disp :
        cv2.waitKey(0)

if __name__ == '__main__':
    sys.exit(main() or 0)
