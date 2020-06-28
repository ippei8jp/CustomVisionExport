# 【参考】
# https://www.tensorflow.org/lite/performance/post_training_quantization

import sys
import os
import glob
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import tensorflow as tf
import cv2
import numpy as np

def build_argparser():
    parser = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                        help='Show this help message and exit.')
    parser.add_argument("-i", "--input", type=str, default="../../SampleExports/TensorFlow/model.pb",
                        help="入力ファイル(pbファイル)")
    parser.add_argument("-o", "--output", type=str, default="./converted_model.tflite",
                        help="出力ファイル(tfliteファイル)")
    parser.add_argument("-d", "--data_dir", type=str, default="./data",
                        help="キャリブレーションデータディレクトリ")
    parser.add_argument('--quantize', nargs='*',
                        help="量子化種別  <weight, integer>")
    parser.add_argument('--input_tensors', type=str, default="Placeholder",
                        help="量子化種別  <weight, integer>")
    parser.add_argument('--output_tensors', type=str, default="model_outputs",
                        help="量子化種別  <weight, integer>")
    return parser

# キャリブレーション用データ生成ルーチン
def representative_dataset_gen():
    # グローバル変数の宣言
    global data_dir
    global input_width
    global input_height

    # ファイル名のリストを作成
    files = glob.glob(os.path.join(data_dir,'**/*.jpg'), recursive=True)

    for filename in files :
        input_file = os.path.abspath(filename)
        assert os.path.isfile(input_file), "Specified input file doesn't exist"
        
        # 画像の読み込み
        image = cv2.imread(input_file)
        # NNに入力できる形式に変換
        resized_image = cv2.resize(image, (input_width, input_height))      # input size of coco ssd mobilenet?
        resized_image = np.expand_dims(resized_image, axis=0)               # 3D -> 4D
        resized_image = resized_image.astype(np.float32)                    # 型変換 uint8 -> float32 
        
        yield [resized_image]                                               # list にwrapして返す

def main():
    # グローバル変数の宣言
    # キャリブレーション用データ生成ルーチンで参照したいのでglobalにしておく
    global data_dir
    global input_width
    global input_height
    
    # コマンドライン引数の解析
    args = build_argparser().parse_args()
    # print(args)
    quantize = args.quantize
    input_tensors  = args.input_tensors
    output_tensors = args.output_tensors
    # オプション quantize が指定されていなければ空のlistを作成
    if not quantize :
        quantize = []
    
    # グローバル変数の設定
    data_dir = args.data_dir
    input_width  = 224
    input_height = 224
    
    tf.compat.v1.enable_eager_execution()
    
    # Full Integer Quantization - Input/Output=int8
    # export_model = os.path.splitext(args.output)[0] + "_full_integer_quant" + os.path.splitext(args.output)[1]  # 出力ファイル名
    export_model = args.output
    # converter = tf.lite.TFLiteConverter.from_saved_model(args.input)
    converter = tf.lite.TFLiteConverter.from_frozen_graph(args.input, [input_tensors], [output_tensors])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    open(export_model, "wb").write(tflite_quant_model)
    print("==== Full Integer Quantization complete! ====")
    
    # if "weight" in quantize :
    if True :
        # Weight Quantization - Input/Output=float32
        export_model = os.path.splitext(args.output)[0] + "_weight_quant" + os.path.splitext(args.output)[1]        # 出力ファイル名
        # converter = tf.lite.TFLiteConverter.from_saved_model(args.input)
        converter = tf.lite.TFLiteConverter.from_frozen_graph(args.input, [input_tensors], [output_tensors])
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_quant_model = converter.convert()
        open(export_model, "wb").write(tflite_quant_model)
        print("==== Weight Quantization complete! ====")
    
    # if "integer" in quantize :
    if True :
        # Integer Quantization - Input/Output=float32
        export_model = os.path.splitext(args.output)[0] + "_integer_quant" + os.path.splitext(args.output)[1]       # 出力ファイル名
        # converter = tf.lite.TFLiteConverter.from_saved_model(args.input)
        converter = tf.lite.TFLiteConverter.from_frozen_graph(args.input, [input_tensors], [output_tensors])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        tflite_quant_model = converter.convert()
        open(export_model, "wb").write(tflite_quant_model)
        print("==== Integer Quantization complete! ====")

if __name__ == '__main__':
    sys.exit(main() or 0)

# converter.representative_dataset で設定する関数の書き方

'''
saved_model_dir="./TensorFlowSavedModel"
export_model = "converted_model.tflite"
export_model_size = "converted_model_size.tflite"

print("**** DEFAULT ****")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open(export_model, "wb").write(tflite_quant_model)

print("**** SIZE ****")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open(export_model_size, "wb").write(tflite_quant_model)


'''
