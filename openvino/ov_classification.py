#!/usr/bin/env python
import sys
import os
import time
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import cv2
import numpy as np

# 環境変数設定スクリプトが実行されているか確認
if not "INTEL_OPENVINO_DIR" in os.environ:
    print("source /opt/intel/openvino/bin/setupvars.sh が 実行されていないようです")
    sys.exit(1)
else:
    # 環境変数を取得するには os.environ['INTEL_OPENVINO_DIR']
    # これを設定されてない変数に対して行うと例外を吐くので注意
    pass

from openvino.inference_engine import get_version as ov_get_version

# openvino.inference_engine のバージョン取得
ov_vession = ov_get_version()
# print(ov_vession)                 # バージョン2019には '2.1.custom_releases/2019/R3_～'という文字列が入っている
                                    # バージョン2020には '2.1.2020.3.0-～'という文字列が入っている

# バージョン2019か？
flag_ver2019 = False
if "/2019/R" in ov_vession :
    # YES!
    flag_ver2019 = True


# IENetwork() での読み込みは非推奨になったので削除
# from openvino.inference_engine import IENetwork, IECore
from openvino.inference_engine import IECore

# IENetwork() での読み込みは非推奨になったのでバージョン2020以降では削除
if flag_ver2019:
    from openvino.inference_engine import IENetwork


# コマンドラインパーサの構築 =====================================================
def build_argparser():
    parser = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    input_args = parser.add_argument_group('Input Options')
    output_args = parser.add_argument_group('Output Options')
    exec_args = parser.add_argument_group('Execution Options')
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                        help='Show this help message and exit.')
    input_args.add_argument("-m", "--model", type=str, default="convert_model/converted_model.xml",
                        help="Optional.\n"
                             "Path to an .xml file with a trained model.")
    input_args.add_argument("-i", "--input", type=str, default="../jpeg/face0.jpg",
                        help="Optional.\n"
                             "Path to a image file.")
    input_args.add_argument("--labels", type=str, default=None, 
                        help="Optional.\n"
                             "Labels mapping file."
                             "Default is to change the extension of the modelfile\n"
                             "to '.labels'.")
    input_args.add_argument("-d", "--device", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer on; \n"
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.\n"
                             "The demo will look for a suitable plugin \n"
                             "for device specified.\n"
                             "Default value is CPU")
    input_args.add_argument("-l", "--cpu_extension", type=str, default=None, 
                        help="Optional.\n"
                             "Required for CPU custom layers. \n"
                             "Absolute path to a shared library\n"
                             "with the kernels implementations.\n"
                             "以前はlibcpu_extension_avx2.so 指定が必須だったけど、\n"
                             "2020.1から不要になった")
    output_args.add_argument("--disp", action='store_true', 
                        help="Optional.\n"
                             "image display")
    return parser
# ================================================================================

def main():
    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    # NCS が指定されていたら MYRIAD に書き換え
    if args.device == 'NCS' :
        args.device = 'MYRIAD'
    
    # モデルファイル名
    model_xml = args.model                                      # モデルファイル名(xml)
    model_bin = os.path.splitext(model_xml)[0] + ".bin"         # モデルファイル名(bin)
    
    # 入力ファイル
    input_file = os.path.abspath(args.input)
    assert os.path.isfile(input_file), "Specified input file doesn't exist"
    
    # ラベルファイル
    labels_file = None
    if args.labels :
        labels_file = args.labels
    else :
        labels_file = os.path.splitext(model_xml)[0] + ".labels"

    if not os.path.isfile(labels_file) :
        print(f"[WARNING] label file is not exist : {labels_file}")
        labels_file = None
    
    labels_map = None
    if labels_file:
        # ラベルファイルの読み込み
        with open(labels_file, 'r') as f:
            print(f"[INFO] Loading label file: {labels_file}")
            labels_map = [x.strip() for x in f]
    
    # 指定されたデバイスの plugin の初期化
    print("[INFO] Creating Inference Engine...")
    ie = IECore()
    
    # 拡張ライブラリのロード(CPU使用時のみ)
    if args.cpu_extension and (args.device == 'CPU') :
        print("[INFO] Loading Extension Library...")
        ie.add_extension(args.cpu_extension, "CPU")
        
    # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
    print(f"[INFO] Loading model files:\n\t{model_xml}\n\t{model_bin}")
    if flag_ver2019:
        # バージョン2019では2020で作成したモデルファイルは使用できないらしい
        # バージョン2019でconvert_model/run.sh を実行してモデルを再変換が必要
        net = IENetwork(model=model_xml, weights=model_bin)
    else :
        # IENetwork() での読み込みは非推奨になったので ie.read_network() に変更 
        net = ie.read_network(model=model_xml, weights=model_bin)
    
    # 未サポートレイヤの確認
    if args.device == 'CPU' :
        # サポートしているレイヤの一覧
        supported_layers = ie.query_network(net, "CPU")
        # netで使用されているレイヤでサポートしているレイヤの一覧にないもの
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        # サポートされていないレイヤがある？
        if len(not_supported_layers) != 0:
            # エラー終了
            print(f"[ERROR] Following layers are not supported by the plugin for specified device {args.device}:\n {', '.join(not_supported_layers)}")
            print(f"[ERROR] Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    
    # このプログラムは1出力のモデルのみサポートしているので、チェック
    # print(net.outputs)
    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    # 入力の準備
    print("[INFO] Preparing inputs")
    # print(net.inputs)
    for blob_name in net.inputs:
        # print(f'{blob_name}   {net.inputs[blob_name].shape}')
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        else:
            raise RuntimeError(f"Unsupported {len(net.inputs[blob_name].shape)} input layer '{ blob_name}'. Only 2D and 4D input layers are supported")
    
    # 入力画像情報の取得
    input_n, input_colors, input_height, input_width = net.inputs[input_blob].shape
    
    # プラグインへモデルをロード
    print("[INFO] Loading model to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    
    # 推論開始
    print("[INFO] Starting inference...")
    
    for loop_cnt in range(10) :         # 2回目以降速いかな？
                                        # そんなことはなかった... でも1割くらいは違ってる？...
                                        # 画像データがキャッシュに乗ってる分？？？
        print(f'==== LOOP COOUNT {loop_cnt} ====')
        # 画像の読み込み
        frame = cv2.imread(input_file)
        # 幅と高さを取得
        img_height, img_width, _ = frame.shape
        if args.disp :
            cv2.imshow("INPUT IMAGE", frame)

        # モデル入力用にリサイズ
        in_frame = cv2.resize(frame, (input_width, input_height))   # input size of coco ssd mobilenet?
        in_frame = in_frame[:, :, [2,1,0]]                          # BGR -> RGB
        in_frame = in_frame.transpose((2, 0, 1))                    # HWC →  CHW
        in_frame = np.expand_dims(in_frame, axis=0)                 # 3D -> 4D
        # if input_dtype != np.uint8 :
        #     in_frame = in_frame.astype(input_dtype)                 # 入力型がuint8以外だったら型変換

        # 画像入力
        feed_dict = {}
        feed_dict[input_blob] = in_frame
        
        inf_start = time.time()                                 # 推論処理開始時刻          --------------------------------
        
        # 片面しか使わないのでrequest_id は固定
        cur_request_id = 0
        
        # 推論予約 =============================================================================
        exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        
        # 推論結果待ち =============================================================================
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()                               # 推論処理終了時刻          --------------------------------
            inf_time = inf_end - inf_start                      # 推論処理時間
            
            # 検出結果の解析 =============================================================================
            parse_start = time.time()                           # 解析処理開始時刻          --------------------------------
            res = exec_net.requests[cur_request_id].outputs
            
            # 出力ノード名
            out_blob = list(net.outputs.keys())[0]
            # print(res[out_blob].shape)

            # 結果
            # スコアのリスト
            scores = res[out_blob][0]
            # 最大スコアのインデックス
            max_index = scores.argmax()
            if labels_map :
                max_label = f'{labels_map[max_index]} ({max_index})'
            else :
                max_label = f'Index={max_index}'
            
            max_score = scores[max_index] * 100
            print(f'     ****RESULT****')
            print(f'     {max_label} : {max_score:.2f}%     ({scores})')
            # 測定データの表示
            print(f'     Inference time : {(inf_time * 1000):.3f} ms')
            
            if args.disp :
                cv2.waitKey(0)

if __name__ == '__main__':
    sys.exit(main() or 0)
