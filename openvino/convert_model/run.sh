#!/bin/bash

INPUT_DIR=../../SampleExports/TensorFlow
INPUT_FILE=model.pb               # 拡張子を付ける
INPUT_SHAPE='[1,224,224,3]'
OUTPUT_DIR=.
OUTPUT_FILE=converted_model        # 拡張子は付けない

# 環境変数定義済みかチェック
if [ -z ${INTEL_OPENVINO_DIR} ]; then
	echo "source /opt/intel/openvino/bin/setupvars.sh が 実行されていないようです"
	exit
fi

# 参考：
# https://github.com/hiouchiy/IntelAI_and_Cloud/blob/master/Azure/demo1/Lesson1_AzureCognitiveService_and_OpenVINO_Collaboration.ipynb
${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py --input_model=${INPUT_DIR}/${INPUT_FILE} --input_shape=${INPUT_SHAPE}  --model_name=${OUTPUT_FILE}

# ラベルファイルをコピー
cp ${INPUT_DIR}/labels.txt "${OUTPUT_DIR}/${OUTPUT_FILE%.*}.labels"
