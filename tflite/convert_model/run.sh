# saved modelから変換する場合のパラメータ
SAVED_DIR=../../SampleExports/TensorFlowSavedModel
OUTPUT_SAVED_FILE=saved_model/converted_model.tflite
WIDTH=224
HIGHT=224

# frozen modelから変換する場合のパラメータ
FROZEN_DIR=../../SampleExports/TensorFlow
FROZEN_FILE=${FROZEN_DIR}/model.pb
OUTPUT_FROZEN_FILE=frozen_model/converted_model.tflite
INPUT_TENSORS="Placeholder"
OUTPUT_TENSORS="model_outputs"

# 共通パラメータ
DATA_DIR="./data"

# 以下のオプションを追加で指定できる
# --quantize [weight] [integer]
 
if [[ $1 == saved ]]; then
	# saved model の場合
	shift 
	
	# 出力ディレクトリを作成
	mkdir -p "${OUTPUT_SAVED_FILE%/*}"
	
	# 変換処理
	python saved_model2tflite.py --input ${SAVED_DIR} --output ${OUTPUT_SAVED_FILE} \
	        --data ${DATA_DIR} --width ${WIDTH} --hight ${HIGHT} $@
	
	# EDGETPUへ変換
	edgetpu_compiler --out_dir "${OUTPUT_SAVED_FILE%/*}" ${OUTPUT_SAVED_FILE}
	
	# ラベルファイルのコピー
	find "${OUTPUT_SAVED_FILE%/*}" -name "*.tflite" | xargs -I {} echo {} | sed -e "s/\.tflite/\.labels/" | xargs -I {} cp ${FROZEN_DIR}/labels.txt {}
else
	# frozen model の場合
	if [[ $1 == frozen ]]; then
		shift 
	fi
	
	# 出力ディレクトリを作成
	mkdir -p "${OUTPUT_FROZEN_FILE%/*}"
	
	# 変換処理
	python frozen_model2tflite.py --input ${FROZEN_FILE} --output ${OUTPUT_FROZEN_FILE} \
	        --data ${DATA_DIR} --input_tensors ${INPUT_TENSORS} --output_tensors ${OUTPUT_TENSORS} $@
	
	# EDGETPUへ変換
	edgetpu_compiler --out_dir "${OUTPUT_FROZEN_FILE%/*}"  ${OUTPUT_FROZEN_FILE}
	
	# ラベルファイルのコピー
	find "${OUTPUT_FROZEN_FILE%/*}" -name "*.tflite" | xargs -I {} echo {} | sed -e "s/\.tflite/\.labels/" | xargs -I {} cp ${FROZEN_DIR}/labels.txt {}
fi



