SAVED_DIR=../../SampleExports/TensorFlowSavedModel
OUTPUT_FILE=converted_model.tflite

python saved_model2tflite.py --input ${SAVED_DIR} --output ${OUTPUT_FILE}

edgetpu_compiler ${OUTPUT_FILE}

# cp ${SAVED_DIR}/labels.txt .
cp ${SAVED_DIR}/labels.txt "${OUTPUT_FILE%.*}.labels"
ln -sf "${OUTPUT_FILE%.*}.labels" "${OUTPUT_FILE%.*}_edgetpu.labels"