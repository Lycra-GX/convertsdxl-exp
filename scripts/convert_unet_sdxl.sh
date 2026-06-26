set -e

# Use SUFFIX from parent script, default to empty if not set
SUFFIX=${SUFFIX:-""}

qairt-converter --onnx_no_simplification --input_network ./unet_sdxl/model.onnx --output_path ./unet_sdxl/model.dlc 

qairt-quantizer --input_dlc ./unet_sdxl/model.dlc --output_dlc ./unet_sdxl/model_quantized.dlc --act_bitwidth 16 --bias_bitwidth 32 --use_per_channel_quantization --input_list ./input_list_unet_sdxl.txt --preserve_io_datatype

qnn-context-binary-generator --dlc_path ./unet_sdxl/model_quantized.dlc --model ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnModelDlc.so --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so --output_dir --output_dir "./output/qnn_models_sdxl$SUFFIX" --binary_file unet --config_file ./htp_backend$SUFFIX.json