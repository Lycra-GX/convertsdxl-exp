set -e

# Use SUFFIX from parent script, default to empty if not set
SUFFIX=${SUFFIX:-""}

./MNNConvert -f ONNX --modelFile clip2_sdxl/model.onnx --MNNModel "output/qnn_models_sdxl$SUFFIX/clip_2.mnn" --fp16
cp clip2_sdxl/pos_emb.bin "output/qnn_models_sdxl$SUFFIX/pos_emb_2.bin"
cp clip2_sdxl/token_emb.bin "output/qnn_models_sdxl$SUFFIX/token_emb_2.bin"
