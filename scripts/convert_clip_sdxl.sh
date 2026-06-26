set -e

# Use SUFFIX from parent script, default to empty if not set
SUFFIX=${SUFFIX:-""}

./MNNConvert -f ONNX --modelFile clip_sdxl/model.onnx --MNNModel "output/qnn_models_sdxl$SUFFIX/clip.mnn" --fp16
cp clip_sdxl/pos_emb.bin "output/qnn_models_sdxl$SUFFIX/pos_emb.bin"
cp clip_sdxl/token_emb.bin "output/qnn_models_sdxl$SUFFIX/token_emb.bin"
