set -e

model_path=~/Downloads/anythingxl.safetensors # Path to your model
model_name=anythingxl # Name used for output files
realistic=false  # Set to true to enable --realistic mode. It will use prompts for realistic images.
scheduler=dpm # dpm lcm eulera
cfg=5,7 # 5-7
steps=15,30 # 15-30

# Define SOC version list
soc_versions=("min")

uv venv -p 3.10.17 --clear
source .venv/bin/activate
uv sync

# Set realistic flag based on realistic variable
realistic_flag=""
if [ "$realistic" = true ]; then
    realistic_flag="--realistic"
fi

# ======== 1024x1024 ========       
echo "Processing base resolution: 1024x1024"
python prepare_data_sdxl.py --model_path $model_path $realistic_flag --scheduler $scheduler --cfg $cfg --step $steps
python gen_quant_data_sdxl.py
python export_onnx_sdxl.py --model_path $model_path

for soc in "${soc_versions[@]}"; do
    bash scripts/convert_all_sdxl.sh --min_soc $soc
done

# ======== Package outputs ========
echo "Packaging output files..."
for soc in "${soc_versions[@]}"; do
    touch output/qnn_models_sdxl_${soc}/SDXL
    zip -r ${model_name}_qnn2.28_${soc}.zip output/qnn_models_sdxl_${soc}
done