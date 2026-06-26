set -e

# Parse command line arguments
MIN_SOC=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --min_soc)
            MIN_SOC="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if min_soc is provided
if [ -z "$MIN_SOC" ]; then
    echo "Error: --min_soc parameter is required"
    exit 1
fi

# Set suffix based on min_soc parameter
SUFFIX="_$MIN_SOC"

mkdir -p "output/qnn_models_sdxl$SUFFIX"
cp ./tokenizer.json "output/qnn_models_sdxl$SUFFIX/"

export LD_LIBRARY_PATH=~/qnn:$LD_LIBRARY_PATH

current_pwd=$(pwd)

QNN_SDK_ROOT=/data/qairt/2.28.0.241029
cd $QNN_SDK_ROOT/bin
source envsetup.sh

cd ${current_pwd}

export SUFFIX

bash scripts/convert_clip_sdxl.sh

bash scripts/convert_clip2_sdxl.sh

bash scripts/convert_vae_encoder_sdxl.sh

bash scripts/convert_vae_decoder_sdxl.sh

bash scripts/convert_unet_sdxl.sh
