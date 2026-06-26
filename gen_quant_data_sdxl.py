from PIL import Image
import numpy as np
import pickle as pkl
import random
import os


def generate_raw(
    data_list, output_dir="./raw", input_list_file="./input_list.txt", names=None
):
    os.makedirs(output_dir, exist_ok=True)

    all_paths = []
    for i, row in enumerate(data_list):
        row_paths = []
        for j, item in enumerate(row):
            if item.dtype == np.int64:
                item = item.astype(np.int32)
            file_path = f"{output_dir}/{i}_{j}.raw"
            item.tofile(file_path)
            row_paths.append(file_path)
        all_paths.append(row_paths)

    with open(input_list_file, "w") as f:
        for row_paths in all_paths:
            if names is not None:
                for i in range(len(row_paths)):
                    row_paths[i] = f"{names[i]}:={row_paths[i]}"
            f.write(" ".join(row_paths) + "\n")


with open("./data_sdxl.pkl", "rb") as f:
    data = pkl.load(f)

# UNet has 5 inputs: sample, timestamp, encoder_hidden_states, text_embeds (pooled), time_ids
processed_data = []
for sample, timestamp, text_embed, pooled_text_embed, time_ids in data["unet"]:
    sample = sample.astype(np.float32)
    text_embed = text_embed.astype(np.float32)
    pooled_text_embed = pooled_text_embed.astype(np.float32)
    time_ids = time_ids.astype(np.float32)

    if np.abs(sample).max() > 7.2:
        continue

    # Split batch dim (negative=0, positive=1) for CFG
    processed_data.append(
        [sample[0], timestamp, text_embed[0], pooled_text_embed[0], time_ids[0]]
    )
    processed_data.append(
        [sample[1], timestamp, text_embed[1], pooled_text_embed[1], time_ids[1]]
    )

print(f"Total valid samples for unet: {len(processed_data)}")
if len(processed_data) > 256:
    processed_data = random.sample(processed_data, 256)

generate_raw(
    processed_data,
    output_dir="./unet_input_raw_sdxl",
    input_list_file="./input_list_unet_sdxl.txt",
    names=["sample", "timestamp", "encoder_hidden_states", "text_embeds", "time_ids"],
)

processed_data = []
for (latent,) in data["vae"]:
    latent = latent.astype(np.float32)
    processed_data.append([latent[0]])
if len(processed_data) > 40:
    processed_data = random.sample(processed_data, 40)

generate_raw(
    processed_data,
    output_dir="./vae_decoder_input_raw_sdxl",
    input_list_file="./input_list_vae_decoder_sdxl.txt",
)


size = 1024

images = os.listdir("images_sdxl")
processed_data = []
for image in images:
    img = Image.open(os.path.join("images_sdxl", image))
    img = img.convert("RGB")
    img = img.resize((size, size))
    img = np.array(img)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = img * 2 - 1
    processed_data.append([img])

if len(processed_data) > 20:
    processed_data = random.sample(processed_data, 20)
generate_raw(
    processed_data,
    output_dir="./vae_encoder_input_raw_sdxl",
    input_list_file="./input_list_vae_encoder_sdxl.txt",
)
