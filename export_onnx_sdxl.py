import torch
from redefined_modules.diffusers.models.unet_2d_condition import UNet2DConditionModel
from redefined_modules.diffusers.models.attention import CrossAttention
import onnx
import pathlib
import shutil
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection
import argparse
import os


def delete_folders(folders):
    """Delete the specified folders if they exist."""
    for folder in folders:
        if os.path.exists(folder):
            print(f"Removing {folder}")
            shutil.rmtree(folder)


# Delete cache at the beginning of the script
folders_to_delete = [
    "qnn_unet_sdxl",
    "qnn_vae_encoder_sdxl",
    "qnn_vae_decoder_sdxl",
]
for folder in os.listdir("."):
    if folder.startswith("output"):
        folders_to_delete.append(folder)
delete_folders(folders_to_delete)


def replace_mha_with_sha_blocks(unet_model):
    for name, module in unet_model.named_modules():
        if isinstance(module, CrossAttention):
            module.replace_linear_to_convs()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--size", type=int, default=1024)
    return parser.parse_args()


args = parse_args()
size = args.size
model_path = args.model_path

for export_path in [
    "clip_sdxl",
    "clip2_sdxl",
    "unet_sdxl",
    "vae_decoder_sdxl",
    "vae_encoder_sdxl",
]:
    pathlib.Path(export_path).mkdir(parents=True, exist_ok=True)

if model_path.endswith("safetensors"):
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
    )
    pipe.save_pretrained("./model_sdxl")
    model_path = "./model_sdxl"
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(model_path)

test_sentence = "a b c d e f g"
test_input_ids = pipe.tokenizer(
    test_sentence,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt",
).input_ids


# ========== VAE Decoder ==========
class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.vae = pipe.vae

    def forward(self, input_ids):
        return self.vae.decode(input_ids, return_dict=False)[0]


vae_decoder = VAEDecoderWrapper(pipe)
vae_decoder.eval()

latents = torch.randn(1, 4, size // 8, size // 8)
torch.onnx.export(
    vae_decoder,
    latents,
    "vae_decoder_sdxl/model.onnx",
    input_names=["input"],
    output_names=["output"],
)


# ========== VAE Encoder ==========
class VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.vae = pipe.vae

    def forward(self, image):
        output = self.vae.encode(image).latent_dist
        return output.mean, output.std


vae_encoder = VAEEncoderWrapper(pipe)
vae_encoder.eval()

dummy_input = torch.randn(1, 3, size, size)
torch.onnx.export(
    vae_encoder,
    dummy_input,
    "vae_encoder_sdxl/model.onnx",
    input_names=["input"],
    output_names=["mean", "std"],
)

del pipe


# ========== Helper ==========
def generate_attn_mask(seq_len, device="cpu", dtype=torch.float32):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    attn_mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    attn_mask.masked_fill_(~mask, torch.finfo(attn_mask.dtype).min)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    return attn_mask


# ========== CLIP 1 (CLIPTextModel, CLIP-L) ==========
text_encoder = CLIPTextModel.from_pretrained(
    model_path,
    subfolder="text_encoder",
    attn_implementation="eager",
)
text_encoder.eval()

with torch.no_grad():
    ref_clip1_out = text_encoder(test_input_ids, output_hidden_states=True)
    ref_clip1_hidden = ref_clip1_out.hidden_states[-2]


class CLIPWrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        text_encoder.text_model.encoder.layers = text_encoder.text_model.encoder.layers[
            :-1
        ]
        self.text_encoder = text_encoder

    def forward(self, input_embedding):
        casual_mask = generate_attn_mask(
            77, device=input_embedding.device, dtype=input_embedding.dtype
        )
        hidden_states = self.text_encoder.text_model.encoder(
            input_embedding, causal_attention_mask=casual_mask
        ).last_hidden_state
        return hidden_states


clip = CLIPWrapper(text_encoder)
clip.eval()

with torch.no_grad():
    test_input_embedding_1 = clip.text_encoder.text_model.embeddings(test_input_ids)
    wrapper_clip1_hidden = clip(test_input_embedding_1)
assert torch.allclose(ref_clip1_hidden, wrapper_clip1_hidden, atol=1e-4), (
    "CLIP1 wrapper output mismatch: "
    f"max abs diff = {(ref_clip1_hidden - wrapper_clip1_hidden).abs().max().item()}"
)
print(
    "CLIP1 wrapper verified (max abs diff = "
    f"{(ref_clip1_hidden - wrapper_clip1_hidden).abs().max().item():.2e})"
)

input_embedding = torch.randn(1, 77, 768)
torch.onnx.export(
    clip,
    input_embedding,
    "clip_sdxl/model.onnx",
    input_names=["input_embedding"],
    output_names=["last_hidden_state"],
)
clip.text_encoder.text_model.embeddings.position_embedding.weight.data.numpy().tofile(
    "clip_sdxl/pos_emb.bin"
)
clip.text_encoder.text_model.embeddings.token_embedding.weight.data.to(
    torch.float16
).numpy().tofile("clip_sdxl/token_emb.bin")


# ========== CLIP 2 (CLIPTextModelWithProjection, OpenCLIP ViT-bigG) ==========
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    model_path,
    subfolder="text_encoder_2",
    attn_implementation="eager",
)
text_encoder_2.eval()

with torch.no_grad():
    ref_clip2_out = text_encoder_2(test_input_ids, output_hidden_states=True)
    ref_clip2_hidden = ref_clip2_out.hidden_states[-2]
    # HF's text_embeds: EOS-token selected from final_layer_norm(encoder.last),
    # then text_projection -> shape (batch, projection_dim)
    ref_clip2_pooled = ref_clip2_out.text_embeds


class CLIP2Wrapper(torch.nn.Module):
    def __init__(self, text_encoder_2):
        super().__init__()
        self.text_encoder_2 = text_encoder_2

    def forward(self, input_embedding):
        casual_mask = generate_attn_mask(
            77, device=input_embedding.device, dtype=input_embedding.dtype
        )
        encoder_outputs = self.text_encoder_2.text_model.encoder(
            input_embedding,
            causal_attention_mask=casual_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_outputs.hidden_states[-2]
        last_hidden_state = self.text_encoder_2.text_model.final_layer_norm(
            encoder_outputs.last_hidden_state
        )
        # Per-token projection; EOS selection is done on the C++ side.
        pooled_output = self.text_encoder_2.text_projection(last_hidden_state)

        return hidden_states, pooled_output


clip2 = CLIP2Wrapper(text_encoder_2)
clip2.eval()

with torch.no_grad():
    test_input_embedding_2 = clip2.text_encoder_2.text_model.embeddings(test_input_ids)
    wrapper_clip2_hidden, wrapper_clip2_pooled_all = clip2(test_input_embedding_2)
    # Emulate the C++-side EOS pick to compare against HF's text_embeds.
    eos_idx = test_input_ids.to(dtype=torch.int).argmax(dim=-1)
    wrapper_clip2_pooled = wrapper_clip2_pooled_all[
        torch.arange(wrapper_clip2_pooled_all.shape[0]), eos_idx
    ]
assert torch.allclose(ref_clip2_hidden, wrapper_clip2_hidden, atol=1e-4), (
    "CLIP2 wrapper hidden_states mismatch: "
    f"max abs diff = {(ref_clip2_hidden - wrapper_clip2_hidden).abs().max().item()}"
)
assert torch.allclose(ref_clip2_pooled, wrapper_clip2_pooled, atol=1e-4), (
    "CLIP2 wrapper pooled_output mismatch: "
    f"max abs diff = {(ref_clip2_pooled - wrapper_clip2_pooled).abs().max().item()}"
)
print(
    "CLIP2 wrapper verified (hidden max abs diff = "
    f"{(ref_clip2_hidden - wrapper_clip2_hidden).abs().max().item():.2e}, "
    "pooled max abs diff = "
    f"{(ref_clip2_pooled - wrapper_clip2_pooled).abs().max().item():.2e})"
)

input_embedding_2 = torch.randn(1, 77, 1280)
torch.onnx.export(
    clip2,
    input_embedding_2,
    "clip2_sdxl/model.onnx",
    input_names=["input_embedding"],
    output_names=["last_hidden_state", "pooled_output"],
)
clip2.text_encoder_2.text_model.embeddings.position_embedding.weight.data.numpy().tofile(
    "clip2_sdxl/pos_emb.bin"
)
clip2.text_encoder_2.text_model.embeddings.token_embedding.weight.data.to(
    torch.float16
).numpy().tofile("clip2_sdxl/token_emb.bin")


# ========== UNet (5 inputs, with redefined_modules) ==========
unet = UNet2DConditionModel.from_pretrained(
    model_path,
    subfolder="unet",
    revision="main",
)
unet.config.return_dict = False

replace_mha_with_sha_blocks(unet)
unet.eval()


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )


unet_wrapper = UNetWrapper(unet)
unet_wrapper.eval()

with torch.no_grad():
    torch.onnx.export(
        unet_wrapper,
        (
            torch.randn(1, 4, size // 8, size // 8),
            torch.tensor([0], dtype=torch.long),
            torch.randn(1, 77, 2048),
            torch.randn(1, 1280),
            torch.randn(1, 6),
        ),
        "unet_sdxl/model.onnx",
        input_names=[
            "sample",
            "timestamp",
            "encoder_hidden_states",
            "text_embeds",
            "time_ids",
        ],
        output_names=["output"],
    )

model = onnx.load("unet_sdxl/model.onnx")
shutil.rmtree("unet_sdxl")
pathlib.Path("unet_sdxl").mkdir(parents=True, exist_ok=True)
onnx.save_model(
    model,
    "unet_sdxl/model.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="weights.pb",
    convert_attribute=False,
)
