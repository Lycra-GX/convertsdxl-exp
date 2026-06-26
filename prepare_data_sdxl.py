import argparse
import random


def _range_pair(cast):
    def _parse(value):
        parts = value.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"expected 'min,max', got {value!r}")
        lo, hi = cast(parts[0]), cast(parts[1])
        if lo > hi:
            raise argparse.ArgumentTypeError(f"min ({lo}) must be <= max ({hi})")
        return (lo, hi)

    return _parse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm",
        choices=["dpm", "eulera", "lcm"],
    )
    parser.add_argument(
        "--cfg",
        type=_range_pair(float),
        default=(5.0, 7.0),
        help="min,max range of guidance scale (float), e.g. 5,7",
    )
    parser.add_argument(
        "--step",
        type=_range_pair(int),
        default=(15, 30),
        help="min,max range of inference steps (int), e.g. 15,30",
    )
    return parser.parse_args()


args = parse_args()
safetensor_path = args.model_path
resolution = (args.width, args.height)
is_realistic = args.realistic
scheduler_name = args.scheduler
cfg_min, cfg_max = args.cfg
step_min, step_max = args.step

# Please use at least 10 prompts, and the prompts should preferably use words related to the model's usage scenarios.
prompts = [
    [
        "masterpiece, best quality, 1girl, solo, school uniform, sailor collar, blue sky, cloud, wind, blowing hair, hand on hat, looking at viewer, smile, sunlight, lens flare, crisp lines,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, fused fingers, deformed, disfigured, mutation, extra limbs",
    ],
    [
        "1girl, solo, pink hair, twintails, pink eyes, frills, ribbon, dress, cupcake, sweets, food, eating, puffy sleeves, pastel color, simple background, kawaii, heart hair ornament,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, monochrome, grayscale, zombie, sketch",
    ],
    [
        "backlighting, sunset, 1girl, solo, silhouette, standing, looking back, classroom, window, orange sky, emotional, atmospheric, shadow, messy hair, school uniform, glowing hair,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad eyes, crossed eyes",
    ],
    [
        "masterpiece, best quality, illustration, 1girl, solo, magical girl, holding staff, casting spell, magic circle, glowing particles, dynamic pose, frilled dress, floating hair, starry sky background, vibrant colors,",
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mutation, deformed limbs",
    ],
    [
        "face focus, 1girl, solo, beautiful eyes, detailed eyes, blush, shy, looking at viewer, scarf, winter, snow, snowflakes, breath vapor, outdoors, cold, upper body,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad proportions, distortion",
    ],
    [
        "1girl, solo, maid, maid headdress, apron, tray, tea set, knee highs, zettai ryouiki, cafe, indoor, table, chair, polite, bowing, long hair, victorian maid,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad legs, extra legs",
    ],
    [
        "absurdres, highres, 1girl, solo, cyberpunk, mechanical ears, glowing headphones, neon lights, jacket, futuristic city, night, rain, wet street, reflection, cool, chromatic aberration,",
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, old, fat, ugly",
    ],
    [
        "best quality, 1girl, solo, white swimsuit, bikini, straw hat, beach, ocean, summer, ripples, wet skin, playing water, splash, barefoot, smile, energetic, blue water,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, missing toes",
    ],
    [
        "1girl, solo, hoodie, oversized clothes, sitting, floor, gaming, holding controller, headphones, messy room, snacks, chips, screen light, dark room, casual, flat color,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, realistic, photorealistic",
    ],
    [
        "masterpiece, 1girl, solo, kimono, floral print, fox ears, fox tail, miko, shrine, torii, cherry blossoms, falling petals, japanese clothes, hair ornament, serene, traditional,",
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, multiple tails, bad tail",
    ],
    [
        "dutch angle, dynamic pose, 1girl, solo, school uniform, running, toast in mouth, panic, bag, outdoors, street, morning, falling leaves, anime logic, motion blur,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, static, boring",
    ],
    [
        "best quality, illustration, 1girl, solo, gothic lolita, black dress, lace, ribbons, red roses, doll, sitting on throne, red eyes, blonde hair, drills, gloomy, dark background,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bright, sunny, happy",
    ],
    [
        "1girl, solo, glasses, sweater, plaid skirt, library, bookshelf, reading book, holding book, quiet, study, sunlight through window, dust particles, intellectual, braid,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad perspective",
    ],
    [
        "watercolor (medium), traditional media, 1girl, solo, white dress, flowers, garden, soft focus, dreamy, pastel palette, artistic, sketch style, flower crown, wet on wet,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, 3d, render, cgi",
    ],
    [
        "1girl, solo, cat ears, cat tail, animal ear fluff, bell collar, paw pose, :3, large hands, white background, simple background, full body, cute pose, maid headpiece,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, complex background",
    ],
    [
        "masterpiece, best quality, 1girl, solo, night, starry sky, milky way, shooting star, telescope, sitting on grass, looking up, shiny eyes, dark blue theme, beautiful detail,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, noise, grain",
    ],
    [
        "absurdres, 1girl, solo, idol, idol clothes, stage, spotlight, singing, microphone, winking, one eye closed, sweat, sparkles, audience, concert, laser lights,",
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly face",
    ],
    [
        "from side, profile, 1girl, solo, high collar, ponytail, looking away, blue sky, cloud, windy, melancholic, tears, crying, emotional, cinematic composition, close up,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, laughing, smile",
    ],
    [
        "best quality, 1girl, solo, elf, pointed ears, archer, green tunic, forest, nature, tree, vines, sunlight, holding bow, hunting, fantasy, cape,",
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, modern clothes, cars",
    ],
    [
        "1girl, solo, white background, simple background, character design, t-shirt, jeans, standing, hands in pockets, short hair, bob cut, tomboy, cool, sneaker,",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, dress, skirt, girly",
    ],
]

if is_realistic:
    prompts = [
        [
            "masterpiece, best quality, award-winning street photography of a young woman with freckles and curly red hair laughing candidly, cinematic golden hour lighting, soft bokeh background, ultra realistic, detailed skin texture",
            "ugly, poorly drawn face, bad anatomy, deformed, disfigured, mutated hands, extra fingers, blurry, out of focus, unrealistic skin texture, jpeg artifacts, signature, watermark, text, painting, cartoon, 3d, render",
        ],
        [
            "high-speed action photograph, best quality, a skateboarder frozen mid-air in a skatepark, dynamic motion blur on the background, tack sharp focus, dramatic low-angle perspective, harsh shadows, photorealistic",
            "static, posed, blurry subject, worst quality, low quality, motion sickness, bad anatomy, contorted limbs, disfigured, malformed, watermark, text, cartoon, 3d, render, painting",
        ],
        [
            "hyperrealistic masterpiece, an evocative black and white portrait of an elderly man, detailed wrinkles and skin texture, looking directly at camera, dramatic chiaroscuro lighting, sharp focus on eyes, 8k",
            "color, young, smooth plastic skin, doll, 3d, cgi, render, flat lighting, blurry, worst quality, low quality, disfigured face, poorly drawn eyes",
        ],
        [
            "heartwarming and cinematic photo, best quality, a father and his child walking hand-in-hand in an autumn park, seen from behind, beautiful warm backlighting, realistic emotional scene",
            "posed, looking at camera, studio lighting, worst quality, bad anatomy, deformed bodies, extra limbs, ugly, blurry, watermark, signature, painting, drawing, anime",
        ],
        [
            "atmospheric film still, masterpiece, a musician playing an acoustic guitar in a dimly lit bar, sharp focus on hands, beautiful lens flare from a single spotlight, grainy film texture, photorealistic",
            "brightly lit, outdoor, posed, empty stage, worst quality, ugly, bad anatomy, extra fingers, deformed hands, blurry instrument, cartoon, 3d, render",
        ],
        [
            "ultra-detailed professional photograph, 8k, a female scientist intently focused on her microscope in a modern laboratory, clean soft lighting, realistic, masterpiece",
            "fantasy, futuristic, messy lab, unprofessional attire, ugly, disfigured, bad anatomy, blurry face, worst quality, painting, drawing, render, cgi",
        ],
        [
            "vibrant and joyful photograph, masterpiece, diverse friends laughing around a campfire at night, warm firelight on their faces, sparks in the dark sky, candid moment, cinematic, best quality",
            "posed, sad, daytime, ugly, blurry faces, worst quality, bad anatomy, distorted faces, watermark, signature, painting, drawing, 3d, render",
        ],
        [
            "dynamic full-body photograph, best quality, an athletic woman running on a coastal path at sunrise, her silhouette defined against the glowing sky, backlighting, sense of motion, realistic",
            "static, posing, gym, indoor, worst quality, bad anatomy, deformed legs, ugly, blurry subject, painting, drawing, render, cartoon",
        ],
        [
            "National Geographic style portrait, masterpiece, a dignified elderly Tibetan woman, detailed weathered skin, wearing beautiful traditional clothing, natural mountain daylight, hyper-detailed",
            "young, smooth skin, cartoon, western clothing, studio lighting, worst quality, ugly, blurry, 3d, cgi, render, painting, disfigured face",
        ],
        [
            "a realistic and cozy photograph, best quality, a man making pasta, soft morning window light, focus on flour-covered hands, shallow depth of field, film aesthetic, masterpiece",
            "clean hands, fake, posed, professional kitchen, dark, ugly, worst quality, bad anatomy, blurry hands, painting, drawing, 3d, cgi, render, cartoon",
        ],
        [
            "breathtaking 8k wallpaper, masterpiece, epic landscape photograph of the Scottish Highlands, misty morning, a lone stag on a hill, dramatic god rays, hyperdetailed, photorealistic",
            "people, ugly, tiling, poorly drawn, out of frame, blurry, worst quality, low quality, oversaturated colors, jpeg artifacts, painting, cartoon, watermark, text",
        ],
        [
            "cinematic interior shot, best quality, a cozy coffee shop with steam rising from a latte, soft window light illuminating dust particles, detailed textures, realistic, masterpiece",
            "ugly, noisy, grainy, distorted, dark, blurry, overexposed, worst quality, fake, plastic textures, render, cgi, painting, drawing, cartoon",
        ],
        [
            "moody photorealistic image, masterpiece, a neon-lit Tokyo street on a rainy night, vivid reflections on wet asphalt, Blade Runner aesthetic, cinematic, high contrast, 8k",
            "dry, daytime, sunny, blurry, worst quality, ugly, oversaturated, dull colors, empty street, painting, drawing, 3d, render, video game",
        ],
        [
            "serene long-exposure photograph, best quality, a tropical beach at sunset, milky smooth waves, palm tree silhouettes against a vibrant sky, wide-angle, photorealistic masterpiece",
            "stormy, crowded, people, trash, ugly, blurry, noisy, jpeg artifacts, worst quality, harsh lighting, painting, drawing, render, cartoon",
        ],
        [
            "professional architectural photography, best quality, a modern minimalist house in a forest, emphasizing clean lines, soft overcast daylight, ultra detailed, realistic, masterpiece",
            "distorted perspective, warped lines, unrealistic materials, dirty, messy, cluttered, worst quality, low quality, painting, drawing, cartoon, signature",
        ],
        [
            "photo capturing the grand atmosphere of an old library, masterpiece, towering bookshelves, shafts of light through arched windows, scholarly mood, high dynamic range, realistic, best quality",
            "empty, modern, messy, ugly, blurry, noisy, worst quality, few books, bright fluorescent lights, painting, drawing, 3d, render, cartoon",
        ],
        [
            "majestic ultra-detailed portrait, masterpiece, a Siberian Husky in a snowy landscape, striking blue eyes in sharp focus, detailed fur texture, professional pet photography, realistic",
            "ugly, deformed, disfigured, blurry, worst quality, cartoon, anime, painting, drawing, 3d, render, skinny, sick, indoor, bad fur texture",
        ],
        [
            "atmospheric photograph, best quality, a vast abandoned industrial warehouse, dramatic light streams from high windows casting long shadows, gritty textures, urban exploration style, masterpiece",
            "clean, new, bright, full, cluttered, people, ugly, blurry, worst quality, unrealistic lighting, painting, drawing, 3d, render, cartoon",
        ],
        [
            "vibrant street photograph, masterpiece, a bustling farmers market, colorful fresh produce, shallow depth of field on a crate of ripe tomatoes, sunny day, realistic textures, best quality",
            "empty, dull, monochrome, indoor, ugly, blurry, worst quality, rotten food, plastic fruit, painting, drawing, 3d, render, cartoon",
        ],
        [
            "hyperrealistic minimalist landscape photo, 8k, a vast desert at dawn, soft light on sand dunes, pastel sky, ultra wide-angle, serene masterpiece",
            "trees, water, people, animals, ugly, harsh midday lighting, blurry, worst quality, saturated colors, jpeg artifacts, painting, drawing, 3d, render, cartoon",
        ],
    ]


import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import pickle as pkl
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
)
import os

data = {"clip": [], "clip2": [], "unet": [], "vae": []}

pipe = StableDiffusionXLPipeline.from_single_file(
    safetensor_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

if scheduler_name == "dpm":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        lower_order_final=True,
        use_karras_sigmas=True,
        beta_schedule="scaled_linear",
    )
elif scheduler_name == "eulera":
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif scheduler_name == "lcm":
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
else:
    raise ValueError(f"Unknown scheduler: {scheduler_name}")

text_encoder = pipe.text_encoder
text_encoder_2 = pipe.text_encoder_2
unet = pipe.unet
vae = pipe.vae
tokenizer = pipe.tokenizer
tokenizer_2 = pipe.tokenizer_2
scheduler = pipe.scheduler

idx = 0


def generate(prompt, negative_prompt, cfg, steps):
    global idx
    idx += 1

    width, height = resolution
    num_inference_steps = steps
    guidance_scale = cfg
    num_images_per_prompt = 1
    do_classifier_free_guidance = guidance_scale > 0

    # ========== CLIP 1 (text_encoder) ==========
    text_inputs = tokenizer(
        [negative_prompt, prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    data["clip"].append([text_input_ids.cpu().numpy()])

    with torch.no_grad():
        prompt_embeds_1 = text_encoder(text_input_ids, output_hidden_states=True)
        # SDXL uses the penultimate hidden state from CLIP 1
        prompt_embeds_1_hidden = prompt_embeds_1.hidden_states[-2]

    # ========== CLIP 2 (text_encoder_2, OpenCLIP) ==========
    text_inputs_2 = tokenizer_2(
        [negative_prompt, prompt],
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids_2 = text_inputs_2.input_ids.to(device)

    data["clip2"].append([text_input_ids_2.cpu().numpy()])

    with torch.no_grad():
        prompt_embeds_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
        # SDXL uses the penultimate hidden state from CLIP 2
        prompt_embeds_2_hidden = prompt_embeds_2.hidden_states[-2]
        # Pooled output from CLIP 2 is used as text_embeds for added_cond_kwargs
        pooled_prompt_embeds = prompt_embeds_2[0]

    # Concatenate the hidden states from both CLIP encoders
    prompt_embeds = torch.cat([prompt_embeds_1_hidden, prompt_embeds_2_hidden], dim=-1)

    # ========== Prepare add_time_ids ==========
    # SDXL time_ids: [original_height, original_width, crop_top, crop_left, target_height, target_width]
    original_size = (height, width)
    crops_coords_top_left = (0, 0)
    target_size = (height, width)

    add_time_ids = torch.tensor(
        [
            list(original_size) + list(crops_coords_top_left) + list(target_size),
        ],
        dtype=prompt_embeds.dtype,
        device=device,
    )
    # Duplicate for negative + positive (batch of 2 for CFG)
    if do_classifier_free_guidance:
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)

    # ========== Prepare latents ==========
    num_channels_latents = unet.config.in_channels
    latents_shape = (
        num_images_per_prompt,
        num_channels_latents,
        height // 8,
        width // 8,
    )

    latents = torch.randn(latents_shape, device=device, dtype=prompt_embeds.dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # UNet 5 inputs: latent_model_input, timestep, encoder_hidden_states, text_embeds, time_ids
        data["unet"].append(
            [
                latent_model_input.cpu().numpy(),
                (
                    t.cpu().numpy().astype(np.int32)
                    if hasattr(t, "cpu")
                    else np.array([t]).astype(np.int32)
                ),
                prompt_embeds.cpu().numpy(),
                pooled_prompt_embeds.cpu().numpy(),
                add_time_ids.cpu().numpy(),
            ]
        )
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
            ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        scheduler_output = scheduler.step(noise_pred, t, latents)
        latents = scheduler_output.prev_sample

    latents = 1 / vae.config.scaling_factor * latents

    data["vae"].append([latents.cpu().numpy()])

    # SDXL VAE is numerically unstable in fp16, must upcast to fp32 for decoding
    vae_dtype = vae.dtype
    needs_upcast = vae.config.get("force_upcast", True)
    if needs_upcast:
        vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)

    with torch.no_grad():
        image = vae.decode(latents).sample

    if needs_upcast:
        vae.to(dtype=vae_dtype)

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = image[0]

    pil_image = Image.fromarray(image)

    os.makedirs("images_sdxl", exist_ok=True)
    pil_image.save(f"images_sdxl/{idx}.png")
    print(f"image saved images_sdxl/{idx}.png")


for prompt, negative_prompt in prompts:
    for _ in range(2):
        cfg = random.uniform(cfg_min, cfg_max)
        steps = random.randint(step_min, step_max)
        print(prompt, negative_prompt, cfg, steps)
        generate(prompt, negative_prompt, cfg, steps)

with open("data_sdxl.pkl", "wb") as f:
    pkl.dump(data, f)
