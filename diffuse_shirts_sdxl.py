''' Reference https://huggingface.co/ByteDance/SDXL-Lightning 
citation:
@misc{lin2024sdxllightning,
      title={SDXL-Lightning: Progressive Adversarial Diffusion Distillation}, 
      author={Shanchuan Lin and Anran Wang and Xiao Yang},
      year={2024},
      eprint={2402.13929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''
# Adjust the memory usage limit of MPS to avoid out-of-memory errors
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from PIL import Image
import requests
from io import BytesIO

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!

# Load model configuration
config = UNet2DConditionModel.load_config(base, subfolder="unet")
unet = UNet2DConditionModel.from_config(config).to("mps")

# Load model weights
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="mps"))

# Initialize the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float32).to("mps")

person_image_path = "data/W+NK+SS+TEE+OC+BB+VINTAGE.png"
new_shirt_image_path = "data/Screenshot_2024-10-29_at_5.06.08_PM-removebg-preview.png"

# Load the base image
base_image = Image.open(person_image_path).convert("RGB")

# Load the new shirt image
new_shirt_image = Image.open(new_shirt_image_path).convert("RGBA")

# Resize both images to a smaller size (e.g., 256x256) to reduce memory usage
base_image = base_image.resize((256, 256))
new_shirt_image = new_shirt_image.resize((256, 256))

# Overlay the new shirt image onto the base image
blended_image = Image.alpha_composite(base_image.convert("RGBA"), new_shirt_image)

# Convert the blended image back to RGB
blended_image = blended_image.convert("RGB")

# Example prompt
prompt = "A model wearing a black t-shirt with another black t-shirt blended seamlessly"

# Clear MPS cache to free up memory
torch.mps.empty_cache()

# Generate an image using the blended image and the prompt
with torch.no_grad():
    generated_images = pipe(prompt=prompt, init_image=blended_image, strength=0.75).images

# Save the generated image
output_image_path = 'output/diffuse_shirts_sdxl_generated_image.png'
generated_images[0].save(output_image_path)

# Display the generated image
generated_images[0].show()