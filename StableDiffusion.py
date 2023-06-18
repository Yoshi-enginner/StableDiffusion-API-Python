import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"

# Specify GPU "CUDA-NVIDIA"
device = "cuda" 

# promp
prompt = "Manhattan, NY in style"

# pipline
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)

# excute
generator = torch.Generator(device).manual_seed(20) 
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]  

# saving
image.save("successful2.png")
