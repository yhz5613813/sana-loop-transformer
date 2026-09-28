import time
from pathlib import Path

import torch
from diffusers import SanaPipeline

root = Path(__file__).resolve().parent
start = time.time()
pipe = SanaPipeline.from_pretrained(root / "model", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda:0")
pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)
print("LOADED", round(time.time() - start, 2), flush=True)
generator = torch.Generator(device="cuda:0").manual_seed(20260927)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = pipe(
        prompt="A red ceramic mug and a green apple sit on a wooden table near a bright window.",
        height=512, width=512, guidance_scale=4.5, num_inference_steps=20,
        generator=generator,
    ).images[0]
image.save(root / "results" / "teacher_smoke.png")
print("GENERATED", round(time.time() - start, 2), flush=True)
