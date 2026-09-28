import copy
import json
import time
from pathlib import Path

import torch
from diffusers import SanaPipeline
from torch import nn

from train_sana_docci_loop import RecurrentBlock, example

root = Path(__file__).resolve().parent
device = torch.device("cuda:1")
torch.cuda.set_device(device)
start = time.time()
pipe = SanaPipeline.from_pretrained(root / "model", variant="fp16", torch_dtype=torch.float16)
pipe.to(device)
pipe.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)
pipe.text_encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
teacher = pipe.transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
student = copy.deepcopy(teacher).to(dtype=torch.float32)
keep = [round(i * 27 / 23) for i in range(24)]
student.transformer_blocks = nn.ModuleList([student.transformer_blocks[i] for i in keep])
width = student.transformer_blocks[0].scale_shift_table.shape[-1]
adapter = RecurrentBlock(student.transformer_blocks[11], width)
student.transformer_blocks[11] = adapter
student.to(device)
teacher_state = {}
teacher.transformer_blocks[14].register_forward_hook(lambda _m, _inp, out: teacher_state.__setitem__("hidden", out))
path = next((root / "images").glob("*.jpg"))
row = {"example_id": path.stem, "image_file": path.name,
       "description": "A photograph of an everyday scene with several visible objects."}
result = example(row, pipe, teacher, student, adapter, teacher_state, device, {}, root,
                 seed=20260927, grad=True, ablate=True)
result["total"].backward()
gradient = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
print(json.dumps({"seconds": time.time() - start,
                  "losses": {k: float(v.detach()) for k, v in result.items() if torch.is_tensor(v)},
                  "grad_norm": float(gradient), "image": path.name}), flush=True)
