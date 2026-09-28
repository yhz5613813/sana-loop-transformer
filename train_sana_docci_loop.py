"""Sana-600M/DOCCI pilot: reduced-depth student with cross-step hidden-state reuse.

Teacher and student receive the same noisy latent, time and caption. The student
first processes a noisier state from the same image/noise pair and carries its
own detached hidden state into the lower-noise prediction. Real flow matching,
teacher velocity and hidden-state losses are logged separately.
"""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers import SanaPipeline
from PIL import Image, ImageOps
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class RecurrentBlock(nn.Module):
    def __init__(self, base: nn.Module, width: int):
        super().__init__()
        self.base = base
        self.gate = nn.Parameter(torch.full((1, 1, width), -4.0))
        self.state = None
        self.capture = None
        self.auto_update = False

    def forward(self, hidden_states, *args, **kwargs):
        if self.state is not None:
            assert self.state.shape == hidden_states.shape
            hidden_states = hidden_states + torch.sigmoid(self.gate).to(hidden_states.dtype) * self.state.to(hidden_states.dtype)
        output = self.base(hidden_states, *args, **kwargs)
        self.capture = output
        if self.auto_update:
            self.state = output.detach()
        return output

    def reset(self, auto_update=False):
        self.state = None
        self.capture = None
        self.auto_update = auto_update


def write_jsonl(path, item):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
        file.flush()


def load_item(row, pipe, device, cache, root):
    key = row["example_id"]
    if key in cache:
        z, emb, mask = cache[key]
        return z.to(device), emb.to(device), mask.to(device)
    with Image.open(root / "images" / row["image_file"]) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image = ImageOps.pad(image, (512, 512), method=Image.Resampling.LANCZOS, color=(127, 127, 127))
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16) / 127.5 - 1.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = pipe.vae.encode(pixels).latent * pipe.vae.config.scaling_factor
        emb, mask, _, _ = pipe.encode_prompt(
            row["description"], do_classifier_free_guidance=False,
            max_sequence_length=300, clean_caption=False,
        )
    z = z.detach().cpu().to(torch.bfloat16)
    emb = emb.detach().cpu().to(torch.bfloat16)
    mask = mask.detach().cpu()
    cache[key] = (z, emb, mask)
    return z.to(device), emb.to(device), mask.to(device)


def run_transformer(model, x, emb, mask, sigma):
    t = torch.full((x.shape[0],), sigma * 1000.0, device=x.device, dtype=torch.float32)
    return model(hidden_states=x, encoder_hidden_states=emb, encoder_attention_mask=mask,
                 timestep=t, return_dict=False)[0]


def example(row, pipe, teacher, student, adapter, teacher_state, device, cache, root, seed,
            sigma_low=None, sigma_high=None, grad=False, ablate=False):
    z, emb, mask = load_item(row, pipe, device, cache, root)
    generator = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(z.shape, device=device, dtype=z.dtype, generator=generator)
    rng = random.Random(seed)
    high = sigma_high if sigma_high is not None else rng.uniform(0.35, 0.95)
    low = sigma_low if sigma_low is not None else rng.uniform(0.02, high - 0.05)
    x_high = (1 - high) * z + high * eps
    x_low = (1 - low) * z + low * eps
    target = (eps - z).float()
    adapter.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        run_transformer(student, x_high, emb, mask, high)
        previous_state = adapter.capture.detach()
        teacher_velocity = run_transformer(teacher, x_low, emb, mask, low)
        teacher_hidden = teacher_state["hidden"].detach()
    adapter.state = previous_state
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        student_velocity = run_transformer(student, x_low, emb, mask, low)
        student_hidden = adapter.capture
        real_loss = (student_velocity.float() - target).square().mean()
        teacher_loss = (student_velocity.float() - teacher_velocity.float()).square().mean()
        hidden_loss = (student_hidden.float() - teacher_hidden.float()).square().mean() / (teacher_hidden.float().square().mean() + 1e-6)
        total = real_loss + teacher_loss + 0.05 * hidden_loss
    result = {
        "real_mse": real_loss,
        "teacher_mse": teacher_loss,
        "hidden_relative_mse": hidden_loss,
        "teacher_real_mse": (teacher_velocity.float() - target).square().mean(),
        "total": total,
        "sigma_low": low,
        "sigma_high": high,
    }
    if ablate:
        adapter.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            no_loop = run_transformer(student, x_low, emb, mask, low)
        result["no_loop_real_mse"] = (no_loop.float() - target).square().mean()
        result["no_loop_teacher_mse"] = (no_loop.float() - teacher_velocity.float()).square().mean()
    adapter.reset()
    return result


def validate(rows, pipe, teacher, student, adapter, teacher_state, device, cache, root, rank, world, step):
    student.eval()
    sums = torch.zeros(6, device=device, dtype=torch.float64)
    for i in range(rank, min(32, len(rows)), world):
        result = example(rows[i], pipe, teacher, student, adapter, teacher_state, device, cache, root,
                         seed=20261000 + i, grad=False, ablate=True)
        for j, key in enumerate(("real_mse", "teacher_mse", "hidden_relative_mse", "teacher_real_mse",
                                 "no_loop_real_mse", "no_loop_teacher_mse")):
            sums[j] += result[key].detach().double()
    dist.all_reduce(sums)
    student.train()
    values = (sums / min(32, len(rows))).tolist()
    return {"step": step, "kind": "fixed_val_32", **dict(zip(
        ("real_mse", "teacher_mse", "hidden_relative_mse", "teacher_real_mse",
         "no_loop_real_mse", "no_loop_teacher_mse"), values))}


def generate(pipe, teacher, student, adapter, rows, output, step, device):
    pipe.transformer = teacher
    teacher.eval()
    student.eval()
    for i, row in enumerate(rows[:4]):
        prompt = row["description"]
        for kind in ("teacher", "student_loop", "student_no_loop"):
            if kind == "teacher" and step != 0:
                continue
            if kind == "teacher":
                pipe.transformer = teacher
            else:
                pipe.transformer = student
                adapter.reset(auto_update=(kind == "student_loop"))
            torch.cuda.synchronize(device)
            generated_at = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipe(prompt=prompt, height=512, width=512, guidance_scale=4.5,
                             num_inference_steps=20, max_sequence_length=300,
                             generator=torch.Generator(device=device).manual_seed(4200 + i)).images[0]
            torch.cuda.synchronize(device)
            write_jsonl(output / "generation_timing.jsonl", {"step": step, "case": i,
                        "kind": kind, "seconds": time.time() - generated_at})
            image.save(output / f"step{step:04d}_{kind}_{i:02d}.png")
            adapter.reset()
    pipe.transformer = teacher
    student.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--generate-every", type=int, default=400)
    args = parser.parse_args()
    root = args.root
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 8, world
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(20260927 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_rows, val_rows = manifest["train"], manifest["val"]
    output = root / "results"
    output.mkdir(exist_ok=True)
    model_path = root / "model"
    pipe = SanaPipeline.from_pretrained(model_path, variant="fp16", torch_dtype=torch.float16)
    pipe.to(device)
    pipe.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    pipe.text_encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    teacher = pipe.transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    assert len(teacher.transformer_blocks) == 28
    student = copy.deepcopy(teacher).to(dtype=torch.float32)
    keep = [round(i * 27 / 23) for i in range(24)]
    assert len(set(keep)) == 24
    student.transformer_blocks = nn.ModuleList([student.transformer_blocks[i] for i in keep])
    width = student.transformer_blocks[0].scale_shift_table.shape[-1]
    adapter = RecurrentBlock(student.transformer_blocks[11], width)
    student.transformer_blocks[11] = adapter
    student.to(device)
    teacher_state = {}
    teacher.transformer_blocks[14].register_forward_hook(lambda _m, _inp, out: teacher_state.__setitem__("hidden", out))
    student_ddp = DDP(student, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    start = 0
    checkpoint = output / "latest.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        student.load_state_dict(saved["student"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        del saved
    config = {"base_model": "Efficient-Large-Model/Sana_600M_512px_diffusers", "teacher_layers": 28,
              "student_layers": 24, "student_keep_indices": keep, "data": "DOCCI", "train_count": len(train_rows),
              "val_count": len(val_rows), "world": world, "steps": args.steps, "lr": args.lr,
              "objective": "real FM MSE + teacher velocity MSE + 0.05 hidden relative MSE",
              "image_size": 512, "text_max_length": 300, "seed": 20260927}
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dist.barrier()
    cache = {}
    if start == 0:
        fixed = validate(val_rows, pipe, teacher, student, adapter, teacher_state,
                         device, cache, root, rank, world, 0)
        if rank == 0:
            write_jsonl(output / "validation.jsonl", fixed)
            print("BASELINE", fixed, flush=True)
            generate(pipe, teacher, student, adapter, val_rows, output, 0, device)
        dist.barrier()
    assert len(train_rows) % world == 0
    steps_per_epoch = len(train_rows) // world
    def row_for_step(step):
        epoch, offset = divmod(step - 1, steps_per_epoch)
        order = list(range(len(train_rows)))
        random.Random(20260927 + epoch).shuffle(order)
        return train_rows[order[offset * world + rank]]
    started = time.time()
    for step in range(start + 1, args.steps + 1):
        row = row_for_step(step)
        seed = 20260927 + step * world + rank
        optimizer.zero_grad(set_to_none=True)
        result = example(row, pipe, teacher, student_ddp, adapter, teacher_state,
                         device, cache, root, seed, grad=True)
        result["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        assert torch.isfinite(grad_norm), (rank, step, grad_norm)
        optimizer.step()
        metrics = torch.tensor([result[k].detach() for k in ("real_mse", "teacher_mse", "hidden_relative_mse",
                                                               "teacher_real_mse", "total")], device=device).float()
        dist.all_reduce(metrics)
        metrics /= world
        if rank == 0:
            write_jsonl(output / "train_history.jsonl", {"step": step, "elapsed_sec": time.time() - started,
                "real_mse": metrics[0].item(), "teacher_mse": metrics[1].item(),
                "hidden_relative_mse": metrics[2].item(), "teacher_real_mse": metrics[3].item(),
                "total": metrics[4].item(), "grad_norm_rank0": float(grad_norm), "cache_rank0": len(cache)})
            if step % 10 == 0:
                print("STEP", step, "real", metrics[0].item(), "teacher", metrics[1].item(),
                      "sec", round(time.time() - started, 1), flush=True)
        write_jsonl(output / f"samples_rank{rank}.jsonl", {"step": step, "example_id": row["example_id"]})
        if step % args.val_every == 0 or step == args.steps:
            fixed = validate(val_rows, pipe, teacher, student, adapter, teacher_state,
                             device, cache, root, rank, world, step)
            if rank == 0:
                write_jsonl(output / "validation.jsonl", fixed)
                print("VALIDATION", fixed, flush=True)
                temp = output / "latest.pt.tmp"
                torch.save({"step": step, "student": student.state_dict(),
                            "optimizer": optimizer.state_dict()}, temp)
                os.replace(temp, checkpoint)
            dist.barrier()
        if step % args.generate_every == 0:
            if rank == 0:
                generate(pipe, teacher, student, adapter, val_rows, output, step, device)
            dist.barrier()
    if rank == 0:
        (output / "PILOT_COMPLETE.json").write_text(json.dumps({"step": args.steps,
            "elapsed_sec_since_resume": time.time() - started, "completed_at": time.time()}), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
