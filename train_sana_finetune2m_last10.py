"""A fixed final-ten-block Sana student reused once per diffusion ODE step.

The student carries its own hidden state between solver steps. Training mixes
real noised latents with one-step student rollouts; the latter use same-input
teacher guidance and never borrow a real flow label from another latent.
"""

import argparse
import copy
import io
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
        z, emb, mask, neg_emb, neg_mask = cache[key]
        return tuple(x.to(device) for x in (z, emb, mask, neg_emb, neg_mask))
    if "tar_file" in row:
        with (root / row["tar_file"]).open("rb") as archive:
            archive.seek(row["offset"])
            blob = archive.read(row["size"])
        source = io.BytesIO(blob)
    else:
        source = root / "images" / row["image_file"]
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image = ImageOps.pad(image, (512, 512), method=Image.Resampling.LANCZOS, color=(127, 127, 127))
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16) / 127.5 - 1.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = pipe.vae.encode(pixels).latent * pipe.vae.config.scaling_factor
        emb, mask, neg_emb, neg_mask = pipe.encode_prompt(
            row["description"], do_classifier_free_guidance=True, negative_prompt="",
            max_sequence_length=300, clean_caption=False,
        )
    z = z.detach().cpu().to(torch.bfloat16)
    emb = emb.detach().cpu().to(torch.bfloat16)
    mask = mask.detach().cpu()
    neg_emb = neg_emb.detach().cpu().to(torch.bfloat16)
    neg_mask = neg_mask.detach().cpu()
    if len(cache) < 512:
        cache[key] = (z, emb, mask, neg_emb, neg_mask)
    return tuple(x.to(device) for x in (z, emb, mask, neg_emb, neg_mask))


def run_transformer(model, x, emb, mask, sigma):
    t = torch.full((x.shape[0],), sigma * 1000.0, device=x.device, dtype=torch.float32)
    return model(hidden_states=x, encoder_hidden_states=emb, encoder_attention_mask=mask,
                 timestep=t, return_dict=False)[0]


def pair_metrics(student_velocity, teacher_velocity, target):
    su, sc = student_velocity.float().chunk(2)
    tu, tc = teacher_velocity.float().chunk(2)
    teacher_mse = ((su - tu).square().mean() + (sc - tc).square().mean()) / 2
    cfg_mse = ((su + 4.5 * (sc - su)) - (tu + 4.5 * (tc - tu))).square().mean()
    return sc, teacher_mse, cfg_mse, tc


def example(row, pipe, teacher, student, adapter, device, cache, root, seed,
            mode="real", sigmas=None, solver=None, grad=False, ablate=False):
    z, emb, mask, neg_emb, neg_mask = load_item(row, pipe, device, cache, root)
    pair_emb = torch.cat((neg_emb, emb))
    pair_mask = torch.cat((neg_mask, mask))
    generator = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(z.shape, device=device, dtype=z.dtype, generator=generator)
    rng = random.Random(seed)
    if mode == "own":
        assert sigmas is not None and len(sigmas) == 21 and solver is not None
        index = rng.randrange(20)
        high, low = sigmas[index], sigmas[index + 1]
    else:
        high = rng.uniform(0.35, 0.95)
        low = rng.uniform(0.02, high - 0.05)
    x_high = (1 - high) * z + high * eps
    x_low_real = (1 - low) * z + low * eps
    target = (eps - z).float()
    adapter.reset()
    if mode != "first":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            high_pair = run_transformer(student, x_high.repeat(2, 1, 1, 1), pair_emb, pair_mask, high)
            previous_state = adapter.capture.detach()
        adapter.state = previous_state
    if mode == "own":
        high_u, high_c = high_pair.float().chunk(2)
        guided_high = high_u + 4.5 * (high_c - high_u)
        local_solver = copy.deepcopy(solver)
        local_solver.set_begin_index(index)
        with torch.no_grad():
            x_low = local_solver.step(guided_high.to(x_high.dtype),
                                      local_solver.timesteps[index], x_high,
                                      return_dict=False)[0].detach()
    else:
        x_low = x_low_real
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher_velocity = run_transformer(teacher, x_low.repeat(2, 1, 1, 1), pair_emb, pair_mask, low)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        student_velocity = run_transformer(student, x_low.repeat(2, 1, 1, 1), pair_emb, pair_mask, low)
        sc, teacher_loss, cfg_loss, tc = pair_metrics(student_velocity, teacher_velocity, target)
        real_loss = (sc - target).square().mean() if mode != "own" else None
        total = teacher_loss + 0.1 * cfg_loss
        if real_loss is not None:
            total = total + real_loss
    result = {"mode": mode, "real_mse": real_loss, "teacher_mse": teacher_loss,
              "cfg_mse": cfg_loss, "teacher_real_mse": (tc - target).square().mean()
              if mode != "own" else None, "total": total,
              "sigma_low": low, "sigma_high": high}
    if mode == "own":
        result["rollout_latent_mse"] = (x_low.float() - x_low_real.float()).square().mean()
    if ablate:
        assert mode == "real"
        adapter.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            no_loop = run_transformer(student, x_low.repeat(2, 1, 1, 1), pair_emb, pair_mask, low)
        no_loop_cond = no_loop.float().chunk(2)[1]
        result["no_loop_real_mse"] = (no_loop_cond - target).square().mean()
    adapter.reset()
    return result


def validate(rows, pipe, teacher, student, adapter, device, cache, root, sigmas, solver, rank, world, step):
    student.eval()
    keys = ("real_mse", "teacher_mse", "cfg_mse", "teacher_real_mse",
            "no_loop_real_mse", "own_teacher_mse", "own_cfg_mse", "own_latent_mse")
    sums = torch.zeros(len(keys), device=device, dtype=torch.float64)
    for i in range(rank, min(32, len(rows)), world):
        real = example(rows[i], pipe, teacher, student, adapter, device, cache, root,
                       seed=20261000 + i, mode="real", grad=False, ablate=True)
        own = example(rows[i], pipe, teacher, student, adapter, device, cache, root,
                      seed=20261000 + i, mode="own", sigmas=sigmas, solver=solver, grad=False)
        values = (real["real_mse"], real["teacher_mse"], real["cfg_mse"],
                  real["teacher_real_mse"], real["no_loop_real_mse"],
                  own["teacher_mse"], own["cfg_mse"], own["rollout_latent_mse"])
        for j, value in enumerate(values):
            sums[j] += value.detach().double()
    dist.all_reduce(sums)
    student.train()
    values = (sums / min(32, len(rows))).tolist()
    return {"step": step, "kind": "fixed_val_32", **dict(zip(keys, values))}


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
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--generate-every", type=int, default=1200)
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
    manifest = json.loads((root / "manifest_finetune2m.json").read_text(encoding="utf-8"))
    train_rows, val_rows = manifest["train"], manifest["val"]
    output = Path(os.environ.get("SANA_OUTPUT_DIR", str(root / "results_finetune2m_last10")))
    output.mkdir(exist_ok=True)
    model_path = root / "model"
    pipe = SanaPipeline.from_pretrained(model_path, variant="fp16", torch_dtype=torch.float16)
    pipe.to(device)
    pipe.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    pipe.text_encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    teacher = pipe.transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    assert len(teacher.transformer_blocks) == 28
    pipe.scheduler.set_timesteps(20, device=device)
    sigmas = [float(s) for s in pipe.scheduler.sigmas.cpu().tolist()]
    assert len(sigmas) == 21 and sigmas[0] > sigmas[-1], sigmas
    solver = copy.deepcopy(pipe.scheduler)
    student = copy.deepcopy(teacher).to(dtype=torch.float32)
    student.requires_grad_(True)
    keep = list(range(18, 28))  # original final ten blocks, zero-based indices
    assert len(set(keep)) == 10
    student.transformer_blocks = nn.ModuleList([student.transformer_blocks[i] for i in keep])
    width = student.transformer_blocks[0].scale_shift_table.shape[-1]
    adapter = RecurrentBlock(student.transformer_blocks[4], width)
    student.transformer_blocks[4] = adapter
    assert all(parameter.requires_grad for parameter in student.parameters())
    student.to(device)
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
              "student_layers": 10, "student_keep_indices": keep,
              "initialization": "original contiguous final ten blocks (18..27), from pretrained teacher", "data": manifest["source"],
              "shard": manifest["shard"], "train_count": len(train_rows),
              "val_count": len(val_rows), "world": world, "steps": args.steps, "lr": args.lr,
              "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
              "objective": "real conditional FM MSE + mean dual-branch teacher MSE + 0.1 CFG MSE; own rollout uses teacher/CFG only",
              "rollout_schedule": "every fourth update after step100, one student DPM solver step with a fresh local solver history; first-step zero-state every sixteenth update",
              "inference": "one shared 10-block network call per ODE time, student state across steps, no teacher refresh or tail",
              "image_size": 512, "text_max_length": 300, "seed": 20260927}
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dist.barrier()
    cache = {}
    if start == 0:
        fixed = validate(val_rows, pipe, teacher, student, adapter,
                         device, cache, root, sigmas, solver, rank, world, 0)
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
        mode = "own" if step > 100 and step % 4 == 0 else ("first" if step % 16 == 1 else "real")
        optimizer.zero_grad(set_to_none=True)
        result = example(row, pipe, teacher, student_ddp, adapter,
                         device, cache, root, seed, mode=mode, sigmas=sigmas, solver=solver, grad=True)
        result["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        assert torch.isfinite(grad_norm), (rank, step, grad_norm)
        optimizer.step()
        metrics = torch.stack((result["real_mse"].detach() if mode != "own" else torch.zeros((), device=device),
                               result["teacher_mse"].detach(), result["cfg_mse"].detach(),
                               result["teacher_real_mse"].detach() if mode != "own" else torch.zeros((), device=device),
                               result["total"].detach())).float()
        dist.all_reduce(metrics)
        metrics /= world
        if rank == 0:
            write_jsonl(output / "train_history.jsonl", {"step": step, "mode": mode,
                "elapsed_sec": time.time() - started,
                "real_mse": metrics[0].item() if mode != "own" else None,
                "teacher_mse": metrics[1].item(), "cfg_mse": metrics[2].item(),
                "teacher_real_mse": metrics[3].item() if mode != "own" else None,
                "total": metrics[4].item(), "grad_norm_rank0": float(grad_norm), "cache_rank0": len(cache)})
            if step % 10 == 0:
                print("STEP", step, mode, "real", metrics[0].item() if mode != "own" else None,
                      "teacher", metrics[1].item(), "cfg", metrics[2].item(),
                      "sec", round(time.time() - started, 1), flush=True)
        write_jsonl(output / f"samples_rank{rank}.jsonl", {"step": step, "mode": mode,
                    "example_id": row["example_id"]})
        if step % args.val_every == 0 or step == args.steps:
            fixed = validate(val_rows, pipe, teacher, student, adapter,
                             device, cache, root, sigmas, solver, rank, world, step)
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
