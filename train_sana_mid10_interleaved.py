"""Shared middle-ten Sana: compare subnet-only training with full/subnet ODE steps.

Original blocks 9..18 and the input/output heads are shared and trainable.
Other full-path blocks are frozen. Full steps suppress incoming memory and
refresh it from the shared middle block. Inference uses this one shared model;
the frozen reference teacher is only used for training targets.
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import SanaPipeline
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from train_sana_finetune2m_shared10 import RecurrentBlock, load_item, pair_metrics, write_jsonl


class SharedMiddleNetwork(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        full = copy.deepcopy(teacher).to(dtype=torch.float32)
        full.requires_grad_(False)
        keep = list(range(9, 19))
        width = full.transformer_blocks[9].scale_shift_table.shape[-1]
        adapter = RecurrentBlock(full.transformer_blocks[13], width)
        full.transformer_blocks[13] = adapter
        # Copy the module containers; preserve exactly the same parameter objects.
        sub = copy.copy(full)
        sub._modules = full._modules.copy()
        sub._parameters = full._parameters.copy()
        sub._buffers = full._buffers.copy()
        sub.transformer_blocks = nn.ModuleList([full.transformer_blocks[i] for i in keep])
        sub.requires_grad_(True)
        assert sub.transformer_blocks[4] is full.transformer_blocks[13]
        assert sub.proj_out is full.proj_out
        for i, block in enumerate(full.transformer_blocks):
            assert all(p.requires_grad == (i in keep) for p in block.parameters())
        self.full = full
        self.sub = sub
        self.config = full.config
        self.anchor_every = 0
        self.memory_enabled = True
        self.step_index = 0
        self.call_trace = []

    @property
    def adapter(self):
        return self.sub.transformer_blocks[4]

    @property
    def dtype(self):
        return next(self.sub.parameters()).dtype

    @property
    def device(self):
        return next(self.sub.parameters()).device

    def reset(self, anchor_every=0, memory=True):
        self.adapter.reset()
        self.anchor_every = anchor_every
        self.memory_enabled = memory
        self.step_index = 0
        self.call_trace = []

    def forward(self, *args, use_full=None, remember=True, **kwargs):
        automatic = use_full is None
        if automatic:
            use_full = self.anchor_every > 0 and self.step_index % self.anchor_every == 0
            remember = self.memory_enabled
        adapter = self.adapter
        if use_full:
            # This full-path call already rebuilds features from all 28 layers.
            adapter.state = None
            adapter.auto_update = False
            result = self.full(*args, **kwargs)
            adapter.state = adapter.capture.detach() if remember else None
        else:
            if not remember:
                adapter.state = None
            adapter.auto_update = remember
            result = self.sub(*args, **kwargs)
        if automatic:
            self.call_trace.append("full" if use_full else "subnet")
            self.step_index += 1
        return result


def run_reference(model, x, emb, mask, sigma):
    t = torch.full((x.shape[0],), sigma * 1000, device=x.device, dtype=torch.float32)
    return model(hidden_states=x, encoder_hidden_states=emb, encoder_attention_mask=mask,
                 timestep=t, return_dict=False)[0]


def run_network(model, x, emb, mask, sigma, use_full=False, remember=True):
    t = torch.full((x.shape[0],), sigma * 1000, device=x.device, dtype=torch.float32)
    return model(hidden_states=x, encoder_hidden_states=emb, encoder_attention_mask=mask,
                 timestep=t, use_full=use_full, remember=remember, return_dict=False)[0]


def example(row, pipe, teacher, network, device, cache, root, seed, anchor_every,
            sigmas, solver, mode="real", grad=False, ablate=False, train_model=None):
    z, emb, mask, neg_emb, neg_mask = load_item(row, pipe, device, cache, root)
    emb = torch.cat((neg_emb, emb))
    mask = torch.cat((neg_mask, mask))
    generator = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(z.shape, device=device, dtype=z.dtype, generator=generator)
    index = random.Random(seed).randrange(20)
    sigma = sigmas[index]
    real_x = (1 - sigma) * z + sigma * eps
    target = (eps - z).float()
    network.reset()
    x = real_x
    prefix = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if mode == "own" and index > 0:
            # A fresh local DPM history at the preceding block boundary.
            # Training uses 1..4-step segments; inference keeps the full history.
            start = ((index - 1) // 4) * 4
            x = (1 - sigmas[start]) * z + sigmas[start] * eps
            local_solver = copy.deepcopy(solver)
            local_solver.set_begin_index(start)
            for k in range(start, index):
                use_full = anchor_every > 0 and k % anchor_every == 0
                velocity = run_network(network, x.repeat(2, 1, 1, 1), emb, mask,
                                       sigmas[k], use_full=use_full)
                vu, vc = velocity.float().chunk(2)
                guided = vu + 4.5 * (vc - vu)
                x = local_solver.step(guided.to(x.dtype), local_solver.timesteps[k],
                                      x, return_dict=False)[0].detach()
                prefix.append("full" if use_full else "subnet")
        elif mode != "first" and index > 0:
            k = index - 1
            previous_x = (1 - sigmas[k]) * z + sigmas[k] * eps
            use_full = anchor_every > 0 and k % anchor_every == 0
            run_network(network, previous_x.repeat(2, 1, 1, 1), emb, mask,
                        sigmas[k], use_full=use_full)
            prefix.append("full" if use_full else "subnet")
        reference = run_reference(teacher, x.repeat(2, 1, 1, 1), emb, mask, sigma)
    use_full = anchor_every > 0 and index % anchor_every == 0
    model = train_model if grad else network
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        velocity = run_network(model, x.repeat(2, 1, 1, 1), emb, mask, sigma, use_full=use_full)
        vc, teacher_loss, cfg_loss, tc = pair_metrics(velocity, reference, target)
        real_loss = (vc - target).square().mean() if mode != "own" else None
        total = teacher_loss + 0.1 * cfg_loss
        if real_loss is not None:
            total = total + real_loss
    result = {"mode": mode, "ode_index": index, "use_full": use_full, "prefix": prefix,
              "real_mse": real_loss, "teacher_mse": teacher_loss, "cfg_mse": cfg_loss,
              "teacher_real_mse": (tc - target).square().mean() if mode != "own" else None,
              "total": total}
    if mode == "own":
        result["rollout_latent_mse"] = (x.float() - real_x.float()).square().mean()
    if ablate:
        network.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            no_memory = run_network(network, x.repeat(2, 1, 1, 1), emb, mask, sigma,
                                    use_full=use_full, remember=False).float().chunk(2)[1]
            subnet = run_network(network, x.repeat(2, 1, 1, 1), emb, mask, sigma,
                                 use_full=False, remember=False).float().chunk(2)[1]
        result["no_memory_real_mse"] = (no_memory - target).square().mean()
        result["subnet_no_memory_real_mse"] = (subnet - target).square().mean()
    network.reset()
    return result


def validate(rows, pipe, teacher, network, device, cache, root, sigmas, solver,
             rank, world, step, anchor_every):
    network.eval()
    keys = ("real_mse", "teacher_mse", "cfg_mse", "teacher_real_mse",
            "no_memory_real_mse", "subnet_no_memory_real_mse",
            "own_teacher_mse", "own_cfg_mse", "own_latent_mse")
    sums = torch.zeros(len(keys), device=device, dtype=torch.float64)
    for i in range(rank, min(32, len(rows)), world):
        real = example(rows[i], pipe, teacher, network, device, cache, root, 20261000 + i,
                       anchor_every, sigmas, solver, grad=False, ablate=True)
        own = example(rows[i], pipe, teacher, network, device, cache, root, 20261000 + i,
                      anchor_every, sigmas, solver, mode="own", grad=False)
        values = (real["real_mse"], real["teacher_mse"], real["cfg_mse"], real["teacher_real_mse"],
                  real["no_memory_real_mse"], real["subnet_no_memory_real_mse"],
                  own["teacher_mse"], own["cfg_mse"], own["rollout_latent_mse"])
        for j, value in enumerate(values):
            sums[j] += value.detach().double()
    dist.all_reduce(sums)
    network.train()
    return {"step": step, "kind": "fixed_val_32_discrete_sigma", "anchor_every": anchor_every,
            **dict(zip(keys, (sums / min(32, len(rows))).tolist()))}


def generate(pipe, teacher, network, rows, output, step, device):
    teacher.eval()
    network.eval()
    for i, row in enumerate(rows[:4]):
        for kind, interval in (("teacher", None), ("student_loop", 0),
                               ("hybrid_loop", 4), ("full_shared", 1)):
            if kind == "teacher" and step != 0:
                continue
            pipe.transformer = teacher if interval is None else network
            if interval is not None:
                network.reset(anchor_every=interval)
            torch.cuda.synchronize(device)
            started = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipe(prompt=row["description"], height=512, width=512, guidance_scale=4.5,
                             num_inference_steps=20, max_sequence_length=300,
                             generator=torch.Generator(device=device).manual_seed(4200 + i)).images[0]
            torch.cuda.synchronize(device)
            elapsed = time.time() - started
            trace = list(network.call_trace) if interval is not None else ["teacher"] * 20
            if interval is not None:
                expected = ["full" if interval and k % interval == 0 else "subnet" for k in range(20)]
                assert trace == expected, (kind, trace, expected)
            write_jsonl(output / "generation_timing.jsonl", {"step": step, "case": i, "kind": kind,
                        "seconds": elapsed, "call_trace": trace})
            image.save(output / f"step{step:04d}_{kind}_{i:02d}.png")
            network.reset()
    pipe.transformer = teacher
    network.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--anchor-every", type=int, choices=(0, 4), default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--generate-every", type=int, default=1200)
    args = parser.parse_args()
    root = args.root
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(20260927 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((root / "manifest_finetune2m.json").read_text(encoding="utf-8"))
    train_rows, val_rows = manifest["train"], manifest["val"]
    label = "hybrid4" if args.anchor_every else "pure"
    output = Path(os.environ.get("SANA_OUTPUT_DIR", str(root / f"results_finetune2m_mid10_{label}")))
    output.mkdir(exist_ok=True)
    pipe = SanaPipeline.from_pretrained(root / "model", variant="fp16", torch_dtype=torch.float16)
    pipe.to(device)
    pipe.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    pipe.text_encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    teacher = pipe.transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    assert len(teacher.transformer_blocks) == 28
    pipe.scheduler.set_timesteps(20, device=device)
    sigmas = [float(s) for s in pipe.scheduler.sigmas.cpu().tolist()]
    assert len(sigmas) == 21
    solver = copy.deepcopy(pipe.scheduler)
    network = SharedMiddleNetwork(teacher).to(device)
    model_ddp = DDP(network, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    trainable = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    config = {"base_model": "Efficient-Large-Model/Sana_600M_512px_diffusers", "full_layers": 28,
              "student_layers": 10, "student_keep_indices": list(range(9, 19)),
              "shared_weight_paths": True, "frozen_full_blocks": list(range(9)) + list(range(19, 28)),
              "anchor_every": args.anchor_every, "full_ode_steps_1based": [1, 5, 9, 13, 17] if args.anchor_every else [],
              "data": manifest["source"], "shard": manifest["shard"], "train_count": len(train_rows),
              "val_count": len(val_rows), "world": world, "steps": args.steps, "lr": args.lr,
              "trainable_parameters": sum(p.numel() for p in trainable),
              "resident_parameters": sum(p.numel() for p in network.parameters()),
              "objective": "real conditional FM MSE + dual-branch teacher MSE + 0.1 CFG MSE; own segments teacher/CFG only",
              "rollout_schedule": "own every fourth update after100; fresh local DPM history at preceding 4-step boundary, 1..4 solver steps; first-state update every16",
              "memory": "full call clears incoming memory and refreshes shared block13 state; subnet reuses it at its block4",
              "image_size": 512, "sampling_steps": 20, "cfg": 4.5, "seed": 20260927}
    checkpoint = output / "latest.pt"
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["config"]["anchor_every"] == args.anchor_every
        network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        del saved
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dist.barrier()
    cache = {}
    if start == 0:
        fixed = validate(val_rows, pipe, teacher, network, device, cache, root, sigmas, solver,
                         rank, world, 0, args.anchor_every)
        if rank == 0:
            write_jsonl(output / "validation.jsonl", fixed)
            print("BASELINE", fixed, flush=True)
            generate(pipe, teacher, network, val_rows, output, 0, device)
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
        result = example(row, pipe, teacher, network, device, cache, root, seed, args.anchor_every,
                         sigmas, solver, mode=mode, grad=True, train_model=model_ddp)
        result["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                "elapsed_sec": time.time() - started, "real_mse": metrics[0].item() if mode != "own" else None,
                "teacher_mse": metrics[1].item(), "cfg_mse": metrics[2].item(),
                "teacher_real_mse": metrics[3].item() if mode != "own" else None,
                "total": metrics[4].item(), "grad_norm_rank0": float(grad_norm),
                "ode_index_rank0": result["ode_index"], "full_rank0": result["use_full"]})
            if step % 10 == 0:
                print("STEP", step, mode, "real", metrics[0].item() if mode != "own" else None,
                      "teacher", metrics[1].item(), "sec", round(time.time() - started, 1), flush=True)
        write_jsonl(output / f"samples_rank{rank}.jsonl", {"step": step, "mode": mode,
                    "example_id": row["example_id"], "ode_index": result["ode_index"],
                    "use_full": result["use_full"], "prefix": result["prefix"]})
        if step % args.val_every == 0 or step == args.steps:
            fixed = validate(val_rows, pipe, teacher, network, device, cache, root, sigmas, solver,
                             rank, world, step, args.anchor_every)
            if rank == 0:
                write_jsonl(output / "validation.jsonl", fixed)
                print("VALIDATION", fixed, flush=True)
                torch.save({"step": step, "network": network.state_dict(),
                            "optimizer": optimizer.state_dict(), "config": config}, output / "latest.pt.tmp")
                os.replace(output / "latest.pt.tmp", checkpoint)
            dist.barrier()
        if step % args.generate_every == 0 or step == args.steps:
            if rank == 0:
                generate(pipe, teacher, network, val_rows, output, step, device)
            dist.barrier()
    if rank == 0:
        (output / "PILOT_COMPLETE.json").write_text(json.dumps({"step": args.steps,
            "elapsed_sec_since_resume": time.time() - started, "completed_at": time.time()}), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
