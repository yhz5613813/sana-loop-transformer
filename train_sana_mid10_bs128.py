"""Continue the shared middle-ten model with a real batch of 16 images per GPU."""
import argparse
import copy
import hashlib
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
from PIL import Image, ImageOps
from diffusers import SanaPipeline
from torch.nn.parallel import DistributedDataParallel as DDP
from train_sana_mid10_interleaved import SharedMiddleNetwork, validate, generate
from train_sana_finetune2m_shared10 import pair_metrics, write_jsonl


class BatchedMiddleNetwork(SharedMiddleNetwork):
    def forward(self, *args, full_mask=None, **kwargs):
        if full_mask is None:
            return super().forward(*args, **kwargs)
        assert not args and kwargs.get('return_dict') is False
        assert full_mask.shape == (kwargs['hidden_states'].shape[0],)
        incoming = self.adapter.state
        output = state = None
        # Both routes belong to one DDP forward and one backward/update.
        for use_full in (True, False):
            ids = torch.where(full_mask == use_full)[0]
            if ids.numel() == 0:
                continue
            self.adapter.state = None if incoming is None else incoming.index_select(0, ids)
            part = {k: v.index_select(0, ids) if torch.is_tensor(v) else v for k, v in kwargs.items()}
            result = super().forward(use_full=use_full, **part)[0]
            if output is None:
                output = result.new_zeros((full_mask.numel(), *result.shape[1:]))
            output = output.index_copy(0, ids, result)
            if self.adapter.state is not None:
                if state is None:
                    state = self.adapter.state.new_zeros((full_mask.numel(), *self.adapter.state.shape[1:]))
                state.index_copy_(0, ids, self.adapter.state)
        self.adapter.state = state
        return (output,)


def load_batch(rows, pipe, device, root):
    pixels = []
    for row in rows:
        with (root / row['tar_file']).open('rb') as archive:
            archive.seek(row['offset'])
            blob = archive.read(row['size'])
        with Image.open(io.BytesIO(blob)) as raw:
            image = ImageOps.pad(ImageOps.exif_transpose(raw).convert('RGB'), (512, 512),
                                 method=Image.Resampling.LANCZOS, color=(127, 127, 127))
            pixels.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    pixels = torch.stack(pixels).to(device, dtype=torch.bfloat16) / 127.5 - 1
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = pipe.vae.encode(pixels).latent * pipe.vae.config.scaling_factor
        emb, mask, neg, neg_mask = pipe.encode_prompt(
            [row['description'] for row in rows], do_classifier_free_guidance=True,
            negative_prompt=[''] * len(rows), max_sequence_length=300, clean_caption=False)
    return z.detach(), torch.cat((neg, emb)).detach(), torch.cat((neg_mask, mask)).detach()


def call(model, x, emb, mask, sigma, full_mask):
    return model(hidden_states=x, encoder_hidden_states=emb, encoder_attention_mask=mask,
                 timestep=sigma.float() * 1000, full_mask=full_mask, return_dict=False)[0]


def batch_example(rows, seeds, pipe, teacher, network, ddp, device, root, sigmas, solver, mode):
    z, emb, mask = load_batch(rows, pipe, device, root)
    batch = len(rows)
    eps = torch.cat([torch.randn(z[i:i+1].shape, device=device, dtype=z.dtype,
                                generator=torch.Generator(device=device).manual_seed(seed))
                     for i, seed in enumerate(seeds)])
    indices = torch.tensor([random.Random(seed).randrange(20) for seed in seeds], device=device)
    grid = torch.tensor(sigmas, device=device)
    sigma = grid[indices]
    weights = sigma.to(z.dtype)[:, None, None, None]
    real_x = (1 - weights) * z + weights * eps
    x = real_x.clone()
    target = (eps - z).float()
    network.reset()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        if mode == 'real':
            active = torch.where(indices > 0)[0]
            if active.numel():
                paired = torch.cat((active, active + batch))
                previous = grid[indices[active] - 1]
                previous_weights = previous.to(z.dtype)[:, None, None, None]
                previous_x = (1 - previous_weights) * z[active] + previous_weights * eps[active]
                full = (indices[active] - 1) % 4 == 0
                call(network, previous_x.repeat(2, 1, 1, 1), emb[paired], mask[paired],
                     previous.repeat(2), full.repeat(2))
                state = network.adapter.state
                memory = state.new_zeros((2 * batch, *state.shape[1:]))
                memory.index_copy_(0, paired, state)
                network.adapter.state = memory
        elif mode == 'own':
            memory = None
            # Group local solver trajectories by their independent target time.
            for index in range(1, 20):
                active = torch.where(indices == index)[0]
                if active.numel() == 0:
                    continue
                paired = torch.cat((active, active + batch))
                start = ((index - 1) // 4) * 4
                local_x = (1 - sigmas[start]) * z[active] + sigmas[start] * eps[active]
                local_solver = copy.deepcopy(solver)
                local_solver.set_begin_index(start)
                network.reset()
                for k in range(start, index):
                    velocity = call(network, local_x.repeat(2, 1, 1, 1), emb[paired], mask[paired],
                                    torch.full((2 * active.numel(),), sigmas[k], device=device),
                                    torch.full((2 * active.numel(),), k % 4 == 0, device=device, dtype=torch.bool))
                    vu, vc = velocity.float().chunk(2)
                    local_x = local_solver.step((vu + 4.5 * (vc - vu)).to(local_x.dtype),
                                               local_solver.timesteps[k], local_x, return_dict=False)[0].detach()
                x.index_copy_(0, active, local_x)
                state = network.adapter.state
                if memory is None:
                    memory = state.new_zeros((2 * batch, *state.shape[1:]))
                memory.index_copy_(0, paired, state)
            network.adapter.state = memory
        reference = teacher(hidden_states=x.repeat(2, 1, 1, 1), encoder_hidden_states=emb,
                            encoder_attention_mask=mask, timestep=sigma.repeat(2).float() * 1000,
                            return_dict=False)[0]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        velocity = call(ddp, x.repeat(2, 1, 1, 1), emb, mask, sigma.repeat(2), (indices % 4 == 0).repeat(2))
        vc, teacher_loss, cfg_loss, tc = pair_metrics(velocity, reference, target)
        real_loss = (vc - target).square().mean() if mode != 'own' else None
        loss = teacher_loss + 0.1 * cfg_loss
        if real_loss is not None:
            loss = loss + real_loss
    return loss, real_loss, teacher_loss, cfg_loss, indices.tolist(), float((tc - target).square().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-per-device', type=int, default=16)
    parser.add_argument('--max-updates', type=int, default=0)
    parser.add_argument('--val-every', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8 and args.batch_per_device == 16
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    torch.manual_seed(20260927 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    root, output = args.root, args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest_finetune2m.json').read_text())
    train, val = manifest['train'], manifest['val']
    assert len(train) % world == 0
    pipe = SanaPipeline.from_pretrained(root / 'model', variant='fp16', torch_dtype=torch.float16).to(device)
    pipe.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    pipe.text_encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    teacher = pipe.transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    network = BatchedMiddleNetwork(teacher).to(device)
    trainable = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01)
    existing = output / 'latest.pt'
    saved = torch.load(existing if existing.exists() else args.resume_from, map_location='cpu', weights_only=False)
    assert saved['config']['anchor_every'] == 4
    network.load_state_dict(saved['network'])
    optimizer.load_state_dict(saved['optimizer'])
    base_step = 6000
    completed = saved.get('additional_updates', 0)
    if not existing.exists():
        assert saved['step'] == base_step
    config = dict(saved['config'])
    del saved
    local_count = len(train) // world
    updates_per_epoch = math.ceil(local_count / args.batch_per_device)
    total_updates = args.epochs * updates_per_epoch
    target_updates = min(total_updates, args.max_updates) if args.max_updates else total_updates
    config.update(batch_per_device=16, gradient_accumulation_steps=1, global_batch_size=128,
                  tail_batch_size=(local_count % 16) * world, additional_epochs=args.epochs,
                  updates_per_epoch=updates_per_epoch, steps=base_step + total_updates,
                  resume_from=str(args.resume_from), base_step=base_step,
                  timestep_sampling='independent per image; mixed routes within one DDP forward',
                  train_mode_sampling='own every4 additional updates; first every16; otherwise real')
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        print('CONFIG', config, flush=True)
    ddp = DDP(network, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    pipe.scheduler.set_timesteps(20, device=device)
    sigmas = [float(v) for v in pipe.scheduler.sigmas.cpu()]
    solver = copy.deepcopy(pipe.scheduler)
    cache = {}
    # Keep the fixed validation unchanged, and record the resumed checkpoint.
    if completed == 0 and not args.max_updates:
        metrics = validate(val, pipe, teacher, network, device, cache, root, sigmas, solver,
                           rank, world, base_step, 4)
        if rank == 0:
            write_jsonl(output / 'validation.jsonl', metrics)
    dist.barrier()
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for update in range(completed + 1, target_updates + 1):
        epoch, offset = divmod(update - 1, updates_per_epoch)
        order = list(range(len(train)))
        random.Random(20260928 + epoch).shuffle(order)
        begin = offset * world * 16
        batch_ids = order[begin:min(begin + world * 16, len(order))][rank::world]
        rows = [train[i] for i in batch_ids]
        assert len(rows) in (16, local_count % 16)
        seeds = [20260927 + 48000 + epoch * len(train) + begin + rank + i * world for i in range(len(rows))]
        mode = 'own' if update % 4 == 0 else ('first' if update % 16 == 1 else 'real')
        optimizer.zero_grad(set_to_none=True)
        loss, real, teacher_mse, cfg_mse, indices, teacher_real = batch_example(
            rows, seeds, pipe, teacher, network, ddp, device, root, sigmas, solver, mode)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(grad_norm), (rank, update, grad_norm)
        optimizer.step()
        network.reset()
        values = torch.stack((loss.detach(), real.detach() if real is not None else loss.new_zeros(()),
                              teacher_mse.detach(), cfg_mse.detach())).float()
        dist.all_reduce(values)
        values /= world
        peaks = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], device=device)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        step = base_step + update
        for row, seed, index in zip(rows, seeds, indices):
            write_jsonl(output / f'samples_rank{rank}.jsonl', dict(step=step, additional_update=update,
                epoch=epoch + 1, mode=mode, example_id=row['example_id'], seed=seed, ode_index=index,
                use_full=index % 4 == 0, local_batch=len(rows), global_batch=len(rows) * world))
        if rank == 0:
            history = dict(step=step, additional_update=update, epoch=epoch + 1, mode=mode,
                elapsed_sec=time.time() - started, global_batch=len(rows) * world,
                total=values[0].item(), real_mse=values[1].item() if real is not None else None,
                teacher_mse=values[2].item(), cfg_mse=values[3].item(),
                peak_allocated_gib=peaks[0].item() / 2**30, peak_reserved_gib=peaks[1].item() / 2**30)
            write_jsonl(output / 'train_history.jsonl', history)
            if update <= 4 or update % 10 == 0:
                print('STEP', history, flush=True)
        end_epoch = update % updates_per_epoch == 0
        if update % args.val_every == 0 or end_epoch or update == target_updates:
            metrics = validate(val, pipe, teacher, network, device, cache, root, sigmas, solver,
                               rank, world, step, 4)
            if rank == 0:
                write_jsonl(output / 'validation.jsonl', metrics)
                print('VALIDATION', metrics, flush=True)
                torch.save(dict(step=step, additional_updates=update, network=network.state_dict(),
                                optimizer=optimizer.state_dict(), config=config), output / 'latest.pt.tmp')
                os.replace(output / 'latest.pt.tmp', existing)
            dist.barrier()
        if (end_epoch or update == target_updates) and not args.max_updates:
            if rank == 0:
                generate(pipe, teacher, network, val, output, step, device)
            dist.barrier()
    if rank == 0:
        marker = 'SMOKE_COMPLETE.json' if args.max_updates else 'PILOT_COMPLETE.json'
        (output / marker).write_text(json.dumps(dict(step=base_step + target_updates,
            additional_updates=target_updates, additional_epochs=args.epochs,
            additional_samples=args.epochs * len(train) if not args.max_updates else None,
            elapsed_sec_since_resume=time.time() - started)))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
