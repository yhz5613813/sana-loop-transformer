"""Fresh BS128 experiments: boundary residual reuse, teacher targets, on-policy inputs."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import SanaPipeline

from train_sana_mid10_bs128_fresh import BatchedMiddleNetwork, load_batch
from train_sana_mid10_interleaved import validate
from train_sana_finetune2m_shared10 import pair_metrics, write_jsonl


class ResidualBridge(nn.Module):
    def __init__(self, width, rank=256):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.nn.functional.silu(self.down(self.norm(x)))).to(x.dtype)


class ExperimentalNetwork(BatchedMiddleNetwork):
    def __init__(self, teacher, bridge=False):
        super().__init__(teacher)
        self.bridge_enabled = bridge
        self.bridge_active = False
        self.boundaries = {}
        if bridge:
            width = self.full.transformer_blocks[9].scale_shift_table.shape[-1]
            self.input_bridge = ResidualBridge(width)
            self.output_bridge = ResidualBridge(width)
            self.full.transformer_blocks[9].register_forward_pre_hook(self.before_mid, with_kwargs=True)
            self.full.transformer_blocks[18].register_forward_hook(self.after_mid)

    def before_mid(self, module, args, kwargs):
        if not self.bridge_active:
            return None
        if 'hidden_states' in kwargs:
            kwargs = dict(kwargs)
            h = self.input_bridge(kwargs['hidden_states'])
            kwargs['hidden_states'] = h
        else:
            h = self.input_bridge(args[0])
            args = (h, *args[1:])
        self.boundaries['input'] = h
        return args, kwargs

    def after_mid(self, module, args, output):
        if not self.bridge_active:
            return None
        assert torch.is_tensor(output)
        h = self.output_bridge(output)
        self.boundaries['output'] = h
        return h

    def forward(self, *args, use_full=None, full_mask=None, **kwargs):
        # Training points use a common route, with independent noise times per image.
        assert full_mask is None
        actual_full = use_full if use_full is not None else (
            self.anchor_every > 0 and self.step_index % self.anchor_every == 0)
        self.bridge_active = self.bridge_enabled and not actual_full
        self.boundaries = {}
        try:
            return super().forward(*args, use_full=use_full, **kwargs)
        finally:
            self.bridge_active = False



class CachedBoundaryNetwork(ExperimentalNetwork):
    """Reuse frozen anchor residuals of blocks 0:9 and 19:28 between full calls."""
    def __init__(self, teacher):
        super().__init__(teacher, bridge=True)
        self.outer_cache = {}
        self.anchor_features = {}
        self.cache_enabled = True
        self.full.transformer_blocks[0].register_forward_pre_hook(self.capture_stem, with_kwargs=True)
        self.full.transformer_blocks[27].register_forward_hook(self.capture_tail)

    def reset(self, anchor_every=0, memory=True):
        super().reset(anchor_every=anchor_every, memory=memory)
        self.outer_cache = {}
        self.anchor_features = {}

    def capture_stem(self, module, args, kwargs):
        self.anchor_features['stem'] = (kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]).detach()

    def before_mid(self, module, args, kwargs):
        h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
        if not self.bridge_active:
            self.anchor_features['mid_input'] = h.detach()
            return None
        if self.cache_enabled and 'input' in self.outer_cache:
            residual = self.outer_cache['input']
            assert residual.shape == h.shape and not residual.requires_grad
            h = h + residual.to(h.dtype)
        h = self.input_bridge(h)
        self.boundaries['input'] = h
        if 'hidden_states' in kwargs:
            kwargs = dict(kwargs, hidden_states=h)
        else:
            args = (h, *args[1:])
        return args, kwargs

    def after_mid(self, module, args, output):
        if not self.bridge_active:
            self.anchor_features['mid_output'] = output.detach()
            return None
        h = output
        if self.cache_enabled and 'output' in self.outer_cache:
            residual = self.outer_cache['output']
            assert residual.shape == h.shape and not residual.requires_grad
            h = h + residual.to(h.dtype)
        h = self.output_bridge(h)
        self.boundaries['output'] = h
        return h

    def capture_tail(self, module, args, output):
        # Only the full route contains block 27. These tensors are runtime state,
        # never parameters or checkpoint contents, and are refreshed every four calls.
        f = self.anchor_features
        self.outer_cache = {
            'input': (f['mid_input'].float() - f['stem'].float()).detach(),
            'output': (output.detach().float() - f['mid_output'].float()).detach(),
        }
        self.anchor_features = {}


class TeacherBoundaries:
    def __init__(self, teacher):
        self.values = {}
        teacher.transformer_blocks[9].register_forward_pre_hook(self.before, with_kwargs=True)
        teacher.transformer_blocks[27].register_forward_hook(self.after)

    def before(self, module, args, kwargs):
        self.values['input'] = (kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]).detach()

    def after(self, module, args, output):
        self.values['output'] = output.detach()


def velocity(model, x, emb, mask, sigma, **kwargs):
    return model(hidden_states=x.repeat(2,1,1,1), encoder_hidden_states=emb,
                 encoder_attention_mask=mask, timestep=sigma.repeat(2).float()*1000,
                 return_dict=False, **kwargs)[0]


def image_errors(v, ref, target):
    vu, vc = v.float().chunk(2)
    tu, tc = ref.float().chunk(2)
    reduce = lambda x: x.square().flatten(1).mean(1)
    return reduce(vc-target), (reduce(vu-tu)+reduce(vc-tc))/2, reduce((vu+4.5*(vc-vu))-(tu+4.5*(tc-tu)))


class SequenceObjective(nn.Module):
    def __init__(self, network, teacher, features, variant, solver):
        super().__init__()
        self.network = network
        object.__setattr__(self, 'teacher', teacher)
        object.__setattr__(self, 'teacher_features', features)
        self.variant = variant
        self.solver = solver
        self.use_rollout = 'rollout' in variant
        self.real_weight = 1.0 if variant == 'cache_bridge' else 0.0

    @torch.no_grad()
    def collect_own_inputs(self, eps, emb, mask, starts, grid):
        # Replay the actual complete DPM prefix from pure noise. Cache a four-call
        # segment independently for each image; keep the real image batch at 128.
        # This is stop-gradient on-policy replay, with no BPTT through the solver.
        network = self.network
        network.reset()
        solver = copy.deepcopy(self.solver)
        x = eps.clone()
        inputs = [torch.empty_like(x) for _ in range(4)]
        covered = torch.zeros((4, len(x)), dtype=torch.bool, device=x.device)
        last = int(starts.max().item()) + 3
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for k in range(last + 1):
                for j in range(4):
                    selected = starts + j == k
                    inputs[j][selected] = x[selected]
                    covered[j, selected] = True
                if k == last:
                    break
                sigma = grid[k].expand(len(x))
                v = velocity(network, x, emb, mask, sigma, use_full=k % 4 == 0)
                vu, vc = v.float().chunk(2)
                x = solver.step((vu + 4.5 * (vc - vu)).to(x.dtype),
                                solver.timesteps[k], x, return_dict=False)[0].detach()
        assert bool(covered.all())
        network.reset()
        return inputs

    def forward(self, z, eps, emb, mask, starts, grid, reset_each=None):
        network = self.network
        reset_each = False if reset_each is None else reset_each
        network.reset()
        losses, diagnostics = [], []
        target = (eps-z).float()
        own_inputs = self.collect_own_inputs(eps, emb, mask, starts, grid) if self.use_rollout else None
        incoming_rms = []
        for j in range(4):
            if reset_each:
                network.reset()
            if network.adapter.state is not None:
                assert not network.adapter.state.requires_grad
                incoming_rms.append(network.adapter.state.float().square().mean().sqrt().detach())
            sigma = grid[starts+j]
            w = sigma.to(z.dtype)[:,None,None,None]
            x = own_inputs[j] if self.use_rollout else (1-w)*z+w*eps
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                ref = velocity(self.teacher,x,emb,mask,sigma)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                v = velocity(network,x,emb,mask,sigma,use_full=j==0)
                vc, teacher_loss, cfg_loss, tc = pair_metrics(v,ref,target)
                real_loss = v.new_zeros((), dtype=torch.float32) if self.use_rollout else (vc-target).square().mean()
                feature_loss = v.new_zeros((),dtype=torch.float32)
                if network.bridge_enabled and j:
                    for name in ('input','output'):
                        pred = network.boundaries[name].float()
                        reference = self.teacher_features.values[name].float()
                        assert pred.shape == reference.shape
                        feature_loss = feature_loss + (pred-reference).square().mean()/(reference.square().mean()+1e-6)
                    feature_loss = feature_loss/2
                losses.append(self.real_weight*real_loss+teacher_loss+0.1*cfg_loss+0.1*feature_loss)
                diagnostics.append(torch.stack((real_loss.detach(),teacher_loss.detach(),
                                                 cfg_loss.detach(),feature_loss.detach())))
        diagnostics = torch.stack(diagnostics)
        state_rms = torch.stack(incoming_rms).mean() if incoming_rms else diagnostics.new_zeros(())
        return torch.stack(losses).mean(), diagnostics, state_rms


@torch.no_grad()
def trajectory_validation(rows,pipe,teacher,network,device,root,sigmas,solver,rank,world,step,case_limit=32,case_offset=0):
    network.eval()
    ids = list(range(rank,min(case_limit,len(rows)),world))
    chosen = [rows[i] for i in ids]
    z,emb,mask = load_batch(chosen,pipe,device,root)
    eps = torch.cat([torch.randn(z[i:i+1].shape,device=device,dtype=z.dtype,
                       generator=torch.Generator(device=device).manual_seed(20261000+case_offset+idx)) for i,idx in enumerate(ids)])
    grid = torch.tensor(sigmas,device=device)
    target = (eps-z).float()
    records = [dict(case=case_offset+i,example_id=rows[i]['example_id'],step=step) for i in ids]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for mode in ('normal','zero','shuffle'):
            network.reset()
            errors, refs, buckets, rms = [],[],[],[]
            for k in range(20):
                full = k%4==0
                sigma = grid[k].expand(len(ids))
                w = sigma.to(z.dtype)[:,None,None,None]
                x = (1-w)*z+w*eps
                if not full and mode == 'zero':
                    network.adapter.state = None
                if not full and mode == 'shuffle' and network.adapter.state is not None:
                    # Rotate examples independently in conditional/unconditional halves.
                    a,b = network.adapter.state.chunk(2)
                    network.adapter.state = torch.cat((a.roll(1,0),b.roll(1,0)))
                ref = velocity(teacher,x,emb,mask,sigma)
                v = velocity(network,x,emb,mask,sigma,use_full=full)
                e = torch.stack(image_errors(v,ref,target),dim=1)
                errors.append(e)
                refs.append((ref.float().chunk(2)[1]-target).square().flatten(1).mean(1))
                buckets.append(full)
                rms.append(float(network.adapter.state.float().square().mean().sqrt()))
            errors = torch.stack(errors,dim=1)
            selector = torch.tensor(buckets,device=device)
            for i,record in enumerate(records):
                record[mode] = dict(real_mse=errors[i,:,0].mean().item(),
                    teacher_mse=errors[i,:,1].mean().item(),cfg_mse=errors[i,:,2].mean().item(),
                    full_real_mse=errors[i,selector,0].mean().item(),
                    subnet_real_mse=errors[i,~selector,0].mean().item(),
                    teacher_real_mse=torch.stack(refs,dim=1)[i].mean().item(),
                    by_ode_step=errors[i].cpu().tolist(),mean_state_rms=sum(rms)/20)
        network.reset()
        own = eps.clone()
        own_solver = copy.deepcopy(solver)
        teacher_x = eps.clone()
        teacher_solver = copy.deepcopy(solver)
        own_errors = []
        for k in range(20):
            sigma = grid[k].expand(len(ids))
            ref = velocity(teacher,own,emb,mask,sigma)
            v = velocity(network,own,emb,mask,sigma,use_full=k%4==0)
            own_errors.append(torch.stack(image_errors(v,ref,target)[1:],dim=1))
            vu,vc = v.float().chunk(2)
            own = own_solver.step((vu+4.5*(vc-vu)).to(own.dtype),own_solver.timesteps[k],own,return_dict=False)[0]
            tv = velocity(teacher,teacher_x,emb,mask,sigma)
            tu,tc = tv.float().chunk(2)
            teacher_x = teacher_solver.step((tu+4.5*(tc-tu)).to(teacher_x.dtype),
                              teacher_solver.timesteps[k],teacher_x,return_dict=False)[0]
        own_errors = torch.stack(own_errors,dim=1)
        endpoints = (own.float()-teacher_x.float()).square().flatten(1).mean(1)
        for i,record in enumerate(records):
            record['own20'] = dict(teacher_mse=own_errors[i,:,0].mean().item(),
                cfg_mse=own_errors[i,:,1].mean().item(),teacher_endpoint_latent_mse=endpoints[i].item(),
                by_ode_step=own_errors[i].cpu().tolist())
    gathered = [None]*world
    dist.all_gather_object(gathered,records)
    flat = sorted([r for group in gathered for r in group],key=lambda r:r['case'])
    network.reset()
    network.train()
    summary = dict(step=step,kind='fixed_images_all20_sigmas',case_offset=case_offset,cases=len(flat),noise_points=len(flat)*20)
    for mode in ('normal','zero','shuffle','own20'):
        summary[mode] = {k:sum(r[mode][k] for r in flat)/len(flat)
                        for k,v in flat[0][mode].items() if isinstance(v,(float,int))}
    return summary,flat


def generate(pipe, teacher, network, rows, output, step, device):
    teacher.eval()
    network.eval()
    for i, row in enumerate(rows[:4]):
        for kind, interval in (("teacher", None), ("hybrid_loop", 4), ("full_shared", 1)):
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
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('cache_bridge','distill_bridge','rollout_distill_bridge','cache_rollout_distill'),required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--max-updates',type=int,default=0)
    parser.add_argument('--val-every',type=int,default=200)
    args = parser.parse_args()
    dist.init_process_group('nccl')
    rank,world = dist.get_rank(),dist.get_world_size()
    assert world==8
    local = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local}')
    torch.cuda.set_device(device)
    torch.manual_seed(20260927+rank)
    torch.backends.cuda.matmul.allow_tf32=True
    root,out = args.root,args.output
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((root/'manifest_finetune2m.json').read_text())
    train,val = manifest['train'],manifest['val']
    assert len(train)%world==0
    pipe = SanaPipeline.from_pretrained(root/'model',variant='fp16',torch_dtype=torch.float16).to(device)
    pipe.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    pipe.text_encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    teacher = pipe.transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    network = (CachedBoundaryNetwork(teacher) if args.variant.startswith('cache_')
               else ExperimentalNetwork(teacher, bridge=True)).to(device)
    features = TeacherBoundaries(teacher)
    pipe.scheduler.set_timesteps(20, device=device)
    objective = SequenceObjective(network,teacher,features,args.variant,copy.deepcopy(pipe.scheduler))
    params = [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=1e-5,weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train)/128)
    total = args.epochs*updates_per_epoch
    target = min(total,args.max_updates) if args.max_updates else total
    config = dict(variant=args.variant,fresh_from_pretrained=True,base_step=0,
        base_model='Efficient-Large-Model/Sana_600M_512px_diffusers',
        initialization='original pretrained network and fresh AdamW',data=manifest['source'],shard=manifest['shard'],
        train_count=len(train),val_count=len(val),epochs=args.epochs,updates_per_epoch=updates_per_epoch,
        steps=total,batch_per_device=16,world=8,global_batch_size=128,tail_batch_size=88,
        gradient_accumulation_steps=1,supervised_points_per_image=4,
        objective='mean of four local (real_weight*conditional FM + dual teacher MSE +0.1 CFG +0.1 normalized boundary MSE)',
        real_weight=objective.real_weight,
        sequence='four points selected independently per image; actual student DPM trajectory from pure noise' if objective.use_rollout else 'uniform four-step full-refresh segment; true interpolants',
        memory='detached middle state across four points',
        boundary_residual_cache=isinstance(network,CachedBoundaryNetwork),
        cache_layers_1based=[[1,9],[20,28]] if isinstance(network,CachedBoundaryNetwork) else [],
        cache_refresh='full calls 1,5,9,13,17; detached input/output residuals',
        own_trajectory_training=objective.use_rollout,full_layers=28,student_layers=10,student_keep_indices=list(range(9,19)),
        anchor_every=4,full_ode_steps_1based=[1,5,9,13,17],image_size=512,sampling_steps=20,cfg=4.5,
        bridge_enabled=network.bridge_enabled,bridge_rank=256,feature_weight=0.1 if network.bridge_enabled else 0,
        bridge_target_layers_1based=[9,28],bridge_initialization='zero output weights and biases; initially identity',
        lr=1e-5,grad_clip=1.0,seed=20260927,
        trainable_parameters=sum(p.numel() for p in params),
        resident_parameters=sum(p.numel() for p in network.parameters()),
        training_forward_calls_per_image=dict(student_with_grad=4,teacher=4,student_replay_max=19 if objective.use_rollout else 0),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
            for name in ['train_sana_mid10_bs128_fresh.py','train_sana_mid10_interleaved.py','train_sana_finetune2m_shared10.py']},
        zero_shuffle_ablation='middle recurrent state only; boundary cache remains enabled',
        comparison_scope='same data/order/noise/4 supervised points; replay adds no-gradient student compute')
    existing=out/'latest.pt'
    completed=0
    if existing.exists():
        raise RuntimeError('Fresh training requires a new output directory: '+str(out))
    if rank==0:
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        print('CONFIG',config,flush=True)
    ddp=DDP(objective,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False)
    pipe.scheduler.set_timesteps(20,device=device)
    sigmas=[float(v) for v in pipe.scheduler.sigmas.cpu()]
    grid=torch.tensor(sigmas,device=device)
    solver=copy.deepcopy(pipe.scheduler)
    cache={}
    def evaluate(step):
        trajectories,records=trajectory_validation(val,pipe,teacher,network,device,root,sigmas,solver,rank,world,step)
        if rank==0:
            write_jsonl(out/'trajectory_validation.jsonl',trajectories)
            (out/f'trajectory_cases_step{step:04d}.json').write_text(json.dumps(records),encoding='utf-8')
            print('VALIDATION',trajectories,flush=True)
    if completed==0 and not args.max_updates:
        evaluate(0)
        if rank==0:
            generate(pipe,teacher,network,val,out,0,device)
    dist.barrier()
    started=time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for update in range(completed+1,target+1):
        epoch,offset=divmod(update-1,updates_per_epoch)
        order=list(range(len(train)))
        random.Random(20260927+epoch).shuffle(order)
        begin=offset*128
        ids=order[begin:min(begin+128,len(train))][rank::world]
        rows=[train[i] for i in ids]
        seeds=[20260927+epoch*len(train)+begin+rank+i*world for i in range(len(rows))]
        z,emb,mask=load_batch(rows,pipe,device,root)
        eps=torch.cat([torch.randn(z[i:i+1].shape,device=device,dtype=z.dtype,
                        generator=torch.Generator(device=device).manual_seed(seed)) for i,seed in enumerate(seeds)])
        starts=torch.tensor([4*random.Random(seed).randrange(5) for seed in seeds],device=device)
        if args.max_updates and update == 1 and isinstance(network, CachedBoundaryNetwork):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                sigma = grid[starts]
                w = sigma.to(z.dtype)[:, None, None, None]
                x = (1-w)*z+w*eps
                network.reset()
                full = velocity(network,x,emb,mask,sigma,use_full=True,remember=False)
                cached = velocity(network,x,emb,mask,sigma,use_full=False,remember=False)
                network.reset()
                cold = velocity(network,x,emb,mask,sigma,use_full=False,remember=False)
                cache_error = (cached.float()-full.float()).square().mean()
                cold_error = (cold.float()-full.float()).square().mean()
                assert cache_error < 0.1*cold_error, (cache_error,cold_error)
                assert cache_error/(full.float().square().mean()+1e-8) < 0.01
                if rank == 0:
                    print('CACHE_SAME_INPUT_CHECK',dict(cached_mse=cache_error.item(),cold_mse=cold_error.item()),flush=True)
                network.reset()
        optimizer.zero_grad(set_to_none=True)
        # One DDP forward, one backward, one update, per-device batch stays 16 images.
        loss,diagnostic,state_rms=ddp(z,eps,emb,mask,starts,grid,
                                   reset_each=None)
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(params,1.0)
        assert torch.isfinite(grad_norm),(rank,update,grad_norm)
        bridge_grad=0.0
        if network.bridge_enabled:
            bridge_grad=sum(float(m.up.weight.grad.float().norm()) for m in
                            (network.input_bridge,network.output_bridge))
            assert bridge_grad>0,(rank,update,'bridge missing gradient')
        optimizer.step()
        network.reset()
        values=torch.cat((loss.detach().reshape(1),diagnostic.detach().reshape(-1),state_rms.detach().reshape(1)))
        dist.all_reduce(values)
        values/=world
        peaks=torch.tensor([torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)],device=device)
        dist.all_reduce(peaks,op=dist.ReduceOp.MAX)
        for row,seed,index in zip(rows,seeds,starts.tolist()):
            write_jsonl(out/f'samples_rank{rank}.jsonl',dict(update=update,step=update,epoch=epoch+1,
                example_id=row['example_id'],seed=seed,ode_indices=list(range(index,index+4)),input_kind='own_trajectory' if objective.use_rollout else 'true_interpolant',
                local_batch=len(rows),global_batch=len(rows)*world,supervised_points=4))
        if rank==0:
            d=values[1:17].reshape(4,4)
            history=dict(step=update,epoch=epoch+1,elapsed_sec=time.time()-started,
                global_batch=len(rows)*world,supervised_noise_points=len(rows)*world*4,
                total=values[0].item(),real_mse=None if objective.use_rollout else d[:,0].mean().item(),teacher_mse=d[:,1].mean().item(),
                cfg_mse=d[:,2].mean().item(),feature_mse=d[:,3].mean().item(),
                full_real_mse=None if objective.use_rollout else d[0,0].item(),subnet_real_mse=None if objective.use_rollout else d[1:,0].mean().item(),
                by_sequence_position=d.cpu().tolist(),incoming_state_rms=values[-1].item(),
                grad_norm=float(grad_norm),gradient_clipped=bool(grad_norm>1),bridge_grad_norm=bridge_grad,
                memory_gate=float(torch.sigmoid(network.adapter.gate.detach()).mean()),
                peak_allocated_gib=peaks[0].item()/2**30,peak_reserved_gib=peaks[1].item()/2**30)
            write_jsonl(out/'train_history.jsonl',history)
            if update<=3 or update%10==0:
                print('STEP',history,flush=True)
        end_epoch=update%updates_per_epoch==0
        if not args.max_updates and (update%args.val_every==0 or end_epoch or update==target):
            evaluate(update)
        if end_epoch or update==target or (not args.max_updates and update%args.val_every==0):
            if rank==0:
                torch.save(dict(step=update,updates=update,network=network.state_dict(),
                                optimizer=optimizer.state_dict(),config=config),out/'latest.pt.tmp')
                os.replace(out/'latest.pt.tmp',existing)
            dist.barrier()
        if not args.max_updates and (end_epoch or update==target):
            if rank==0:
                generate(pipe,teacher,network,val,out,update,device)
            dist.barrier()
    if not args.max_updates:
        held,held_cases=trajectory_validation(val[32:96],pipe,teacher,network,device,root,
            sigmas,solver,rank,world,target,case_limit=64,case_offset=32)
        if rank==0:
            (out/'heldout_summary.json').write_text(json.dumps(held,indent=2),encoding='utf-8')
            (out/'heldout_cases.json').write_text(json.dumps(held_cases),encoding='utf-8')
            print('HELDOUT',held,flush=True)
    if rank==0:
        marker='SMOKE_COMPLETE.json' if args.max_updates else 'PILOT_COMPLETE.json'
        (out/marker).write_text(json.dumps(dict(step=target,updates=target,epochs=args.epochs,
            samples=args.epochs*len(train) if not args.max_updates else None,
            supervised_noise_points=4*args.epochs*len(train) if not args.max_updates else None,
            elapsed_training_sec=time.time()-started)))
    dist.destroy_process_group()


if __name__=='__main__':
    main()
