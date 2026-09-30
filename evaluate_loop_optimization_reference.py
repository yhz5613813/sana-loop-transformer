"""Evaluate the frozen previous bridge checkpoint on the new held-out protocol."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import SanaPipeline

from train_sana_loop_sequence import ExperimentalNetwork
from train_sana_loop_optimized import trajectory_validation, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    device = torch.device('cuda:' + os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    root = args.root
    output = root / 'results_loop_opt_reference'
    output.mkdir(exist_ok=True)
    checkpoint = root / 'results_loop_verify_sequence4_bridge/latest.pt'
    pipe = SanaPipeline.from_pretrained(root/'model', variant='fp16', torch_dtype=torch.float16).to(device)
    pipe.vae.to(torch.bfloat16).eval().requires_grad_(False)
    pipe.text_encoder.to(torch.bfloat16).eval().requires_grad_(False)
    teacher = pipe.transformer.to(torch.bfloat16).eval().requires_grad_(False)
    network = ExperimentalNetwork(teacher, bridge=True).to(device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['config']['variant'] == 'sequence4_bridge' and saved['updates'] == 774
    network.load_state_dict(saved['network'])
    del saved
    val = json.loads((root/'manifest_finetune2m.json').read_text())['val']
    pipe.scheduler.set_timesteps(20, device=device)
    sigmas = [float(v) for v in pipe.scheduler.sigmas.cpu()]
    solver = copy.deepcopy(pipe.scheduler)
    summary, cases = trajectory_validation(val[32:96], pipe, teacher, network, device,
        root, sigmas, solver, rank, world, 774, case_limit=64, case_offset=32)
    if rank == 0:
        (output/'heldout_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (output/'heldout_cases.json').write_text(json.dumps(cases),encoding='utf-8')
        generate(pipe, teacher, network, val, output, 0, device)
        (output/'REFERENCE_COMPLETE.json').write_text(json.dumps({
            'checkpoint':str(checkpoint), 'source_updates':774,
            'checkpoint_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
            'heldout_cases':64, 'generation_file_step_label':0}),encoding='utf-8')
        print('REFERENCE_COMPLETE',summary,flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
