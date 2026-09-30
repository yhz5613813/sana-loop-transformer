"""Validate completed optimization runs and compare matched development/held-out cases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
VARIANTS = ('cache_bridge','distill_bridge','rollout_distill_bridge','cache_rollout_distill')


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def paired(a, b):
    assert [x['example_id'] for x in a] == [x['example_id'] for x in b]
    result = {}
    for mode, metric in [('normal','real_mse'),('normal','teacher_mse'),('own20','cfg_mse'),('own20','teacher_endpoint_latent_mse')]:
        values = [y[mode][metric]-x[mode][metric] for x,y in zip(a,b)]
        rng = random.Random(20260930)
        boot = sorted(mean(rng.choices(values,k=len(values))) for _ in range(10000))
        result[mode+'_'+metric] = dict(delta=mean(values),ci95=[boot[250],boot[9749]],cases=len(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args = parser.parse_args()
    old = json.loads((ROOT/'loop_verification_summary.json').read_text())['results']['sequence4_bridge']
    base = ROOT/'results_loop_verify_sequence4_bridge'
    reference = ROOT/'results_loop_opt_reference'
    assert (reference/'REFERENCE_COMPLETE.json').exists()
    dev_cases = json.loads((base/'trajectory_cases_step0774.json').read_text())
    held_cases = json.loads((reference/'heldout_cases.json').read_text())
    ref_time = rows(reference/'generation_timing.jsonl')
    baseline = dict(dev=old['final_trajectory'],heldout=json.loads((reference/'heldout_summary.json').read_text()),
        seconds=mean(x['seconds'] for x in ref_time if x['kind']=='hybrid_loop'),
        teacher_seconds=mean(x['seconds'] for x in ref_time if x['kind']=='teacher'))
    results = {}
    schedule = None
    for name in VARIANTS:
        output = ROOT/('results_loop_opt_'+name)
        if not (output/'PILOT_COMPLETE.json').exists():
            if args.partial: continue
            raise RuntimeError('Incomplete: '+name)
        config = json.loads((output/'config.json').read_text())
        done = json.loads((output/'PILOT_COMPLETE.json').read_text())
        assert config['variant']==name and config['fresh_from_pretrained'] and config['base_step']==0
        assert config['global_batch_size']==128 and config['gradient_accumulation_steps']==1
        assert done['updates']==774 and done['samples']==98992 and done['supervised_noise_points']==395968
        coverage={1:Counter(),2:Counter()}
        order=[]
        for rank in range(8):
            for record in rows(output/f'samples_rank{rank}.jsonl'):
                coverage[record['epoch']][record['example_id']]+=1
                assert record['supervised_points']==4 and record['ode_indices'][0]%4==0
                order.append((record['update'],rank,record['example_id'],record['seed'],record['ode_indices']))
        assert coverage[1]==coverage[2] and len(coverage[1])==49496 and set(coverage[1].values())=={1}
        digest=hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest()
        if schedule is None: schedule=digest
        assert digest==schedule==old['sample_schedule_sha256']
        hashes=(ROOT/f'logs/loop_opt_{name}_sha256.txt').read_text().splitlines()
        assert len(hashes)==2 and hashes[0].split()[0]==hashes[1].split()[0]
        history=rows(output/'train_history.jsonl')
        assert len(history)==774 and [x['step'] for x in history if x['global_batch']==88]==[387,774]
        trajectory=rows(output/'trajectory_validation.jsonl')
        assert trajectory[0]['step']==0 and trajectory[-1]['step']==774
        cases=json.loads((output/'trajectory_cases_step0774.json').read_text())
        held=json.loads((output/'heldout_cases.json').read_text())
        timings=rows(output/'generation_timing.jsonl')
        selected=[x for x in timings if x['step']==774 and x['kind']=='hybrid_loop']
        assert sorted(x['case'] for x in selected)==list(range(4))
        assert all(x['call_trace']==['full' if k%4==0 else 'subnet' for k in range(20)] for x in selected)
        results[name]=dict(config=config,completion=done,dev=trajectory[-1],initial_dev=trajectory[0],
            heldout=json.loads((output/'heldout_summary.json').read_text()),
            seconds=mean(x['seconds'] for x in selected),
            teacher_seconds=mean(x['seconds'] for x in timings if x['step']==0 and x['kind']=='teacher'),
            dev_delta_vs_bridge=paired(dev_cases,cases),heldout_delta_vs_bridge=paired(held_cases,held),
            checkpoint_sha256=hashes[0].split()[0],sample_schedule_sha256=digest,
            peak_allocated_gib=max(x['peak_allocated_gib'] for x in history))
    payload=dict(completed=list(results),pending=[v for v in VARIANTS if v not in results],baseline=baseline,results=results,
        scope='single training seed; 32 development and 64 held-out images; replay adds training compute')
    (ROOT/'loop_optimization_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    table=[]
    for name,r in [('previous_bridge',baseline),*results.items()]:
        table.append(f"| {name} | {r['dev']['normal']['real_mse']:.5f} | {r['dev']['own20']['cfg_mse']:.5f} | {r['heldout']['normal']['real_mse']:.5f} | {r['heldout']['own20']['cfg_mse']:.5f} | {r['heldout']['own20']['teacher_endpoint_latent_mse']:.5f} | {r['seconds']:.4f} |")
    report='# Sana 循环模型优化结果\n\n已完成：'+(', '.join(results) or '无')+'。待完成：'+(', '.join(payload['pending']) or '无')+'。\n\n'
    report+='| 配置 | 开发真实MSE | 开发自身轨迹CFG MSE | 留出真实MSE | 留出自身轨迹CFG MSE | 留出终点latent MSE | 秒/图 |\n|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(table)+'\n\n'
    report+='各组真实BS128、从头训练两轮774更新。图像顺序、噪声和四点监督时间表与旧桥接对照一致。自身轨迹组额外执行无梯度生成前缀，训练成本不等同。留出64图在固定预算后统一评估，配对区间以图像为单位；完整数值和耗时见JSON。单种子结果不能说明跨种子稳定性，MSE也不能直接代表画质。\n\n'
    for name,r in results.items():
        d=r['heldout_delta_vs_bridge']['own20_cfg_mse']
        report+=f"- {name}：留出自身轨迹CFG误差相对旧桥接的配对差值 {d['delta']:.6f}，95%区间 [{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}]；训练循环 {r['completion']['elapsed_training_sec']/60:.1f} 分钟。\n"
    (ROOT/'LOOP_OPTIMIZATION_REPORT.md').write_text(report,encoding='utf-8')
    columns=[('Teacher',reference,0,'teacher'),('Previous bridge',reference,0,'hybrid_loop')]
    columns += [(name,ROOT/('results_loop_opt_'+name),774,'hybrid_loop') for name in results]
    image=Image.new('RGB',(512*len(columns),550*4),'white')
    draw=ImageDraw.Draw(image)
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    except OSError: font=ImageFont.load_default()
    for col,(label,path,step,kind) in enumerate(columns):
        for case in range(4):
            draw.text((col*512+8,case*550+8),label,font=font,fill='black')
            with Image.open(path/f'step{step:04d}_{kind}_{case:02d}.png') as sample:
                image.paste(sample.convert('RGB'),(col*512,case*550+38))
    image.save(ROOT/'comparison_loop_optimization.jpg',quality=94)
    print(json.dumps(dict(completed=list(results),pending=payload['pending']),indent=2),flush=True)


if __name__=='__main__':
    main()
