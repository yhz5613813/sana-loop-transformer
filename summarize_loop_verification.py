import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from PIL import Image,ImageDraw,ImageFont

parser=argparse.ArgumentParser()
parser.add_argument('--partial',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parent
names=('control4','sequence4','sequence4_bridge')

def read(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def paired_delta(a,b,mode,metric):
    assert [r['example_id'] for r in a]==[r['example_id'] for r in b]
    d=[y[mode][metric]-x[mode][metric] for x,y in zip(a,b)]
    rng=random.Random(20260928)
    boots=sorted(mean(rng.choices(d,k=len(d))) for _ in range(10000))
    return dict(delta=mean(d),ci95=[boots[250],boots[9749]],unit='original validation image',cases=len(d))

summary={}
cases={}
for name in names:
    p=root/f'results_loop_verify_{name}'
    if not (p/'PILOT_COMPLETE.json').exists():
        if args.partial:
            continue
        raise RuntimeError(f'Incomplete experiment: {name}')
    cfg=json.loads((p/'config.json').read_text())
    done=json.loads((p/'PILOT_COMPLETE.json').read_text())
    assert cfg['variant']==name and cfg['fresh_from_pretrained'] and cfg['base_step']==0
    assert cfg['global_batch_size']==128 and cfg['gradient_accumulation_steps']==1
    assert done['updates']==774 and done['samples']==98992 and done['supervised_noise_points']==395968
    counts={1:Counter(),2:Counter()}
    noise=[]
    for rank in range(8):
        rs=read(p/f'samples_rank{rank}.jsonl')
        assert len(rs)==12374
        for r in rs:
            counts[r['epoch']][r['example_id']]+=1
            assert r['supervised_points']==4
            assert r['ode_indices']==list(range(r['ode_indices'][0],r['ode_indices'][0]+4))
            assert r['ode_indices'][0]%4==0
            noise.append((r['update'],rank,r['example_id'],r['seed'],r['ode_indices']))
    assert counts[1]==counts[2]
    assert len(counts[1])==49496 and set(counts[1].values())=={1}
    history=read(p/'train_history.jsonl')
    assert len(history)==774 and [r['step'] for r in history if r['global_batch']==88]==[387,774]
    val=read(p/'validation.jsonl')
    traj=read(p/'trajectory_validation.jsonl')
    assert val[0]['step']==traj[0]['step']==0
    assert val[-1]['step']==traj[-1]['step']==774
    assert abs(val[0]['teacher_real_mse']-val[-1]['teacher_real_mse'])<1e-6
    timings={}
    timing_rows=read(p/'generation_timing.jsonl')
    for kind,step,interval in [('teacher',0,None),('hybrid_loop',774,4),('student_loop',774,0),('full_shared',774,1)]:
        ts=[r for r in timing_rows if r['kind']==kind and r['step']==step]
        assert sorted(r['case'] for r in ts)==list(range(4))
        trace=['teacher']*20 if interval is None else ['full' if interval and k%interval==0 else 'subnet' for k in range(20)]
        assert all(r['call_trace']==trace for r in ts)
        timings[kind]=mean(r['seconds'] for r in ts)
    import hashlib
    sample_hash=hashlib.sha256(json.dumps(noise,sort_keys=True).encode()).hexdigest()
    hashfile=root/f'logs/loop_verify_{name}_sha256.txt'
    hashes=hashfile.read_text().splitlines()
    assert len(hashes)==2 and hashes[0].split()[0]==hashes[1].split()[0]
    summary[name]=dict(config=cfg,completion=done,initial_validation=val[0],final_validation=val[-1],
        initial_trajectory=traj[0],final_trajectory=traj[-1],timing_seconds=timings,
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in history),
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in history),
        last250_clip_fraction=mean(r['gradient_clipped'] for r in history[-250:]),
        sample_schedule_sha256=sample_hash,checkpoint_sha256=hashes[0].split()[0],
        exact_two_epoch_coverage=True,validation_history=val,trajectory_history=traj)
    cases[name]=json.loads((p/'trajectory_cases_step0774.json').read_text())

if summary:
    assert len({s['sample_schedule_sha256'] for s in summary.values()})==1
    teacher=[s['final_trajectory']['normal']['teacher_real_mse'] for s in summary.values()]
    assert max(teacher)-min(teacher)<1e-6
comparisons={}
for a,b in [('control4','sequence4'),('sequence4','sequence4_bridge')]:
    if a in cases and b in cases:
        comparisons[f'{b}_minus_{a}']={f'{mode}_{metric}':paired_delta(cases[a],cases[b],mode,metric)
            for mode,metric in [('normal','real_mse'),('normal','subnet_real_mse'),('own20','cfg_mse'),
                                 ('own20','teacher_endpoint_latent_mse')]}
payload=dict(completed=list(summary),results=summary,paired_comparisons=comparisons,
             interpretation='single training seed; bootstrap resamples original validation cases only')
(root/'loop_verification_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
table='\n'.join(f"| {n} | {s['final_validation']['real_mse']:.5f} | {s['final_trajectory']['normal']['real_mse']:.5f} | {s['final_trajectory']['normal']['subnet_real_mse']:.5f} | {s['final_trajectory']['own20']['cfg_mse']:.5f} | {s['final_trajectory']['own20']['teacher_endpoint_latent_mse']:.5f} |" for n,s in summary.items())
tm='\n'.join(f"| {n} | {s['completion']['elapsed_training_sec']/60:.2f} | {s['timing_seconds']['hybrid_loop']:.4f} | {s['timing_seconds']['teacher']/s['timing_seconds']['hybrid_loop']:.3f} | {s['peak_allocated_gib']:.2f} |" for n,s in summary.items())
ci='\n'.join(f"- {name} / {metric}: 差值 {v['delta']:.6f}，按图像重采样95%区间 [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]。" for name,metrics in comparisons.items() for metric,v in metrics.items())
report=f'''# Sana 四步局部监督与边界桥接验证

## 完成状态

已完成：{', '.join(summary) or '无'}。{'三组尚未全部完成，当前报告为中间记录。' if len(summary)<3 else '三组均完成两轮，检查点已复制到CFS且哈希一致。'}

从相同Sana-600M基座和全新AdamW初始化，真实8×16=BS128，梯度累积1，每更新四步损失平均后一次反向和更新；每组774更新、98,992次图像使用、395,968监督噪声点。每轮49,496张恰好一次，逐样本顺序、噪声种子及时间段一致。

control4：每个监督点清空记忆；sequence4：同图像/噪声的四个有序真实插值点间传递停止梯度的状态；sequence4_bridge：增加子网输入/输出残差MLP桥接，并分别对齐teacher第9、28层后的特征。桥接瓶颈256、输出零初始化，特征损失权重0.1。学习率1e-5、裁剪1。三个新实验均不混入own轨迹训练。

推理为512×512、20步、CFG4.5，第1/5/9/13/17步完整28层，其余中间10层。对照训练时清空状态，评估时仍执行相同交替采样。

## 数值

| 配置 | 旧32点真实MSE | 32图×20点真实MSE | 其中子网真实MSE | 完整自身20步CFG MSE | 对teacher终点latent MSE |
|---|---:|---:|---:|---:|---:|
{table}

旧指标沿用此前的32个固定离散时间点；新指标覆盖同32图全部20个点，不能跨列直接比较。teacher真实MSE是参考值，不是理论下限。生成轨迹CFG误差在模型自身实际输入上对齐teacher；终点差异来自相同初始噪声的两条完整求解轨迹。

### 配对差异

以下差值为后者减前者，负值代表对应误差更小。按32条原始图像重采样，不将同图的20个点视为独立样本。区间不包含训练随机种子的变异。

{ci or '待配对实验完成。'}

## 时间和显存

| 配置 | 训练循环分钟 | 交替秒/张 | 对同运行teacher速度比 | 峰值已分配GiB |
|---|---:|---:|---:|---:|
{tm}

训练循环含验证、样图与检查点，排除初始化/CFS复制。生成时间是四条相同提示词和种子的平均，包含文本编码、采样、VAE解码，排除加载/PNG保存。桥接组参数与额外计算量应同时考虑。

## 判读边界

这验证的是三组Sana迁移消融，不能称为原论文的完整复现。旧BS128实验仅每图一个监督点，并混入局部自身轨迹；历史数值0.8463仅为参考。三组新对照使用相同四步监督预算。

单训练种子、32条验证及四张样图是探索性证据。必须结合样图判断语义与人物结构，不能把MSE下降直接称为画质恢复或无损加速。置零/打乱记忆及逐时间误差见JSON原始记录。

## 文件

`results_loop_verify_control4/`、`results_loop_verify_sequence4/`、`results_loop_verify_sequence4_bridge/`保存配置、检查点、逐样本日志、验证和样图。`loop_verification_summary.json`保存数值、覆盖核对、配对差异和哈希；`comparison_loop_verification.jpg`为最终样图对照。
'''
(root/'LOOP_VERIFICATION_REPORT.md').write_text(report,encoding='utf-8')
if len(summary)==3:
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    except OSError:
        font=ImageFont.load_default()
    columns=[('Teacher 28',root/'results_loop_verify_control4',0,'teacher'),
             ('Previous BS128 / 1 point',root/'results_finetune2m_mid10_bs128_fresh',774,'hybrid_loop'),
             ('Control: 4 points / reset',root/'results_loop_verify_control4',774,'hybrid_loop'),
             ('Sequence: 4 points / carry',root/'results_loop_verify_sequence4',774,'hybrid_loop'),
             ('Sequence + boundary bridge',root/'results_loop_verify_sequence4_bridge',774,'hybrid_loop')]
    canvas=Image.new('RGB',(512*len(columns),550*4),'white')
    draw=ImageDraw.Draw(canvas)
    for i in range(4):
        for j,(label,p,step,kind) in enumerate(columns):
            draw.text((j*512+8,i*550+8),label,font=font,fill='black')
            with Image.open(p/f'step{step:04d}_{kind}_{i:02d}.png') as im:
                canvas.paste(im.convert('RGB'),(j*512,i*550+38))
    canvas.save(root/'comparison_loop_verification.jpg',quality=94)
print(json.dumps(dict(completed=list(summary),paired_comparisons=comparisons),indent=2))
