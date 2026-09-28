import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
folder = root / 'results_finetune2m_mid10_bs128_fresh'
old = root / 'results_finetune2m_mid10_hybrid4'
def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
config = json.loads((folder / 'config.json').read_text())
done = json.loads((folder / 'PILOT_COMPLETE.json').read_text())
assert config['fresh_from_pretrained'] is True and config['base_step'] == 0
assert config['batch_per_device'] == 16 and config['gradient_accumulation_steps'] == 1
assert done['step'] == done['updates'] == 774 and done['samples'] == 98992
counts = {1: Counter(), 2: Counter()}
for rank in range(8):
    rows = read(folder / f'samples_rank{rank}.jsonl')
    assert len(rows) == 12374
    for r in rows:
        counts[r['epoch']][r['example_id']] += 1
        assert r['global_batch'] in (88,128)
assert all(len(v) == 49496 and set(v.values()) == {1} for v in counts.values())
assert counts[1] == counts[2]
history = read(folder / 'train_history.jsonl')
assert len(history) == 774
assert [r['step'] for r in history if r['global_batch'] == 88] == [387,774]
assert all(r['full_count'] + r['subnet_count'] == r['global_batch'] for r in history)
validation = read(folder / 'validation.jsonl')
before, after = validation[0], validation[-1]
assert before['step'] == 0 and after['step'] == 774
assert abs(before['teacher_real_mse'] - after['teacher_real_mse']) < 1e-6
timings = {}
all_times = read(folder / 'generation_timing.jsonl')
for kind, step, interval in [('teacher',0,None),('student_loop',774,0),('hybrid_loop',774,4),('full_shared',774,1)]:
    chosen = [r for r in all_times if r['kind'] == kind and r['step'] == step]
    assert sorted(r['case'] for r in chosen) == list(range(4))
    expected = ['teacher'] * 20 if interval is None else ['full' if interval and j % interval == 0 else 'subnet' for j in range(20)]
    assert all(r['call_trace'] == expected for r in chosen)
    timings[kind] = mean(r['seconds'] for r in chosen)
last = history[-250:]
summary = dict(config=config, completion=done, initialization_validation=before, final_validation=after,
    timing_seconds_per_image=timings, exact_two_epoch_coverage=True,
    max_allocated_gib=max(r['peak_allocated_gib'] for r in history),
    max_reserved_gib=max(r['peak_reserved_gib'] for r in history),
    last250_gradient_clip_fraction=mean(r['gradient_clipped'] for r in last),
    last250_gradient_norm_median=median(r['grad_norm'] for r in last))
(root / 'mid10_bs128_fresh_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False), encoding='utf-8')
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
except OSError:
    font = ImageFont.load_default()
columns = [('Teacher 28',folder,0,'teacher'),('BS8: 6000 updates / hybrid',old,6000,'hybrid_loop'),
           ('BS128: fresh 2 epochs / hybrid',folder,774,'hybrid_loop'),
           ('BS128: subnet sampling',folder,774,'student_loop'),('BS128: full shared',folder,774,'full_shared')]
canvas = Image.new('RGB',(512*len(columns),4*550),'white')
draw = ImageDraw.Draw(canvas)
for i in range(4):
    for j,(label,p,step,kind) in enumerate(columns):
        draw.text((j*512+8,i*550+8),label,fill='black',font=font)
        with Image.open(p / f'step{step:04d}_{kind}_{i:02d}.png') as im:
            canvas.paste(im.convert('RGB'),(j*512,i*550+38))
canvas.save(root / 'comparison_mid10_bs128_fresh.jpg',quality=94)
vm = '\n'.join(f"| {v['step']} | {v['real_mse']:.4f} | {v['subnet_no_memory_real_mse']:.4f} | {v['teacher_mse']:.4f} | {v['own_cfg_mse']:.4f} |" for v in validation)
tm = '\n'.join(f'| {kind} | {seconds:.4f} | {timings["teacher"]/seconds:.3f} |' for kind,seconds in timings.items())
report = f'''# 中间十层交替模型：从基座以BS128重训

## 初始化与训练

从原始Sana-600M预训练基座重新初始化网络和AdamW，没有加载BS8或BS128续训的模型、优化器状态。每卡直接16张图，8张H20，全局128，梯度累积1；每张图独立噪声和ODE时间点，按完整/子网路径分组进入一次DDP前向、一次反向传播和参数更新。学习率固定1e-5，梯度裁剪1.0。

中间第10–19层和输入/输出模块共享于28层完整路径及10层子网，前后18层冻结。可训练参数{config['trainable_parameters']:,}，常驻参数{config['resident_parameters']:,}。20个ODE步的第1、5、9、13、17步完整，其余15步子网。完整路径使用当前共享权重并刷新记忆；冻结原teacher用于监督和参考成图。

训练2轮，每轮49,496张、387更新；每轮末批88张，其余批128。完成774更新，共98,992次样本使用。逐rank记录核对每轮每张图恰好一次。数据仍是text-to-image-2M第一个分片，没有覆盖完整2M。纯子网对照和先前续训已按用户指示取消，不能作为完成的配对实验。

真实点训练目标为conditional FM MSE + 双分支teacher MSE +0.1 CFG MSE。每4次更新使用自身局部轨迹，每16次包括一次零记忆真实点；自身轨迹从真实带噪latent上的四步边界展开1–4步，未从纯噪声完整展开20步。

## 固定32条验证

| 更新 | 交替路径真实MSE | 子网零记忆真实MSE | teacher MSE | 自身轨迹CFG MSE |
|---|---:|---:|---:|---:|
{vm}

原始teacher在同一验证上的真实MSE为{after['teacher_real_mse']:.4f}，它是参考值，不是理论下限。时间使用20步离散网格；不与此前连续随机时间的MSE直接比较。配置路径MSE含全量/子网调用混合，不能单独证明子网能力；子网零记忆MSE作为共同路径补充指标。

日志额外保存每个更新的full/subnet样本数、真实MSE、teacher MSE和裁剪前梯度范数。最后250次更新裁剪比例{summary['last250_gradient_clip_fraction']:.1%}，梯度范数中位数{summary['last250_gradient_norm_median']:.4f}。训练数据中的随机时间点与模式不同，逐更新路径误差仅用于诊断，不当作固定验证的替代。

## 耗时

训练循环{done['elapsed_training_sec']/60:.2f}分钟，包含验证、样图及检查点，排除初始化和CFS复制。8卡中最大的PyTorch峰值已分配显存{summary['max_allocated_gib']:.2f}GiB，预留显存{summary['max_reserved_gib']:.2f}GiB。

| 采样方式 | 平均秒/张 | 相对原模型速度比 |
|---|---:|---:|
{tm}

512×512、20步、CFG4.5、相同4条提示词和种子4200–4203；计时包含文本编码、采样、VAE解码，排除加载和PNG保存，仅为初步测速。

## 证据边界与文件

`comparison_mid10_bs128_fresh.jpg`给出原teacher、此前BS8的6000更新、此次BS128两轮重训、此次纯子网采样和完整共享路径。此前BS8看过48,000张、此次看过98,992张，更新次数也不同；两者的差异不能单独归因于batch。四张视觉案例和32条MSE验证不替代完整生图benchmark。

- `results_finetune2m_mid10_bs128_fresh/`：完整检查点、配置、逐更新日志、逐样本记录、验证、样图和完成标记。
- `train_sana_mid10_bs128_fresh.py`、`run_mid10_bs128_fresh.sh`：模型代码及真实命令。
- `mid10_bs128_fresh_summary.json`：覆盖核验和数值。
- `logs/mid10_bs128_fresh_checkpoint_sha256.txt`：快速盘与CFS检查点哈希。
'''
(root / 'MID10_BS128_FRESH_REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(summary,indent=2,ensure_ascii=False))
