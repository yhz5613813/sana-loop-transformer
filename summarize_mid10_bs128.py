import json
from collections import Counter
from pathlib import Path
from statistics import mean
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
old = root / 'results_finetune2m_mid10_hybrid4'
new = root / 'results_finetune2m_mid10_bs128'
def read_rows(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
config = json.loads((new / 'config.json').read_text())
done = json.loads((new / 'PILOT_COMPLETE.json').read_text())
assert done['step'] == 6774 and done['additional_updates'] == 774
assert done['additional_samples'] == 98992
assert config['batch_per_device'] == 16 and config['gradient_accumulation_steps'] == 1
epoch_counts = {1: Counter(), 2: Counter()}
for rank in range(8):
    rows = read_rows(new / f'samples_rank{rank}.jsonl')
    assert len(rows) == 12374
    for r in rows:
        epoch_counts[r['epoch']][r['example_id']] += 1
        assert r['global_batch'] in (88, 128)
assert all(len(v) == 49496 and set(v.values()) == {1} for v in epoch_counts.values())
assert epoch_counts[1] == epoch_counts[2]
history = read_rows(new / 'train_history.jsonl')
assert len(history) == 774
assert [r['additional_update'] for r in history if r['global_batch'] == 88] == [387, 774]
validation = read_rows(new / 'validation.jsonl')
before, after = validation[0], validation[-1]
assert before['step'] == 6000 and after['step'] == 6774
original = read_rows(old / 'validation.jsonl')[-1]
assert abs(before['real_mse'] - original['real_mse']) < 1e-5
times = {}
for label, folder, step in [('before', old, 6000), ('after', new, 6774)]:
    records = read_rows(folder / 'generation_timing.jsonl')
    times[label] = {}
    for kind in ('student_loop', 'hybrid_loop', 'full_shared'):
        chosen = [r for r in records if r['step'] == step and r['kind'] == kind]
        assert len(chosen) == 4
        interval = {'student_loop': 0, 'hybrid_loop': 4, 'full_shared': 1}[kind]
        expected = ['full' if interval and i % interval == 0 else 'subnet' for i in range(20)]
        assert all(r['call_trace'] == expected for r in chosen)
        times[label][kind] = mean(r['seconds'] for r in chosen)
times['teacher'] = mean(r['seconds'] for r in read_rows(old / 'generation_timing.jsonl') if r['kind'] == 'teacher')
summary = dict(config=config, completion=done, before=before, after=after, timings=times,
               exact_two_epoch_coverage=True, max_allocated_gib=max(r['peak_allocated_gib'] for r in history),
               max_reserved_gib=max(r['peak_reserved_gib'] for r in history))
(root / 'mid10_bs128_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
columns = [('Teacher 28', old, 0, 'teacher'), ('Before: BS8 / hybrid', old, 6000, 'hybrid_loop'),
           ('After: BS128 / hybrid', new, 6774, 'hybrid_loop'), ('After: BS128 / subnet', new, 6774, 'student_loop'),
           ('After: full shared', new, 6774, 'full_shared')]
canvas = Image.new('RGB', (512 * len(columns), 4 * 550), 'white')
draw = ImageDraw.Draw(canvas)
for i in range(4):
    for j, (label, folder, step, kind) in enumerate(columns):
        draw.text((j * 512 + 8, i * 550 + 8), label, fill='black', font=font)
        with Image.open(folder / f'step{step:04d}_{kind}_{i:02d}.png') as im:
            canvas.paste(im.convert('RGB'), (j * 512, i * 550 + 38))
canvas.save(root / 'comparison_mid10_bs128.jpg', quality=94)
table = '\n'.join(f"| {label} | {v['real_mse']:.4f} | {v['subnet_no_memory_real_mse']:.4f} | {v['own_cfg_mse']:.4f} |" for label,v in [('续训前',before),('续训后',after)])
timing = '\n'.join(f"| {label} | {kind} | {seconds:.4f} |" for label, v in times.items() if isinstance(v,dict) for kind,seconds in v.items())
report = f'''# 中间十层交替模型：BS128 续训两轮

## 设置与覆盖

从交替版6000更新检查点加载模型和AdamW状态，每卡直接16张图，8张H20，全局128，梯度累积1。每张图独立采样噪声和ODE时间点；同一次DDP前向中按完整/子网路径分组，整批一次反向传播、一次参数更新。学习率保持1e-5。中间第10–19层与完整28层路径共享，前后18层冻结；采样第1、5、9、13、17步完整，其余15步子网。

续训2轮，每轮49,496张训练图、387次更新；每轮末批88张，其余128。新增774更新、98,992张图。逐rank日志核验每轮每个训练样本恰好一次，两轮样本集合相同。累计更新6774。数据仍为text-to-image-2M第一个分片，未覆盖完整2M。纯子网对照按用户要求停止，不能形成完成的配对训练结论。

## 相同32条固定验证

| 阶段 | 实际交替路径真实MSE | 子网零记忆真实MSE | 自身局部轨迹CFG MSE |
|---|---:|---:|---:|
{table}

原始teacher真实MSE为{after['teacher_real_mse']:.4f}。验证仍使用同一离散20步时间网格，不与旧的连续随机时间指标直接比较。自身轨迹训练只展开局部1–4步，推理则连续20步。大batch续训同时改变batch和训练数据量，这组前后比较不能单独识别batch的因果作用。

## 计时与显存

训练循环约{done['elapsed_sec_since_resume']/60:.2f}分钟，包含验证、样图和检查点，不包含模型初始化和CFS同步。8卡中的最大PyTorch峰值已分配显存{summary['max_allocated_gib']:.2f}GiB，峰值预留显存{summary['max_reserved_gib']:.2f}GiB。

| 阶段 | 采样方式 | 平均秒/张 |
|---|---|---:|
{timing}

原始teacher参考计时{times['teacher']:.4f}秒/张。512×512、20步、CFG4.5、相同4提示词和种子4200–4203；计时包含文本编码、采样、VAE解码，排除加载与图片保存，仅属初步测速。

## 文件

- `comparison_mid10_bs128.jpg`：原模型、续训前交替、续训后交替、续训后纯子网、续训后完整共享路径。
- `results_finetune2m_mid10_bs128/`：配置、检查点、逐更新日志、每个样本记录、验证、样图、真实调用追踪和完成标记。
- `mid10_bs128_summary.json`：覆盖核验和数值汇总。
- `train_sana_mid10_bs128.py`与`run_mid10_bs128.sh`：代码和实际运行命令。
'''
(root / 'MID10_BS128_REPORT.md').write_text(report, encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
