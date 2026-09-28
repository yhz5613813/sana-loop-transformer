"""Validate matched coverage and summarize two completed middle-ten runs."""

import hashlib
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
folders = {label: root / f"results_finetune2m_mid10_{label}" for label in ("pure", "hybrid4")}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


configs = {k: json.loads((p / "config.json").read_text()) for k, p in folders.items()}
for key in configs["pure"]:
    if key not in ("anchor_every", "full_ode_steps_1based"):
        assert configs["pure"][key] == configs["hybrid4"][key], key
assert configs["pure"]["anchor_every"] == 0
assert configs["hybrid4"]["anchor_every"] == 4
assert configs["pure"]["student_keep_indices"] == list(range(9, 19))
unique = set()
for rank in range(8):
    a = rows(folders["pure"] / f"samples_rank{rank}.jsonl")
    b = rows(folders["hybrid4"] / f"samples_rank{rank}.jsonl")
    assert len(a) == len(b) == 6000
    for x, y in zip(a, b):
        assert all(x[k] == y[k] for k in ("step", "mode", "example_id", "ode_index")), (rank, x, y)
        unique.add(x["example_id"])
assert len(unique) == 48000
final = {k: rows(p / "validation.jsonl")[-1] for k, p in folders.items()}
assert abs(final["pure"]["teacher_real_mse"] - final["hybrid4"]["teacher_real_mse"]) < 1e-5
completion = {k: json.loads((p / "PILOT_COMPLETE.json").read_text()) for k, p in folders.items()}
assert all(x["step"] == 6000 for x in [*final.values(), *completion.values()])
timings = {}
for label, folder in folders.items():
    all_rows = rows(folder / "generation_timing.jsonl")
    timings[label] = {}
    for kind, interval, step in (("teacher", None, 0), ("student_loop", 0, 6000),
                                ("hybrid_loop", 4, 6000), ("full_shared", 1, 6000)):
        selected = [r for r in all_rows if r["kind"] == kind and r["step"] == step]
        assert sorted(r["case"] for r in selected) == list(range(4))
        expected = (["teacher"] * 20 if interval is None else
                    ["full" if interval and j % interval == 0 else "subnet" for j in range(20)])
        assert all(r["call_trace"] == expected for r in selected)
        timings[label][kind] = mean(r["seconds"] for r in selected)
initial_equal = {}
for kind in ("teacher", "student_loop", "hybrid_loop", "full_shared"):
    initial_equal[kind] = [
        hashlib.sha256((folders["pure"] / f"step0000_{kind}_{i:02d}.png").read_bytes()).hexdigest()
        == hashlib.sha256((folders["hybrid4"] / f"step0000_{kind}_{i:02d}.png").read_bytes()).hexdigest()
        for i in range(4)
    ]
summary = {"matched_samples_noise_time_and_order": True, "unique_samples_per_run": len(unique),
           "configs": configs, "validation": final, "completion": completion, "timings": timings,
           "initial_png_hash_equal": initial_equal}
(root / "mid10_pair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
fm_table = "\n".join(
    f"| {label} | {v['real_mse']:.4f} | {v['subnet_no_memory_real_mse']:.4f} | {v['teacher_mse']:.4f} | {v['own_cfg_mse']:.4f} |"
    for label, v in final.items()
)
timing_table = "\n".join(
    f"| {label} | {kind} | {seconds:.4f} | {timings[label]['teacher']/seconds:.3f} |"
    for label, values in timings.items() for kind, seconds in values.items()
)
elapsed = ", ".join(f"{k}: {d['elapsed_sec_since_resume']:.1f} 秒" for k, d in completion.items())
report = f"""# 中间十层与全量交替 ODE：匹配训练对照

## 实现

中间十层为原 Sana 的零起始第 9–18 层，即第 10–19 层。完整 28 层路径和十层子网共享这些层及输入、输出模块；前后十八层冻结。两版可训练参数均为 {configs['pure']['trainable_parameters']:,}，常驻参数均为 {configs['pure']['resident_parameters']:,}。因此交替版减少部分采样步的计算，仍需要常驻完整网络。

交替配置：20 个 ODE 步的第 1、5、9、13、17 步执行完整路径，其余十五步执行子网。完整路径使用当前模型的共享权重，清空输入记忆后执行，并用共享第 14 层输出刷新记忆；子网在相对第 5 层使用该记忆。冻结的原始教师只在训练监督及参考图中使用，交替推理没有额外教师模型。实际调用记录已校验：每张图恰好 5 次 full + 15 次 subnet。纯子网恰好 20 次 subnet。两种采样都使用连续的 DPM 调度历史。

## 训练对照

`pure` 全部训练点都使用子网；`hybrid4` 按同一 1 full + 3 subnet 规则选择训练网络路径。均从同一预训练基座初始化，8 张 H20、全局 batch 8、学习率 1e-5、6,000 更新。逐个 rank 和逐个更新核对样本 ID、训练模式、ODE 时间索引，完全相同；每版 48,000 个互异训练样本。

本轮按更新数匹配训练预算。交替版额外执行完整路径，训练计算开销更高，尚未做相等 GPU 时间的对照。

数据为 text-to-image-2M 的第一个 50,000 样本分片：49,496 训练、500 留出，未训练完整 2M。噪声与时间由逐样本种子确定。真实点使用 conditional flow MSE + 双分支教师 MSE + 0.1 CFG MSE；100 步后每 4 步训练局部自身轨迹，仅用教师和 CFG 监督。自身轨迹从最近的四步分段边界上的真实带噪 latent 展开 1–4 个 solver 步，局部调度历史重新建立；它不是从纯噪声展开全部二十步的训练。每 16 步还包括零记忆的真实点。验证和成图保留这一证据边界。

## 固定 32 条验证

| 训练方式 | 实际配置路径真实 MSE | 子网零记忆真实 MSE | 实际路径教师 MSE | 自身轨迹 CFG MSE |
|---|---:|---:|---:|---:|
{fm_table}

同一验证点上的原始教师真实 MSE 为 **{final['pure']['teacher_real_mse']:.4f}**，两版核对一致。

实际配置路径指标会受到全量/子网调用比例影响，不能单凭它断言子网权重训练得更好。子网零记忆列在两版中都运行十层子网，用来检查子网本身的预测能力。验证样本、噪声和二十步离散时间网格固定；这个时间网格与此前均匀十层/最后十层实验的连续随机时间验证不同，不直接拼接旧 MSE 数字。

## 交叉成图与时间

每个训练版本都分别使用纯子网采样和交替采样，形成训练方式 × 采样方式的 2×2 对照；另外保存全程完整共享路径的成图，用于检查完整路径的能力保留。四个案例的提示词相同，种子 4200–4203，512×512、20 步、CFG 4.5。

| 训练方式 | 采样方式 | 平均秒/张 | 相对本轮原始教师速度比 |
|---|---|---:|---:|
{timing_table}

计时包括文本编码、采样和 VAE 解码，不包括加载与图片保存，仅四张，属于初步测速。训练循环时间（包括验证、样图和本地检查点）：{elapsed}；不包括初始化及 CFS 同步。

`comparison_mid10_2x2.jpg` 展示原模型及四个交叉条件。`comparison_training_effect_hybrid_sampler.jpg` 固定交替采样，比较 pure-trained 和 hybrid-trained，用于判断训练方式的影响。四个案例的视觉检查不代替完整图像生成 benchmark。

## 结果文件

- `train_sana_mid10_interleaved.py`、`run_mid10_pair.sh`：模型实现与实际训练命令。
- `results_finetune2m_mid10_pure/`、`results_finetune2m_mid10_hybrid4/`：各自的完整检查点、验证、逐步训练和样本记录、调用追踪、成图、完成标记。
- `mid10_pair_summary.json`：匹配检查与数值汇总。
- `logs/mid10_*_train.log`、`logs/mid10_pair_runner.log`：原始日志。
"""
(root / "MID10_INTERLEAVED_REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps({"validation": final, "timings": timings, "elapsed": elapsed,
                  "matched_samples": True, "initial_png_hash_equal": initial_equal}, indent=2))
