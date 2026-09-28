"""Read completed runs and write an evidence-based comparison report."""

import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
current = root / "results_finetune2m_last10"
previous = root / "results_finetune2m_shared10"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def timing(folder, kind, step):
    rows = [row for row in read_jsonl(folder / "generation_timing.jsonl")
            if row["kind"] == kind and row["step"] == step]
    assert sorted(row["case"] for row in rows) == list(range(4)), rows
    return mean(row["seconds"] for row in rows)


new_config = json.loads((current / "config.json").read_text())
old_config = json.loads((previous / "config.json").read_text())
for key in ("base_model", "teacher_layers", "student_layers", "data", "shard", "train_count", "val_count",
            "world", "steps", "lr", "trainable_parameters", "objective", "rollout_schedule", "inference",
            "image_size", "text_max_length", "seed"):
    assert new_config[key] == old_config[key], (key, new_config[key], old_config[key])
assert new_config["student_keep_indices"] == list(range(18, 28))
assert old_config["student_keep_indices"] == list(range(0, 28, 3))
completed = json.loads((current / "PILOT_COMPLETE.json").read_text())
assert completed["step"] == 6000
samples = []
for rank in range(8):
    new = read_jsonl(current / f"samples_rank{rank}.jsonl")
    old = read_jsonl(previous / f"samples_rank{rank}.jsonl")
    assert len(new) == 6000 and new == old, (rank, len(new), len(old))
    samples.extend(new)
assert len({row["example_id"] for row in samples}) == 48000
validation = read_jsonl(current / "validation.jsonl")
new = validation[-1]
old = read_jsonl(previous / "validation.jsonl")[-1]
assert new["step"] == old["step"] == 6000
teacher_seconds = timing(current, "teacher", 0)
student_seconds = timing(current, "student_loop", 6000)
summary = {
    "last10_indices": new_config["student_keep_indices"],
    "spread10_indices": old_config["student_keep_indices"],
    "matched_training_samples_and_order": True,
    "unique_training_samples": 48000,
    "training_elapsed_seconds": completed["elapsed_sec_since_resume"],
    "last10_validation": new,
    "spread10_validation": old,
    "teacher_generation_seconds": teacher_seconds,
    "last10_generation_seconds": student_seconds,
    "speedup_vs_matched_teacher": teacher_seconds / student_seconds,
    "latency_reduction_percent": 100 * (1 - student_seconds / teacher_seconds),
    "real_mse_reduction_vs_spread10_percent": 100 * (1 - new["real_mse"] / old["real_mse"]),
}
(current / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
table = "\n".join(
    f"| {v['step']:,} | {v['real_mse']:.4f} | {v['teacher_mse']:.4f} | {v['own_teacher_mse']:.4f} |"
    for v in validation if v["step"] in (0, 1200, 2000, 4000, 6000)
)
change = summary["real_mse_reduction_vs_spread10_percent"]
change_description = f"{'降低' if change >= 0 else '增加'} {abs(change):.2f}%"
report = f"""# Sana 最后十层跨 ODE 步循环实验

## 改动与受控比较

学生初始化由均匀抽取的 `[0,3,6,9,12,15,18,21,24,27]` 改为连续最后十层 `[18,19,20,21,22,23,24,25,26,27]`，即原模型第 19–28 层。原模型输入与输出模块保留；推理直接执行这十层，前十八层不参与。循环状态仍在十层网络第 5 层注入，跨 20 个采样时间步传递。无教师刷新或额外尾层。

两版均从预训练 Sana-600M 初始化并训练 6,000 步，未加载上一版十层学生检查点。8 张 H20、全局 batch 8、学习率 1e-5、相同随机种子、相同训练目标（真实 conditional flow MSE + 双分支教师 MSE + 0.1 CFG MSE，间隔训练自身一步轨迹）。脚本校验两版 8 个 rank 的逐步样本 ID 和训练模式完全一致，均使用 48,000 条互异样本。数据仅为 text-to-image-2M 的第一个 50,000 条分片（49,496 训练、500 留出），没有覆盖全部 2M。

## 固定 32 条验证样本

| 步数 | 真实 MSE | 双分支教师 MSE | 学生自身一步轨迹教师 MSE |
|---:|---:|---:|---:|
{table}

6000 步时：上一版均匀十层真实 MSE **{old['real_mse']:.4f}**，最后十层 **{new['real_mse']:.4f}**；误差相对上一版 **{change_description}**。本轮原始教师真实 MSE **{new['teacher_real_mse']:.4f}**，关闭循环记忆的最后十层 MSE **{new['no_loop_real_mse']:.4f}**。MSE 是固定噪声和时间上的速度预测误差，不等同于感知画质；仅 32 条样本，不代表完整图像生成 benchmark。

## 时间

训练循环耗时 **{completed['elapsed_sec_since_resume']:.1f} 秒**，包括期间验证、样图生成与本地检查点保存，不包括模型初始化、下载和最终 CFS 同步。

本轮同一张 H20，512×512、20 步、CFG 4.5、四个相同提示词/种子：教师平均 **{teacher_seconds:.4f} 秒/张**，最后十层循环平均 **{student_seconds:.4f} 秒/张**，速度比 **{teacher_seconds/student_seconds:.3f} 倍**，耗时降低 **{summary['latency_reduction_percent']:.2f}%**。计时包括文本编码、采样及 VAE 解码，不包括模型加载和保存图片；四张属于初步测速。

## 文件

- `train_sana_finetune2m_last10.py`：最后十层训练代码。
- `results_finetune2m_last10/latest.pt`：最终学生和优化器检查点。
- `results_finetune2m_last10/comparison_teacher_spread10_last10.jpg`：相同提示词/种子的原模型、均匀十层、最后十层循环、最后十层关闭记忆的四列对照，共四个案例。
- `results_finetune2m_last10/comparison_summary.json`：可核查的数值和数据顺序检查结果。
- `logs/last10_train.log`、`results_finetune2m_last10/validation.jsonl`、`train_history.jsonl`、`samples_rank*.jsonl`、`PILOT_COMPLETE.json`：原始日志。
"""
(root / "LAST10_REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(root / "LAST10_REPORT.md")
