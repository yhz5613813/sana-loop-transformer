# Sana 跨 ODE Transformer 循环实验

本仓库包含 2026-09-28 的实验代码快照及 2026-09-30 的优化实验入口。基座为 `Efficient-Large-Model/Sana_600M_512px_diffusers`，当前数据来自 `jackyhate/text-to-image-2M` 的一个 WebDataset shard。`SOURCE_SNAPSHOT.json` 记录原始快照文件的 SHA-256；新增优化文件不在原始快照清单中。

## 2026-09-30 优化实验

新增四组固定预算实验：外层残差缓存、移除真实 FM 损失的 teacher 蒸馏、在 student 自身生成轨迹上蒸馏，以及缓存与自身轨迹的组合。均从原始预训练权重开始，真实 BS128、两轮训练。方案与完成条件见 [LOOP_OPTIMIZATION_PROTOCOL.md](LOOP_OPTIMIZATION_PROTOCOL.md)。这些新配置的完整效果尚待训练验证；下方结果属于此前已完成的三组实验。

训练入口为 `train_sana_loop_optimized.py`；原实验机器可用 `bash run_loop_optimization.sh` 顺序运行。队列需要旧桥接组的完整检查点和评估产物，且使用 `/tmp/sana_finetune2m` 作为临时输出。迁移机器时先修改队列路径和准备旧对照产物。单独运行新配置的例子：

```bash
torchrun --standalone --nproc_per_node=8 train_sana_loop_optimized.py \
  --root "$PWD" --variant cache_bridge --epochs 2 --val-every 200 \
  --output "$PWD/results_loop_opt_cache_bridge"
```

输出目录必须尚无检查点。完成后由 `summarize_loop_optimization.py` 检查样本覆盖、噪声时间表和检查点复制哈希，并输出对照报告。

## 当前方案

- 完整网络 28 层；子网使用中间第 10–19 层（从零计数为 9–18）。
- 推理为 512×512、20 次 ODE 调用、CFG 4.5。第 1、5、9、13、17 次运行完整网络，其余 15 次运行子网。
- 真实 global batch size 128：8 卡 × 每卡 16，梯度累积为 1。末批 88。
- 每组从原始预训练权重和全新 AdamW 初始化。49,496 张训练图像、500 张预留验证图像，每组两轮、774 次更新。
- 每张图使用四个共享噪声的有序真实插值点，四个局部损失取平均后反向和更新一次；这不是四倍独立图像 batch。

| 变体 | 四步之间的状态 | 桥接 |
|---|---|---|
| `control4` | 每个监督点前清空 | 无 |
| `sequence4` | 传递状态，停止梯度 | 无 |
| `sequence4_bridge` | 传递状态，停止梯度 | 输入/输出残差 MLP，瓶颈 256，teacher 边界监督权重 0.1 |

当前方案压缩每次调用的网络计算，仍执行 20 次 ODE 调用。它不是 1–4 步采样蒸馏。

## 文件入口

- `train_sana_loop_sequence.py`：当前三组训练入口。它依赖仓库内 `train_sana_mid10_bs128_fresh.py`、`train_sana_mid10_interleaved.py`、`train_sana_finetune2m_shared10.py`。
- `prepare_finetune2m.py`：索引 tar shard，按提示词 hash 分组划分训练和验证。
- `configs/`：本次三组实际运行配置。
- `LOOP_VERIFICATION_PROTOCOL.md`：实验协议。
- `LOOP_VERIFICATION_REPORT.md`、`loop_verification_summary.json`：三组最终数值和配对分析。
- `summarize_loop_verification.py`：从完整运行产物重新生成报告和样图对照。
- 其他训练、预览及汇总脚本记录历史实验；历史 `.sh` 包含原机器绝对路径，复用时需调整。

## 环境和数据准备

实际训练使用 Python 3.11.11 和 CUDA PyTorch，关键包版本见 `environment_versions.json`、`requirements.txt`。先安装适合目标 GPU/CUDA 的 PyTorch，再安装其余依赖；本快照没有在另一台 GPU 上重跑验证。

在 Linux 仓库根目录运行：

```bash
pip install -r requirements.txt
hf download Efficient-Large-Model/Sana_600M_512px_diffusers --local-dir model
hf download jackyhate/text-to-image-2M data_512_2M/data_000000.tar \
  --repo-type dataset --local-dir data/finetune2m
python prepare_finetune2m.py --root "$PWD" \
  --shard data/finetune2m/data_512_2M/data_000000.tar --val-count 500
```

索引脚本会生成 `manifest_finetune2m.json`。模型及数据需单独下载，并遵守各自的使用条款。原运行将 shard 放在临时盘；新路径不影响索引逻辑，但应核对 split 数量和数据内容。

## 从头运行当前三组

以下命令需要 8 张 CUDA GPU，输出目录应为新的空目录，不要覆盖已有检查点：

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
for variant in control4 sequence4 sequence4_bridge; do
  target="$PWD/results_loop_verify_${variant}"
  test ! -e "$target/latest.pt" || exit 1
  torchrun --standalone --nproc_per_node=8 train_sana_loop_sequence.py \
    --root "$PWD" --variant "$variant" --epochs 2 --val-every 100 \
    --output "$target" >"logs/loop_verify_${variant}.log" 2>&1 || exit 1
done
python summarize_loop_verification.py
```

运行结束后各目录应有 `PILOT_COMPLETE.json`。仅有本仓库的配置与摘要不足以重新执行汇总程序：它还需要本地生成的检查点、逐样本日志、评估记录和样图。

## 当前结果及边界

三组均已完成。本轮 32 张验证图像 × 20 个点的真实 MSE：`control4` 0.81359，`sequence4` 0.81365，`sequence4_bridge` 0.81105。连续状态组没有明确优于对照；桥接组在这组局部误差评估中小幅改善。完整配对区间、轨迹误差及耗时见最终报告。

这是单训练种子、32 张主要评估图像和四条生成提示词的探索性实验。MSE 改善不能直接代表画质恢复或无损加速。

## 快照范围

上传代码、配置、研究笔记和报告摘要。训练数据、逐图数据 manifest、模型权重、检查点、缓存、原始日志和生成图片不在仓库中。仓库默认私有；没有为第三方模型或数据重新指定许可。
