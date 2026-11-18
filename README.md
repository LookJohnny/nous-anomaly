# nous-anomaly

在大模型内部激活上做异常检测。同一套特征提取，上面接两类检测器：

- **单类异常检测器**：只用"正常"样本训练，偏离就报警，不需要任何风险样本。
- **有监督线性探针**：作为参照上限，格式兼容原仓库的 `ProbeSpec`。

状态：M1（代码和单元测试）完成；M2 第一轮已在 Qwen3-4B 上跑完验证集，结果见下一节。测试集没有开封。

## 第一轮结果：Qwen3-4B-Instruct-2507，验证集（2026-11-18）

一张 RTX PRO 6000，4160 条记录的激活提取用时 138 秒，0 条失败。原始数字在 `results/`。
标签全部来自原仓库的 LLM 评分者。验证集很小（敌意只有 12 个阳性），所有数字都是探索性的。

**结论：纯单类异常检测能发现情绪语气，但发现不了敌意和附和。**

| 概念 | 单类检测器（没见过任何阳性样本） | 有监督线性探针 | 文本分类器 |
|---|---|---|---|
| 愤怒 / 喜悦 / 悲伤 | **0.99**（Mahalanobis，回复均值，第 18 层） | 1.00 | 1.00 |
| 用户困扰（关键词重的 V1 数据） | 0.67–0.80 | 0.997 | 0.994 |
| 用户困扰（去关键词的 M4 数据） | 0.44–0.65 | 未测 | 未测 |
| 敌意 | 多数设置 0.43–0.77，接近随机 | 0.94–0.97 | 0.94 |
| 附和 | 0.32–0.53，低于或等于随机 | 0.87（回复前，第 27 层） | 0.71 |

怎么读这张表：

1. **提取是对的。** 有监督探针复现了原仓库的量级：敌意 0.94–0.97（V1.1 报告 0.93），附和 0.83–0.87（原报告 0.83）。
2. **情绪在中层最容易被发现。** 单类检测器在相对深度 0.5（第 18 层）达到 0.99，往两头下降，和 E02 的结论一致。
   但情绪数据的阳性样本带有明确的情绪开头句，文本分类器也是 1.00，所以这只说明"语气明显变了能被发现"。
3. **敌意和附和不是"离群"。** 敌意回复在整体激活分布里并不比正常回复更远；附和回复甚至比正常回复更"典型"（低于 0.5）。
   只在同一数据集的正常样本上拟合（`results/per_source_val_qwen3-4b.json`）也没有改善。这符合高维稀释的预期：
   风险信号是少数几个方向上的偏移，淹没在其余两千多维的正常波动里。
4. **去内容处理在真实数据上几乎没有额外收益**，和普通 Mahalanobis 的差别在 0.01 以内。换话题误报本来就不高（新话题正常样本的误报率 3–11%，目标 5%）。
5. 敌意那一行里个别设置能到 0.87（Isolation Forest，前 16 个 token，第 18 层）。这是 60 个设置里的最大值，阳性只有 12 个，不能当作结果。

**一次偏离，如实记录**：按来源单独拟合的那次探索里，M4 数据用的是我自己划分的 "test" 部分（原研究本来就在全部 M4 上评估过）。
它没有经过冻结流程，只能算探索性数字。其余数据集的测试集都没有开封。

**这对方向意味着什么**：要检测敌意、附和这类风险，仍然需要知道往哪个方向看，也就是需要标注或对比数据。
纯单类检测可以留作"语气明显异常"的兜底信号。下一步值得试的是带少量已知风险样本的半监督方法，以及在已知探针方向之外的子空间里再做异常检测（`ConceptSubspace`）。

## 为什么要"去内容"

E02 的结论是：内容解释约 90% 的激活方差，情绪只占 3–5%。在原始空间做异常检测，报出来的主要是"话题新"。
`ContentFreeMahalanobis` 先去掉正常数据里随话题变化的方向，再算距离。

合成数据上的实测（`tests/test_detectors.py`）：

| | 普通 Mahalanobis | 去内容 Mahalanobis |
|---|---|---|
| 没见过的风险偏移（AUROC，越高越好） | 0.995 | 0.996 |
| 远离训练话题的新话题（AUROC，越接近 0.5 越好） | 1.000 | 0.63 |

去内容能大幅降低换话题误报，但内容方向是从数据估计的，做不到完全消除。

## 已知局限

- **高维稀释**。单类检测器不知道该看哪个方向。风险信号只落在一个方向上时，维度越高越难发现（合成数据 96 维、5σ 偏移时 AUROC 约 0.90；真实模型是 2560 维以上）。真实效果要等 M2 的数据。
- 标签来自原仓库的 LLM 评分者，没有人工校准。
- 原仓库的测试集已被原研究开封过，这里的"测试"不是全新封存集。

## 用法

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                   # 22 项（没有原仓库时跳过 1 项）

# 1. 把原仓库的数据转成统一格式（只读原仓库）
python -m nous_anomaly.data --repo ~/ai-emotion-experiment/ai-emotion-experiment-main --out data/records.jsonl

# 2. 提取激活（需要 GPU；多卡时每张卡跑一个分片，中断后重跑会跳过已完成的块）
CUDA_VISIBLE_DEVICES=0 python -m nous_anomaly.extract --model Qwen/Qwen3-4B-Instruct-2507 \
    --records data/records.jsonl --out feats/qwen3-4b --shard 0/2
CUDA_VISIBLE_DEVICES=1 python -m nous_anomaly.extract ... --shard 1/2

# 3. 验证集上扫描层和池化方式 -> 冻结 -> 测试集只开一次
python -m nous_anomaly.evaluate sweep  --records data/records.jsonl --feats feats/qwen3-4b --out out/sweep_val.json
python -m nous_anomaly.evaluate freeze --sweep out/sweep_val.json --out out/FREEZE.json
python -m nous_anomaly.evaluate test   --records data/records.jsonl --feats feats/qwen3-4b --freeze out/FREEZE.json --out out/test.json
```

换模型只需要改 `--model`：层按相对深度选（0.25 / 0.4 / 0.5 / 0.6 / 0.75），不写死层号。

## 模块

| 文件 | 内容 |
|---|---|
| `nous_anomaly/data.py` | 统一记录格式；敌意、用户困扰、情绪、附和四个数据集的适配器；防止同一模板跨划分 |
| `nous_anomaly/extract.py` | 任意 HuggingFace 因果语言模型的激活提取：回复前、回复均值、前 16 个 token |
| `nous_anomaly/detectors.py` | Mahalanobis、去内容 Mahalanobis、kNN、Isolation Forest、自编码器、概念子空间、线性探针 |
| `nous_anomaly/evaluate.py` | AUROC、固定误报率下的召回、按模板分组的 bootstrap、换话题对照、冻结与一次性测试 |

## 数据保密

`data/` 和 `feats/` 含原仓库（专利提交前保密）的文本和由它算出的激活，已在 `.gitignore` 中排除，不要提交或上传到第三方。
`results/` 里只有汇总指标（AUROC、样本数、层号），不含任何原文。
