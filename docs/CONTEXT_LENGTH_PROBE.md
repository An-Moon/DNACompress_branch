# DNACorpus 上下文长度机制实验

## 研究目标与固定设计

本实验检验：当 genomic language model（gLM）仍采用窗口独立推理时，扩大单窗口上下文能否消除序列尺度 STAP 统计信息带来的压缩收益。

首个正式实验对象为 OrSa。实验从原始序列的前、中、后三个等长坐标区间中，各选取一个确定性的连续片段。三个正式片段长度均为 2,359,296 bp，起点分别为 6,094,848、20,447,232 和 34,799,616 bp。三段合计 7,077,888 bp，占 OrSa 总长度的 16.36%。三个片段互不重叠，并且都完整落在源序列范围内。此前每段 1,179,648 bp 的结果只作为管线 smoke test 保留。

对齐窗口长度依次为 6,144、12,288、24,576、49,152、98,304 和 196,608 bp。所有长度都能整除固定的 196,608 bp 片段对齐单位，并且均可被 Carbon 的 6-mer 长度整除。主要比较是在完全相同的片段和窗口边界上，对比 Carbon/Evo2 standalone 与对应的 gLM+STAP online-Hedge 融合结果。

## 2026-08-09 已完成进度

### 早期 smoke test

- 已生成三个 1,179,648 bp 的早期片段，并保存来源与片段 SHA-256。
- 已完成三个片段、六种窗口长度的 18 组 STAP depth-major 和 position-major trace。
- 已完成三个片段在 6,144 bp 下的 Carbon-3B trace 和 Carbon+STAP 汇总。
- Segment A/B/C 的 Carbon bpb 分别为 1.641209、1.597961、1.629128；STAP 分别为 1.858295、1.767691、1.867243；融合分别为 1.614472、1.526326、1.596265。
- 三段融合相对 Carbon 的改善分别为 -0.026736、-0.071635、-0.032863 bpb，说明初始收益并非由单一片段驱动。

### 扩大后的正式 v2 实验

- 正式结果独立保存在 `outputs/dnacorpus_context_length_probe_orsa_v2`，不会与早期 smoke trace 混合。
- 已完成三个 2,359,296 bp 正式片段的 manifest、源序列切片和 SHA-256 校验。
- 已完成正式片段的全部 18 组 STAP trace。
- 已完成三个正式片段在 6,144 bp 下的 Carbon trace，A/B/C 的融合改善分别为 -0.037277、-0.098182 和 -0.072919 bpb。
- 已完成三个正式片段在 12,288 bp 下的 Carbon trace，A/B/C 的融合改善分别为 -0.032836、-0.087117 和 -0.068266 bpb。
- 已完成三个正式片段在 24,576 bp 下的 Carbon trace。A/B/C 的 Carbon bpb 分别为 1.623851、1.555765、1.587810，融合分别为 1.594851、1.479755、1.524862，融合改善分别为 -0.029000、-0.076010、-0.062948 bpb。
- Carbon 正式网格已全部完成，共 18/18 组 trace。Evo2 无 kernel 主路径通过数学等价的 `compute_filter` 分块扩展到 98,304 bp，共完成 15/18 组；只剩 196,608 bp 的三个片段。单纯的单卡、三卡或四卡按层切分无法解决原始 filter 峰值，最终采用沿独立 system 维的 chunk=128 实现，并通过原长度逐位置零差异验证。
- common-mask 下 Carbon、STAP 和融合分别为 1.648935、1.842516 和 1.611652 bpb，说明该收益不是由基础块首位置评分差异造成的。
- 自定义融合汇总与仓库原有离线 trace 融合函数在浮点精度内一致。

### 三段平均正式结果

下表的标准差是三个固定连续片段之间的样本标准差；融合增益定义为 `fused bpb - standalone bpb`，负值表示融合更好。

| 模型 | 窗口长度 | standalone bpb（均值±SD） | +STAP bpb（均值） | 融合增益（均值±SD） |
|---|---:|---:|---:|---:|
| Carbon | 6,144 | 1.622871 ± 0.023978 | 1.553412 | -0.069459 ± 0.030599 |
| Carbon | 12,288 | 1.603898 ± 0.028696 | 1.541158 | -0.062740 ± 0.027559 |
| Carbon | 24,576 | 1.589142 ± 0.034062 | 1.533156 | -0.055986 ± 0.024266 |
| Carbon | 49,152 | 1.576051 ± 0.039486 | 1.526722 | -0.049329 ± 0.020732 |
| Carbon | 98,304 | 1.568621 ± 0.042152 | 1.523491 | -0.045130 ± 0.018995 |
| Carbon | 196,608 | 1.558898 ± 0.044873 | 1.519489 | -0.039409 ± 0.016229 |
| Evo2 | 6,144 | 1.562406 ± 0.034627 | 1.506551 | -0.055854 ± 0.025046 |
| Evo2 | 12,288 | 1.547070 ± 0.038713 | 1.496022 | -0.051048 ± 0.023035 |
| Evo2 | 24,576 | 1.534329 ± 0.043387 | 1.488830 | -0.045499 ± 0.019991 |
| Evo2 | 49,152 | 1.522455 ± 0.048304 | 1.483098 | -0.039357 ± 0.016728 |
| Evo2 | 98,304 | 1.515869 ± 0.050184 | 1.479141 | -0.036728 ± 0.015605 |
| Evo2-optimized | 196,608 | 1.550402 ± 0.048035 | 1.501315 | -0.049087 ± 0.018268 |

Carbon 的 standalone bpb 随上下文增长单调下降，证明扩大窗口确实提升独立窗口 gLM 的压缩建模能力；但在最大 196,608 bp 窗口下，STAP 仍提供平均 0.039409 bpb 的额外改善。当前结果支持“更长窗口逐步吸收部分 sequence-scale signal，但尚未消除与 STAP 的互补信息”，而不是声称融合收益完全不随窗口变化。

## Trace 保留约定

正式实验保留模型和 STAP 的 target-probability shards、manifest、来源哈希和 checksum。STAP 同时保留原始 depth-major trace 与便于位置分析的 position-major trace。融合结果保存指标、参数和两条来源 trace 的 checksum，不额外复制逐位置融合 trace；后续绘图、固定权重融合和 Hedge 参数分析都可以从冻结的来源 trace 重新计算。

每个模型 depth-major trace 的 manifest 还保留效率字段，包括 `trace_generation_seconds`、`model_seconds`、概率分解或 softmax 时间、概率传输时间、batch 数和 window 数。其中 `trace_generation_seconds` 从模型加载完成后开始计时，不包含 checkpoint/tokenizer 加载，也不应称为完整端到端耗时。总控脚本现另存 `timings/<segment>/w<length>_<model>.json`，记录子进程开始/结束时间、墙钟秒数、退出码和命令。后续统一报告子进程墙钟吞吐量、概率生成吞吐量与模型前向吞吐量；融合阶段另行记录 CPU 汇总耗时。例如 Carbon Segment A × 49,152 bp 的概率生成时间为 2032.14 秒，模型前向为 31.88 秒，说明当前实现的主要工程瓶颈还包括 6-mer 条件分解、概率传输与 trace 构建。

## 后续任务

1. 从冻结 summary 生成逐片段曲线、三段平均曲线、common-mask 对比和逐长度融合增益图。
2. 汇总 manifest 内的概率生成/模型前向耗时，并结合 launcher 记录生成效率表。
3. 决定 Evo2 长窗口采用等价的 `compute_filter` 分块实现，还是从最短窗口开始重跑一条独立 kernel 曲线；不同概率路径不直接拼接。
4. 完成 OrSa 单物种结论审查后，再决定是否扩展到其他 DNACorpus 来源。

正式运行阶段曾使用 GPU0 续跑 Carbon，并用 GPU1、GPU2、GPU3 分别运行 Evo2 Segment C、A、B。Carbon 网格和 Evo2 可运行网格现均已结束，统一 summary 已从冻结 trace 重建。STAP 融合在 CPU 上离线完成，不单独占用 GPU。

已对 Carbon 的 `streaming_cache`（KV cache）路径做独立一致性检查，不覆盖正式 `full_forward` trace。单个 6,144 bp 窗口中，KV 与 full-forward 分别得到 1.710188 和 1.710983 bpb；逐位置目标概率平均绝对差为 0.00673，最大差为 0.1243，超过可接受的浮点误差。当前 KV 实现还因逐 token Python 循环而更慢（约 24.0 秒对 2.25 秒）。因此正式批量 trace 继续使用已验证的 full-forward teacher-forcing 路径；KV cache 保留为后续缓存位置和 attention mask 调试项，在修复并通过逐位置一致性验证前不进入论文结果。

Evo2 使用独立环境 `/home/Hu_xuanwei/.conda/envs/evo2_py311`：Python 3.11.15、PyTorch 2.7.1+cu128、Vortex 1.1.0、FlashAttention 2.8.0.post2、Evo2 0.6.0、Triton 3.3.1。CUDA 可用。Transformer Engine 未安装时 Evo2 7B 回退到 BF16 路径，当前 6,144–49,152 bp 正式 trace 均已完成。PyArrow 未安装，但 trace 入口已解除不必要的 PyArrow 间接依赖，本实验无需为此额外安装。

为解决 98,304 bp 无 kernel 路径在 `HyenaCascade.compute_filter` 中物化 `(D, state_size, L)` 张量导致的 OOM，trace 入口新增可选参数 `--evo2-filter-chunk-systems`。实现只沿相互独立的 D/system 维分块，仍在每个 system 内按原顺序对完整 state 维执行 `sum(1)`，并预分配最终 `(D,L)` 输出。chunk=128 在 6,144 和 49,152 bp 单窗口上与原实现逐位置完全一致（非零差异数为 0，trace bpb 与 checksum 一致），98,304 bp 单窗口与正式首窗口也逐位置完全一致。完整片段在窗口之间显式释放已经复制到 CPU 的 logits/FFT 张量。三个正式片段均已完成，每段 launcher 墙钟约 434 秒。

196,608 bp 的完整优化路径已经验证成功。本地 Evo2 0.6.0 原始 `use_kernels=true` 只覆盖部分 HCS/HCM/HCL 分支：非 gated 短 FIR 仍走普通 PyTorch `conv1d`，HCS全通道grid会越界，HCM/HCL全通道FFT会产生单卡峰值。仓库侧运行时补丁将非 gated FIR接入已有Triton `hcs_depthwise_conv`，并沿完全独立的channel维分别对HCS（1024 channels）、HCM（128）和HCL（128）分块。每次分块都在6,144 bp与未分块kernel trace取得完全相同的checksum。最终196,608 bp单窗口在单张RTX 4090上完成：模型前向65.88秒，输出196,608行，checksum为 `c4e22ca632534ef987362a1e15c5b683199239d96ac40a5c6d15fcd2e3e9fce1`。现有无kernel trace未被覆盖；独立标签 `evo2_optimized` 的完整六长度曲线已完成。

`evo2_optimized` 正式网格现已完成18/18。三段平均standalone bpb从6,144到98,304依次为1.562412、1.547080、1.534333、1.522456、1.515876，随后在196,608回升到1.550402；对应融合bpb为1.506560、1.496032、1.488836、1.483100、1.479144、1.501315。196,608的融合仍比standalone改善0.049087 bpb。回升在三个片段上均出现：相对98,304，A/B/C分别增加约0.03436、0.03819和0.03103 bpb。GPU0和GPU3分别重复Segment B/C的首个196,608窗口，均与正式trace逐位置完全相同（非零差异0、最大差异0），排除了随机kernel故障和trace写入错误。该拐点应作为待解释结果保留，后续可增加98,304与196,608之间的诊断长度定位起点。

在6,144–98,304共同长度上，optimized与reference backend的三段平均bpb差异仅约8e-7到1e-5，且两条曲线给出相同的融合趋势，说明“STAP收益随上下文增长而缩小但未消失”的结论对后端选择稳健。

## 文档语言约定

实验过程、进度、结果解读和 TODO 文档优先使用中文，便于当前研究协作。代码参数、JSON 字段、trace schema 和模型官方名称保留英文，以维持脚本兼容性。论文定稿阶段再整理一份统一、专业的英文总述。
