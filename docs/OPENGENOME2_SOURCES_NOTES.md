# OpenGenome2 FASTA 子集来源概览
本文档基于 `/data/students/Liang_junnan/opengenome2_subset/index/manifest.json` 的索引统计生成。索引只统计原始 FASTA 的 record 和连续纯 `ACGT` 区间，不预先分窗，也不复制清洗后的序列。
## 总览
- FASTA 文件数：56,919
- FASTA 总大小：587.65 GiB
- FASTA records：72,997,330
- 连续纯 ACGT runs：92,914,322
- 随机访问 anchors：93,374,967

## Source 统计表
| Source | 生物/物种来源概括 | Files | Records | ACGT GiB | 来源占比 | 平均 record 长度 | 平均 ACGT run 长度 | ACGT 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gtdb_v220` | GTDB v220 bacterial/archaeal genome assemblies | 56,576 | 9,420,951 | 162.92 | 28.00% | 18,576.7 bp | 13,926.1 bp | 99.96% |
| `metagenomes` | filtered metagenomic sequences | 2 | 4,408,515 | 186.26 | 32.01% | 45,366.8 bp | 36,781.4 bp | 100.00% |
| `mrna` | representative mRNA sequences | 1 | 26,355,147 | 54.00 | 9.28% | 2,199.9 bp | 2,139.0 bp | 100.00% |
| `ncbi_eukaryotic_genomes` | NCBI eukaryotic genome assemblies | 334 | 1,443,369 | 108.56 | 18.65% | 80,764.3 bp | 26,328.8 bp | 100.00% |
| `ncrna` | representative ncRNA sequences | 1 | 25,206,175 | 17.53 | 3.01% | 749.9 bp | 843.1 bp | 99.60% |
| `organelles` | organelle sequences | 1 | 32,240 | 2.62 | 0.45% | 87,370.5 bp | 64,988.8 bp | 100.00% |
| `plasmids_phage` | plasmid and phage sequences | 2 | 3,172,445 | 41.63 | 7.15% | 14,091.3 bp | 12,253.6 bp | 99.98% |
| `promoters` | promoter representative sequences | 1 | 173,241 | 0.10 | 0.02% | 600.0 bp | 594.8 bp | 99.77% |
| `transcripts` | representative transcript sequences | 1 | 2,785,247 | 8.33 | 1.43% | 3,217.5 bp | 520.3 bp | 99.78% |

## 各来源简述
### `gtdb_v220`
主要是原核生物基因组，覆盖细菌和古菌。文件多、assembly 多，平均 record 约十几 kb，适合中长片段抽样。

- 规模：56,576 个文件，9,420,951 条 record，纯 ACGT 碱基约 162.92 GiB。
- 长度：平均 record 长度 18,576.7 bp；平均连续 ACGT run 长度 13,926.1 bp。
- 字符情况：非 ACGT 字母 71,130,957 个，小写字母 0 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：通常能支持中长片段抽样，但具体仍取决于 `seq_length` 和连续 ACGT run 长度。

### `metagenomes`
宏基因组来源，通常来自环境或混合样本中的微生物群落 contig/scaffold；平均 record 较长，是本子集中最大的碱基来源之一。

- 规模：2 个文件，4,408,515 条 record，纯 ACGT 碱基约 186.26 GiB。
- 长度：平均 record 长度 45,366.8 bp；平均连续 ACGT run 长度 36,781.4 bp。
- 字符情况：非 ACGT 字母 1,113,357 个，小写字母 0 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：通常能支持中长片段抽样，但具体仍取决于 `seq_length` 和连续 ACGT run 长度。

### `mrna`
mRNA / coding transcript 代表序列，记录数很多但单条较短，适合短到中等片段；长 seq_length 下可采样窗口会明显减少。

- 规模：1 个文件，26,355,147 条 record，纯 ACGT 碱基约 54.00 GiB。
- 长度：平均 record 长度 2,199.9 bp；平均连续 ACGT run 长度 2,139.0 bp。
- 字符情况：非 ACGT 字母 780,368 个，小写字母 0 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：适合短到中等 `seq_length`；若使用很长片段，需要先检查 eligible windows。

### `ncbi_eukaryotic_genomes`
真核基因组 assembly，包含动物、植物、真菌、原生生物等真核来源；平均 record 长，但 N/非 ACGT 会把 ACGT run 切短。

- 规模：334 个文件，1,443,369 条 record，纯 ACGT 碱基约 108.56 GiB。
- 长度：平均 record 长度 80,764.3 bp；平均连续 ACGT run 长度 26,328.8 bp。
- 字符情况：非 ACGT 字母 3,035,639 个，小写字母 43,341,422,511 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：通常能支持中长片段抽样，但具体仍取决于 `seq_length` 和连续 ACGT run 长度。

### `ncrna`
非编码 RNA 代表序列，单条序列普遍很短；适合短片段训练，不适合很长 seq_length 的 full-window 采样。

- 规模：1 个文件，25,206,175 条 record，纯 ACGT 碱基约 17.53 GiB。
- 长度：平均 record 长度 749.9 bp；平均连续 ACGT run 长度 843.1 bp。
- 字符情况：非 ACGT 字母 74,808,199 个，小写字母 71,203,034 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：该来源序列偏短，较大的 `seq_length` 会显著减少可采样窗口。

### `organelles`
细胞器序列，包括线粒体、叶绿体/质体等；record 平均较长，ACGT 纯度高。

- 规模：1 个文件，32,240 条 record，纯 ACGT 碱基约 2.62 GiB。
- 长度：平均 record 长度 87,370.5 bp；平均连续 ACGT run 长度 64,988.8 bp。
- 字符情况：非 ACGT 字母 15,996 个，小写字母 0 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：通常能支持中长片段抽样，但具体仍取决于 `seq_length` 和连续 ACGT run 长度。

### `plasmids_phage`
质粒和噬菌体/病毒相关序列，偏微生物移动遗传元件；长度中等，适合中等片段抽样。

- 规模：2 个文件，3,172,445 条 record，纯 ACGT 碱基约 41.63 GiB。
- 长度：平均 record 长度 14,091.3 bp；平均连续 ACGT run 长度 12,253.6 bp。
- 字符情况：非 ACGT 字母 7,128,701 个，小写字母 0 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：通常能支持中长片段抽样，但具体仍取决于 `seq_length` 和连续 ACGT run 长度。

### `promoters`
启动子序列，通常是短调控区域；平均长度约 600 bp，长 seq_length 下基本不适合 full-window 抽样。

- 规模：1 个文件，173,241 条 record，纯 ACGT 碱基约 0.10 GiB。
- 长度：平均 record 长度 600.0 bp；平均连续 ACGT run 长度 594.8 bp。
- 字符情况：非 ACGT 字母 236,649 个，小写字母 0 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：该来源序列偏短，较大的 `seq_length` 会显著减少可采样窗口。

### `transcripts`
转录本代表序列，整体较短且被非 ACGT 字符切成较多短 run；适合较短 seq_length。

- 规模：1 个文件，2,785,247 条 record，纯 ACGT 碱基约 8.33 GiB。
- 长度：平均 record 长度 3,217.5 bp；平均连续 ACGT run 长度 520.3 bp。
- 字符情况：非 ACGT 字母 19,401,896 个，小写字母 0 个；采样时小写会转大写，`N` 和其他非 ACGT 字母会切断 run。
- 训练提示：该来源序列偏短，较大的 `seq_length` 会显著减少可采样窗口。

## 采样含义
当前 megaDNA 索引采样只使用长度不小于 `seq_length` 的连续纯 `ACGT` run。默认情况下，run 的抽样权重为 `run_base_length - seq_length + 1`，即按可形成的完整窗口数加权。若指定顶层 source 采样权重，则先按 source 权重选来源，再在该来源内部按完整窗口数抽样。

因此，`ncrna`、`promoters`、`transcripts`、`mrna` 这类短序列来源在长 `seq_length` 下可能贡献很少，甚至没有可采样窗口；训练前应按目标 `seq_length` 从 `acgt_runs.parquet` 统计各 source 的 eligible windows。
