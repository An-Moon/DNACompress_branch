# DNACorpus Species Notes

This directory contains 17 flat DNA files under `datasets/DNACorpus`. Each file is a single raw byte sequence with only `A/C/G/T` symbols; there are no FASTA headers, chromosome labels, scaffold names, or internal sequence boundaries.

## Species Table

| Code | Species / source name | 中文名/译名 | Broad group | Local size (bytes) | Chromosome interpretation in this repo |
|---|---|---|---:|---:|---|
| HoSa | *Homo sapiens* | 人 | Eukaryota, animal | 189,752,667 | Single flat sequence; literature describes this benchmark as a human chromosome extract. |
| GaGa | *Gallus gallus* | 原鸡/鸡 | Eukaryota, animal | 148,532,294 | Single flat sequence; literature describes this benchmark as a chicken chromosome extract. |
| AnCa | *Antilo capra* / antelope-like source name used by DNACorpus tables | 叉角羚类（暂译） | Eukaryota, animal | 142,189,675 | Single flat sequence; no local chromosome labels. |
| DaRe | *Danio rerio* | 斑马鱼 | Eukaryota, animal | 62,565,020 | Single flat sequence; no local chromosome labels. |
| OrSa | *Oryza sativa* | 水稻 | Eukaryota, plant | 43,262,523 | Single flat sequence; literature describes this benchmark as a rice chromosome extract. |
| DrMe | *Drosophila miranda* | 米兰达果蝇 | Eukaryota, animal | 32,181,429 | Single flat sequence; literature describes this benchmark as a chromosome-scale fly sequence. |
| EnIn | *Entamoeba invadens* | 侵袭内阿米巴 | Eukaryota, Amoebozoa | 26,403,087 | Single flat sequence, generally treated as a genome-level microbial eukaryote benchmark. |
| ScPo | *Schizosaccharomyces pombe* | 粟酒裂殖酵母 | Eukaryota, fungi | 10,652,155 | Single flat sequence, genome-level fungal benchmark. |
| WaMe | *Wallemia muriae* | 穆里瓦勒霉（直译） | Eukaryota, fungi | 9,144,432 | Single flat sequence, genome-level fungal benchmark. |
| PlFa | *Plasmodium falciparum* | 恶性疟原虫 | Eukaryota, apicomplexan/protozoan | 8,986,712 | Single flat sequence, genome or genome-part benchmark. |
| EsCo | *Escherichia coli* | 大肠杆菌 | Bacteria | 4,641,652 | Single bacterial genome-scale sequence. |
| HaHi | *Haloarcula hispanica* | 西班牙盐盒菌 | Archaea | 3,890,005 | Single archaeal genome-scale sequence. |
| HePy | *Helicobacter pylori* | 幽门螺杆菌 | Bacteria | 1,667,825 | Single bacterial genome-scale sequence. |
| AeCa | *Aeropyrum camini* | 卡米尼气火菌（直译） | Archaea | 1,591,049 | Single archaeal genome-scale sequence. |
| YeMi | Yellowstone Lake mimivirus | 黄石湖巨病毒 | Virus, Mimiviridae-like large DNA virus | 73,689 | Single viral sequence. |
| AgPh | Aggregatibacter phage S1249 | 聚杆菌噬菌体 S1249 | Virus, bacteriophage | 43,970 | Single phage sequence. |
| BuEb | Bundibugyo ebolavirus | 本迪布焦埃博拉病毒 | Virus | 18,940 | Single viral sequence. |

Chinese names are common Chinese names where available. Rare microbial and benchmark-specific names are transliterations or direct translations and should be treated as reader aids rather than authoritative taxonomy labels.

## Chromosome-Level Differences

DNACorpus is not organized in a way that lets the loader compare chromosomes within a species. In this repo, each entry is one continuous file, so `load_splits(...)` divides the file by byte position rather than by chromosome or scaffold.

That makes DNACorpus useful for cross-source compression comparisons across very different biological regimes:

- Large animal/plant eukaryotic chromosome-scale sequences: HoSa, GaGa, AnCa, DaRe, OrSa, DrMe.
- Smaller microbial eukaryotes and parasites: EnIn, ScPo, WaMe, PlFa.
- Compact bacterial and archaeal genomes: EsCo, HePy, AeCa, HaHi.
- Very small viral genomes or phage sequences: YeMi, AgPh, BuEb.

The important caveat is that within-species chromosome heterogeneity is hidden. For example, a mammalian species may have autosomes, sex chromosomes, mitochondrial DNA, repeats, and assembly gaps in the real genome, but the DNACorpus file does not preserve those boundaries. Any train/val/test split therefore samples consecutive regions of the flat benchmark sequence, not biologically named chromosomes.

## Practical Notes For Experiments

- The alphabet is clean `ACGT` only; there are no `N` bases in the local files.
- Size varies by four orders of magnitude, from BuEb at 18,940 bytes to HoSa at 189,752,667 bytes.
- Because the files are already flattened, `multi_sequence_mode` has no effect on DNACorpus.
- Report DNACorpus results as source-level benchmarks rather than chromosome-aware per-organism benchmarks.

## Sources

- Local files: `datasets/DNACorpus/*`.
- DNACorpus species names and broad types are consistent with the DNACorpus table cited in DNA-compression literature, including the GigaScience neural compression paper and the Cambridge MLMI lossless DNA compression dissertation.
