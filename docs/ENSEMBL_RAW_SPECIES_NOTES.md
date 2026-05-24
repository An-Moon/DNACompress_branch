# Ensembl Raw Species Notes

`datasets/ensembl_raw` contains Ensembl-style FASTA files organized as one directory per species and one file per chromosome, primary assembly sequence, organelle, or nonchromosomal collection. Unlike DNACorpus, this dataset preserves within-species sequence boundaries, so `multi_sequence_mode="separate"` treats each chromosome/sequence as its own source.

The length statistics below use the local clean-cache metadata in `datasets/ensembl_raw/.dna_cache/clean/**/*.json`, where `cleaned_length` counts cleaned `ACGTN` sequence length after removing FASTA headers and line breaks.

## Species Table

| Species key | Scientific name | 中文名 | Broad group | Assembly in filenames | Files | Cleaned length | Nuclear sequences | Extra sequences |
|---|---|---|---|---|---:|---:|---:|---|
| `homo_sapiens` | *Homo sapiens* | 人 | Mammal, primate | GRCh38 | 26 | 3,099.75 Mbp | 24 | MT, nonchromosomal |
| `mus_musculus` | *Mus musculus* | 小鼠 | Mammal, rodent | GRCm39 | 23 | 2,728.22 Mbp | 21 | MT, nonchromosomal |
| `bos_taurus` | *Bos taurus* | 牛/家牛 | Mammal, ruminant | ARS-UCD2.0 | 33 | 2,770.69 Mbp | 31 | MT, nonchromosomal |
| `danio_rerio` | *Danio rerio* | 斑马鱼 | Vertebrate, teleost fish | GRCz11 | 27 | 1,373.47 Mbp | 25 | MT, nonchromosomal |
| `drosophila_melanogaster` | *Drosophila melanogaster* | 黑腹果蝇 | Arthropod, insect | BDGP6.54 | 9 | 143.73 Mbp | 7 | mitochondrion genome, nonchromosomal |
| `caenorhabditis_elegans` | *Caenorhabditis elegans* | 秀丽隐杆线虫 | Nematode | WBcel235 | 7 | 100.29 Mbp | 6 | MtDNA |
| `saccharomyces_cerevisiae` | *Saccharomyces cerevisiae* | 酿酒酵母 | Fungi, budding yeast | R64-1-1 | 17 | 12.16 Mbp | 16 | Mito |
| `arabidopsis_thaliana` | *Arabidopsis thaliana* | 拟南芥 | Plant, angiosperm | TAIR10 | 7 | 119.67 Mbp | 5 | Mt, Pt |

## Within-Species Chromosome Differences

### Vertebrates

- `homo_sapiens`: 22 autosomes plus X/Y and MT. Nuclear chromosome lengths range from 46.71 Mbp (21) to 248.96 Mbp (1). The largest autosomes dominate the total sequence mass; MT is tiny at 0.017 Mbp, and the nonchromosomal collection is 11.46 Mbp.
- `mus_musculus`: 19 autosomes plus X/Y and MT. Nuclear sequence lengths range from 61.42 Mbp (19) to 195.15 Mbp (1). X is among the largest sequences; MT is 0.016 Mbp and nonchromosomal sequence is 4.79 Mbp.
- `bos_taurus`: 29 autosomes plus X/Y and MT. Nuclear sequence lengths range from 42.35 Mbp (25) to 158.53 Mbp (1). The nonchromosomal collection is large at 82.80 Mbp, so it is not a negligible side sequence for compression experiments.
- `danio_rerio`: 25 nuclear chromosomes plus MT and nonchromosomal sequence. Nuclear lengths range from 37.50 Mbp (25) to 78.09 Mbp (4). Chromosomes are more evenly sized than mammalian genomes, and the nonchromosomal collection is 28.35 Mbp.

### Invertebrate Model Organisms

- `drosophila_melanogaster`: primary assembly sequences include chromosome arms 2L/2R and 3L/3R, plus 4, X, Y, mitochondrion genome, and nonchromosomal sequence. Nuclear lengths range from 1.35 Mbp (4) to 32.08 Mbp (3R). Chromosome 4 and Y are much smaller than the major autosomal arms, which can make per-source compression behavior noticeably different.
- `caenorhabditis_elegans`: six nuclear chromosomes I-V plus X and MtDNA. Nuclear lengths range from 13.78 Mbp (III) to 20.92 Mbp (V). The chromosomes are relatively balanced compared with mammals; MtDNA is only 0.014 Mbp.

### Fungi And Plant

- `saccharomyces_cerevisiae`: 16 compact nuclear chromosomes plus mitochondrial DNA. Nuclear lengths range from 0.23 Mbp (I) to 1.53 Mbp (IV), so per-chromosome train/val/test splits can be very short for the smallest chromosomes.
- `arabidopsis_thaliana`: five nuclear chromosomes plus mitochondrial and plastid sequences. Nuclear lengths range from 18.59 Mbp (4) to 30.43 Mbp (1). The organellar genomes are explicit: Mt is 0.367 Mbp and Pt is 0.154 Mbp.

## Practical Notes For Experiments

- The repo's default `multi_sequence_mode="separate"` means each chromosome/sequence becomes its own source and is split independently into train/val/test.
- `multi_sequence_mode="concat"` would instead join sequences with an `N` boundary before splitting, which changes the biological meaning of the split.
- Mammalian and fish species contribute most bytes, while yeast, nematode, fly, and Arabidopsis provide compact model-organism regimes.
- Organelle and nonchromosomal files should be kept visible in reports because their compression behavior can differ sharply from large nuclear chromosomes.

## Sources

- Local files: `datasets/ensembl_raw/<species>/dna/*.fa`.
- Local clean-cache metadata: `datasets/ensembl_raw/.dna_cache/clean/**/*.json`.
- Species/assembly naming follows the Ensembl FASTA filenames present in this repo.
