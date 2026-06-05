#!/usr/bin/env bash
set -euo pipefail

cd /home/Liang_junnan/DNACompress

OUT_DIR="outputs/dna_megabyte_large_b128_ensembl_all/statistics_fullsplit"
GECO2_DIR="outputs/dna_geco2_paper_modes_0p6_0p2_0p2_fullsplit"

echo "[start] $(date -Is)"
echo "[compression] MEGABYTE b128 Ensembl-pretrained checkpoint on DNACorpus full split, arithmetic backend=fast_cpp"

env \
  PYTHONUNBUFFERED=1 \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python -u scripts/run_dna_compression.py \
    --run-dir outputs/dna_megabyte_large_b128_ensembl_all \
    --output-json "${OUT_DIR}/compression_compare.json" \
    --export-out-dir "${OUT_DIR}" \
    --checkpoint-tag best \
    --dataset-dir datasets/DNACorpus \
    --sequence-source-mode auto \
    --multi-sequence-mode separate \
    --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
    --split all \
    --compression-modes windows_nonoverlap \
    --compression-sample-bytes 0 \
    --eval-batch-size 32 \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --device cuda:2 \
    --skip-codec-baselines \
    --arithmetic-backend fast_cpp

echo "[plot] $(date -Is)"
python scripts/plot_fullsplit_geco2_comparison.py \
  --model-dir "${OUT_DIR}" \
  --geco2-dir "${GECO2_DIR}" \
  --out-dir "${OUT_DIR}/geco2_comparison_curves" \
  --model-label "MEGABYTE b128 fast_cpp"

echo "[done] $(date -Is)"
