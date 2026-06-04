#!/usr/bin/env bash
set -euo pipefail

cd /home/Liang_junnan/DNACompress

OUT_DIR="outputs/dna_megabyte_large_b128_ensembl_all_finetune/statistics_fullsplit"
GECO2_DIR="outputs/dna_geco2_paper_modes_0p6_0p2_0p2_fullsplit"

echo "[start] $(date -Is)"
echo "[compression] MEGABYTE fullsplit with arithmetic backend=fast_cpp"

env PYTHONUNBUFFERED=1 python -u scripts/run_dna_compression.py \
  --run-dir outputs/dna_megabyte_large_b128_ensembl_all_finetune \
  --output-json "${OUT_DIR}/compression_compare.json" \
  --export-out-dir "${OUT_DIR}" \
  --checkpoint-tag best \
  --split all \
  --compression-modes windows_nonoverlap \
  --compression-sample-bytes 0 \
  --eval-batch-size 32 \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --device cuda:3 \
  --skip-codec-baselines \
  --arithmetic-backend fast_cpp

echo "[plot] $(date -Is)"
python scripts/plot_fullsplit_geco2_comparison.py \
  --model-dir "${OUT_DIR}" \
  --geco2-dir "${GECO2_DIR}" \
  --out-dir "${OUT_DIR}/geco2_comparison_curves" \
  --model-label "MEGABYTE fast_cpp"

echo "[done] $(date -Is)"
