#!/usr/bin/env bash
set -euo pipefail

cd /home/Liang_junnan/DNACompress

exec env PYTHONUNBUFFERED=1 python -u scripts/run_dna_compression.py \
  --run-dir outputs/dna_megabyte_large_b128_ensembl_all_finetune \
  --output-json outputs/dna_megabyte_large_b128_ensembl_all_finetune/statistics_fullsplit_nocodec/compression_compare.json \
  --export-out-dir outputs/dna_megabyte_large_b128_ensembl_all_finetune/statistics_fullsplit_nocodec \
  --checkpoint-tag best \
  --split all \
  --compression-modes windows_nonoverlap \
  --compression-sample-bytes 0 \
  --eval-batch-size 32 \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --device cuda:3 \
  --skip-codec-baselines
