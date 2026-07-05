#!/usr/bin/env bash
set -euo pipefail

# Wait for the Carbon HF dataset download tmux job, verify/resume the download,
# then convert local parquet files to FASTA and build the existing indexed FASTA
# runtime input. This script is intentionally conservative: it re-runs the HF
# download command after the watched tmux session exits, so a transient failure
# will be resumed before indexing begins.

REPO_ROOT="${REPO_ROOT:-/home/Liang_junnan/DNACompress}"
DATASET_ROOT="${DATASET_ROOT:-/data/students/Liang_junnan/carbon-pretraining-corpus}"
FASTA_ROOT="${FASTA_ROOT:-/data/students/Liang_junnan/carbon-pretraining-corpus_fasta}"
INDEX_DIR="${INDEX_DIR:-/data/students/Liang_junnan/carbon-pretraining-corpus_index}"
DOWNLOAD_SESSION="${DOWNLOAD_SESSION:-carbon_hf_download}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
MIN_PARQUET_FILES="${MIN_PARQUET_FILES:-500}"
POSTPROCESS_LOG="${POSTPROCESS_LOG:-${DATASET_ROOT}/postprocess.log}"

mkdir -p "${DATASET_ROOT}"
cd "${REPO_ROOT}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${DOWNLOAD_SESSION}" 2>/dev/null; then
  log "waiting for tmux session ${DOWNLOAD_SESSION} to finish"
  while tmux has-session -t "${DOWNLOAD_SESSION}" 2>/dev/null; do
    parquet_count=$(find "${DATASET_ROOT}" -name '*.parquet' 2>/dev/null | wc -l || true)
    size_text=$(du -sh "${DATASET_ROOT}" 2>/dev/null | awk '{print $1}' || true)
    log "download still running; parquet_count=${parquet_count}; size=${size_text:-unknown}; sleeping ${WAIT_SECONDS}s"
    sleep "${WAIT_SECONDS}"
  done
else
  log "download session ${DOWNLOAD_SESSION} is not running; continuing with verification/resume"
fi

log "verifying/resuming HF dataset download via hf-mirror"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download HuggingFaceBio/carbon-pretraining-corpus \
    --repo-type dataset \
    --local-dir "${DATASET_ROOT}" \
    --local-dir-use-symlinks False

parquet_count=$(find "${DATASET_ROOT}" -name '*.parquet' | wc -l)
log "download verification finished; parquet_count=${parquet_count}"
if [ "${parquet_count}" -lt "${MIN_PARQUET_FILES}" ]; then
  log "ERROR: expected at least ${MIN_PARQUET_FILES} parquet files before full indexing"
  log "Set MIN_PARQUET_FILES lower only for intentional partial/smoke indexing."
  exit 2
fi

log "converting Carbon parquet corpus to FASTA and building indexed FASTA"
python scripts/build_carbon_hf_fasta_index.py \
  --dataset-root "${DATASET_ROOT}" \
  --fasta-root "${FASTA_ROOT}" \
  --index-dir "${INDEX_DIR}" \
  --skip-existing \
  --build-index

log "done; fasta_root=${FASTA_ROOT}; index_dir=${INDEX_DIR}; log=${POSTPROCESS_LOG}"
