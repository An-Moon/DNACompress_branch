#!/usr/bin/env bash
set -euo pipefail

# Run Python from the isolated Evo2 1B environment.
#
# Evo2 1B uses Transformer Engine / FP8 and should not be installed into the
# active DNACompress training .venv. This wrapper keeps the environment isolated
# and adds the CUDA libraries shipped with the pip nvidia-* packages to
# LD_LIBRARY_PATH, which is required for transformer_engine / flash-attn imports.
#
# Example:
#   CUDA_VISIBLE_DEVICES=3 scripts/run_evo2_1b_env_python.sh \
#     scripts/run_dna_region_bpb_probe.py \
#     --dataset dnacorpus --dataset-dir datasets/DNACorpus --species BuEb \
#     --model evo2:third_party/evo2_1b_base:evo2_1b_base \
#     --evo2-context-bases 3072 --device cuda:0 --batch-size 32 \
#     --random-region --region-bases 50000 \
#     --output-dir outputs/evo2_1b_base_region_bpb_probe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${REPO_ROOT}/.conda_evo2_1b"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing Evo2 1B environment: ${ENV_DIR}" >&2
  echo "Expected ${ENV_DIR}/bin/python to exist." >&2
  exit 1
fi

NVIDIA_ROOT="${ENV_DIR}/lib/python3.11/site-packages/nvidia"
NVIDIA_LIBS=""
if [[ -d "${NVIDIA_ROOT}" ]]; then
  NVIDIA_LIBS="$(find "${NVIDIA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
fi

if [[ -n "${NVIDIA_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${ENV_DIR}/lib:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

exec "${ENV_DIR}/bin/python" "$@"
