# megaDNA 145M Reproduction Notes

## Source and model shape

This setup uses the official `lingxusb/megaDNA` code in `third_party/megaDNA`
and the non-gated Hugging Face checkpoint
`lingxusb/megaDNA_updated/megaDNA_phage_145M.pt`.

megaDNA is a long-context autoregressive genome model built on the MEGABYTE
multiscale Transformer idea. Like this repository's Megabyte experiments, it
uses coarse global context plus lower-level local prediction. The official
checkpoint differs in two important ways:

- It is saved as a full `torch.save(model)` object, not as this repository's
  usual `{"model_state": ...}` checkpoint payload.
- Its vocabulary is `["**", "A", "T", "C", "G", "#"]`. There is no token for
  `N`, so the local adapter rejects sequences outside uppercase `A/T/C/G`.

The upstream code and model are licensed as `CC BY-NC 4.0`.

## Install

Install only the missing runtime dependencies and avoid replacing the existing
Torch installation:

```bash
python -m pip install beartype MEGABYTE_pytorch==0.2.1
git clone https://github.com/lingxusb/megaDNA.git third_party/megaDNA
python -m pip install -e third_party/megaDNA --no-deps
```

## Download the 145M checkpoint

Use the Hugging Face mirror and explicitly remove HTTP proxy variables during
the download:

```bash
env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  HF_ENDPOINT=https://hf-mirror.com \
  python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="lingxusb/megaDNA_updated",
    filename="megaDNA_phage_145M.pt",
    local_dir="third_party/megaDNA/checkpoints",
)
PY
```

The repository `.gitignore` already ignores `*.pt`, so this checkpoint should
remain local.

## Smoke test

```bash
python scripts/run_megadna_smoke.py \
  --weight third_party/megaDNA/checkpoints/megaDNA_phage_145M.pt \
  --device cpu \
  --primer ATCGATCG
```

For generation, set a total length greater than or equal to the primer length:

```bash
python scripts/run_megadna_smoke.py \
  --device cuda \
  --primer ATCGATCG \
  --generate-len 32
```

The gated `megaDNA_variants` and `megaDNA_finetuned` repositories are not part
of this initial reproduction setup.
