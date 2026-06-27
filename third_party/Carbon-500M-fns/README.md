---
library_name: transformers
license: apache-2.0
language:
  - dna
tags:
  - dna
  - genomic
  - transformers
  - speculative-decoding
---

# Carbon-500M-remote

A small generative DNA model from the **Carbon** family with base-pair-level generation and scoring.

**Carbon-500M-remote** is a variant of [Carbon-500M](https://huggingface.co/HuggingFaceBio/Carbon-500M) that uses a custom `CarbonForCausalLM` class (loaded via `trust_remote_code=True`) to expose base-pair-level generation (`generate`) and per-nucleotide sequence scoring (`score_sequence`). For a plain `LlamaForCausalLM` version that does not require remote code, see [Carbon-500M](https://huggingface.co/HuggingFaceBio/Carbon-500M).

**Carbon-500M-remote is intended primarily as a draft model for speculative decoding** — it shares the tokenizer and DNA template format of [Carbon-3B-remote](https://huggingface.co/HuggingFaceBio/Carbon-3B-remote) and [Carbon-8B-remote](https://huggingface.co/HuggingFaceBio/Carbon-8B-remote), so it can be paired with either as the target model to reduce wall-clock generation cost at no quality loss.

The weights, tokenizer, and training are identical to Carbon-500M — only the model class and available methods differ.

For the full design rationale, tokenizer specification, evaluation protocol, and usage notes (DNA tag wrapping, 6-mer constraints, scoring helpers), please refer to the **[Carbon-3B-remote model card](https://huggingface.co/HuggingFaceBio/Carbon-3B-remote)** — this card focuses only on facts specific to Carbon-500M-remote.

## Facts

- **500M-parameter decoder-only autoregressive DNA model** (Llama-style architecture).
- **Hybrid tokenizer** shared with the rest of the Carbon family (6-mer for DNA + Qwen3 BPE for English text; each DNA token ≈ 6 bp).
- **Pre-training tokens:** 600B 6-mer tokens (≈ 3.6 T DNA base pairs).
- **Sequence length:** 8,192 tokens (≈ 49 kbp).
- **Loss schedule:** cross-entropy 0 → 300 B tokens, then switch to the hybrid Factorised Nucleotide Supervision (FNS) loss from 300 B → 600 B tokens.
- **Data mixture:** identical to the decay-phase mixture used by Carbon-3B — 50 % Generator-style eukaryotic genes / 25 % mature mRNA / 10 % splice-enriched mRNA / 15 % GTDB bacterial genomes.
- **Precision:** bfloat16. **Optimizer:** AdamW. **Positional embedding:** RoPE.
- **No long-context training stage** — the model stays at its 8,192-token native context (≈ 49 kbp).
- Released as a `CarbonForCausalLM` model (requires `trust_remote_code=True`).

## How to use

Both the tokenizer and model require `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "HuggingFaceBio/Carbon-500M-remote"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, dtype=torch.bfloat16, trust_remote_code=True,
).cuda().eval()

prompt = "<dna>ATGCGCTAGCTACGATCGATCGTAGCTAGCTAGCTAGCTACG"
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
out = model.generate_bp(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### Recommended use: speculative decoding with Carbon-3B-remote / Carbon-8B-remote

Carbon-500M-remote is most useful when paired with a larger Carbon model as the verifier:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok    = AutoTokenizer.from_pretrained("HuggingFaceBio/Carbon-3B-remote", trust_remote_code=True)
draft  = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceBio/Carbon-500M-remote", dtype=torch.bfloat16, trust_remote_code=True,
).cuda().eval()
target = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceBio/Carbon-3B-remote", dtype=torch.bfloat16, trust_remote_code=True,
).cuda().eval()

prompt = "<dna>ATGCGCTAGCTACGATCGATCGTAGCTAGCTAGCTAGCTACG"
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
out = target.generate(
    **inputs, max_new_tokens=256, do_sample=False,
    assistant_model=draft,
)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Output is guaranteed identical to greedy decoding with the target model alone; only wall-clock latency is reduced.

## License

Apache 2.0.
