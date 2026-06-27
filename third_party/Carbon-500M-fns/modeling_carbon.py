"""
Carbon with bp-level generation and scoring.

generate_bp() plugs into the standard HF generate() pipeline via a
LogitsProcessor — no internal methods are overridden, so it is compatible
with any transformers version.
"""
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LogitsProcessor, LogitsProcessorList
from typing import Union

BASE_TO_IDX = {"A": 0, "T": 1, "C": 2, "G": 3, "N": -1}
IDX_TO_BASE = {0: "A", 1: "T", 2: "C", 3: "G", -1: "N"}


class _BPLogitsProcessor(LogitsProcessor):
    """Forces token selection to use per-base marginal probabilities.

    Runs LAST in the logits-processor chain so that temperature / top-k /
    top-p etc. influence the marginal distributions before base selection.
    """

    def __init__(self, kmer_ids, bp_base_index, flat_idx_to_token_id, bp_powers, k, do_sample):
        self.kmer_ids = kmer_ids
        self.bp_base_index = bp_base_index
        self.flat_idx_to_token_id = flat_idx_to_token_id
        self.bp_powers = bp_powers
        self.k = k
        self.do_sample = do_sample

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = scores.shape[0]
        kmer_probs = F.softmax(scores[:, self.kmer_ids].float(), dim=-1)  # [B, num_kmers]

        # Marginalise to per-base probabilities [B, k, 4]
        bp_probs = torch.zeros(B, self.k, 4, device=scores.device, dtype=kmer_probs.dtype)
        for pos in range(self.k):
            idx = self.bp_base_index[pos]  # [num_kmers] in {0,1,2,3}
            for nt in range(4):
                bp_probs[:, pos, nt] = kmer_probs[:, idx == nt].sum(dim=-1)

        if self.do_sample:
            base_indices = torch.multinomial(bp_probs.view(-1, 4), 1).view(B, self.k)
        else:
            base_indices = bp_probs.argmax(dim=-1)  # [B, k]

        flat_idx = (base_indices * self.bp_powers).sum(dim=-1)   # [B]
        selected = self.flat_idx_to_token_id[flat_idx]           # [B]

        # One-hot: both argmax and multinomial land on the bp-selected token
        new_scores = torch.full_like(scores, float("-inf"))
        new_scores.scatter_(1, selected.unsqueeze(1), 0.0)
        return new_scores


class CarbonForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM with bp-level autoregressive generation.

    Inherits all standard functionality (forward, generate, etc.)
    and adds generate_bp() for base-pair independent generation.

    The tokenizer is automatically set up when loading the model with from_pretrained().
    """

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load model and automatically setup tokenizer if available."""
        model = super().from_pretrained(*args, **kwargs)

        model_path = args[0] if len(args) > 0 else kwargs.get('pretrained_model_name_or_path')

        if model_path:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model.setup_tokenizer(tokenizer)
                print(f"Tokenizer automatically loaded and configured for bp-level scoring")
            except Exception as e:
                print(f"Could not auto-load tokenizer: {e}")
                print(f"  Call model.setup_tokenizer(tokenizer) manually if needed")

        return model

    def setup_tokenizer(self, tokenizer):
        """Cache tokenizer and precompute lookup tables for bp-level operations."""
        self.tokenizer = tokenizer
        k = tokenizer.k
        self.k = k

        device = next(self.parameters()).device

        # Build ordered kmer list from the tokenizer's DNA vocab
        kmer_items = sorted(
            [
                (kmer, tid)
                for kmer, tid in tokenizer.dna_token_to_id.items()
                if len(kmer) == k and all(b in "ATCG" for b in kmer)
            ],
            key=lambda x: x[1],
        )
        kmers = [item[0] for item in kmer_items]
        kmer_ids = [item[1] for item in kmer_items]
        num_kmers = len(kmer_ids)

        kmer_ids_tensor = torch.tensor(kmer_ids, dtype=torch.long, device=device)
        self.register_buffer("_kmer_ids", kmer_ids_tensor, persistent=False)

        # bp_base_index[pos, j] = base index (0-3) of kmer j at position pos
        bp_base_index = torch.zeros(k, num_kmers, dtype=torch.long)
        for j, kmer in enumerate(kmers):
            for pos, base in enumerate(kmer):
                bp_base_index[pos, j] = BASE_TO_IDX[base]
        self.register_buffer("_bp_base_index", bp_base_index.to(device), persistent=False)

        bp_powers = torch.tensor(
            [4 ** i for i in range(k - 1, -1, -1)], dtype=torch.long, device=device
        )
        self.register_buffer("_bp_powers", bp_powers, persistent=False)

        # flat kmer index -> token id (flat index = sum base_idx[i] * 4^(k-1-i))
        flat_to_tid = torch.zeros(num_kmers, dtype=torch.long, device=device)
        for j, (kmer, tid) in enumerate(kmer_items):
            flat_idx = sum(BASE_TO_IDX[c] * (4 ** (k - 1 - i)) for i, c in enumerate(kmer))
            flat_to_tid[flat_idx] = tid
        self.register_buffer("_flat_idx_to_token_id", flat_to_tid, persistent=False)

    def compute_bp_probs(self, logits):
        """Compute per-base marginal probabilities from token logits.

        Args:
            logits: [B, V] or [B, L, V]
        Returns:
            bp_probs: [B, k, 4] or [B, L, k, 4]
        """
        squeeze = logits.dim() == 2
        if squeeze:
            logits = logits.unsqueeze(1)

        kmer_logits = logits[:, :, self._kmer_ids]
        kmer_probs = F.softmax(kmer_logits.float(), dim=-1)
        B, L, _ = kmer_probs.shape
        bp_probs = torch.zeros(B, L, self.k, 4, device=logits.device, dtype=kmer_probs.dtype)
        for pos in range(self.k):
            idx = self._bp_base_index[pos]
            for nt in range(4):
                bp_probs[:, :, pos, nt] = kmer_probs[:, :, idx == nt].sum(dim=-1)

        return bp_probs.squeeze(1) if squeeze else bp_probs

    def generate(self, inputs=None, generation_config=None, **kwargs):
        """Like generate(), but each token is selected base-by-base from marginal distributions.

        Temperature, top_k, top_p, repetition_penalty etc. all apply as usual —
        they run before the bp processor and shift the marginal distributions.
        Output shape and type are identical to generate().
        """
        assert hasattr(self, "_bp_base_index"), "Call setup_tokenizer(tokenizer) first"

        gc = generation_config or self.generation_config
        do_sample = kwargs.get("do_sample", getattr(gc, "do_sample", False))

        bp_proc = _BPLogitsProcessor(
            kmer_ids=self._kmer_ids,
            bp_base_index=self._bp_base_index,
            flat_idx_to_token_id=self._flat_idx_to_token_id,
            bp_powers=self._bp_powers,
            k=self.k,
            do_sample=do_sample,
        )
        existing = list(kwargs.pop("logits_processor", None) or [])
        kwargs["logits_processor"] = LogitsProcessorList(existing + [bp_proc])

        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)


    def generate_bp(self, inputs=None, generation_config=None, **kwargs):
        """Base-pair-level generation. Delegates to generate(), which applies bp-level token selection."""
        return self.generate(inputs=inputs, generation_config=generation_config, **kwargs)

    @torch.no_grad()
    def score_sequence(self, sequences: Union[str, list]):
        """Score DNA sequence(s) at base resolution.

        Returns per-base probability distributions and the probability of the
        actual base at each position, given all preceding context.

        Args:
            sequences: single DNA string or list of DNA strings (ACGT only)

        Returns:
            (bp_probs, actual_probs) for a single sequence, or
            (list of bp_probs, list of actual_probs) for a batch.
            bp_probs[i]: [seq_len_i, 4] — P(base | context) at each position
            actual_probs[i]: [seq_len_i] — P(actual base | context)
        """
        assert hasattr(self, "tokenizer"), "Call setup_tokenizer(tokenizer) first"

        is_single = isinstance(sequences, str)
        if is_single:
            sequences = [sequences]

        original_lens = [len(s) for s in sequences]

        # Right-pad to multiple of k with 'A' (matches tokenizer convention)
        padded = []
        for s in sequences:
            r = len(s) % self.k
            padded.append(s + "A" * (self.k - r) if r else s)

        # Prepend <dna> tag manually (training format)
        tagged = ["<dna>" + s for s in padded]

        inputs = self.tokenizer(
            tagged, return_tensors="pt", padding=True, add_special_tokens=False
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        logits = self(input_ids, attention_mask=attention_mask, return_dict=True).logits
        bp_probs_all = self.compute_bp_probs(logits)  # [B, L, k, 4]

        bp_results, actual_results = [], []
        for i, (seq, orig_len, pad_seq) in enumerate(zip(sequences, original_lens, padded)):
            num_tokens = len(pad_seq) // self.k
            # logits[t] predicts token t+1; logits[0] (from <dna>) predicts token 1
            seq_bp = bp_probs_all[i, :num_tokens]          # [num_tokens, k, 4]
            seq_bp = seq_bp.reshape(-1, 4)[:orig_len]      # [orig_len, 4]
            actual = self._extract_actual_probs(seq_bp, seq)
            bp_results.append(seq_bp)
            actual_results.append(actual)

        if is_single:
            return bp_results[0], actual_results[0]
        return bp_results, actual_results

    def _extract_actual_probs(self, bp_probs: torch.Tensor, sequence: str) -> torch.Tensor:
        actual = torch.zeros(len(sequence), device=bp_probs.device, dtype=bp_probs.dtype)
        for i, base in enumerate(sequence):
            actual[i] = bp_probs[i].max() if base == "N" else bp_probs[i, BASE_TO_IDX[base]]
        return actual
