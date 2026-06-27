"""
HybridDNATokenizer: Combines Qwen3 BPE tokenization with DNA 6-mer tokenization.

DNA sequences wrapped in <dna>...</dna> tags are tokenized as 6-mers.
All other text uses Qwen3's BPE tokenization.

Supports token_mask for Fine-grained Nucleotide Supervision (FNS):
  -2: padding token
  -1: text token (BPE)
   0: DNA special token (<dna>, </dna>, <oov>)
  1-5: partial 6-mer token — valid_length real bases at positions [0, valid_length),
       right-padded with 'A' at positions [valid_length, k) so loss can supervise
       positions 0..valid_len-1 via pos_mask = (valid_len > pos)
   6: full 6-mer
"""

import os
import json
import warnings
import itertools
from typing import List, Optional, Tuple, Dict, Union, Any

from transformers import PreTrainedTokenizer, AutoTokenizer, BatchEncoding


class HybridDNATokenizer(PreTrainedTokenizer):
    """
    Hybrid tokenizer combining Qwen3 BPE with DNA 6-mer tokenization.

    DNA regions must be wrapped in <dna>...</dna> tags to be tokenized as 6-mers.
    Without tags, DNA sequences are tokenized as regular BPE text.

    For pure-DNA input (no metadata tokens), pass auto_dna_tags=True to have
    <dna>...</dna> tags added automatically when they are absent.  Do NOT set
    this if the input may contain BPE metadata such as species tags
    (<fungi_species> etc.) — those must appear outside <dna>...</dna> and would
    be incorrectly k-mer encoded if auto-wrapping fired.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        base_tokenizer_path: Optional[str] = None,
        k: int = 6,
        auto_dna_tags: bool = False,
        **kwargs
    ):
        self.k = k

        # Load base tokenizer (Qwen3-4B-Base)
        self._base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")

        # Get base vocabulary
        self._base_vocab = self._base_tokenizer.get_vocab()
        self._base_vocab_size = len(self._base_vocab)

        # Initialize DNA vocabulary
        self._init_dna_vocab()

        # Build combined vocabulary
        self._build_combined_vocab()

        # Set special tokens
        self._eos_token = kwargs.pop('eos_token', None) or "<|endoftext|>"
        self._pad_token = kwargs.pop('pad_token', None) or self._base_tokenizer.pad_token or "<|endoftext|>"

        # Initialize parent class
        super().__init__(
            eos_token=self._eos_token,
            pad_token=self._pad_token,
            **kwargs
        )

        self.special_tokens = self.dna_special_tokens + [self._eos_token, self._pad_token]
        self.auto_dna_tags = auto_dna_tags

    def _init_dna_vocab(self):
        """Initialize DNA vocabulary (special tokens + k-mers + padding for 128 alignment)."""
        bases = ['A', 'T', 'C', 'G']

        # DNA special tokens
        self.dna_special_tokens = ["<dna>", "</dna>", "<oov>"]

        # Generate all k-mer combinations (4^k = 4096 for k=6)
        self.kmers = [''.join(kmer) for kmer in itertools.product(bases, repeat=self.k)]

        # DNA tokens start after base vocabulary
        self.dna_start_id = self._base_vocab_size

        # All DNA tokens get new IDs (no reuse of base vocab IDs, even for
        # overlapping tokens like CCCCCC — they have different semantics in
        # DNA context vs BPE context, per Qiuyi's recommendation)
        base_dna_tokens = self.dna_special_tokens + self.kmers

        # Calculate padding for 128 alignment
        total_vocab_unpadded = self._base_vocab_size + len(base_dna_tokens)
        target_vocab_size = ((total_vocab_unpadded + 127) // 128) * 128
        num_padding_tokens = target_vocab_size - total_vocab_unpadded

        # Add unused padding tokens
        self.padding_tokens = [f"<unused_{i}>" for i in range(num_padding_tokens)]

        # Create DNA token mappings — all get sequential new IDs
        self.dna_token_to_id = {}
        self.dna_id_to_token = {}

        current_id = self.dna_start_id
        for token in base_dna_tokens:
            self.dna_token_to_id[token] = current_id
            self.dna_id_to_token[current_id] = token
            current_id += 1

        # Add padding tokens
        for token in self.padding_tokens:
            self.dna_token_to_id[token] = current_id
            self.dna_id_to_token[current_id] = token
            current_id += 1

        self.dna_vocab_size = len(base_dna_tokens) + len(self.padding_tokens)

        # Set DNA special token IDs
        self.dna_begin_token_id = self.dna_token_to_id["<dna>"]
        self.dna_end_token_id = self.dna_token_to_id["</dna>"]
        self.oov_token_id = self.dna_token_to_id["<oov>"]

    def _build_combined_vocab(self):
        """Build combined vocabulary (base + DNA)."""
        self._vocab = self._base_vocab.copy()

        for token, token_id in self.dna_token_to_id.items():
            if token not in self._vocab:
                self._vocab[token] = token_id

        self._id_to_token = {v: k for k, v in self._vocab.items()}
        for token_id, token in self.dna_id_to_token.items():
            if token_id not in self._id_to_token:
                self._id_to_token[token_id] = token

    @property
    def vocab_size(self) -> int:
        return max(self._vocab.values()) + 1

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab.copy()

    def __len__(self):
        # Override default (len(get_vocab())) because get_vocab() deduplicates
        # CCCCCC which exists as both BPE (ID 91443) and DNA 6-mer (ID 154402).
        return self.vocab_size

    def _split_by_dna_tags(self, text: str) -> List[Tuple[str, bool]]:
        segments = []
        i = 0
        n = len(text)

        while i < n:
            start_pos = text.find('<dna>', i)
            end_pos = text.find('</dna>', i)

            if start_pos == -1 and end_pos == -1:
                remaining = text[i:]
                if remaining:
                    segments.append((remaining, False))
                break

            if start_pos == -1 and end_pos != -1:
                dna_region = text[i:end_pos + 6]
                if dna_region:
                    segments.append((dna_region, True))
                i = end_pos + 6
                continue

            if start_pos != -1 and end_pos == -1:
                if i < start_pos:
                    normal_text = text[i:start_pos]
                    if normal_text:
                        segments.append((normal_text, False))
                dna_region = text[start_pos:]
                if dna_region:
                    segments.append((dna_region, True))
                break

            if start_pos < end_pos:
                if i < start_pos:
                    normal_text = text[i:start_pos]
                    if normal_text:
                        segments.append((normal_text, False))
                dna_region = text[start_pos:end_pos + 6]
                if dna_region:
                    segments.append((dna_region, True))
                i = end_pos + 6
            else:
                dna_region = text[i:end_pos + 6]
                if dna_region:
                    segments.append((dna_region, True))
                i = end_pos + 6

        return segments

    def _parse_dna_region(self, dna_region: str) -> Tuple[str, bool, bool]:
        if dna_region == '<dna>':
            return '', True, False
        elif dna_region == '</dna>':
            return '', False, True

        has_start = dna_region.startswith('<dna>')
        has_end = dna_region.endswith('</dna>')

        content = dna_region
        if has_start:
            content = content[5:]
        if has_end and content.endswith('</dna>'):
            content = content[:-6]

        return content.strip(), has_start, has_end

    def _process_dna_sequence(self, dna_seq: str) -> Dict:
        k = self.k
        dna_seq = dna_seq.upper()

        kmer_tokens = []
        valid_bases = set('ATCG')

        def is_valid_kmer(kmer):
            return len(kmer) == k and all(base in valid_bases for base in kmer)

        for i in range(0, len(dna_seq) - k + 1, k):
            kmer = dna_seq[i:i+k]
            if is_valid_kmer(kmer):
                kmer_tokens.append(kmer)
            else:
                kmer_tokens.append("<oov>")

        processed_length = len(kmer_tokens) * k
        remaining = dna_seq[processed_length:]
        padding_length = 0
        valid_length = k

        if remaining:
            padding_needed = k - len(remaining)
            # Right-pad with A: real bases occupy positions [0, valid_length).
            # The hybrid BP loss supervises positions 0..valid_len-1 via
            #   pos_mask = (valid_len > pos)
            # so padding must be at the END, not the start.
            padded = remaining + 'A' * padding_needed

            if is_valid_kmer(padded):
                kmer_tokens.append(padded)
            else:
                kmer_tokens.append("<oov>")

            padding_length = padding_needed
            valid_length = len(remaining)

        return {
            "kmer_tokens": kmer_tokens,
            "padding_length": padding_length,
            "valid_length": valid_length,
        }

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.dna_token_to_id:
            return self.dna_token_to_id[token]
        return self._base_vocab.get(token, self._base_tokenizer.unk_token_id or 0)

    def _convert_id_to_token(self, index: int) -> str:
        if index in self.dna_id_to_token:
            return self.dna_id_to_token[index]
        return self._id_to_token.get(index, "<oov>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_token_mask: bool = False,
        auto_dna_tags: Optional[bool] = None,
        **kwargs
    ) -> Union[List[int], Tuple[List[int], List[int]]]:
        use_auto = self.auto_dna_tags if auto_dna_tags is None else auto_dna_tags
        if use_auto and '<dna>' not in text:
            text = f'<dna>{text}</dna>'

        segments = self._split_by_dna_tags(text)

        token_ids = []
        token_mask = [] if return_token_mask else None

        for segment_content, is_dna in segments:
            if is_dna:
                dna_content, has_start, has_end = self._parse_dna_region(segment_content)

                if has_start:
                    token_ids.append(self.dna_begin_token_id)
                    if return_token_mask:
                        token_mask.append(0)

                if dna_content:
                    result = self._process_dna_sequence(dna_content)

                    for idx, kmer in enumerate(result["kmer_tokens"]):
                        token_id = self.dna_token_to_id.get(kmer, self.oov_token_id)
                        token_ids.append(token_id)

                        if return_token_mask:
                            if kmer == "<oov>":
                                token_mask.append(0)
                            elif idx == len(result["kmer_tokens"]) - 1 and result["padding_length"] > 0:
                                token_mask.append(result["valid_length"])
                            else:
                                token_mask.append(self.k)

                if has_end:
                    token_ids.append(self.dna_end_token_id)
                    if return_token_mask:
                        token_mask.append(0)
            else:
                base_ids = self._base_tokenizer.encode(
                    segment_content,
                    add_special_tokens=add_special_tokens
                )
                token_ids.extend(base_ids)
                if return_token_mask:
                    token_mask.extend([-1] * len(base_ids))

        # Do NOT append EOS when add_special_tokens=True. Qwen3 doesn't add
        # BOS/EOS either, and appending EOS here breaks lighteval's
        # tok_encode_pair: it relies on
        #   len(encode(ctx)) + len(encode(answer)) == len(encode(ctx + answer))
        # which the extra EOS violates by shifting the split by 1.

        if return_token_mask:
            return token_ids, token_mask
        return token_ids

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        **kwargs
    ) -> str:
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            special_ids = {self.eos_token_id, self.pad_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        parts = []
        i = 0

        while i < len(token_ids):
            tid = token_ids[i]

            if tid == self.dna_begin_token_id:
                dna_tokens = []
                i += 1

                while i < len(token_ids) and token_ids[i] != self.dna_end_token_id:
                    if token_ids[i] in self.dna_id_to_token:
                        dna_tokens.append(self.dna_id_to_token[token_ids[i]])
                    i += 1

                dna_seq = ''.join(dna_tokens)

                if skip_special_tokens:
                    parts.append(dna_seq)
                else:
                    parts.append(f"<dna>{dna_seq}")
                    if i < len(token_ids) and token_ids[i] == self.dna_end_token_id:
                        parts.append("</dna>")
                        i += 1

            elif tid in self.dna_id_to_token:
                # This branch handles k-mer tokens that appear without a <dna>
                # wrapper — the common generation case where <dna> was in the
                # prompt but only the generated portion is being decoded.
                # K-mer tokens are content, not special tokens, so always decode
                # them.  Only drop true DNA special tokens (<dna>, </dna>, <oov>)
                # when skip_special_tokens=True.
                is_dna_special = tid in (self.dna_begin_token_id, self.dna_end_token_id, self.oov_token_id)
                if not (skip_special_tokens and is_dna_special):
                    parts.append(self.dna_id_to_token[tid])
                i += 1

            else:
                text_ids = []
                while i < len(token_ids):
                    curr_id = token_ids[i]
                    if curr_id in self.dna_id_to_token or curr_id == self.dna_begin_token_id:
                        break
                    text_ids.append(curr_id)
                    i += 1

                if text_ids:
                    decoded = self._base_tokenizer.decode(text_ids, skip_special_tokens=skip_special_tokens)
                    parts.append(decoded)

        return ''.join(parts)

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = False,
        **kwargs
    ) -> List[str]:
        return [
            self.decode(
                seq.tolist() if hasattr(seq, 'tolist') else list(seq),
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )
            for seq in sequences
        ]

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_mask: bool = False,
        auto_dna_tags: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if add_special_tokens:
            warnings.warn(
                "HybridTokenizer does not support add_special_tokens=True, ignoring.",
                UserWarning
            )
            add_special_tokens = False
            
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]

        all_ids = []
        all_masks = [] if return_token_mask else None

        for t in texts:
            if return_token_mask:
                ids, mask = self.encode(t, add_special_tokens=add_special_tokens, return_token_mask=True, auto_dna_tags=auto_dna_tags)
                all_ids.append(ids)
                all_masks.append(mask)
            else:
                ids = self.encode(t, add_special_tokens=add_special_tokens, return_token_mask=False, auto_dna_tags=auto_dna_tags)
                all_ids.append(ids)

        if padding:
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)

            padded_ids = []
            attention_masks = []
            padded_token_masks = [] if return_token_mask else None

            for idx, ids in enumerate(all_ids):
                pad_len = max_len - len(ids)

                if pad_len > 0:
                    ids = ids + [self.pad_token_id] * pad_len
                    attn = [1] * (max_len - pad_len) + [0] * pad_len
                    if return_token_mask:
                        mask = all_masks[idx] + [-2] * pad_len
                else:
                    ids = ids[:max_len]
                    attn = [1] * max_len
                    if return_token_mask:
                        mask = all_masks[idx][:max_len]

                padded_ids.append(ids)
                attention_masks.append(attn)
                if return_token_mask:
                    padded_token_masks.append(mask)

            all_ids = padded_ids
            all_masks = padded_token_masks
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]

        result = {
            "input_ids": all_ids if is_batch else all_ids[0],
            "attention_mask": attention_masks if is_batch else attention_masks[0],
        }

        if return_token_mask:
            result["token_mask"] = all_masks if is_batch else all_masks[0]

        if return_tensors == "pt":
            import torch
            if is_batch:
                result["input_ids"] = torch.tensor(result["input_ids"])
                result["attention_mask"] = torch.tensor(result["attention_mask"])
                if return_token_mask:
                    result["token_mask"] = torch.tensor(result["token_mask"])
            else:
                result["input_ids"] = torch.tensor([result["input_ids"]])
                result["attention_mask"] = torch.tensor([result["attention_mask"]])
                if return_token_mask:
                    result["token_mask"] = torch.tensor([result["token_mask"]])

        return BatchEncoding(result, tensor_type=return_tensors)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer files
        self._base_tokenizer.save_pretrained(save_directory)

        # Save DNA config
        dna_config = {
            "k": self.k,
            "dna_start_id": self.dna_start_id,
            "dna_vocab_size": self.dna_vocab_size,
            "dna_special_tokens": self.dna_special_tokens,
            "auto_dna_tags": self.auto_dna_tags,
        }

        dna_config_path = os.path.join(save_directory, "dna_config.json")
        with open(dna_config_path, "w", encoding="utf-8") as f:
            json.dump(dna_config, f, indent=2)

        # Update tokenizer_config.json with auto_map
        config_path = os.path.join(save_directory, "tokenizer_config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        config.update({
            "tokenizer_class": "HybridDNATokenizer",
            "auto_map": {
                "AutoTokenizer": ["tokenizer.HybridDNATokenizer", None]
            },
            "k": self.k,
            "auto_dna_tags": self.auto_dna_tags,
        })

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Copy this tokenizer.py to save directory
        import shutil
        src_py = os.path.abspath(__file__)
        dst_py = os.path.join(save_directory, "tokenizer.py")
        if os.path.exists(src_py) and src_py != dst_py:
            shutil.copy2(src_py, dst_py)

        return (save_directory,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        k = 6
        auto_dna_tags = False

        dna_config_path = os.path.join(pretrained_model_name_or_path, "dna_config.json")
        tok_config_path = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")

        if os.path.exists(dna_config_path):
            with open(dna_config_path, "r") as f:
                dna_config = json.load(f)
            k = dna_config.get("k", 6)
            auto_dna_tags = dna_config.get("auto_dna_tags", False)
        elif os.path.exists(tok_config_path):
            with open(tok_config_path, "r") as f:
                tok_config = json.load(f)
            k = tok_config.get("k", 6)
            auto_dna_tags = tok_config.get("auto_dna_tags", False)

        return cls(base_tokenizer_path=pretrained_model_name_or_path, k=k, auto_dna_tags=auto_dna_tags, **kwargs)
