# DNACompress Data Pipeline and Training Resumption

## Overview

This document describes the data reading pipeline in DNACompress, with a focus on training resumption behavior, randomness guarantees, and known trade-offs.

## Data Pipeline Modes

### 1. `source_batch_file_stream` Mode (Recommended for Large-Scale Training)

**Purpose**: Sequential disk reading with controlled source mixing and efficient resumption.

**Key Features**:
- Reads DNA sequences sequentially from indexed FASTA files
- Organizes reads by genomic source (e.g., `gtdb_v220`, `metagenomes`)
- Supports configurable source mixing ratios
- Enables mid-epoch training resumption
- Excellent disk I/O efficiency (sequential reads)

**Configuration Parameters**:
```bash
--indexed-window-mode source_batch_file_stream
--indexed-source-mix-chunk-batches 64        # Batches per source mixing window
--indexed-source-read-chunk-windows 8192     # Windows to read before optional shuffle
--indexed-source-read-chunk-shuffle          # Enable in-memory shuffle after sequential read
--indexed-source-file-order-seed 0           # Seed for file order randomization
```

**Data Flow**:
1. Load pre-built Parquet index from disk
2. Split runs into train/val/test using deterministic seed-based hashing
3. Group files by genomic source
4. For each batch:
   - Determine source assignment based on sampling weights
   - Read `source_read_chunk_windows` sequentially from disk
   - Optionally shuffle in-memory (if `source_read_chunk_shuffle=True`)
   - Emit samples to training loop

### 2. Other Window Modes

- **`sliding_random`**: Random start positions with overlap (poor resume support)
- **`nonoverlap_random`**: Non-overlapping windows in random order (poor resume support)
- **`nonoverlap_file_stream`**: Sequential file-by-file reading (good resume support)

## Training Resumption Mechanism

### Checkpoint State

When saving a checkpoint, the following data pipeline state is persisted:

```python
data_state = {
    "schema_version": 1,
    "train": {
        "dataset_kind": str,                    # e.g., "indexed_fasta_source_batch"
        "completed_global_steps": int,
        "batches_per_epoch": int,
        "current_epoch_index": int,
        "current_epoch": int,                   # Human-readable (1-indexed)
        "current_batch_index_in_epoch": int,
        "next_epoch_index": int,                # Where to resume
        "next_epoch": int,
        "next_batch_index_in_epoch": int,       # Where to resume
    }
}
```

### Resume Process

**For Batch-Producing Datasets** (`source_batch_file_stream`):

1. Load checkpoint and extract `next_epoch_index` and `next_batch_index_in_epoch`
2. Create dataset with `set_start_batch_index(next_batch_index_in_epoch)`
3. Dataset calculates skip counts per source:
   - Replay batch assignment algorithm deterministically
   - Compute how many items each source contributed up to target batch
4. Each `_SourceSequentialReader` calls `skip_items(count)` to advance file position
5. Training continues from correct position

**For Non-Batch-Producing Datasets** (RandomWindowDataset, SequentialWindowDataset):

1. Training loop iterates through DataLoader
2. For each batch where `enumerated_batch_index < resume_batch_index`:
   - Skip the batch without training (lines in experiment.py:1286-1288)
   - This is inefficient but simple

**NOTE**: As of this optimization, non-batch-producing datasets now support efficient skip via `set_start_batch_index()`.

## Randomness and Reproducibility

### Seed Hierarchy

All randomness in the data pipeline is controlled by configuration seeds:

```python
config.train.seed                              # Base seed for model and training
config.data.indexed_split_seed                 # Seed for train/val/test split
config.data.indexed_source_file_order_seed     # Seed for file order within sources
```

### Deterministic Guarantees

Given the same seed configuration, the data pipeline produces **identical sequences** across runs:

1. **File order per cycle**: `seed + source_file_order_seed + source_id * 1_000_003 + cycle * 97_003`
2. **Shuffle within chunk**: Adds `chunk_index * 65_537` to file order seed
3. **Source slot assignment**: `seed + source_file_order_seed + balance_window * 1_000_003`
4. **DDP rank assignment**: Modulo arithmetic on `rank * num_workers + worker_id`

### RandomWindowDataset Reproducibility

For `RandomWindowDataset` (used in non-indexed modes):

```python
def __getitem__(self, index: int):
    rng = random.Random(self.seed + index)  # Deterministic per index
    source_index = rng.choices(...)
    start = rng.randrange(...)
```

- Each index generates the same sample given the same seed
- After resume, indices replay from 0 with the same seed
- This means **resumed training sees the same data again** (acceptable for reproducibility)

## Known Trade-offs and Limitations

### 1. Shuffle + Mid-Epoch Resume

**Trade-off**: When `source_read_chunk_shuffle=True` and resuming mid-epoch:

- The `skip_items()` method must physically read and discard items from shuffle buffers
- Up to `source_read_chunk_windows` items (default 8192) may be re-read on resume
- This is because shuffle reorders buffer contents, so skip count landing mid-buffer requires loading the buffer

**Impact**:
- Negligible for large-scale training (8192 out of millions of windows)
- Adds ~1-5 seconds to resume startup time
- Maintains correctness: no data duplication or loss

**Alternative**: Disable shuffle (`--no-indexed-source-read-chunk-shuffle`) for perfectly efficient skip, at the cost of less training variance.

### 2. Persistent Workers + Mid-Epoch Resume

**Limitation**: Cannot resume `source_batch_file_stream` mid-epoch with `persistent_workers=True`.

**Reason**:
- Persistent workers maintain file handles and reader state across epochs
- Mid-epoch resume requires resetting reader state via `skip_items()`
- This needs fresh reader objects, incompatible with persistent workers

**Current Behavior**: Raises error at startup if this combination is detected.

**Workaround**: Use `--no-persistent-workers` when mid-epoch resume is needed.

**Recommendation**: 
- For long training runs: disable persistent workers (resume flexibility)
- For short epochs: enable persistent workers (faster epoch transitions)

### 3. Evaluation Does Not Affect Training Data

**Design**: Evaluation runs on completely separate data:
- Training: split determined by `train_ratio` (default 0.9)
- Validation: split determined by `val_ratio` (default 0.05)
- Test: split determined by `test_ratio` (default 0.05)

**Behavior**:
- Evaluation pauses the training loop but does not affect training data ordering
- Validation uses `SequentialWindowDataset` (deterministic)
- Training uses `RandomWindowDataset` or streaming modes (randomized)

**Seeds**:
```python
train_seed = config.train.seed
val_seed = config.train.seed + 1_000_000
test_seed = config.train.seed + 2_000_000
```

## Performance Characteristics

### Disk I/O Efficiency

| Mode | Read Pattern | Resume Efficiency | Randomness |
|------|--------------|-------------------|------------|
| `source_batch_file_stream` + shuffle | Sequential read + in-memory shuffle | Excellent | High |
| `source_batch_file_stream` | Sequential read only | Excellent | Medium |
| `nonoverlap_file_stream` | Sequential file-by-file | Good | Medium |
| `sliding_random` | Random seeks | Poor | Highest |
| `nonoverlap_random` | Random seeks | Poor | High |

### Resume Startup Time

- **Batch-producing datasets**: O(skip_count) but optimized (skip without reading most data)
- **Non-batch-producing datasets (optimized)**: O(1) with new skip support
- **Shuffle mode**: +1-5 seconds for buffer reconstruction

### Memory Usage

- Shuffle buffer: `source_read_chunk_windows * seq_length * token_merge_size` bytes per source
- Default: 8192 windows * 1024 tokens * 1 byte ≈ 8 MB per source
- With 10 sources: ~80 MB shuffle buffer memory

## Best Practices

### For Large-Scale Training

```bash
--indexed-window-mode source_batch_file_stream
--indexed-source-read-chunk-shuffle           # Enable for better randomness
--no-persistent-workers                       # Allow mid-epoch resume
--eval-interval 100                           # Checkpoint frequently
```

### For Maximum Resume Efficiency

```bash
--indexed-window-mode source_batch_file_stream
--no-indexed-source-read-chunk-shuffle        # Disable shuffle for zero-overhead skip
--no-persistent-workers
```

### For Maximum Randomness

```bash
--indexed-window-mode sliding_random
# Note: Poor resume efficiency, use only for final training runs
```

## Testing and Validation

### Reproducibility Test

```bash
# Train for 100 steps
python scripts/run_dna_experiment.py --seed 42 --epochs 1 ...

# Resume from step 50
python scripts/run_dna_experiment.py --seed 42 --init-from resume --out path/to/checkpoint ...

# Verify: losses at steps 51-100 should match continuation of first run
```

### Determinism Test

```bash
# Run 1
python scripts/run_dna_experiment.py --seed 42 --epochs 1 ...

# Run 2 (same seed)
python scripts/run_dna_experiment.py --seed 42 --epochs 1 ...

# Verify: training losses should be identical step-by-step
```

## Troubleshooting

### "Resuming source_batch_file_stream from the middle of an epoch requires persistent DataLoader workers to be disabled"

**Solution**: Add `--no-persistent-workers` to your command.

### Resume seems to re-train on same data

**Check**: Are you using `RandomWindowDataset`? This is expected behavior for reproducibility.

**Solution**: Use `source_batch_file_stream` mode for sequential data coverage.

### Resume is very slow

**Check**: Are you using a non-batch-producing dataset in an old version?

**Solution**: Update to latest version with efficient skip support, or switch to `source_batch_file_stream`.

## Implementation Details

### Key Files

- `/dna_compress/fasta_fragment_index.py`: IndexedMegabyteSourceBatchStreamDataset, _SourceSequentialReader
- `/dna_compress/data.py`: RandomWindowDataset, SequentialWindowDataset
- `/dna_compress/experiment.py`: Training loop, checkpoint save/load, resume logic

### Key Methods

- `IndexedMegabyteSourceBatchStreamDataset._source_skip_counts_before_batch()`: Calculate skip counts
- `_SourceSequentialReader.skip_items()`: Advance file read position
- `_resume_training_position()`: Extract resume position from checkpoint
- `_training_data_state()`: Build data state for checkpoint save

## Changelog

### 2026-06-04: Data Pipeline Optimization
- Added efficient skip support for RandomWindowDataset and SequentialWindowDataset
- Removed legacy parameter storage (source_balance_batches, source_read_block_windows)
- Added comprehensive documentation for shuffle + resume trade-off
- Added deprecation warnings for old parameter names
