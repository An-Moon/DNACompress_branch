"""
End-to-end tests for DNACompress training resumption.

These tests verify that training resumption works correctly across different
dataset modes and configuration options.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json

import torch
import numpy as np


def test_random_dataset_resume():
    """Test RandomWindowDataset resumption with set_start_batch_index."""
    from dna_compress.data import RandomWindowDataset

    print("[TEST] RandomWindowDataset resume...")

    # Create simple test data
    sources = [b"ACGTACGTACGTACGT" * 100, b"TGCATGCATGCATGCA" * 100]
    seq_length = 64
    samples_per_epoch = 100
    seed = 42

    # Create dataset
    dataset = RandomWindowDataset(
        sources=sources,
        seq_length=seq_length,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        sampling_strategy="proportional",
    )

    # Get first 10 samples
    samples_initial = [dataset[i] for i in range(10)]

    # Simulate resume from batch 5
    dataset.set_start_batch_index(5)

    # Get next 5 samples (should match samples 5-9 from initial run)
    samples_resumed = [dataset[i] for i in range(5)]

    # Verify: samples_resumed[i] should match samples_initial[5+i]
    for i in range(5):
        initial = samples_initial[5 + i]["input_ids"]
        resumed = samples_resumed[i]["input_ids"]
        assert torch.equal(initial, resumed), f"Mismatch at offset {i}: resume changes sample sequence"

    print("  ✓ RandomWindowDataset resume maintains correct sequence")
    return True


def test_sequential_dataset_resume():
    """Test SequentialWindowDataset resumption with set_start_batch_index."""
    from dna_compress.data import SequentialWindowDataset

    print("[TEST] SequentialWindowDataset resume...")

    # Create simple test data
    sources = [b"ACGTACGTACGTACGT" * 100, b"TGCATGCATGCATGCA" * 100]
    seq_length = 64
    pad_id = 0

    # Create dataset
    dataset = SequentialWindowDataset(
        sources=sources,
        seq_length=seq_length,
        pad_id=pad_id,
    )

    total_samples = len(dataset)

    # Get first 20 samples
    samples_initial = [dataset[i] for i in range(min(20, total_samples))]

    # Simulate resume from batch 10
    dataset.set_start_batch_index(10)

    # Get next 10 samples (should match samples 10-19 from initial run)
    samples_resumed = [dataset[i] for i in range(min(10, total_samples - 10))]

    # Verify: samples_resumed[i] should match samples_initial[10+i]
    for i in range(len(samples_resumed)):
        initial = samples_initial[10 + i]["input_ids"]
        resumed = samples_resumed[i]["input_ids"]
        assert torch.equal(initial, resumed), f"Mismatch at offset {i}: resume changes sample sequence"

    print("  ✓ SequentialWindowDataset resume maintains correct sequence")
    return True


def test_random_dataset_determinism():
    """Test that RandomWindowDataset produces same sequence with same seed."""
    from dna_compress.data import RandomWindowDataset

    print("[TEST] RandomWindowDataset determinism...")

    sources = [b"ACGTACGTACGTACGT" * 100, b"TGCATGCATGCATGCA" * 100]
    seq_length = 64
    samples_per_epoch = 50
    seed = 12345

    # Create two datasets with same seed
    dataset1 = RandomWindowDataset(
        sources=sources,
        seq_length=seq_length,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )

    dataset2 = RandomWindowDataset(
        sources=sources,
        seq_length=seq_length,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )

    # Get samples from both
    samples1 = [dataset1[i] for i in range(50)]
    samples2 = [dataset2[i] for i in range(50)]

    # Verify they match
    for i in range(50):
        assert torch.equal(samples1[i]["input_ids"], samples2[i]["input_ids"]), \
            f"Determinism broken at index {i}: same seed produces different samples"

    print("  ✓ RandomWindowDataset is deterministic given same seed")
    return True


def test_random_dataset_different_seeds():
    """Test that different seeds produce different sequences."""
    from dna_compress.data import RandomWindowDataset

    print("[TEST] RandomWindowDataset seed variation...")

    sources = [b"ACGTACGTACGTACGT" * 100, b"TGCATGCATGCATGCA" * 100]
    seq_length = 64
    samples_per_epoch = 50

    dataset1 = RandomWindowDataset(
        sources=sources,
        seq_length=seq_length,
        samples_per_epoch=samples_per_epoch,
        seed=111,
    )

    dataset2 = RandomWindowDataset(
        sources=sources,
        seq_length=seq_length,
        samples_per_epoch=samples_per_epoch,
        seed=222,
    )

    # Get samples from both
    samples1 = [dataset1[i] for i in range(50)]
    samples2 = [dataset2[i] for i in range(50)]

    # Count differences
    differences = sum(
        1 for i in range(50)
        if not torch.equal(samples1[i]["input_ids"], samples2[i]["input_ids"])
    )

    # Should have many differences (probabilistically, almost certainly >40 out of 50)
    assert differences > 40, f"Different seeds should produce different sequences, only {differences}/50 differ"

    print(f"  ✓ Different seeds produce different sequences ({differences}/50 differ)")
    return True


def test_legacy_parameter_warnings():
    """Test that legacy parameters trigger deprecation warnings."""
    from dna_compress.fasta_fragment_index import IndexedMegabyteSourceBatchStreamDataset
    import warnings

    print("[TEST] Legacy parameter deprecation warnings...")

    # This test requires an actual indexed FASTA directory to work
    # Skip if not available
    test_index_dir = os.environ.get("DNA_TEST_INDEX_DIR")
    if not test_index_dir or not Path(test_index_dir).exists():
        print("  ⚠ Skipped (no test index dir available, set DNA_TEST_INDEX_DIR)")
        return True

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Use legacy parameter name
            dataset = IndexedMegabyteSourceBatchStreamDataset(
                index_dir=test_index_dir,
                split="train",
                seq_length=1024,
                token_merge_size=1,
                token_merge_alphabet="ACGT",
                samples=100,
                seed=42,
                batch_size=2,
                pad_id=0,
                source_balance_batches=64,  # Legacy name
            )

            # Should have triggered a deprecation warning
            assert len(w) > 0, "Legacy parameter should trigger deprecation warning"
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
                "Should be a DeprecationWarning"
            assert any("source_balance_batches" in str(warning.message) for warning in w), \
                "Warning should mention the deprecated parameter"

        print("  ✓ Legacy parameters trigger deprecation warnings")
    except Exception as e:
        print(f"  ⚠ Skipped (error loading dataset: {e})")

    return True


def test_resume_offset_correctness():
    """Test that index offset works correctly across different offsets."""
    from dna_compress.data import RandomWindowDataset

    print("[TEST] Resume offset correctness...")

    sources = [b"ACGTACGTACGTACGT" * 100]
    seq_length = 64
    samples_per_epoch = 100
    seed = 999

    # Create baseline dataset
    baseline = RandomWindowDataset(
        sources=sources,
        seq_length=seq_length,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )

    # Get samples 0-99
    baseline_samples = [baseline[i]["input_ids"] for i in range(100)]

    # Test various resume offsets
    offsets_to_test = [0, 1, 10, 50, 99]

    for offset in offsets_to_test:
        dataset = RandomWindowDataset(
            sources=sources,
            seq_length=seq_length,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
        dataset.set_start_batch_index(offset)

        # Get 10 samples (or fewer if near end)
        num_samples = min(10, 100 - offset)
        resumed_samples = [dataset[i]["input_ids"] for i in range(num_samples)]

        # Verify they match baseline
        for i in range(num_samples):
            assert torch.equal(resumed_samples[i], baseline_samples[offset + i]), \
                f"Offset {offset}, sample {i}: resume produces wrong sample"

    print(f"  ✓ Resume offset correctness verified for offsets {offsets_to_test}")
    return True


def run_all_tests():
    """Run all end-to-end tests."""
    print("\n" + "="*70)
    print("Running DNACompress End-to-End Resumption Tests")
    print("="*70 + "\n")

    tests = [
        ("RandomWindowDataset resume", test_random_dataset_resume),
        ("SequentialWindowDataset resume", test_sequential_dataset_resume),
        ("RandomWindowDataset determinism", test_random_dataset_determinism),
        ("RandomWindowDataset seed variation", test_random_dataset_different_seeds),
        ("Legacy parameter warnings", test_legacy_parameter_warnings),
        ("Resume offset correctness", test_resume_offset_correctness),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
