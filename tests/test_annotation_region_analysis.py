from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dna_compress.annotation_region_analysis import (
    CLASS_TO_ID,
    aggregate_trace_by_annotation_regions,
    build_annotation_interval_index,
    class_ids_for_positions,
    load_coordinate_records,
    map_official_interval_to_local_acgt,
)
from dna_compress.probability_trace import (
    fused_depth_major_emit_positions,
    write_target_probability_trace,
)


class AnnotationRegionAnalysisTests(unittest.TestCase):
    def _write_synthetic_annotation(self, root: Path) -> Path:
        annotation_dir = root / "datasets" / "DNACorpus_annotations_official"
        species_dir = annotation_dir / "Fake"
        species_dir.mkdir(parents=True)
        gff_path = species_dir / "fake.gff3"
        gff_path.write_text(
            "##gff-version 3\n"
            "chr1\tRefSeq\tgene\t1\t14\t.\t+\t.\tID=gene1;gene_biotype=protein_coding\n"
            "chr1\tRefSeq\texon\t1\t4\t.\t+\t.\tID=exon1;Parent=gene1\n"
            "chr1\tRefSeq\tCDS\t3\t8\t.\t+\t0\tID=cds1;Parent=gene1\n"
            "chr1\tRefSeq\ttRNA\t10\t12\t.\t+\t.\tID=rna1;Parent=gene1\n"
            "chr1\tRefSeq\ttandem_repeat\t11\t14\t.\t+\t.\tID=rep1\n",
            encoding="utf-8",
        )
        (species_dir / "coordinate_records.tsv").write_text(
            "record_index\tseqid\tdescription\tofficial_length\tofficial_acgt_length\t"
            "local_acgt_start_0based\tlocal_acgt_end_0based_exclusive\n"
            "0\tchr1\tfake\t14\t12\t0\t12\n",
            encoding="utf-8",
        )
        (species_dir / "removed_non_acgt_intervals.tsv").write_text(
            "record_index\tseqid\tofficial_start_1based\tofficial_end_1based\tlength\tsymbols\n"
            "0\tchr1\t5\t6\t2\tN\n",
            encoding="utf-8",
        )
        with (annotation_dir / "official_mapping_validation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "code",
                    "species",
                    "source",
                    "accession",
                    "mode",
                    "chromosome",
                    "note",
                    "status",
                    "local_length",
                    "official_record_count",
                    "official_total_length",
                    "official_acgt_length",
                    "local_sha256",
                    "official_raw_sha256",
                    "official_acgt_sha256",
                    "exact_raw_match",
                    "exact_acgt_match",
                    "local_is_acgt_substring",
                    "acgt_substring_offset",
                    "fasta_path",
                    "gff3_path",
                    "seq_report_path",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "code": "Fake",
                    "species": "Fake species",
                    "source": "synthetic",
                    "accession": "fake",
                    "mode": "nuccore",
                    "chromosome": "",
                    "note": "",
                    "status": "exact_acgt_filtered_match",
                    "local_length": "12",
                    "official_record_count": "1",
                    "official_total_length": "14",
                    "official_acgt_length": "12",
                    "local_sha256": "",
                    "official_raw_sha256": "",
                    "official_acgt_sha256": "",
                    "exact_raw_match": "False",
                    "exact_acgt_match": "True",
                    "local_is_acgt_substring": "True",
                    "acgt_substring_offset": "0",
                    "fasta_path": "",
                    "gff3_path": str(gff_path),
                    "seq_report_path": "",
                    "error": "",
                }
            )
        return annotation_dir

    def test_coordinate_mapping_splits_removed_non_acgt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_dir = self._write_synthetic_annotation(Path(tmpdir))
            record = load_coordinate_records(annotation_dir / "Fake")["chr1"]
            self.assertEqual(
                map_official_interval_to_local_acgt(record, 3, 8, core_base_count=12),
                [(2, 4), (4, 6)],
            )

    def test_build_index_uses_overlap_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotation_dir = self._write_synthetic_annotation(root)
            index = build_annotation_interval_index(
                species="Fake",
                annotation_dir=annotation_dir,
                output_dir=root / "out",
                repo_root=root,
                overwrite=True,
            )
            labels = class_ids_for_positions(index, np.arange(12, dtype=np.int64))
            self.assertEqual(labels[2], CLASS_TO_ID["cds"])
            self.assertEqual(labels[7], CLASS_TO_ID["rna"])
            self.assertTrue(np.all(labels[8:12] == CLASS_TO_ID["repeat_mobile_existing"]))
            self.assertTrue(index.feature_table_path.exists())

    def test_streaming_region_aggregation_matches_direct_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotation_dir = self._write_synthetic_annotation(root)
            sequence = "ACGTACGTACGT"
            positions = fused_depth_major_emit_positions(core_base_count=12, window_bases=4, token_merge_size=1)
            probs_by_position = np.linspace(0.2, 0.9, 12, dtype=np.float64)
            probs = probs_by_position[positions]
            symbols = np.asarray([{"A": 0, "C": 1, "G": 2, "T": 3}[sequence[pos]] for pos in positions], dtype=np.int16)
            trace_dir = root / "trace"
            write_target_probability_trace(
                trace_dir,
                model_family="fake",
                model_id="fake",
                source_payload=sequence.encode("ascii"),
                normalized_sequence=sequence,
                core_sequence=sequence,
                tail_sequence="",
                target_prob=probs,
                target_symbol=symbols,
                emit_position=positions,
                window_bases=4,
                token_merge_size=1,
                producer_config={},
                shard_rows=5,
                overwrite=True,
            )
            index = build_annotation_interval_index(
                species="Fake",
                annotation_dir=annotation_dir,
                output_dir=root / "out",
                trace_dir=trace_dir,
                repo_root=root,
                overwrite=True,
            )
            result = aggregate_trace_by_annotation_regions(trace_dir=trace_dir, annotation_index=index)
            rows = {row["region_class"]: row for row in result["region_rows"]}

            labels = class_ids_for_positions(index, positions)
            bits = -np.log2(probs)
            for class_name, class_id in CLASS_TO_ID.items():
                mask = labels == class_id
                expected_count = int(np.sum(mask))
                self.assertEqual(rows[class_name]["base_count"], expected_count)
                if expected_count:
                    self.assertAlmostEqual(rows[class_name]["mean_bpb"], float(np.mean(bits[mask])), places=7)


if __name__ == "__main__":
    unittest.main()
