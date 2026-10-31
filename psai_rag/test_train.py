"""CPU-only checks for retrieval split integrity and contrastive batching."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psai_rag.train import (
    SourceAwareBatchSampler, _base_provenance, _publish_checkpoint, _settings, _text_key,
    _write_json, validate_training_data,
)


def pair(number, group=None):
    group = group or f"g{number}"
    return {
        "query_id": f"q{number}", "query": f"question {number}",
        "positive_id": f"p{number}", "positive": f"relevant text {number}",
        "negative_id": f"n{number}", "negative": f"distractor text {number}",
        "doc_id": f"doc{group}", "source_group_id": group, "split": "train",
        "supervision": "synthetic_genq_teacher_filtered",
    }


def corpus_for(rows):
    return [{"chunk_id": row[f"{key}_id"], "text": row[key],
             "doc_id": row["doc_id"], "source_group_id": row["source_group_id"],
             "split": row["split"]}
            for row in rows for key in ("positive", "negative")]


class TrainingDataTests(unittest.TestCase):
    def test_cross_partition_family_leakage_fails(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_training_data([pair(1)], {
                "split_groups": {"train": ["g1"], "validation": ["g1"]}})

    def test_known_positive_cannot_be_explicit_negative(self):
        rows = [pair(1), pair(2)]
        rows[1]["query"] = rows[0]["query"]
        rows[1]["positive"] = rows[0]["negative"]
        with self.assertRaisesRegex(ValueError, "known positive"):
            validate_training_data(rows, {"split_groups": {"train": ["g1", "g2"]}})

    def test_corpus_id_text_and_source_are_verified(self):
        rows = [pair(1)]
        manifest = {"split_groups": {"train": ["g1"], "validation": ["heldout"]}}
        corpus = corpus_for(rows)
        result = validate_training_data(rows, manifest, corpus)
        self.assertTrue(result["corpus_references_verified"])
        self.assertFalse(result["labels_are_human_ground_truth"])
        corpus[1]["source_group_id"] = "heldout"
        with self.assertRaisesRegex(ValueError, "training document"):
            validate_training_data(rows, manifest, corpus)

    def test_negative_cannot_equal_positive_after_unicode_normalization(self):
        row = pair(1)
        row["negative"] = "  RELEVANT   text 1 "
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_training_data([row], {"split_groups": {"train": ["g1"]}})

    def test_batching_preserves_every_row_and_excludes_source_and_text_collisions(self):
        rows = [pair(i, f"g{i % 4}") for i in range(25)]
        rows[3]["positive"] = rows[0]["positive"].upper()
        sampler = SourceAwareBatchSampler(rows, 16, seed=42)
        batches = sampler.batches(0)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(rows))))
        self.assertEqual(batches, sampler.batches(0))
        self.assertNotEqual(batches, sampler.batches(1))
        for batch in batches:
            families = [rows[index]["source_group_id"] for index in batch]
            documents = [rows[index]["doc_id"] for index in batch]
            texts = [_text_key(rows[index][column]) for index in batch
                     for column in ("query", "positive", "negative")]
            self.assertEqual(len(families), len(set(families)))
            self.assertEqual(len(documents), len(set(documents)))
            self.assertEqual(len(texts), len(set(texts)))

    def test_single_source_triplet_is_useful_but_pair_only_singleton_is_rejected(self):
        row = pair(1)
        self.assertEqual(SourceAwareBatchSampler([row], 16).batches(), [[0]])
        del row["negative"]
        with self.assertRaisesRegex(ValueError, "no contrastive negatives"):
            SourceAwareBatchSampler([row], 16).batches()

    def test_manifest_nested_format_is_also_accepted(self):
        result = validate_training_data([pair(1)], {
            "splits": {"train": {"source_group_ids": ["g1"]},
                       "validation": {"source_group_ids": ["g2"]}}})
        self.assertEqual(result["source_groups"], 1)


class TrainingProvenanceTests(unittest.TestCase):
    def test_failed_checkpoint_publication_preserves_previous_committed_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, pending, new = root / "step_1", root / ".pending", root / "step_2"
            old.mkdir()
            pending.mkdir()
            (old / "commit.json").write_text('{"global_step":1}')
            (pending / "commit.json").write_text('{"global_step":2}')
            latest = root / "latest_checkpoint.json"
            _write_json(latest, {"adapter_dir": str(old), "global_step": 1})
            previous = latest.read_bytes()
            with patch("psai_rag.train._write_json", side_effect=OSError("simulated interruption")):
                with self.assertRaises(OSError):
                    _publish_checkpoint(pending, new, latest, 2)
            self.assertEqual(latest.read_bytes(), previous)
            self.assertEqual(json.loads((old / "commit.json").read_text())["global_step"], 1)
            self.assertTrue((new / "commit.json").exists())

    def test_checkpoint_pointer_only_selects_a_complete_committed_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending, committed = root / ".pending", root / "step_1"
            pending.mkdir()
            (pending / "commit.json").write_text('{"global_step":1}')
            latest = root / "latest_checkpoint.json"
            _publish_checkpoint(pending, committed, latest, 1)
            pointer = json.loads(latest.read_text())
            self.assertEqual(pointer["adapter_dir"], str(committed))
            self.assertEqual(len(pointer["commit_sha256"]), 64)
            self.assertFalse(pending.exists())

    def test_output_cannot_escape_project_or_overwrite_project_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for candidate in (root, root.parent / "outside"):
                with self.assertRaisesRegex(ValueError, "child"):
                    _settings(root, {"output_dir": candidate})

    def test_atomic_receipt_replaces_previous_complete_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            _write_json(path, {"status": "training"})
            _write_json(path, {"status": "completed"})
            self.assertEqual(json.loads(path.read_text())["status"], "completed")
            self.assertFalse(path.with_name(path.name + ".pending").exists())

    def test_base_requires_native_normalized_cls_snapshot_and_no_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / ("a" * 40)
            (base / "1_Pooling").mkdir(parents=True)
            files = {
                "config.json": {"model_type": "xlm-roberta", "hidden_size": 1024,
                                "architectures": ["XLMRobertaModel"]},
                "modules.json": [{"type": "sentence_transformers.models.Normalize"}],
                "tokenizer_config.json": {},
                "1_Pooling/config.json": {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False},
            }
            for name, value in files.items():
                (base / name).write_text(json.dumps(value), encoding="utf-8")
            (base / "pytorch_model.bin").write_bytes(b"fake CPU-test weight fixture")
            provenance = _base_provenance(base)
            self.assertEqual(provenance["revision"], "a" * 40)
            self.assertEqual(len(provenance["weights_sha256"]["pytorch_model.bin"]), 64)
            (base / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adapter"):
                _base_provenance(base)


if __name__ == "__main__":
    unittest.main()
