"""CPU-only retrieval mechanics tests using synthetic labels and fake vectors."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from psai_rag import evaluate


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.closed = False

    def synchronize(self):
        pass

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[1.0, 0.0] if "alpha" in text else [0.0, 1.0]
                           for text in texts], dtype=np.float32)

    def summary(self):
        return {"device": "cpu", "dtype": "fake_float32", "pooling": "normalized_CLS",
                "normalization_dtype": "float32", "cls_cast_before_normalization": True,
                "batch_size": 1, "max_length": 256, "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0}

    def close(self):
        self.closed = True


class BM25Tests(unittest.TestCase):
    def test_retrieval_scores_unicode_and_deterministic_ties(self):
        index = evaluate.BM25([
            {"chunk_id": "z", "text": "Yêu cầu lưu nhật ký"},
            {"chunk_id": "a", "text": "Build a dashboard"},
            {"chunk_id": "b", "text": "Yêu cầu lưu nhật ký"},
        ])
        ranked = index.rank("YÊU CẦU", top_k=2)
        self.assertEqual([row["chunk_id"] for row in ranked], ["b", "z"])
        self.assertGreater(ranked[0]["bm25_score"], 0)
        self.assertEqual(index.rank("unseen", candidate_ids=["z", "a"])[0]["chunk_id"], "a")
        self.assertEqual(evaluate.BM25([]).scores("query").shape, (0,))

    def test_negative_mining_excludes_all_known_and_duplicate_positives(self):
        index = evaluate.BM25([
            {"chunk_id": "p", "text": "Alpha requirement"},
            {"chunk_id": "duplicate", "text": " ALPHA   requirement "},
            {"chunk_id": "other_positive", "text": "Alpha milestone"},
            {"chunk_id": "duplicate_other", "text": "alpha MILESTONE"},
            {"chunk_id": "n1", "text": "Alpha release"},
            {"chunk_id": "n2", "text": "Alpha release"},
            {"chunk_id": "n3", "text": "Beta testing"},
        ])
        negatives = index.negative_candidates("alpha", "p", ["other_positive"], top_k=10)
        self.assertEqual([row["chunk_id"] for row in negatives], ["n1", "n3"])
        self.assertEqual(index.negative_candidates("alpha", "p", top_k=0), [])
        with self.assertRaises(ValueError):
            index.negative_candidates("alpha", "missing")


class MetricTests(unittest.TestCase):
    def test_multiple_positive_metrics_and_cutoff(self):
        metrics = evaluate.ranking_metrics(["negative", "p1", "p2"], ["p1", "p2"])
        self.assertEqual(metrics["Recall@1"], 0.0)
        self.assertEqual(metrics["Recall@5"], 1.0)
        self.assertEqual(metrics["MRR@10"], 0.5)
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["nDCG@10"], expected)
        metrics = evaluate.ranking_metrics([str(index) for index in range(11)], ["10"])
        self.assertEqual(metrics["Recall@10"], 0)
        self.assertEqual(metrics["MRR@10"], 0)
        with self.assertRaises(ValueError):
            evaluate.ranking_metrics(["p", "p"], ["p"])

    def test_rrf_uses_both_rankings_and_rejects_duplicates(self):
        fused = evaluate.reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        self.assertAlmostEqual(fused["a"], 1 / 61 + 1 / 62)
        self.assertEqual(fused["a"], fused["b"])
        with self.assertRaises(ValueError):
            evaluate.reciprocal_rank_fusion([["a", "a"]])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="psai-rag-eval-")
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.study.mkdir()
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"hidden_size":2}', encoding="utf-8")
        (self.base / "model.safetensors").write_bytes(b"fake weights; never loaded")
        self.corpus = [
            {"chunk_id": "train", "doc_id": "A", "source_group_id": "train-group", "split": "train", "text": "alpha"},
            {"chunk_id": "test", "doc_id": "A", "source_group_id": "test-group", "split": "test", "text": "alpha"},
            {"chunk_id": "A1", "doc_id": "A", "source_group_id": "GA", "split": "validation", "text": "alpha requirement"},
            {"chunk_id": "A2", "doc_id": "A", "source_group_id": "GA", "split": "validation", "text": "beta details"},
            {"chunk_id": "B1", "doc_id": "B", "source_group_id": "GB", "split": "validation", "text": "beta activity"},
        ]
        self.pairs = [
            {"query_id": "qa", "query": "alpha question", "positive_id": "A1", "positive": "alpha requirement", "doc_id": "A", "source_group_id": "GA"},
            {"query_id": "qb", "query": "beta question", "positive_id": "B1", "doc_id": "B", "source_group_id": "GB"},
        ]
        self.write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self):
        for name, rows in (("corpus.jsonl", self.corpus), ("validation_pairs.jsonl", self.pairs)):
            (self.study / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def evaluate(self, adapter=None):
        encoder = FakeEncoder()
        with patch.object(evaluate, "_make_encoder", return_value=encoder):
            result = evaluate.evaluate_retriever(self.root, self.study, self.base, adapter, {})
        self.assertTrue(encoder.closed)
        return result, encoder

    def test_import_does_not_load_model_libraries(self):
        command = (
            "import sys; import psai_rag.evaluate; "
            "assert not ({'torch','transformers','sentence_transformers','peft','chromadb'} & set(sys.modules))"
        )
        subprocess.run([sys.executable, "-c", command], cwd=Path(__file__).resolve().parents[1],
                       check=True, capture_output=True, text=True)

    def test_full_evaluation_is_validation_only_and_writes_provenance(self):
        result, encoder = self.evaluate()
        self.assertEqual(result["validation_chunks"], 3)
        self.assertEqual(result["metrics"]["within_doc"]["dense"]["Recall@1"], 1.0)
        self.assertEqual(result["label_kind"], "source_chunk_pseudopositive")
        self.assertIn("not human ground truth", result["weak_label_notice"])
        predictions = list(evaluate._read_jsonl(self.study / "base_predictions.jsonl"))
        self.assertEqual(len(predictions), 12)
        for row in predictions:
            retrieved = {hit["chunk_id"] for hit in row["ranking"]}
            self.assertFalse(retrieved & {"train", "test"})
            if row["scope"] == "within_doc":
                allowed = {"A1", "A2"} if row["doc_id"] == "A" else {"B1"}
                self.assertTrue(retrieved.issubset(allowed))
        self.assertEqual(len(encoder.calls), 2)
        self.assertTrue((self.study / "base_predictions.csv").is_file())
        self.assertEqual(result["runtime"]["peak_allocated_bytes"], 0)
        self.assertEqual(result["cache_provenance"]["inference"]["normalization_dtype"], "float32")
        self.assertTrue(result["cache_provenance"]["inference"]["cls_cast_before_normalization"])

    def test_cache_reuse_and_invalidation_by_tokenizer_corpus_and_adapter(self):
        first, _ = self.evaluate()
        repeated, encoder = self.evaluate()
        self.assertEqual(encoder.calls, [])
        self.assertTrue(repeated["embedding_cache"]["passages"]["reused"])
        self.assertEqual(first["embedding_cache"]["passages"]["key"], repeated["embedding_cache"]["passages"]["key"])
        (self.base / "tokenizer_config.json").write_text('{"version":2}', encoding="utf-8")
        tokenizer_changed, encoder = self.evaluate()
        self.assertEqual(len(encoder.calls), 2)
        self.assertNotEqual(first["embedding_cache"]["passages"]["key"], tokenizer_changed["embedding_cache"]["passages"]["key"])
        self.corpus[3]["text"] = "beta changed details"
        self.write_inputs()
        corpus_changed, _ = self.evaluate()
        self.assertNotEqual(tokenizer_changed["embedding_cache"]["passages"]["key"], corpus_changed["embedding_cache"]["passages"]["key"])
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"fake adapter")
        adapted, _ = self.evaluate(adapter)
        self.assertEqual(adapted["variant"], "adapter")
        self.assertNotEqual(adapted["embedding_cache"]["passages"]["key"], corpus_changed["embedding_cache"]["passages"]["key"])

    def test_tampered_vector_cache_is_recomputed(self):
        result, _ = self.evaluate()
        path = Path(result["embedding_cache"]["passages"]["path"])
        np.save(path, np.ones((3, 2), dtype=np.float32))
        result, encoder = self.evaluate()
        self.assertFalse(result["embedding_cache"]["passages"]["reused"])
        self.assertEqual(len(encoder.calls), 1)

    def test_nonvalidation_positive_rejected_before_model_load(self):
        self.pairs[0]["positive_id"] = "test"
        self.write_inputs()
        with patch.object(evaluate, "_make_encoder") as make, self.assertRaises(ValueError):
            evaluate.evaluate_retriever(self.root, self.study, self.base, None, {})
        make.assert_not_called()

    def test_inconsistent_query_or_positive_provenance_is_rejected(self):
        self.pairs[0]["source_group_id"] = "wrong"
        self.write_inputs()
        with self.assertRaises(ValueError):
            evaluate._load_validation_inputs(self.study / "corpus.jsonl", self.study / "validation_pairs.jsonl")
        self.pairs[0]["source_group_id"] = "GA"
        self.pairs[0]["positive"] = "invented positive"
        self.write_inputs()
        with self.assertRaises(ValueError):
            evaluate._load_validation_inputs(self.study / "corpus.jsonl", self.study / "validation_pairs.jsonl")

    def test_repeated_query_rows_merge_multiple_known_positives(self):
        self.pairs.append({"query_id": "qa", "query": "alpha question", "positive_id": "A2", "doc_id": "A", "source_group_id": "GA"})
        self.write_inputs()
        _, queries = evaluate._load_validation_inputs(self.study / "corpus.jsonl", self.study / "validation_pairs.jsonl")
        self.assertEqual(queries[0]["positive_ids"], ["A1", "A2"])


if __name__ == "__main__":
    unittest.main()
