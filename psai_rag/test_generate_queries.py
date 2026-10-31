"""CPU-only fixtures for GenQ resume, filtering and partition invariants."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psai_rag.generate_queries import (
    _deduplicate_queries, _journal, _load_inputs, _round_robin_cap,
    generate_pairs,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _FakeGenerator:
    calls = 0

    def __init__(self, *_):
        type(self).calls += 1

    def generate(self, passages, seed):
        return [[f"what is the outcome of {passage}", f"how does the method for {passage} work"] for passage in passages]

    def close(self):
        pass


class _FakeTeacher:
    calls = 0

    def __init__(self, *_):
        type(self).calls += 1

    def score(self, pairs):
        return [2.0 if "NEGATIVE_HIGH" in passage else 0.0 if "NEGATIVE_LOW" in passage else 3.0
                for _, passage in pairs]

    def close(self):
        pass


class GenerationTests(unittest.TestCase):
    def _fixture(self, root):
        study = root / "reports/study"
        study.mkdir(parents=True)
        corpus, candidates = [], []
        split_groups = {"train": [], "validation": [], "test": ["test-group"]}
        for split, count in [("train", 48), ("validation", 12)]:
            for index in range(count):
                doc = f"{split}-{index:03d}"
                split_groups[split].append(doc)
                base = {"doc_id": doc, "source_group_id": doc, "split": split,
                        "source_sha256": "source-" + doc, "source_file": str(root / (doc + ".md"))}
                positive = {**base, "chunk_id": doc + "-positive", "text": f"subject {doc} project workflow summary"}
                corpus.append(positive)
                candidates.append({**positive, "candidate_id": "cand-" + positive["chunk_id"]})
                for negative in ["NEGATIVE_HIGH", "NEGATIVE_LOW"]:
                    corpus.append({**base, "chunk_id": doc + "-" + negative,
                                   "text": f"subject {doc} project workflow {negative}"})
        corpus.append({"chunk_id": "test-positive", "text": "never expose test sources to inference",
                       "doc_id": "test", "source_group_id": "test-group", "split": "test"})
        _jsonl(study / "corpus.jsonl", corpus)
        _jsonl(study / "candidate_passages.jsonl", candidates)
        (study / "split_manifest.json").write_text(json.dumps({"split_groups": split_groups}), encoding="utf-8")
        (study / "source_manifest.json").write_text("{}", encoding="utf-8")
        models = []
        for role, repo in [("query_generator", "doc2query/msmarco-t5-small-v1"),
                           ("pseudo_label_teacher", "cross-encoder/ms-marco-MiniLM-L6-v2")]:
            folder = root / "aux" / role
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}", encoding="utf-8")
            (folder / "model.safetensors").write_bytes(b"fixture")
            models.append({"role": role, "repo_id": repo, "revision": "a" * 40, "local_path": str(folder)})
        return study, {"models": models}

    def test_resume_uses_journals_and_keeps_hardest_same_document_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, models = self._fixture(root)
            _FakeGenerator.calls = _FakeTeacher.calls = 0
            with patch("psai_rag.generate_queries._QueryGenerator", _FakeGenerator), patch("psai_rag.generate_queries._Teacher", _FakeTeacher):
                first = generate_pairs(root, study, models, {})
                before = (study / "train_pairs.jsonl").read_bytes()
                second = generate_pairs(root, study, models, {})
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(before, (study / "train_pairs.jsonl").read_bytes())
            self.assertEqual((_FakeGenerator.calls, _FakeTeacher.calls), (1, 1))
            training = [json.loads(line) for line in before.splitlines()]
            validation = [json.loads(line) for line in (study / "validation_pairs.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(len(training), 96)
            self.assertEqual(len(validation), 24)
            self.assertTrue(all(row["negative_id"].startswith(row["doc_id"]) for row in training + validation))
            self.assertTrue(all(row["negative_id"].endswith("NEGATIVE_HIGH") for row in training + validation))
            self.assertTrue(all(row["teacher_margin"] == 1.0 for row in training + validation))
            self.assertTrue(all(not row["human_relevance_verified"] for row in training + validation))
            self.assertFalse({row["query_id"] for row in training} & {row["query_id"] for row in validation})
            self.assertEqual(first["coverage"]["train"]["selected_source_groups"], 48)
            # Strict source fingerprint rejects altered inputs before models load.
            with (study / "corpus.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                generate_pairs(root, study, models, {})

    def test_cross_partition_query_duplicates_are_removed_in_both_partitions(self):
        candidates = {name: {"candidate_id": name, "chunk_id": name, "split": split, "doc_id": name,
                             "text": "A source passage with content that differs from generated questions completely"}
                      for name, split in [("t", "train"), ("v", "validation")]}
        generated = [{"candidate_id": "t", "queries": ["What shared project result", "unique train project query"]},
                     {"candidate_id": "v", "queries": ["what shared project result", "unique validation project query"]}]
        rows, counts = _deduplicate_queries(generated, candidates)
        self.assertEqual(len(rows), 2)
        self.assertEqual(counts["cross_partition_duplicate_queries_removed"], 2)
        self.assertTrue(all("shared" not in row["query"].lower() for row in rows))

    def test_quality_filter_and_cross_document_ambiguity(self):
        passage = "one two three four five six seven eight nine ten"
        candidates = {name: {"candidate_id": name, "chunk_id": name, "split": "train", "doc_id": name,
                             "text": passage} for name in ["a", "b", "c", "d"]}
        generated = [
            {"candidate_id": "a", "queries": ["too few words", passage]},
            {"candidate_id": "b", "queries": ["one two three four five six seven eight", "word " * 41]},
            {"candidate_id": "c", "queries": ["what shared project result appears", "which unique question belongs here"]},
            {"candidate_id": "d", "queries": ["WHAT shared project result appears", "which other question belongs elsewhere"]},
        ]
        rows, counts = _deduplicate_queries(generated, candidates)
        self.assertEqual(len(rows), 2)
        self.assertEqual(counts["queries_rejected_word_count_4_to_40"], 2)
        self.assertEqual(counts["queries_rejected_verbatim_or_long_source_span"], 2)
        self.assertEqual(counts["cross_document_duplicate_queries_removed"], 2)

    def test_interrupted_terminal_journal_recovers_but_changed_fingerprint_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.jsonl"
            first = {"candidate_id": "a", "fingerprint": "f", "queries": ["one", "two"]}
            path.write_bytes((json.dumps(first) + "\n").encode() + b'{"candidate_id":')
            rows = _journal(path, "f", {"a"}, "candidate_id")
            self.assertEqual(set(rows), {"a"})
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                _journal(path, "changed", {"a"}, "candidate_id")

    def test_test_candidate_is_rejected_and_cap_spreads_source_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, _ = self._fixture(root)
            rows = [json.loads(line) for line in (study / "candidate_passages.jsonl").read_text("utf-8").splitlines()]
            rows[0]["split"] = "test"
            _jsonl(study / "candidate_passages.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "test-source"):
                _load_inputs(study)
        rows = [{"source_group_id": group, "query_id": group + str(index)} for group in "abc" for index in range(5)]
        self.assertEqual({row["source_group_id"] for row in _round_robin_cap(rows, 3)}, set("abc"))


if __name__ == "__main__":
    unittest.main()
