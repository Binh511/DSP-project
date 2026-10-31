"""Meaningful metric, dependence, and human-label integrity checks on CPU."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from psai_rag_quality.analysis import grouped_bootstrap, metrics
from psai_rag_quality.gates import inspect_training, quality_flags
from psai_rag_quality.offline import write_review_html
from psai_rag_quality.review import import_reviews, make_review


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MetricTests(unittest.TestCase):
    def test_multiple_positives_and_missing_top_ten(self):
        result = metrics(["negative", "a", "other", "b"], ["a", "b"])
        self.assertEqual(result["Recall@1"], 0)
        self.assertEqual(result["Recall@5"], 1)
        self.assertEqual(result["MRR@10"], 0.5)
        self.assertEqual(metrics([str(i) for i in range(12)], ["11"])["MRR@10"], 0)
        with self.assertRaises(ValueError):
            metrics(["a", "a"], ["a"])

    def test_cluster_resampling_preserves_queries_within_a_source(self):
        rows = [{"source_group_id": "large", "delta": 1} for _ in range(10)] + [{"source_group_id": "small", "delta": -1}]
        result = grouped_bootstrap(rows, samples=1000)
        self.assertAlmostEqual(result["estimate"], 9 / 11)
        self.assertEqual(result["source_macro_delta"], 0)
        self.assertEqual((result["ci95_low"], result["ci95_high"]), (-1, 1))
        self.assertEqual(result, grouped_bootstrap(rows, samples=1000))
        with self.assertRaises(ValueError):
            grouped_bootstrap(rows[:10], samples=1000)

    def test_train_quality_never_accepts_validation(self):
        with self.assertRaises(ValueError):
            inspect_training([{"split": "validation"}])
        self.assertIn("unicode_replacement_character", quality_flags("what does this source explain", "broken\ufffd text"))


class ReviewTests(unittest.TestCase):
    def fixture(self, root):
        study = root / "study"
        study.mkdir()
        corpus = [{"chunk_id": name, "text": "=untrusted formula" if name == "a" else "A plain source passage",
                   "doc_id": "D", "source_group_id": "G", "split": "validation"} for name in ["a", "b"]]
        query = {"query_id": "q", "query": "what does the source say", "doc_id": "D", "source_group_id": "G",
                 "positive_id": "a", "known_positive_ids": ["a"], "negative_id": "b", "split": "validation"}
        prediction = {"query_id": "q", "scope": "within_doc", "method": "dense", "ranking": [{"chunk_id": "a", "rank": 1}, {"chunk_id": "b", "rank": 2}]}
        jsonl(study / "corpus.jsonl", corpus)
        jsonl(study / "validation_pairs.jsonl", [query])
        for variant in ["base", "adapter"]:
            jsonl(study / (variant + "_predictions.jsonl"), [prediction])
        packet = root / "packet"
        make_review(study, packet)
        return study, packet

    def fill(self, path, reviewer, *, change_text=False, invalid_label=False):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row.update(query_realism="realistic", source_quality="readable", relevance="2" if row["chunk_id"] == "a" else "0", reviewer=reviewer)
        if change_text:
            rows[0]["text"] += " changed"
        if invalid_label:
            rows[0]["relevance"] = "=1+1"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_labels_never_export_and_existing_packet_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, packet = self.fixture(root)
            report = import_reviews(packet, packet / "reviewer_A.csv", packet / "reviewer_B.csv", root / "import")
            self.assertEqual(report["missing_rater_item_judgments"], 4)
            self.assertFalse(report["human_validation_exported"])
            self.assertFalse((root / "import/human_validation_queries.jsonl").exists())
            with self.assertRaises(FileExistsError):
                make_review(study, packet)

    def test_offline_javascript_preserves_escape_sequences_and_untrusted_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, packet = self.fixture(root)
            target = write_review_html(packet)
            html = target.read_text("utf-8")
            self.assertIn(r"/^[=+@\t\r-]/", html)
            self.assertNotIn("\r", html)
            self.assertIn(r".join('\r\n')", html)
            self.assertIn("textContent=r.text", html)
            self.assertNotIn("innerHTML", html)

    def test_complete_independent_forms_export_with_explicit_unknown_unjudged_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, packet = self.fixture(root)
            self.assertIn("'=untrusted formula", (packet / "reviewer_A.csv").read_text("utf-8-sig"))
            self.fill(packet / "reviewer_A.csv", "Alice")
            self.fill(packet / "reviewer_B.csv", "Bob")
            report = import_reviews(packet, packet / "reviewer_A.csv", packet / "reviewer_B.csv", root / "import")
            self.assertTrue(report["human_validation_exported"])
            row = json.loads((root / "import/human_validation_queries.jsonl").read_text("utf-8"))
            self.assertEqual(row["positive_ids"], ["a"])
            self.assertEqual(row["unjudged_policy"], "unknown_not_automatically_irrelevant")
            self.assertEqual(row["split"], "validation")
            with self.assertRaises(FileExistsError):
                import_reviews(packet, packet / "reviewer_A.csv", packet / "reviewer_B.csv", root / "import")

    def test_disagreement_requires_independent_adjudication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, packet = self.fixture(root)
            self.fill(packet / "reviewer_A.csv", "Alice")
            self.fill(packet / "reviewer_B.csv", "Bob")
            with (packet / "reviewer_B.csv").open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                if row["chunk_id"] == "a":
                    row["relevance"] = "1"
            with (packet / "reviewer_B.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            first = import_reviews(packet, packet / "reviewer_A.csv", packet / "reviewer_B.csv", root / "first")
            self.assertFalse(first["human_validation_exported"])
            self.assertEqual(first["unresolved_items"], 1)
            adjudication = root / "first/adjudication_to_complete.csv"
            self.fill(adjudication, "Carol")
            final = import_reviews(packet, packet / "reviewer_A.csv", packet / "reviewer_B.csv", root / "final", adjudication_path=adjudication)
            self.assertTrue(final["human_validation_exported"])

    def test_rejects_edited_evidence_and_expression_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, packet = self.fixture(root)
            self.fill(packet / "reviewer_A.csv", "Alice", change_text=True)
            self.fill(packet / "reviewer_B.csv", "Bob")
            with self.assertRaisesRegex(ValueError, "edited"):
                import_reviews(packet, packet / "reviewer_A.csv", packet / "reviewer_B.csv", root / "import")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, packet = self.fixture(root)
            self.fill(packet / "reviewer_A.csv", "Alice", invalid_label=True)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                import_reviews(packet, packet / "reviewer_A.csv", packet / "reviewer_B.csv", root / "import")


if __name__ == "__main__":
    unittest.main()
