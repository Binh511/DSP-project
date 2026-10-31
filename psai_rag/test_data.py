"""Small CPU fixtures for source leakage and token-budget invariants."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from psai_rag.data import _normalize, _sha256, _source_groups, prepare_dataset


class _FixtureTokenizer:
    def __init__(self) -> None:
        self.words = []

    def encode(self, text, add_special_tokens=False):
        matches = list(re.finditer(r"\S+", text))
        ids = []
        for match in matches:
            self.words.append(match.group())
            ids.append(len(self.words) - 1)
        return SimpleNamespace(ids=ids, offsets=[match.span() for match in matches])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[index] for index in ids)


class PreparationTests(unittest.TestCase):
    def test_near_duplicate_groups_include_aliases_before_split(self):
        original = " ".join(f"term{index}" for index in range(180))
        modified = original.replace("term100", "changed100")
        unrelated = " ".join(f"different{index}" for index in range(180))
        texts = [original, original.upper(), modified, unrelated]
        sources = [{"doc_id": f"D{index}", "normalized_sha256": _sha256(_normalize(text).encode()), "_text": text} for index, text in enumerate(texts)]
        groups, pairs = _source_groups(sources)
        self.assertEqual([group["alias_doc_ids"] for group in groups], [["D0", "D1", "D2"], ["D3"]])
        self.assertEqual(len(pairs), 1)
        self.assertGreaterEqual(pairs[0]["jaccard"], 0.85)

    def test_preparation_is_partitioned_bounded_and_excludes_shared_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path = root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            common = " ".join(["This shared passage describes ordinary planning and resource coordination for a project."] * 4)
            for index in range(8):
                folder = root / "data_cleaned" / f"D{index}"
                folder.mkdir(parents=True)
                # A common paragraph is intentionally too small a fraction of
                # these different sources to trigger source-level grouping.
                unique = "\n\n".join(" ".join(f"document{index}section{section}word{word}" for word in range(80)) for section in range(4))
                (folder / f"D{index}.md").write_text("# Shared heading\n\n" + common + "\n\n" + unique, encoding="utf-8")
            alias = root / "data_cleaned" / "D8"
            alias.mkdir()
            (alias / "D8.md").write_bytes((root / "data_cleaned/D0/D0.md").read_bytes())
            study = root / "reports/study"
            with patch("psai_rag.data._load_tokenizer", side_effect=lambda _: _FixtureTokenizer()):
                result = prepare_dataset(root, study, tokenizer_path, max_tokens=64)
                first_bytes = (study / "corpus.jsonl").read_bytes()
                second = prepare_dataset(root, study, tokenizer_path, max_tokens=64)
            self.assertEqual(first_bytes, (study / "corpus.jsonl").read_bytes())
            self.assertEqual(result["split_groups"], second["split_groups"])
            manifest = json.loads((study / "split_manifest.json").read_text("utf-8"))
            corpus = [json.loads(line) for line in (study / "corpus.jsonl").read_text("utf-8").splitlines()]
            candidates = [json.loads(line) for line in (study / "candidate_passages.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(manifest["counts"]["source_documents"], 9)
            self.assertEqual(manifest["counts"]["source_groups"], 8)
            self.assertGreater(manifest["counts"]["shared_text_chunks_excluded"], 0)
            self.assertTrue(all(row["token_count"] <= 64 for row in corpus))
            self.assertTrue(all(32 <= row["token_count"] <= 64 for row in candidates))
            self.assertTrue(all(row["body_token_count"] >= 32 for row in candidates))
            self.assertTrue(all(row["split"] != "test" for row in candidates))
            self.assertEqual(len(corpus), len({row["normalized_text_sha256"] for row in corpus}))
            sets = [set(value) for value in manifest["split_groups"].values()]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertTrue(all(row["source_start_line"] <= row["source_end_line"] for row in corpus))
            self.assertTrue(any(row["alias_doc_ids"] == ["D0", "D8"] for row in corpus))
            self.assertTrue(all(row["source_span_kind"].startswith("enclosing_block") for row in corpus))

    def test_refuses_output_inside_existing_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "isolated"):
                prepare_dataset(root, root / "models/new", tokenizer)
            self.assertFalse((root / "models").exists())


if __name__ == "__main__":
    unittest.main()
