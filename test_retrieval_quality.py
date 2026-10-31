"""CPU-only integrity checks; no real embedding model, database, or corpus writes."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import chunking
import cleaning
import embedding
import vector_db


class CharacterEncoder:
    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def get_sentence_embedding_dimension(self):
        return 2


class FakeCollection:
    def __init__(self, rows=None, metadata=None, fail_upsert=None):
        self.rows = dict(rows or {})
        self.metadata = metadata or {
            "hnsw:space": "cosine", "embedding_model": embedding.MODEL_NAME,
            "embedding_normalize": embedding.NORMALIZE, "embedding_dimension": 2,
        }
        self.fail_upsert = fail_upsert
        self.upserts = 0
        self.queries = []

    def count(self):
        return len(self.rows)

    def get(self, where, include):
        return {"ids": [key for key, value in self.rows.items() if value["doc_id"] == where["doc_id"]]}

    def upsert(self, ids, embeddings, metadatas, documents):
        self.upserts += 1
        if self.upserts == self.fail_upsert:
            raise RuntimeError("simulated storage failure")
        for chunk_id, metadata, text in zip(ids, metadatas, documents):
            self.rows[chunk_id] = dict(metadata, text=text)

    def delete(self, ids):
        for chunk_id in ids:
            del self.rows[chunk_id]

    def query(self, **kwargs):
        self.queries.append(kwargs)
        matches = [key for key, value in self.rows.items() if value["doc_id"] == kwargs["where"]["doc_id"]][:kwargs["n_results"]]
        return {"ids": [matches], "documents": [[self.rows[key].get("text", "text") for key in matches]],
                "metadatas": [[self.rows[key] for key in matches]], "distances": [[0.1] * len(matches)]}


class FakeClient:
    def __init__(self, collection):
        self.collection = collection
        self.closed = False

    def get_collection(self, name):
        return self.collection

    def get_or_create_collection(self, name, metadata):
        return self.collection

    def close(self):
        self.closed = True


class RetrievalQualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="psai-retrieval-test-")
        self.root = Path(self.tmp.name)
        self.encoder_patch = patch.object(chunking, "_ENCODER", CharacterEncoder())
        self.encoder_patch.start()
        self.output_capture = contextlib.redirect_stdout(io.StringIO())
        self.output_capture.__enter__()

    def tearDown(self):
        self.output_capture.__exit__(None, None, None)
        self.encoder_patch.stop()
        self.tmp.cleanup()

    def source(self, filename, text):
        source = self.root / "input" / "D1" / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(text, encoding="utf-8")
        return source

    def embedded_fixture(self, count=1):
        chunks = [{"chunk_id": f"D1-C{index:04d}", "doc_id": "D1", "chunk_index": index,
                   "text": f"Requirement {index}", "section": "Requirements", "heading_path": ["Requirements"],
                   "source_start_line": index + 2} for index in range(count)]
        source = self.source("D1.jsonl", "\n".join(json.dumps(item) for item in chunks))
        target = self.root / "embedded"
        with patch.object(embedding, "DEVICE", "cpu"):
            embedding.process_document(source, FakeEncoder(), output_dir=target)
        return source, target

    def test_imports_and_help_do_not_load_gpu_or_optional_retrieval_packages(self):
        command = "import sys,cleaning,chunking,embedding,vector_db; assert not ({'torch','tiktoken','chromadb','sentence_transformers'} & set(sys.modules))"
        subprocess.run([sys.executable, "-c", command], cwd=Path(__file__).parent, check=True, capture_output=True)

    def test_cleaning_refuses_overwrite_and_records_source_hash(self):
        source = self.source("D1.md", "# Requirements\nUsers must sign in.\n")
        target = self.root / "clean"
        cleaning.process_document(source, output_dir=target)
        output = target / "D1" / "D1.md"
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            cleaning.process_document(source, output_dir=target)
        self.assertEqual(output.read_bytes(), before)
        manifest = json.loads((target / "D1" / "D1.provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_invalid_utf8_is_rejected_without_replacement_text(self):
        source = self.source("D1.md", "text")
        source.write_bytes(b"\xff")
        for processor in (cleaning.process_document, chunking.process_document):
            with self.subTest(processor=processor.__module__), self.assertRaises(UnicodeDecodeError):
                processor(source, output_dir=self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_empty_input_never_removes_prior_outputs(self):
        empty = self.root / "empty"
        empty.mkdir()
        target = self.root / "out"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        for module in (cleaning, chunking, embedding):
            with self.subTest(module=module.__name__), self.assertRaises(ValueError):
                module.main(["--input-dir", str(empty), "--output-dir", str(target)])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_short_content_and_unclosed_fence_are_retained(self):
        blocks = chunking.parse_blocks("# Scope\nMust log in.\n~~~python\nprint('saved')")
        self.assertEqual(len(blocks), 2)
        chunks = chunking.build_chunks("D1", blocks)
        self.assertTrue(any("Must log in." in chunk["text"] for chunk in chunks))
        code = next(chunk for chunk in chunks if chunk["block_type"] == "diagram")
        self.assertIn("print('saved')", code["text"])
        self.assertTrue(code["unterminated_fence"])
        self.assertEqual(code["source_start_line"], 3)
        self.assertEqual(code["source_end_line"], 4)

    def test_nested_fence_and_headings_keep_source_context(self):
        text = "# Scope\n````python\n```\n# code, not a heading\n````\n## Requirements\nMust work."
        blocks = chunking.parse_blocks(text)
        self.assertEqual(blocks[-1]["heading_path"], ["Scope", "Requirements"])
        self.assertIn("# code, not a heading", blocks[0]["content"])

    def test_unicode_splitting_preserves_source_characters_and_budget(self):
        text = "Yêu cầu tiếng Việt 😀. " * 15
        pieces = list(chunking.split_text_by_tokens(text, 48, 7, prefix="# Scope\n\n"))
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(chunking.count_tokens(piece), 48)
            self.assertNotIn("\ufffd", piece)
            self.assertIn(piece.removeprefix("# Scope\n\n"), text)
        with self.assertRaises(ValueError):
            list(chunking.split_text_by_tokens("content", 5, 1, prefix="too long heading"))

    def test_long_diagram_and_table_respect_budget_without_dropping_extra_columns(self):
        text = "# Scope\n```\n" + ("abc " * 30) + "\n```\n| Key | Value |\n|---|---|\n| A | B | EXTRA |"
        with patch.object(chunking, "MAX_TOKENS", 48), patch.object(chunking, "OVERLAP_TOKENS", 4):
            chunks = chunking.build_chunks("D1", chunking.parse_blocks(text))
        self.assertTrue(all(chunk["token_count"] <= 48 for chunk in chunks))
        self.assertTrue(any("EXTRA" in chunk["text"] for chunk in chunks))
        self.assertTrue(all("source_start_line" in chunk for chunk in chunks))

    def test_chunk_process_provenance_and_no_default_overwrite(self):
        source = self.source("D1.md", "# Scope\nMust work.")
        target = self.root / "chunks"
        chunking.process_document(source, output_dir=target)
        chunks = embedding.load_chunks(target / "D1" / "D1.jsonl")
        self.assertEqual(chunks[0]["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(chunks[0]["source_start_line"], 2)
        with self.assertRaises(FileExistsError):
            chunking.process_document(source, output_dir=target)

    def test_embedding_integrity_empty_input_and_no_default_overwrite(self):
        source, target = self.embedded_fixture()
        loaded = list(vector_db.load_embeddings(target))
        self.assertEqual(loaded[0][1].shape, (1, 2))
        self.assertEqual(loaded[0][2][0]["source_start_line"], 2)
        with self.assertRaises(FileExistsError):
            embedding.process_document(source, FakeEncoder(), output_dir=target)
        self.assertEqual(embedding.embed_chunks(FakeEncoder(), []).shape, (0, 2))

    def test_embedding_rejects_duplicate_ids_and_invalid_vectors(self):
        source = self.source("D1.jsonl", '\n'.join([json.dumps({"chunk_id": "same", "doc_id": "D1", "chunk_index": 0, "text": "content"})] * 2))
        with self.assertRaises(ValueError):
            embedding.load_chunks(source)
        for values in ([[float("nan"), 0]], [[0.0, 0.0]], [[2.0, 0.0]]):
            model = FakeEncoder()
            model.encode = lambda *args, result=values, **kwargs: np.asarray(result)
            with self.subTest(values=values), patch.object(embedding, "DEVICE", "cpu"), self.assertRaises(ValueError):
                embedding.embed_chunks(model, ["content"])

    def test_index_inputs_fail_before_database_open(self):
        _, target = self.embedded_fixture()
        metadata = target / "D1" / "D1.meta.json"
        records = json.loads(metadata.read_text(encoding="utf-8"))
        records[0]["text"] = "tampered text"
        metadata.write_text(json.dumps(records), encoding="utf-8")
        with patch.object(vector_db, "_open_client") as open_client, self.assertRaises(ValueError):
            vector_db.build_collection(embed_dir=target)
        open_client.assert_not_called()

    def test_failed_replacement_does_not_delete_old_document_chunks(self):
        _, target = self.embedded_fixture(count=2)
        collection = FakeCollection({"old-id": {"doc_id": "D1", "text": "old content"}}, fail_upsert=2)
        with patch.object(vector_db, "_open_client", return_value=FakeClient(collection)), patch.object(vector_db, "BATCH_ADD", 1), self.assertRaises(RuntimeError):
            vector_db.build_collection(embed_dir=target)
        self.assertIn("old-id", collection.rows)

    def test_successful_replacement_removes_only_obsolete_ids(self):
        _, target = self.embedded_fixture()
        collection = FakeCollection({"old-id": {"doc_id": "D1"}, "other-id": {"doc_id": "D2"}})
        with patch.object(vector_db, "_open_client", return_value=FakeClient(collection)):
            vector_db.build_collection(embed_dir=target)
        self.assertEqual(set(collection.rows), {"D1-C0000", "other-id"})

    def test_retrieval_is_document_scoped_and_closes_client(self):
        collection = FakeCollection({"one": {"doc_id": "D1"}, "two": {"doc_id": "D2"}, "three": {"doc_id": "D2"}})
        client = FakeClient(collection)
        with patch.object(vector_db, "_open_client", return_value=client), patch.object(vector_db, "get_model", return_value=FakeEncoder()):
            results = vector_db.retrieve_context_batch([("D1", "query"), ("missing", "query")], top_k=5)
        self.assertEqual([item["chunk_id"] for item in results["D1"]], ["one"])
        self.assertEqual(results["missing"], [])
        self.assertEqual(collection.queries[0]["n_results"], 1)
        self.assertTrue(client.closed)
        with self.assertRaises(ValueError):
            vector_db.retrieve_context_batch([("D1", "one"), ("D1", "two")])

    def test_retrieval_rejects_wrong_model_before_loading_encoder(self):
        collection = FakeCollection({"one": {"doc_id": "D1"}})
        collection.metadata["embedding_model"] = "different-model"
        client = FakeClient(collection)
        with patch.object(vector_db, "_open_client", return_value=client), patch.object(vector_db, "get_model") as get_model, self.assertRaises(ValueError):
            vector_db.retrieve_context_batch([("D1", "query")])
        get_model.assert_not_called()
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
