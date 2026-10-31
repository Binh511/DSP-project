"""CPU-only source integrity, scope and ranking tests; fake vectors prove no quality claim."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from psai_retrieval import build_index, load_index, retrieve_document, search_index
from psai_retrieval import runtime


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(char) for char in text], offsets=[(index, index + 1) for index in range(len(text))])

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(value) for value in ids)


def fake_encode(texts, encoder, device):
    import numpy as np
    values = np.zeros((len(texts), 1024), dtype=np.float32)
    for index, text in enumerate(texts):
        values[index, 0 if "alpha" in text.casefold() else 1] = 1.0
    return values, {"device": "cpu", "dtype": "fake", "normalization_dtype": "float32"}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="psai-runtime-test-")
        self.root = Path(self.temp.name)
        self.docs, self.resources = self.root / "docs", self.root / "resources"
        self.docs.mkdir()
        self.resources.mkdir()
        (self.docs / "A.md").write_text("# Delivery\n\nAlpha deployment must record audit events.\n\n## Schedule\n\nBeta release follows testing.\n", encoding="utf-8")
        (self.docs / "B.md").write_text("# Other\n\nGamma monitoring tracks power use.\n", encoding="utf-8")
        (self.resources / "guide.md").write_text("# Manual\n\nAlpha deployment requires a rollback runbook.\n", encoding="utf-8")
        self.output = self.root / "index"

    def tearDown(self):
        self.temp.cleanup()

    def test_bm25_import_and_execution_require_no_ml_packages(self):
        code = (
            "import sys; from psai_retrieval import retrieve_document; "
            "r=retrieve_document('# Scope\\n\\nUsers must log audit events.','D',queries=['audit']); "
            "assert len(r['hits'])==1; "
            "assert not ({'torch','transformers','sentence_transformers','peft','numpy','tokenizers','chromadb'} & set(sys.modules))"
        )
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                       check=True, capture_output=True, text=True)

    def test_exact_source_offsets_unicode_crlf_and_repeated_text(self):
        text = "# Yêu cầu\r\n\r\nPhải ghi nhật ký 😀.\r\n\r\n## Đợt hai\r\n\r\nPhải ghi nhật ký 😀.\r\n"
        result = retrieve_document(text, "VN", queries=["nhật ký", "ghi nhật ký"], top_k=5)
        self.assertEqual(len(result["hits"]), 2)
        for hit in result["hits"]:
            self.assertEqual(hit["quote"], text[hit["char_start"]:hit["char_end"]])
            self.assertEqual(hit["text"], hit["quote"])
            self.assertEqual(hit["quote"], "Phải ghi nhật ký 😀.")
            self.assertEqual(hit["source_sha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertTrue(hit["verified"])
            self.assertFalse(hit["relevance_verified"])
            self.assertEqual(len(hit["query_matches"]), 2)
        self.assertNotEqual(result["hits"][0]["char_start"], result["hits"][1]["char_start"])

    def test_zero_overlap_query_returns_no_arbitrary_evidence(self):
        result = retrieve_document("# Plants\n\nLeaves use sunlight.", "D", queries="quantum blockchain", top_k=3)
        self.assertEqual(result["hits"], [])

    def test_fences_tables_short_content_and_long_windows_preserve_quotes(self):
        text = "# Scope\n\nYes.\n\n```python\n# Not a real heading\nprint('audit')\n```\n\n| Audit | Result |\n|---|---|\n| alpha | beta |\n\n" + ("logging telemetry " * 150)
        source = runtime._source(text, "D")
        chunks = runtime._chunk_sources([source], max_chars=100)
        runtime._validate_chunks(chunks, [source])
        self.assertTrue(any(chunk["quote"] == "Yes." for chunk in chunks))
        self.assertTrue(any(chunk["block_type"] == "code" and "# Not a real heading" in chunk["quote"] for chunk in chunks))
        self.assertTrue(any(chunk["block_type"] == "table_rows" for chunk in chunks))
        self.assertTrue(all(chunk["quote"] == text[chunk["char_start"]:chunk["char_end"]] for chunk in chunks))
        self.assertTrue(all(len(chunk["quote"]) <= 100 for chunk in chunks))

    def test_index_roundtrip_document_scope_and_resource_opt_in(self):
        built = build_index(self.docs, self.output, resource_dir=self.resources)
        self.assertEqual(built["manifest"]["source_count"], 3)
        scoped = search_index(self.output, "alpha", document_ids=["A"], top_k=10)
        self.assertTrue(scoped["hits"])
        self.assertEqual({hit["doc_id"] for hit in scoped["hits"]}, {"A"})
        unrelated = search_index(self.output, "alpha", document_ids=["B"])
        self.assertEqual(unrelated["hits"], [])
        resources = search_index(self.output, "alpha", document_ids=["A"], include_resources=True, top_k=10)
        self.assertEqual({hit["role"] for hit in resources["hits"]}, {"document", "resource"})
        for hit in resources["hits"]:
            exact = Path(hit["source_file"]).read_text(encoding="utf-8-sig")
            self.assertEqual(hit["quote"], exact[hit["char_start"]:hit["char_end"]])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            search_index(self.output, "alpha", document_ids=["missing"])

    def test_changed_live_source_is_rejected_at_load_and_again_at_search(self):
        build_index(self.docs, self.output)
        index = load_index(self.output)
        (self.docs / "A.md").write_text("Changed source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_index(self.output)
        with self.assertRaisesRegex(ValueError, "changed"):
            index.search("alpha")

    def test_corrupt_chunks_manifest_and_wrong_index_are_rejected(self):
        build_index(self.docs, self.output)
        manifest_path = self.output / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["source_count"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            load_index(self.output)
        manifest_path.write_bytes(original)
        with (self.output / "chunks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "artifact SHA256"):
            load_index(self.output)
        with self.assertRaises(FileNotFoundError):
            load_index(self.root / "not-an-index")

    def test_rehashed_invalid_source_spans_still_fail_grounding(self):
        build_index(self.docs, self.output)
        chunks_path = self.output / "chunks.jsonl"
        rows = runtime._jsonl(chunks_path)
        rows[0]["char_start"] += 1
        chunks_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files_sha256"]["chunks.jsonl"] = runtime._file_hash(chunks_path)
        manifest["corpus_sha256"] = runtime._digest(rows)
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = runtime._digest(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "spans"):
            load_index(self.output)

    def test_existing_output_empty_inputs_invalid_queries_and_overlapping_trees(self):
        build_index(self.docs, self.output)
        before = runtime._file_hash(self.output / "manifest.json")
        with self.assertRaises(FileExistsError):
            build_index(self.docs, self.output)
        self.assertEqual(runtime._file_hash(self.output / "manifest.json"), before)
        with self.assertRaises(ValueError):
            build_index(self.docs, self.docs / "nested-index")
        for text, query in ((" ", "query"), ("content", ""), ("content", [])):
            with self.subTest(text=text, query=query), self.assertRaises(ValueError):
                retrieve_document(text, "D", queries=query)
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            build_index(empty, self.root / "unused")

    def test_bm25_index_does_not_silently_use_dense_or_adapter(self):
        build_index(self.docs, self.output)
        with patch.object(runtime, "_encode") as encode, self.assertRaisesRegex(ValueError, "lexical-only"):
            search_index(self.output, "alpha", mode="dense")
        encode.assert_not_called()
        with self.assertRaisesRegex(ValueError, "adapter"):
            retrieve_document("content", "D", queries="content", adapter_dir=self.root)

    @contextlib.contextmanager
    def dense_stack(self):
        from psai_rag import data
        encoder = {"base": {"path": str(self.root / ("a" * 40)), "sha256": "base"},
                   "adapter": None, "encoder_code_sha256": "code", "dimension": 1024}
        with patch.object(runtime, "_encoder_provenance", return_value=encoder) as provenance, \
                patch.object(data, "_load_tokenizer", return_value=CharacterTokenizer()), \
                patch.object(runtime, "_encode", side_effect=fake_encode) as encode:
            yield encoder, provenance, encode

    def test_dense_in_memory_encodes_passages_and_multiple_queries_in_one_load(self):
        with self.dense_stack() as (_, provenance, encode):
            result = retrieve_document("# Requirements\n\nAlpha deployment.\n\nBeta metrics.", "D",
                                       queries=["alpha", "beta"], mode="dense", top_k=2)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(provenance.call_count, 1)
        self.assertEqual(len(result["hits"]), 2)
        self.assertEqual(result["provenance"]["query_fusion"], "RRF_k60")

    def test_dense_heading_context_does_not_become_a_fabricated_source_quote(self):
        text = '# Requirements\n\nAlpha deployment.\n\nBeta metrics.'
        with self.dense_stack() as (_, _, encode):
            result = retrieve_document(text, 'D', queries='alpha', mode='dense', top_k=2)
        encoded_texts = encode.call_args.args[0]
        self.assertTrue(encoded_texts[0].startswith('# Requirements\n\n'))
        for hit in result['hits']:
            self.assertEqual(hit['quote'], text[hit['char_start']:hit['char_end']])
            self.assertNotIn('# Requirements', hit['quote'])
            self.assertEqual(hit['embedding_text'], hit['embedding_prefix'] + hit['quote'])
            self.assertLessEqual(hit['embedding_tokens_without_special_tokens'], 254)

    def test_relocated_adapter_requires_matching_base_file_hash_binding(self):
        adapter = self.root / 'portable_adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps({
            'base_model_name_or_path': 'BAAI/bge-m3', 'peft_type': 'LORA',
            'task_type': 'FEATURE_EXTRACTION'}), encoding='utf-8')
        base = {'path': str(self.root / ('a' * 40)), 'revision': 'a' * 40,
                'files_sha256': {'model.safetensors': 'weights'}, 'sha256': 'base'}
        def model_files(path, **kwargs):
            return {'path': str(adapter), 'sha256': 'adapter'} if kwargs.get('adapter') else base
        with patch.object(runtime, '_model_files', side_effect=model_files):
            with self.assertRaisesRegex(ValueError, 'relocated'):
                runtime._encoder_provenance(Path(base['path']), adapter)
            binding = {'model_id': 'BAAI/bge-m3', 'revision': 'a' * 40,
                       'files_sha256': {'model.safetensors': 'weights'}}
            (adapter / 'base_provenance.json').write_text(json.dumps(binding), encoding='utf-8')
            self.assertIsNotNone(runtime._encoder_provenance(Path(base['path']), adapter)['adapter'])
            binding['files_sha256']['model.safetensors'] = 'different'
            (adapter / 'base_provenance.json').write_text(json.dumps(binding), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hashes'):
                runtime._encoder_provenance(Path(base['path']), adapter)

    def test_long_heading_and_body_stay_inside_native_dense_budget(self):
        text = '# ' + ('Long heading ' * 25) + '\n\n' + ('Alpha deployment details. ' * 100)
        tokenizer = CharacterTokenizer()
        chunks = runtime._chunk_sources([runtime._source(text, 'D')], tokenizer=tokenizer)
        self.assertGreater(len(chunks), 1)
        for row in chunks:
            self.assertLessEqual(len(tokenizer.encode(row['embedding_text']).ids), 254)
            self.assertEqual(text[row['char_start']:row['char_end']], row['quote'])

    def test_dense_manifest_model_alignment_and_vector_shape(self):
        with self.dense_stack() as (encoder, provenance, _):
            build_index(self.docs, self.output, mode="dense")
            result = search_index(self.output, "alpha", mode="hybrid", document_ids=["A"])
            self.assertTrue(result["hits"])
            changed = copy.deepcopy(encoder)
            changed["base"]["sha256"] = "different-model-or-tokenizer"
            provenance.return_value = changed
            with self.assertRaisesRegex(ValueError, "does not match"):
                search_index(self.output, "alpha", mode="dense")
        import numpy as np
        for values in (np.zeros((1, 1024)), np.ones((1, 2)), np.full((1, 1024), np.nan)):
            with self.assertRaises(ValueError):
                runtime._validate_vectors(values, 1)

    def test_optional_reranking_happens_after_dense_encoder_returns(self):
        events = []
        def encode(*args):
            events.append("dense_model_closed")
            return fake_encode(*args)
        def rerank(pairs, model_dir):
            self.assertEqual(events, ["dense_model_closed"])
            return list(range(len(pairs))), {"device": "cpu", "score_kind": "fake"}
        with self.dense_stack(), patch.object(runtime, "_encode", side_effect=encode), \
                patch.object(runtime, "_rerank", side_effect=rerank):
            result = retrieve_document("# Delivery\n\nAlpha deployment.\n\nBeta metrics.", "D", queries="alpha",
                                       mode="dense", rerank_model_dir=self.root, top_k=2)
        self.assertEqual(result["provenance"]["reranker"]["device"], "cpu")
        self.assertTrue(all(hit["query_matches"][0]["rerank_score"] is not None for hit in result["hits"]))


if __name__ == "__main__":
    unittest.main()
