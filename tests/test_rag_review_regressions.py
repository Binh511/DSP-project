"""Regression checks for source identity and non-LF line citation fidelity."""
import hashlib
from html import escape
import json
from pathlib import Path
import tempfile
import unittest

import psai
from psai_retrieval import retrieve_document


TEXT = '# Scope\n\nThe API must protect credentials.\n'


class CitationOriginTests(unittest.TestCase):
    def _analyze(self, root: Path, resources: list[tuple[str, str]]):
        proposal = root / 'proposal.md'
        proposal.write_text(TEXT, encoding='utf-8')
        resource_dir = root / 'resources'
        resource_dir.mkdir()
        for name, text in resources:
            (resource_dir / name).write_text(text, encoding='utf-8')
        report = psai.analyze(
            proposal, root / 'result', 'rules', retrieval='bm25',
            queries=['API credentials'], resource_dir=resource_dir, top_k=10,
        )
        payload = json.loads((report.parent / 'retrieval.json').read_text(encoding='utf-8'))
        html = (report.parent / 'report.html').read_text(encoding='utf-8')
        return report.parent, payload, html, resource_dir

    def _assert_origin_snapshot(self, folder, payload, html, hit, expected_path, expected_text):
        self.assertIn('snapshot_id', hit)
        source = payload['sources'][hit['snapshot_id']]
        self.assertEqual(source['role'], 'resource')
        self.assertEqual(Path(source['origin_path']), expected_path.resolve())
        self.assertEqual(source['source_sha256'], hashlib.sha256(expected_text.encode('utf-8')).hexdigest())
        self.assertEqual(source['artifact'], hit['citation_source'])
        self.assertNotEqual(hit['citation_source'], 'source.txt')
        snapshot = (folder / hit['citation_source']).read_text(encoding='utf-8')
        self.assertEqual(snapshot, expected_text)
        self.assertEqual(snapshot[hit['char_start']:hit['char_end']], hit['quote'])
        self.assertIn(escape(str(expected_path.resolve()), quote=True), html)
        self.assertIn(f'id="src-{hit["snapshot_id"]}-L', html)

    def test_identical_resource_and_main_keep_distinct_citation_origins(self):
        with tempfile.TemporaryDirectory(prefix='psai-origin-review-') as directory:
            folder, payload, html, resource_dir = self._analyze(Path(directory), [('mirror.md', TEXT)])
            document_hits = [hit for hit in payload['hits'] if hit['role'] == 'document']
            resources = [hit for hit in payload['hits'] if hit['role'] == 'resource']
            self.assertTrue(document_hits)
            self.assertEqual(len(resources), 1)
            self._assert_origin_snapshot(folder, payload, html, resources[0], resource_dir / 'mirror.md', TEXT)
            self.assertEqual(document_hits[0]['citation_source'], 'source.txt')
            self.assertNotEqual(document_hits[0]['snapshot_id'], resources[0]['snapshot_id'])

    def test_two_identical_resources_keep_both_file_origins(self):
        resource_text = '# Manual\n\nThe API credentials must be stored in environment variables.\n'
        with tempfile.TemporaryDirectory(prefix='psai-origin-review-') as directory:
            folder, payload, html, resource_dir = self._analyze(
                Path(directory), [('first.md', resource_text), ('second.md', resource_text)],
            )
            hits = [hit for hit in payload['hits'] if hit['role'] == 'resource']
            self.assertEqual(len(hits), 2)
            self.assertEqual(len({hit['snapshot_id'] for hit in hits}), 2)
            self.assertEqual(len({hit['citation_source'] for hit in hits}), 2)
            for hit in hits:
                path = Path(hit['source_file'])
                self.assertEqual(path.parent, resource_dir.resolve())
                self._assert_origin_snapshot(folder, payload, html, hit, path, resource_text)


class UnicodeLineCitationTests(unittest.TestCase):
    def test_runtime_lines_match_splitlines_for_non_lf_separators(self):
        for separator in ('\n', '\r\n', '\r', '\u2028', '\u0085'):
            with self.subTest(separator=repr(separator)):
                text = '# Scope' + separator * 2 + 'The API must protect credentials.' + separator
                result = retrieve_document(text, 'D', queries=['API'])
                self.assertEqual(len(result['hits']), 1)
                hit = result['hits'][0]
                self.assertEqual(hit['source_start_line'], 3)
                self.assertEqual(hit['source_end_line'], 3)
                self.assertEqual(text[hit['char_start']:hit['char_end']], hit['quote'])


if __name__ == '__main__':
    unittest.main()
