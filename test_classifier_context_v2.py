"""Synthetic regression checks for Section-only context preparation."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from classifier_context_v2 import REQUIRED, parse_source, prepare, revise_rows, resolve_section


def source(text, path='source.md'):
    return [{'path': path, 'parsed': parse_source(text)}], [{'path': path, 'status': 'readable_utf8'}]


class ContextRevisionTests(unittest.TestCase):
    def test_hashtag_shebang_and_indented_code_are_not_headings(self):
        sources, evidence = source('# Title\n#tag\n#define VAR 1\n#!/bin/bash\n    # indented code\nbody sentence')
        self.assertEqual(resolve_section(sources, evidence, 'body sentence')['candidate_section'], 'Title')

    def test_backtick_and_tilde_fences_require_matching_closers(self):
        for opener, bad_closers, closer in [('````python', ['~~~', '```', '```` not a closer'], '````'),
                                              ('~~~~', ['```', '~~~', '~~~~ still code'], '~~~~~')]:
            with self.subTest(opener=opener):
                text = '# Title\n' + opener + '\n# code heading\n' + '\n'.join(bad_closers) + '\n# still code\n' + closer + '\n#### Real\nbody sentence'
                sources, evidence = source(text)
                self.assertEqual(resolve_section(sources, evidence, 'body sentence')['candidate_section'], 'Title > Real')
                self.assertEqual(resolve_section(sources, evidence, '# still code')['candidate_section'], 'Title')

    def test_heading_level_jump_replaces_same_level_heading(self):
        sources, evidence = source('# Title\n#### First\nfirst sentence\n#### Second\nsecond sentence')
        result = resolve_section(sources, evidence, 'second sentence')
        self.assertTrue(result['resolved'])
        self.assertEqual(result['candidate_section'], 'Title > Second')

    def test_repetition_same_section_is_supported_but_different_section_unresolved(self):
        sources, evidence = source('# A\nrepeat text\nrepeat text')
        self.assertTrue(resolve_section(sources, evidence, 'repeat text')['resolved'])
        sources, evidence = source('# A\nrepeat text\n# B\nrepeat text')
        result = resolve_section(sources, evidence, 'repeat text')
        self.assertFalse(result['resolved'])
        self.assertEqual(result['status'], 'multiple_section_paths_needs_review')

    def test_multiple_matching_files_and_preheading_are_unresolved(self):
        sources_a, evidence_a = source('# A\nrepeat text', 'one.md')
        sources_b, evidence_b = source('# A\nrepeat text', 'two.md')
        result = resolve_section(sources_a + sources_b, evidence_a + evidence_b, 'repeat text')
        self.assertEqual(result['status'], 'multiple_matching_sources_needs_review')
        sources, evidence = source('before heading\n# Heading\nbody')
        self.assertEqual(resolve_section(sources, evidence, 'before heading')['status'], 'before_first_heading_needs_review')

    def test_query_crossing_distinct_heading_paths_unresolved(self):
        sources, evidence = source('# A\nbody one\n# B\nbody two')
        result = resolve_section(sources, evidence, 'body one # B body two')
        self.assertFalse(result['resolved'])

    def test_missing_source_retains_old_section_and_preserves_nonsection_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'DOC1'
            folder.mkdir()
            (folder / 'main.md').write_text('# Title\n#### First\nfirst\n#### Second\nnew sentence', encoding='utf-8')
            rows = pd.DataFrame([
                ['DOC1', 'new sentence', 'REQUIREMENT', 'bad old path', 'previous', '0.123400', 'next'],
                ['MISSING', 'no source sentence', 'OTHER', 'retain this', '', '1.0', ''],
            ], columns=REQUIRED)
            revised, audit, _ = revise_rows(rows, root, 'train')
            pd.testing.assert_frame_equal(rows.drop(columns='section'), revised.drop(columns='section'))
            self.assertEqual(revised.section.tolist(), ['Title > Second', 'retain this'])
            self.assertEqual(audit.status.tolist(), ['resolved_unique_heading_path', 'missing_source'])
            self.assertEqual(rows.section.iloc[0], 'bad old path')

    def test_preparation_preserves_rows_records_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = pd.DataFrame([['TRAIN', 'a sentence', 'ACTIVITY', '', '', '0.0000', '']], columns=REQUIRED)
            validation = pd.DataFrame([['VAL', 'a sentence', 'EXPECTED_RESULT', '', '', '1.0000', '']], columns=REQUIRED)
            train_path, validation_path = root / 'train.csv', root / 'validation.csv'
            rows.to_csv(train_path, index=False)
            validation.to_csv(validation_path, index=False)
            for document in ['TRAIN', 'VAL']:
                folder = root / 'cleaned' / document
                folder.mkdir(parents=True)
                (folder / 'main.md').write_text('# Heading\na sentence', encoding='utf-8')
            output = root / 'version'
            manifest = prepare(train_path, validation_path, root / 'cleaned', output)
            self.assertEqual(manifest['status'], 'preparation_only_not_train_evaluated')
            self.assertFalse(manifest['training_or_inference_performed'])
            self.assertNotEqual(manifest['splits']['train']['original_ordered_row_sha256'], manifest['splits']['train']['revised_ordered_row_sha256'])
            saved = pd.read_csv(output / 'training_rows.csv', keep_default_na=False, dtype=str)
            pd.testing.assert_frame_equal(saved.drop(columns='section'), rows.drop(columns='section'))
            source_manifest = json.loads((output / 'source_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(source_manifest), 2)
            self.assertEqual(len(source_manifest[0]['sources'][0]['sha256']), 64)
            with self.assertRaises(FileExistsError):
                prepare(train_path, validation_path, root / 'cleaned', output)


if __name__ == '__main__':
    unittest.main()
