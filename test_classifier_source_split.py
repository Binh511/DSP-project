"""Source split regression tests with no model/dependency initialization."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from classifier_source_split import COLS, prepare, read_rows, normalized_source_hash


class SourceSplitTests(unittest.TestCase):
    def test_full_source_normalization_handles_unicode_equivalence(self):
        self.assertEqual(normalized_source_hash('# TITLE\nCaf\u00e9  project'),
                         normalized_source_hash('# title\nCafe\u0301 project'))

    def test_original_and_context_inputs_are_bound_to_preparation_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, _, _, _ = self.fixture(root)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            metadata = {'splits': {'train': {'original_input_file_sha256': digest(paths[0])},
                                   'validation': {'original_input_file_sha256': digest(paths[1])}},
                        'evidence_files': {'training_rows.csv': digest(paths[2]),
                                           'validation_rows.csv': digest(paths[3])}}
            meta_path = root / 'preparation.json'
            meta_path.write_text(json.dumps(metadata), encoding='utf-8')
            for path in [paths[0], paths[2]]:
                rows = read_rows(path)
                rows['label'] = 'OTHER'
                rows.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                prepare(*paths, root / 'cleaned', root / 'version', expected_preparation_manifest=meta_path)
            self.assertFalse((root / 'version').exists())

    def fixture(self, root):
        training = pd.DataFrame([
            ['RAW', 'train raw labeled sentence', 'REQUIREMENT', '', 'previous', '0.11', 'next'],
            ['KEEP', 'train distinct sentence', 'ACTIVITY', 'old Section', '', '0.22', ''],
            ['NORMALIZED', 'another train sentence', 'OTHER', '', '', '0.33', ''],
        ], columns=COLS)
        validation = pd.DataFrame([
            ['VAL', 'validation second sentence', 'EXPECTED_RESULT', 'old val', 'before', '0.50', 'after'],
            ['VAL', 'validation first sentence', 'OTHER', '', '', '1.0', ''],
        ], columns=COLS)
        context_training, context_validation = training.copy(), validation.copy()
        context_training['section'] = ['new source Section', 'new distinct Section', 'new source Section']
        context_validation['section'] = ['new validation Section', 'new validation Section']
        paths = []
        for name, rows in [('base_train', training), ('base_val', validation), ('ctx_train', context_training), ('ctx_val', context_validation)]:
            path = root / f'{name}.csv'
            rows.to_csv(path, index=False, encoding='utf-8-sig')
            paths.append(path)
        texts = {'RAW': '# Source\nA full source document.\n',
                 'NORMALIZED': '# source\n\nA   FULL source DOCUMENT.\n',
                 'VAL': '# Source\nA full source document.\n',
                 'KEEP': '# Completely different document\nDistinct source content.'}
        for document, text in texts.items():
            folder = root / 'cleaned' / document
            folder.mkdir(parents=True)
            (folder / 'main.md').write_text(text, encoding='utf-8')
        return paths, training, validation, context_validation

    def test_raw_and_normalized_source_duplicates_purged_despite_no_labeled_sentence_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, training, validation, context_validation = self.fixture(root)
            output = root / 'version'
            manifest = prepare(*paths, root / 'cleaned', output)
            self.assertEqual(manifest['removed_train_documents'], ['NORMALIZED', 'RAW'])
            self.assertEqual(manifest['removed_train_rows'], 2)
            self.assertEqual(manifest['train_rows'], 1)
            self.assertEqual(manifest['train_validation_source_overlap_after_purge'], 0)
            pd.testing.assert_frame_equal(read_rows(output / 'baseline_validation_rows.csv'), validation)
            pd.testing.assert_frame_equal(read_rows(output / 'context_validation_rows.csv'), context_validation)
            pd.testing.assert_frame_equal(read_rows(output / 'baseline_training_rows.csv'), training.iloc[[1]].reset_index(drop=True))
            self.assertEqual(manifest['class_counts']['removed_training'], {'OTHER': 1, 'REQUIREMENT': 1})
            for name, value in manifest['evidence_files'].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), value)
            with self.assertRaises(FileExistsError):
                prepare(*paths, root / 'cleaned', output)

    def test_context_cannot_change_labels_order_or_other_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _, _, _ = self.fixture(root)
            rows = read_rows(paths[2])
            rows.at[0, 'label'] = 'OBJECTIVE'
            rows.to_csv(paths[2], index=False)
            with self.assertRaises(AssertionError):
                prepare(*paths, root / 'cleaned', root / 'version')
            self.assertFalse((root / 'version').exists())

    def test_missing_and_multiple_source_files_fail_before_output_creation(self):
        for mode in ['missing', 'multiple']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths, _, _, _ = self.fixture(root)
                if mode == 'missing':
                    (root / 'cleaned/KEEP/main.md').unlink()
                else:
                    (root / 'cleaned/KEEP/second.txt').write_text('another source', encoding='utf-8')
                with self.assertRaises(ValueError):
                    prepare(*paths, root / 'cleaned', root / 'version')
                self.assertFalse((root / 'version').exists())

    def test_changed_source_against_context_manifest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _, _, _ = self.fixture(root)
            source_manifest = []
            for folder in sorted((root / 'cleaned').iterdir()):
                path = folder / 'main.md'
                source_manifest.append({'document_id': folder.name, 'sources': [{'path': str(path.resolve()),
                                                                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]})
            context_manifest = root / 'sources.json'
            context_manifest.write_text(json.dumps(source_manifest), encoding='utf-8')
            (root / 'cleaned/KEEP/main.md').write_text('Changed after context preparation', encoding='utf-8')
            with self.assertRaises(ValueError):
                prepare(*paths, root / 'cleaned', root / 'version', context_manifest)


if __name__ == '__main__':
    unittest.main()
