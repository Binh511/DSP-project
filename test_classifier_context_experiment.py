"""Checks for the new prepared-data experiment contract; no model loading."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from classifier_context_experiment import validated_rows
from study_data import sha, write_json
from train_human_cv import COLS


class ContextExperimentTests(unittest.TestCase):
    def fixture(self, root):
        for split, doc in [('training', 'TRAIN'), ('validation', 'VAL')]:
            rows = pd.DataFrame([[doc, 'Implement it.', 'ACTIVITY', 'Old', '', .5, '']], columns=COLS)
            rows.to_csv(root / f'baseline_{split}_rows.csv', index=False)
            rows['section'] = 'New section'
            rows.to_csv(root / f'context_{split}_rows.csv', index=False)
        self.manifest(root)

    def manifest(self, root, overlap=0):
        write_json(root / 'preparation_manifest.json', dict(train_validation_source_overlap_after_purge=overlap,
            evidence_files={path.name: sha(path) for path in root.glob('*.csv')}))

    def test_both_variants_preserve_pairing_and_only_context_section_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            baseline = validated_rows('baseline', root)
            context = validated_rows('context', root)
            for old, new in zip(baseline, context):
                pd.testing.assert_frame_equal(old.drop(columns='section'), new.drop(columns='section'))
                self.assertNotEqual(old.section.tolist(), new.section.tolist())

    def test_changed_prepared_file_is_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / 'context_training_rows.csv'
            path.write_text(path.read_text() + '\n')
            with self.assertRaises(ValueError):
                validated_rows('context', root)

    def test_nonsection_change_and_source_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / 'context_training_rows.csv'
            rows = pd.read_csv(path, keep_default_na=False)
            rows['label'] = 'OTHER'
            rows.to_csv(path, index=False)
            self.manifest(root)
            with self.assertRaises(AssertionError):
                validated_rows('context', root)
            self.manifest(root, overlap=1)
            with self.assertRaises(ValueError):
                validated_rows('baseline', root)


if __name__ == '__main__':
    unittest.main()
