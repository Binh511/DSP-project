"""Selection completeness and frozen-artifact integrity without model loading."""
from pathlib import Path
import tempfile
import unittest

from tools.finalize_classifier_release import check_bindings, choose_run, safe_file, sha


class ClassifierReleaseTests(unittest.TestCase):
    def results(self):
        return [dict(run=f'{variant}_seed_{seed}', variant=variant, seed=seed, macro_f1=value)
                for variant, values in [('baseline', [.60, .61, .62]), ('context', [.95, .20, .20])]
                for seed, value in zip([123, 42, 2026], values)]

    def test_selection_uses_configuration_mean_before_best_single_run(self):
        chosen, groups = choose_run(self.results())
        self.assertEqual(chosen['run'], 'baseline_seed_2026')
        self.assertEqual(len(groups), 2)

    def test_incomplete_or_duplicated_seed_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            choose_run(self.results()[:-1])
        rows = self.results()
        rows[-1] = dict(rows[-2])
        with self.assertRaises(ValueError):
            choose_run(rows)

    def test_nonfinite_metric_cannot_be_selected(self):
        rows = self.results()
        rows[0]['macro_f1'] = float('nan')
        with self.assertRaises(ValueError):
            choose_run(rows)

    def test_changed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'artifact.json'
            path.write_text('original')
            bindings = {'artifact.json': sha(path)}
            check_bindings(root, bindings)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                check_bindings(root, bindings)

    def test_artifact_path_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                safe_file(Path(directory), '../outside.json')


if __name__ == '__main__':
    unittest.main()
