"""Package integrity tests use tiny fixtures, never production model weights."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import finalize_project_release as release


class ProjectReleaseVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='psai-release-verify-')
        self.root = Path(self.temp.name)
        self.folder = self.root / 'reports/project_completion'
        self.folder.mkdir(parents=True)
        self.source = self.write('source.py', b'print("source fixture")\n')
        self.adapter = 'models/rag/adapter.safetensors'
        self.base = 'models/base/weights.bin'
        self.source_manifest = self.root / 'SOURCE_BUNDLE_MANIFEST.json'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def manifest(self, optional=None):
        data = {'files_sha256': {'source.py': self.source}}
        if optional is not None:
            data['optional_components'] = optional
        self.source_manifest.write_text(json.dumps(data), encoding='utf-8')
        return data

    def source_verify(self):
        with redirect_stdout(io.StringIO()):
            return release.verify_source_bundle(self.source_manifest, root=self.root)

    def historical_manifest(self):
        data = {
            'release_version': 'frozen-fixture',
            'functional_acceptance': 'historical_authored_fixture',
            'files_sha256': {'source.py': self.source},
            'classifier_model_files_sha256': {
                'models/classifier/weights.bin': self.write('models/classifier/weights.bin', b'classifier fixture'),
            },
            'embedding_base': {'files_sha256': {'weights.bin': self.write(self.base, b'base fixture')}},
            'planner_model': {'files_sha256': {
                'weights.bin': self.write('models/planner/weights.bin', b'planner fixture'),
            }},
        }
        (self.folder / 'PROJECT_RELEASE_MANIFEST.json').write_text(json.dumps(data), encoding='utf-8')
        (self.folder / 'local_planner_model.json').write_text(
            json.dumps({'local_path': str(self.root / 'models/planner')}), encoding='utf-8')
        return data

    def test_source_bundle_reports_missing_optional_models_and_adapter_as_partial(self):
        expected = hashlib.sha256(b'optional fixture').hexdigest()
        self.manifest({
            'base_model': {'files_sha256': {self.base: expected}},
            'retrieval_adapter': {'files_sha256': {self.adapter: expected}},
        })
        before = self.source_manifest.read_bytes()
        result = self.source_verify()
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['source_status'], 'verified')
        self.assertEqual(result['files_checked'], 1)
        self.assertEqual(result['missing_optional_components'], ['base_model', 'retrieval_adapter'])
        self.assertEqual(result['optional_components']['base_model']['status'], 'missing_optional')
        self.assertFalse(result['historical_release_verified'])
        self.assertFalse(result['full_runtime_verified'])
        self.assertFalse(result['independent_quality_evaluation_complete'])
        self.assertEqual(self.source_manifest.read_bytes(), before)

    def test_all_present_optional_assets_are_hash_checked_without_claiming_model_execution(self):
        expected = self.write(self.adapter, b'adapter fixture')
        self.manifest({'retrieval_adapter': {'files_sha256': {self.adapter: expected}}})
        result = self.source_verify()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['verification_scope'], 'source_bundle_integrity')
        self.assertEqual(result['optional_components']['retrieval_adapter']['files_checked'], 1)
        self.assertFalse(result['full_runtime_verified'])

    def test_partially_installed_optional_component_is_reported_explicitly(self):
        expected = self.write('models/base/config.json', b'{}')
        self.manifest({'base_model': {'files_sha256': {
            'models/base/config.json': expected, self.base: self.source,
        }}})
        component = self.source_verify()['optional_components']['base_model']
        self.assertEqual(component['status'], 'partial')
        self.assertEqual(component['files_checked'], 1)
        self.assertEqual(component['missing_files'], [self.base])

    def test_missing_or_changed_required_source_fails_and_never_rewrites_manifest(self):
        self.manifest()
        before = self.source_manifest.read_bytes()
        source = self.root / 'source.py'
        source.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'Source bundle integrity mismatch'):
            self.source_verify()
        source.unlink()
        with self.assertRaisesRegex(ValueError, 'Source bundle integrity mismatch'):
            self.source_verify()
        self.assertEqual(self.source_manifest.read_bytes(), before)

    def test_present_optional_hash_mismatch_is_fatal_even_when_other_models_are_missing(self):
        self.write(self.adapter, b'wrong adapter')
        self.manifest({
            'base_model': {'files_sha256': {self.base: self.source}},
            'retrieval_adapter': {'files_sha256': {self.adapter: self.source}},
        })
        with self.assertRaisesRegex(ValueError, 'Optional component retrieval_adapter integrity mismatch'):
            self.source_verify()

    def test_existing_directory_cannot_stand_in_for_a_missing_optional_file(self):
        (self.root / self.adapter).mkdir(parents=True)
        self.manifest({'retrieval_adapter': {'files_sha256': {self.adapter: self.source}}})
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            self.source_verify()

    def test_original_handoff_inventory_format_is_supported(self):
        self.source_manifest.write_text(json.dumps({'files': {'source.py': self.source}}), encoding='utf-8')
        self.assertEqual(self.source_verify()['files_checked'], 1)

    def test_unsafe_paths_bad_hashes_empty_inventory_and_duplicate_members_fail(self):
        for name, expected in (
                ('../outside.py', self.source), ('C:/outside.py', self.source),
                ('folder\\source.py', self.source), ('./source.py', self.source),
                ('folder//source.py', self.source), ('source.py', 'not-a-sha256')):
            with self.subTest(name=name):
                self.source_manifest.write_text(json.dumps({'files_sha256': {name: expected}}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'unsafe path|invalid SHA-256'):
                    self.source_verify()
        self.source_manifest.write_text(json.dumps({'files_sha256': {}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.source_verify()
        self.source_manifest.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'must be an object'):
            self.source_verify()
        self.manifest({'model': {'files_sha256': {'source.py': self.source}}})
        with self.assertRaisesRegex(ValueError, 'multiple components'):
            self.source_verify()

    def test_optional_components_cannot_conflict_with_each_other(self):
        self.manifest({
            'first': {'files_sha256': {self.adapter: self.source}},
            'second': {'files_sha256': {self.adapter: self.source}},
        })
        with self.assertRaisesRegex(ValueError, 'multiple components'):
            self.source_verify()

    def test_historical_source_mismatch_remains_strict_and_build_preserves_frozen_manifest(self):
        self.historical_manifest()
        frozen = self.folder / 'PROJECT_RELEASE_MANIFEST.json'
        before = frozen.read_bytes()
        self.write('source.py', b'newer source')
        with patch.object(release, 'ROOT', self.root), patch.object(release, 'FOLDER', self.folder):
            for operation in (release.verify, release.build):
                with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                    operation()
        self.assertEqual(frozen.read_bytes(), before)

    def test_historical_manifest_does_not_make_listed_adapter_optional(self):
        data = self.historical_manifest()
        data['files_sha256'][self.adapter] = self.source
        (self.folder / 'PROJECT_RELEASE_MANIFEST.json').write_text(json.dumps(data), encoding='utf-8')
        with patch.object(release, 'ROOT', self.root), patch.object(release, 'FOLDER', self.folder):
            with self.assertRaisesRegex(ValueError, 'Project source/artifact integrity mismatch'):
                release.verify()

    def test_full_historical_verification_rejects_each_missing_or_changed_model(self):
        data = self.historical_manifest()
        model_paths = [
            ('models/classifier/weights.bin', 'Classifier model'),
            (self.base, 'Embedding base'),
            ('models/planner/weights.bin', 'Planner model'),
        ]
        with patch.object(release, 'ROOT', self.root), patch.object(release, 'FOLDER', self.folder), \
                patch('psai_retrieval.runtime.DEFAULT_BASE', self.root / 'models/base'), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(release.verify(models=True), data)
            for name, label in model_paths:
                path = self.root / name
                raw = path.read_bytes()
                try:
                    path.unlink()
                    with self.subTest(name=name, state='missing'), self.assertRaisesRegex(ValueError, label):
                        release.verify(models=True)
                    path.write_bytes(b'changed model fixture')
                    with self.subTest(name=name, state='changed'), self.assertRaisesRegex(ValueError, label):
                        release.verify(models=True)
                finally:
                    path.write_bytes(raw)

    def test_source_only_cli_runs_without_site_packages_and_does_not_touch_historical_manifest(self):
        self.historical_manifest()
        frozen = self.folder / 'PROJECT_RELEASE_MANIFEST.json'
        before = frozen.read_bytes()
        self.manifest({'retrieval_adapter': {'files_sha256': {self.adapter: self.source}}})
        script = self.root / 'tools/finalize_project_release.py'
        script.parent.mkdir()
        shutil.copyfile(release.__file__, script)
        result = subprocess.run([sys.executable, '-I', '-S', str(script), '--verify-only', '--source-only'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'partial')
        self.assertEqual(frozen.read_bytes(), before)

    def test_cli_rejects_source_mode_combined_with_full_model_verification(self):
        result = subprocess.run([sys.executable, '-I', '-S', release.__file__,
                                 '--source-only', '--models'], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn('do not combine', result.stderr)


if __name__ == '__main__':
    unittest.main()
