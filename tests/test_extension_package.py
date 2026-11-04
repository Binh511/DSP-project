"""Release-level checks for complete, reproducible and immutable handoff files."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import package_extension as release


class ExtensionPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.extension = self.root / 'extension/psai'
        self.extension.mkdir(parents=True)
        self.manifest = {
            'manifest_version': 3,
            'version': release.VERSION,
            'permissions': ['storage'],
            'host_permissions': ['http://127.0.0.1/*'],
            'background': {'service_worker': 'service_worker.js'},
        }
        self.write_manifest()
        self.write('app.html', '<link rel="stylesheet" href="app.css"><script type="module" src="app.mjs"></script>')
        self.write('app.css', 'body { margin: 0; }\n')
        self.write('app.mjs', 'import { value } from "./client.mjs";\n')
        self.write('client.mjs', 'export const value = 1;\n')
        self.write('service_worker.js', '// Toolbar entry point.\n')
        self.write('tests/excluded.test.mjs', 'throw new Error("not a shipped asset");\n')
        package = self.root / 'psai_extension'
        package.mkdir()
        (package / '__init__.py').write_text('from .settings import PORT\n', encoding='utf-8')
        (package / 'settings.py').write_text('PORT = 8765\n', encoding='utf-8')
        (self.root / 'README_EXTENSION.md').write_text('Local source handoff.\n', encoding='utf-8')
        self.output = self.root / 'release'
        for name, value in (
                ('ROOT', self.root), ('PACKAGES', ('psai_extension',)),
                ('ROOT_FILES', ('README_EXTENSION.md',))):
            replacement = patch.object(release, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def write(self, name, content):
        target = self.extension / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')

    def write_manifest(self):
        self.write('manifest.json', json.dumps(self.manifest))

    def test_archives_and_unpacked_folder_are_complete_and_reproducible(self):
        receipt = release.package(self.output)
        snapshots = {path.name: path.read_bytes() for path in self.output.glob('*.zip')}
        self.assertEqual(receipt, release.package(self.output))
        for name, data in snapshots.items():
            self.assertEqual(data, (self.output / name).read_bytes())
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any('tests/' in item for item in archive.namelist()))
                if 'CHROME_EDGE' in name:
                    self.assertIn('manifest.json', archive.namelist())
                    self.assertIn('client.mjs', archive.namelist())
                    unpacked = self.output / receipt['unpacked_extension']
                    for item in archive.namelist():
                        self.assertEqual(archive.read(item), (unpacked / item).read_bytes())

    def test_missing_frontend_module_is_rejected_before_publication(self):
        self.write('app.mjs', 'import "./missing.mjs";\n')
        with self.assertRaisesRegex(ValueError, 'Missing extension resource'):
            release.package(self.output)
        self.assertFalse(self.output.exists())

    def test_missing_backend_relative_import_is_rejected(self):
        (self.root / 'psai_extension/__init__.py').write_text('from .missing import PORT\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unbundled local imports'):
            release.package(self.output)

    def test_manifest_and_backend_versions_must_agree(self):
        self.manifest['version'] = '99.0.0'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Manifest V3'):
            release.package(self.output)

    def test_release_keeps_loopback_permissions(self):
        self.manifest['host_permissions'] = ['<all_urls>']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'loopback'):
            release.package(self.output)

    def test_modified_release_is_preserved_instead_of_overwritten(self):
        release.package(self.output)
        original = {path.name: path.read_bytes() for path in self.output.glob('*.zip')}
        self.write('client.mjs', 'export const value = 2;\n')
        with self.assertRaises(FileExistsError):
            release.package(self.output)
        for name, data in original.items():
            self.assertEqual(data, (self.output / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
