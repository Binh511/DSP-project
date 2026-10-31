"""Windows' existing WinRT PDF/OCR capability; no download or installation."""
from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import tempfile

from .extract import OCRDependencyError


def _powershell() -> str:
    if os.name != 'nt':
        raise OCRDependencyError('Windows native OCR is available only on Windows')
    path = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    if not path.is_file():
        raise OCRDependencyError('Windows PowerShell 5.1 is unavailable')
    return str(path)


def _invoke(arguments: list[str], result_path: Path, *, timeout: int = 120) -> dict:
    # This option applies only to the child process, allowing this reviewed local
    # script to run without changing the user's persisted execution policy.
    command = [_powershell(), '-NoLogo', '-NoProfile', '-NonInteractive',
               '-ExecutionPolicy', 'Bypass',
               '-File', str(Path(__file__).with_name('windows_native.ps1')),
               '-ResultPath', str(result_path), *arguments]
    try:
        result = subprocess.run(command, shell=False, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=timeout,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OCRDependencyError(f'Windows native OCR helper could not run: {exc}') from exc
    if result.returncode != 0 or not result_path.is_file():
        detail = (result.stderr or result.stdout).strip()[-3000:]
        raise OCRDependencyError(f'Windows native OCR capability failed: {detail}')
    value = json.loads(result_path.read_text(encoding='utf-8-sig'))
    if isinstance(value.get('engine'), dict):
        value['engine']['helper_script_sha256'] = sha256(
            Path(__file__).with_name('windows_native.ps1').read_bytes()).hexdigest()
    return value


def probe_windows_ocr() -> dict:
    with tempfile.TemporaryDirectory(prefix='psai-winocr-') as temp:
        return _invoke(['-Probe'], Path(temp) / 'capability.json', timeout=30)


class WindowsPDFRenderer:
    """PDF rendering and OCR occur together inside the installed OS capability."""
    name = 'Windows.Data.Pdf'

    def __init__(self, path: Path, *, language: str = 'en'):
        self.path = Path(path).resolve()
        capability = probe_windows_ocr()
        languages = capability.get('languages', [])
        if isinstance(languages, str):
            languages = [languages]
        matching = [tag for tag in languages if tag.casefold() == language.casefold()
                    or tag.casefold().startswith(language.casefold() + '-')]
        if not matching:
            raise OCRDependencyError(f'No installed Windows OCR language matches {language}; available={languages}')
        self.language = matching[0]
        self.version = capability.get('os_version', 'unknown')

    def render(self, index: int, *, dpi: int, max_pixels: int):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='psai-winocr-page-') as temp:
            image_path = Path(temp) / 'page.png'
            output = _invoke([
                '-PdfPath', str(self.path), '-PageIndex', str(index), '-Dpi', str(dpi),
                '-MaxPixels', str(max_pixels), '-LanguageTag', self.language,
                '-OutputImage', str(image_path),
            ], Path(temp) / 'result.json')
            with Image.open(image_path) as source:
                image = source.convert('RGB').copy()
            if image.width * image.height > max_pixels:
                image.close()
                raise ValueError('Windows PDF renderer exceeded the configured pixel limit')
            image._psai_native_ocr = output
            return image

    def close(self):
        pass


class WindowsOCREngine:
    def __init__(self, *, language='en', model_dir=None):
        if model_dir is not None:
            raise ValueError('Windows OCR uses installed OS language packs, not a model directory')

    def recognize(self, image) -> dict:
        result = getattr(image, '_psai_native_ocr', None)
        if result is None:
            raise ValueError('Windows OCR expects a page rendered by WindowsPDFRenderer')
        return result
