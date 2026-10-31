"""Standalone local OCR command, with no automatic installation or download."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from . import extract_pdf_with_ocr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=['none', 'auto', 'always'], default='auto')
    parser.add_argument('--backend', choices=['auto', 'rapidocr', 'windows'], default='auto')
    parser.add_argument('--language', choices=['en'], default='en')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--page-image-dir', type=Path)
    parser.add_argument('--doctor', action='store_true', help='Report local dependencies; does not download anything')
    args = parser.parse_args()
    if args.doctor:
        dependencies = {}
        for name in ('pypdf', 'pypdfium2', 'Pillow', 'rapidocr', 'onnxruntime'):
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = 'not_installed'
        try:
            from .windows import probe_windows_ocr
            native = probe_windows_ocr()
        except Exception as exc:
            native = {'available': False, 'reason': str(exc)}
        print(json.dumps({'dependencies': dependencies, 'windows_native': native,
                          'real_ocr_executed_by_this_doctor': False,
                          'functional_smoke_command': 'python -m psai_ocr.verify_native',
                          'note': 'This command probes availability; run the authored PDF smoke to verify actual OCR.'}, indent=2))
        return 0
    if args.input is None or args.output is None:
        parser.error('--input and --output are required without --doctor')
    if args.output.exists():
        parser.error('Output already exists; choose a new path to preserve evidence')
    result = extract_pdf_with_ocr(args.input, mode=args.mode, backend=args.backend, language=args.language,
                                  dpi=args.dpi, model_dir=args.model_dir,
                                  cache_dir=args.page_image_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'ocr_pages': result['ocr_pages'],
                      'method': result['method'], 'pdf_visual_review_complete': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
