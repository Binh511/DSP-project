"""Create reproducible, source-only Chrome/Edge and local backend handoff ZIPs.

No model weights, user documents, environments or machine-specific manifests are
included. Existing releases are immutable: an identical rerun is accepted, and a
changed build requires another --output-dir.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
PACKAGES = (
    "psai_extension", "psai_project", "psai_planner", "psai_ocr", "psai_rules",
    "psai_retrieval", "psai_rag",
)
ROOT_FILES = (
    "extraction.py", "ensemble_classifier.py", "planning.py", "psai.py",
    "psai_reports.py", "pdf_preprocessing.py", "README_EXTENSION.md",
    "requirements.txt", "requirements-project.txt", "requirements-rag.txt",
    "requirements-pdf.txt", "requirements-retrieval.txt", "examples/proposal.md",
    "tools/start_extension_backend.ps1", "tools/package_extension.py", "tools/verify_extension_demo.py",
    "tools/verify_extension_browser.mjs",
    "psai_ocr/windows_native.ps1", "psai_ocr/requirements.txt", "psai_ocr/README.md",
)
EXTENSION_SUFFIXES = {".html", ".css", ".js", ".mjs", ".svg", ".png", ".ico"}


def _source(path: Path) -> bytes:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(ROOT):
        raise ValueError(f"Refusing source outside the project: {path}")
    if not resolved.is_file():
        raise ValueError(f"Source is not a regular file: {path}")
    return resolved.read_bytes()


def _extension_files() -> dict[str, bytes]:
    directory = ROOT / "extension/psai"
    payload = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(directory).as_posix()
        if "tests" in Path(name).parts or path.name.endswith((".test.mjs", ".test.js")):
            continue
        if name != "manifest.json" and path.suffix.lower() not in EXTENSION_SUFFIXES:
            raise ValueError(f"Unexpected file in extension source: {name}")
        if any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts):
            raise ValueError(f"Unexpected hidden/cache file in extension: {name}")
        payload[name] = _source(path)
    if "manifest.json" not in payload:
        raise ValueError("Extension manifest is missing: extension/psai/manifest.json")
    manifest = json.loads(payload["manifest.json"])
    if manifest.get("manifest_version") != 3 or manifest.get("version") != VERSION:
        raise ValueError(f"Expected Manifest V3 with version {VERSION}")
    if "app.html" not in payload:
        raise ValueError("Extension application is missing: extension/psai/app.html")
    return payload


def _backend_files(extension: dict[str, bytes]) -> dict[str, bytes]:
    payload = {name: _source(ROOT / name) for name in ROOT_FILES}
    for package in PACKAGES:
        directory = ROOT / package
        if not (directory / "__init__.py").is_file():
            raise ValueError(f"Required Python package is missing: {package}")
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(directory)
            if path.name.startswith("test_") or any(part in {"tests", "__pycache__"} for part in relative.parts):
                continue
            payload[path.relative_to(ROOT).as_posix()] = _source(path)
    payload.update({"extension/psai/" + name: data for name, data in extension.items()})
    _verify_local_imports(payload)
    return payload


def _verify_local_imports(payload: dict[str, bytes]) -> None:
    """Reject a runtime import omitted from the curated source bundle."""
    local_modules = {path.stem for path in ROOT.glob("*.py")}
    local_modules.update(path.name for path in ROOT.iterdir()
                         if path.is_dir() and (path / "__init__.py").is_file())
    bundled_modules = {Path(name).stem for name in payload if "/" not in name and name.endswith(".py")}
    bundled_modules.update(name.split("/", 1)[0] for name in payload if name.endswith("/__init__.py"))
    missing = set()
    for name, data in payload.items():
        if not name.endswith(".py") or name.startswith("tools/"):
            continue
        tree = ast.parse(data.decode("utf-8-sig"), filename=name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules = [node.module.split(".", 1)[0]]
            missing.update(f"{name}: {module}" for module in modules
                           if module in local_modules and module not in bundled_modules)
    if missing:
        raise ValueError("Unbundled local imports: " + ", ".join(sorted(missing)))


def _zip(payload: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    result = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"ZIP CRC check failed: {bad_member}")
    return result


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package(output_dir: Path) -> dict:
    extension = _extension_files()
    backend = _backend_files(extension)
    builds = {
        f"PSAI_EXTENSION_CHROME_EDGE_v{VERSION}.zip": extension,
        f"PSAI_EXTENSION_BACKEND_v{VERSION}.zip": backend,
    }
    archives = {name: _zip(payload) for name, payload in builds.items()}
    receipt = {
        "release_version": VERSION,
        "distribution": "source_only_local_backend_required",
        "reproducible_zip_timestamps": True,
        "existing_outputs": "preserved_unless_byte_identical",
        "excluded": ["model_weights", "checkpoints", "datasets", "user_documents", "outputs",
                     "virtual_environments", "credentials", "machine_specific_model_manifests"],
        "archives": {
            name: {
                "sha256": _digest(data), "size_bytes": len(data), "crc_verified": True,
                "files": [{"path": path, "size_bytes": len(value), "sha256": _digest(value)}
                          for path, value in sorted(builds[name].items())],
            }
            for name, data in archives.items()
        },
    }
    outputs = {**archives, "package_receipt.json":
               (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = output_dir / name
        if target.exists() and (not target.is_file() or target.is_symlink() or target.read_bytes() != data):
            raise FileExistsError(f"Existing release is preserved: {target}. Choose a new --output-dir.")
    for name, data in outputs.items():
        target = output_dir / name
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/extension")
    args = parser.parse_args()
    receipt = package(args.output_dir)
    for name, details in receipt["archives"].items():
        print(f"{args.output_dir / name}\n  SHA256 {details['sha256']} ({details['size_bytes']} bytes)")
    print(args.output_dir / "package_receipt.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
