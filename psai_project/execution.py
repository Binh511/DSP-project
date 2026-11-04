"""Shared launch policy for demos, the extension backend and serial workers.

These helpers only construct arguments and environment copies. They never load
models, probe the network or replace the caller's requested pipeline mode.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from pathlib import Path
import sys


_PROJECT_MODES = {
    'fast': ('rules', 'bm25', 'evidence'),
    'full': ('ensemble', 'hybrid', 'local'),
}
_OFFLINE_ENVIRONMENT = {
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'PYTHONIOENCODING': 'utf-8',
}


def runtime_python(root: Path, explicit: str | Path | None = None) -> str:
    """Honor an explicit interpreter, then the existing dedicated runtime."""
    if explicit:
        return str(explicit)
    runtime = Path(root) / '.venv/psai_runtime/Scripts/python.exe'
    return str(runtime if runtime.is_file() else sys.executable)


def offline_environment(base: Mapping[str, str] | None = None, *,
                        disable_tokenizer_parallelism: bool = False) -> dict[str, str]:
    """Copy the caller's environment and enforce local-only inference settings."""
    environment = dict(os.environ if base is None else base)
    environment.update(_OFFLINE_ENVIRONMENT)
    if disable_tokenizer_parallelism:
        environment['TOKENIZERS_PARALLELISM'] = 'false'
    return environment


def local_path_arguments(options: Mapping[str, str | Path | None]) -> list[str]:
    """Resolve supplied local paths without changing their flag order."""
    arguments = []
    for name, value in options.items():
        if value is not None:
            arguments.extend(['--' + name, str(Path(value).resolve())])
    return arguments


def project_command(python_executable: str | Path, input_path: Path, output_dir: Path, *,
                    mode: str = 'full', device: str = 'auto', ocr: str = 'auto',
                    path_options: Mapping[str, str | Path | None] | None = None,
                    queries: Iterable[str] | None = None) -> list[str]:
    """Build the same project command for an explicit fast or full launch.

    Callers choose which optional paths apply to their interface. In particular,
    the extension omits ML model paths in fast mode; the demo CLI forwards its
    explicit flags so the project runtime can reject unused configuration.
    """
    if mode not in _PROJECT_MODES:
        raise ValueError('Project mode must be fast or full')
    classifier, retrieval, planner = _PROJECT_MODES[mode]
    command = [str(python_executable), '-u', '-m', 'psai_project',
               '--input', str(Path(input_path).resolve()),
               '--output', str(Path(output_dir).resolve()),
               '--device', device, '--ocr', ocr,
               '--classifier', classifier, '--retrieval', retrieval, '--planner', planner]
    command.extend(local_path_arguments(path_options or {}))
    for query in queries or ():
        command.extend(['--query', query])
    return command
