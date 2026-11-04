"""Command-line startup for the authenticated local extension backend."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import secrets
import sys

from . import __version__
from .config import ROOT
from .http import create_server
from .jobs import JobManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'outputs/extension/jobs')
    parser.add_argument('--python', dest='python_executable')
    for option in ('resource-dir', 'base-model-dir', 'planner-model-dir'):
        parser.add_argument('--' + option, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bridge until interrupted, with a new pairing token each start."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    for name in ('resource_dir', 'base_model_dir', 'planner_model_dir'):
        value = getattr(args, name)
        if value is not None and not value.is_dir():
            parser.error('--' + name.replace('_', '-') + ' must name an existing local directory')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    token = secrets.token_urlsafe(32)
    manager = JobManager(args.data_dir, python_executable=args.python_executable,
                         resource_dir=args.resource_dir, base_model_dir=args.base_model_dir,
                         planner_model_dir=args.planner_model_dir)
    try:
        server = create_server(manager, token, args.port)
    except OSError as error:
        manager.close()
        parser.exit(1, f'Cannot start loopback backend on port {args.port}: {error}\n')
    print(f'PSAI extension backend v{__version__}\nURL: http://127.0.0.1:{args.port}\n'
          f'Mã ghép nối / Pairing token: {token}\n'
          f'Kết quả được lưu tại: {manager.data_dir}\n'
          'Giữ cửa sổ này mở. Nhấn Ctrl+C để dừng. Mỗi lần khởi động dùng mã ghép nối mới.', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print('\nBackend đã dừng.', flush=True)
    finally:
        server.server_close()
        manager.close()
    return 0
