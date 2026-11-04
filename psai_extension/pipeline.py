"""Launch local project inference and stop only its owned process tree."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import signal
import subprocess
from typing import Any, TYPE_CHECKING

from psai_project.execution import offline_environment, project_command

from .config import ROOT

if TYPE_CHECKING:
    from .jobs import JobManager


_STAGE_PATTERN = re.compile(r'PSAI stage: (extract|retrieve|plan)')


def run_pipeline(job: Mapping[str, Any], directory: Path, manager: JobManager) -> None:
    """Stream subprocess logs and stage updates back into the durable job."""
    paths = {name: value for name, value in manager.model_paths.items()
             if job['mode'] == 'full' or name == 'resource-dir'}
    command = project_command(
        manager.python_executable, directory / ('source' + job['_suffix']), directory / 'result',
        mode=job['mode'], device=job['device'], ocr=job['ocr'], path_options=paths)
    with (directory / 'pipeline.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=ROOT, env=offline_environment(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                              start_new_session=os.name != 'nt') as process:
            with manager._lock:
                manager._process = process
            try:
                if manager._closing.is_set():
                    manager._stop_process(process)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    match = _STAGE_PATTERN.search(line)
                    if match:
                        manager.update(job['id'], stage=match.group(1))
                code = process.wait()
            finally:
                with manager._lock:
                    manager._process = None
        if code:
            raise RuntimeError(f'Pipeline dừng với mã {code}. Xem nhật ký để biết nguyên nhân.')


def stop_process(process: subprocess.Popen[str]) -> None:
    """Stop the pipeline and its descendants without targeting other backends."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        stopped = subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                                 capture_output=True,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=10)
        if stopped.returncode and process.poll() is None:
            raise RuntimeError(f'Không dừng được cây tiến trình xử lý PID {process.pid}. '
                               'Hãy chờ xử lý kết thúc trước khi khởi động lại backend.')
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
