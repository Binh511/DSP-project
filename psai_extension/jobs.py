"""Durable job records, result capabilities, and the serial processing queue."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import secrets
import threading
from typing import Any, TYPE_CHECKING
import uuid

from psai_project.execution import runtime_python

from .config import JOB_ID_PATTERN, MAX_ACTIVE_JOBS, RESULT_FILES, ROOT, same_secret
from .pipeline import run_pipeline, stop_process
from .uploads import validate_upload

if TYPE_CHECKING:
    from subprocess import Popen


Job = dict[str, Any]
JobRunner = Callable[[Job, Path, 'JobManager'], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueueFullError(ValueError):
    """The backend is stopping or has reached its active-job limit."""


class JobManager:
    """Persist job state and execute one uploaded document at a time."""

    def __init__(self, data_dir: Path, python_executable: str | None = None,
                 runner: JobRunner | None = None, start_worker: bool = True,
                 resource_dir: Path | str | None = None,
                 base_model_dir: Path | str | None = None,
                 planner_model_dir: Path | str | None = None):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.python_executable = str(runtime_python(ROOT, python_executable))
        self.model_paths = {'resource-dir': resource_dir, 'base-model-dir': base_model_dir,
                            'planner-model-dir': planner_model_dir}
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._closing = threading.Event()
        self._runner = runner or self._run_pipeline
        self._thread: threading.Thread | None = None
        self._process: Popen[str] | None = None
        self._restore()
        if start_worker:
            self._thread = threading.Thread(target=self._work, name='psai-extension-worker', daemon=True)
            self._thread.start()

    def _restore(self) -> None:
        for directory in self.data_dir.iterdir():
            if not directory.is_dir() or not JOB_ID_PATTERN.fullmatch(directory.name):
                continue
            try:
                job = json.loads((directory / 'status.json').read_text('utf-8'))
                if job['id'] != directory.name or not isinstance(job.get('_access'), str):
                    continue
                if job.get('status') in ('queued', 'running'):
                    job.update(status='failed', stage='interrupted', updated_at=_now(),
                               error='Backend đã dừng trước khi hoàn tất. Hãy gửi lại tài liệu.')
                elif job.get('status') not in ('completed', 'failed'):
                    continue
                self._jobs[job['id']] = job
                self._save(job)
            except (OSError, ValueError, KeyError, TypeError):
                # A partial or unrelated folder must not stop the backend.
                continue

    def _save(self, job: Job) -> None:
        target = self.data_dir / job['id'] / 'status.json'
        pending = target.with_suffix('.pending')
        pending.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        pending.replace(target)

    def _public(self, job: Job) -> Job:
        result = {key: value for key, value in job.items() if not key.startswith('_')}
        result['results'] = {}
        if job['status'] == 'completed':
            directory = self.data_dir / job['id'] / 'result'
            for key, filename in RESULT_FILES.items():
                if (directory / filename).is_file():
                    result['results'][key] = f"/results/{job['id']}/{job['_access']}/{filename}"
        log = self.data_dir / job['id'] / 'pipeline.log'
        try:
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                result['log_tail'] = stream.read(8192).decode('utf-8', errors='replace')
        except OSError:
            result['log_tail'] = ''
        return result

    def create_job(self, payload: object) -> Job:
        """Validate and persist an upload before placing it on the worker queue."""
        upload = validate_upload(payload)
        with self._lock:
            if self._closing.is_set():
                raise QueueFullError('Backend đang dừng. Hãy kết nối lại sau.')
            active = sum(job['status'] in ('queued', 'running') for job in self._jobs.values())
            if active >= MAX_ACTIVE_JOBS:
                raise QueueFullError('Hàng đợi đã đầy (tối đa 4 tài liệu). Hãy chờ một lượt hoàn tất.')
            identity = uuid.uuid4().hex
            directory = self.data_dir / identity
            directory.mkdir()
            (directory / ('source' + upload.suffix)).write_bytes(upload.content)
            job = dict(id=identity, filename=upload.filename, mode=upload.mode,
                       device=upload.device, ocr=upload.ocr, status='queued', stage='queued',
                       created_at=_now(), updated_at=_now(), error=None,
                       _suffix=upload.suffix, _access=secrets.token_urlsafe(32))
            self._jobs[identity] = job
            self._save(job)
            self._queue.put(identity)
            return self._public(job)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields, updated_at=_now())
            self._save(self._jobs[job_id])

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public(job) if job else None

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return [self._public(job) for job in sorted(self._jobs.values(),
                    key=lambda value: value['created_at'], reverse=True)[:100]]

    def result_file(self, job_id: str, access: str, filename: str) -> Path | None:
        """Resolve only a completed job's explicitly supported capability export."""
        with self._lock:
            job = self._jobs.get(job_id)
            if (not job or job['status'] != 'completed' or filename not in RESULT_FILES.values()
                    or not same_secret(job['_access'], access)):
                return None
            directory = (self.data_dir / job_id / 'result').resolve()
            path = (directory / filename).resolve()
            return path if path.is_file() and path.is_relative_to(directory) else None

    def _work(self) -> None:
        while not self._closing.is_set():
            try:
                identity = self._queue.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    if self._closing.is_set():
                        continue
                    self.update(identity, status='running', stage='source')
                    job = dict(self._jobs[identity])
                directory = self.data_dir / identity
                self._runner(job, directory, self)
                if not all((directory / 'result' / name).is_file() for name in RESULT_FILES.values()):
                    raise RuntimeError('Pipeline chưa tạo đủ báo cáo, workspace và analysis.json.')
                with self._lock:
                    if self._closing.is_set():
                        raise RuntimeError('Backend đã dừng trong khi xử lý. Hãy gửi lại tài liệu.')
                    self.update(identity, status='completed', stage='completed', error=None)
            except Exception as error:
                self.update(identity, status='failed', stage='failed', error=str(error)[:4000])
            finally:
                self._queue.task_done()

    def _run_pipeline(self, job: Job, directory: Path, manager: JobManager) -> None:
        run_pipeline(job, directory, manager)

    @staticmethod
    def _stop_process(process: Popen[str]) -> None:
        stop_process(process)

    def close(self) -> None:
        """Reject new work, interrupt queued jobs, and stop this backend's child."""
        self._closing.set()
        with self._lock:
            process = self._process
            for job in self._jobs.values():
                if job['status'] == 'queued':
                    job.update(status='failed', stage='interrupted', updated_at=_now(),
                               error='Backend đã dừng trước khi xử lý. Hãy gửi lại tài liệu.')
                    self._save(job)
        if process is not None:
            self._stop_process(process)
        if self._thread:
            self._thread.join(timeout=5)
