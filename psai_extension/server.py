"""Small authenticated HTTP bridge; inference stays in serial local processes.

Only explicitly uploaded documents enter the pipeline. Result links carry a
separate per-job capability so reports can open in ordinary browser tabs without
putting the API pairing token into a URL.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import subprocess
import sys
import threading
from urllib.parse import urlsplit
import uuid

from . import __version__

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 65536
MAX_ACTIVE_JOBS = 4
_ID = re.compile(r'[0-9a-f]{32}\Z')
_ORIGIN = re.compile(r'chrome-extension://[a-p]{32}\Z')
_RESULT_FILES = {'report': 'report.html', 'workspace': 'workspace.html', 'analysis': 'analysis.json'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _same_secret(left, right):
    return secrets.compare_digest(left.encode('utf-8'), right.encode('utf-8'))


class QueueFullError(ValueError):
    pass


class JobManager:
    """Durable job metadata and one inference worker per backend instance."""

    def __init__(self, data_dir: Path, python_executable: str | None = None,
                 runner=None, start_worker: bool = True, resource_dir=None,
                 base_model_dir=None, planner_model_dir=None):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        runtime = ROOT / '.venv/psai_runtime/Scripts/python.exe'
        self.python_executable = str(python_executable or (runtime if runtime.is_file() else sys.executable))
        self.model_paths = {'resource-dir': resource_dir, 'base-model-dir': base_model_dir,
                            'planner-model-dir': planner_model_dir}
        self._lock = threading.RLock()
        self._jobs = {}
        self._queue = queue.Queue()
        self._closing = threading.Event()
        self._runner = runner or self._run_pipeline
        self._thread = None
        self._process = None
        self._restore()
        if start_worker:
            self._thread = threading.Thread(target=self._work, name='psai-extension-worker', daemon=True)
            self._thread.start()

    def _restore(self):
        for directory in self.data_dir.iterdir():
            if not directory.is_dir() or not _ID.fullmatch(directory.name):
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

    def _save(self, job):
        target = self.data_dir / job['id'] / 'status.json'
        pending = target.with_suffix('.pending')
        pending.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        pending.replace(target)

    def _public(self, job):
        result = {key: value for key, value in job.items() if not key.startswith('_')}
        result['results'] = {}
        if job['status'] == 'completed':
            directory = self.data_dir / job['id'] / 'result'
            for key, filename in _RESULT_FILES.items():
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

    def create_job(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Nội dung yêu cầu phải là một JSON object.')
        filename = payload.get('filename')
        if (not isinstance(filename, str) or not filename.strip() or len(filename) > 180
                or any(char in filename for char in '/\\:')
                or any(ord(char) < 32 for char in filename)):
            raise ValueError('Tên tài liệu không hợp lệ; chỉ gửi tên file, không gửi đường dẫn.')
        suffix = Path(filename).suffix.lower()
        if suffix not in ('.pdf', '.md', '.txt'):
            raise ValueError('Chỉ hỗ trợ tài liệu PDF, Markdown (.md) hoặc text (.txt).')
        options = {}
        for key, default, choices in (
                ('mode', 'full', ('full', 'fast')),
                ('device', 'auto', ('auto', 'cpu', 'cuda')),
                ('ocr', 'auto', ('auto', 'none', 'always'))):
            value = payload.get(key, default)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f'{key} phải là một trong: {", ".join(choices)}.')
            options[key] = value
        encoded = payload.get('content_base64')
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
            raise ValueError('Nội dung tài liệu thiếu hoặc vượt quá giới hạn 20 MiB.')
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError('Nội dung tài liệu không phải base64 hợp lệ.') from error
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise ValueError('Tài liệu phải có nội dung và không vượt quá 20 MiB.')
        if suffix in ('.md', '.txt'):
            try:
                if not content.decode('utf-8-sig').strip():
                    raise ValueError('Tài liệu text không có nội dung đọc được.')
            except UnicodeDecodeError as error:
                raise ValueError('Tài liệu Markdown/text cần mã hóa UTF-8.') from error
        with self._lock:
            if self._closing.is_set():
                raise QueueFullError('Backend đang dừng. Hãy kết nối lại sau.')
            if sum(job['status'] in ('queued', 'running') for job in self._jobs.values()) >= MAX_ACTIVE_JOBS:
                raise QueueFullError('Hàng đợi đã đầy (tối đa 4 tài liệu). Hãy chờ một lượt hoàn tất.')
            identity = uuid.uuid4().hex
            directory = self.data_dir / identity
            directory.mkdir()
            (directory / ('source' + suffix)).write_bytes(content)
            job = dict(id=identity, filename=filename, **options, status='queued', stage='queued',
                       created_at=_now(), updated_at=_now(), error=None,
                       _suffix=suffix, _access=secrets.token_urlsafe(32))
            self._jobs[identity] = job
            self._save(job)
            self._queue.put(identity)
            return self._public(job)

    def update(self, job_id, **fields):
        with self._lock:
            self._jobs[job_id].update(fields, updated_at=_now())
            self._save(self._jobs[job_id])

    def get_job(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public(job) if job else None

    def list_jobs(self):
        with self._lock:
            return [self._public(job) for job in sorted(self._jobs.values(),
                    key=lambda value: value['created_at'], reverse=True)[:100]]

    def result_file(self, job_id, access, filename):
        with self._lock:
            job = self._jobs.get(job_id)
            if (not job or job['status'] != 'completed' or filename not in _RESULT_FILES.values()
                    or not _same_secret(job['_access'], access)):
                return None
            directory = (self.data_dir / job_id / 'result').resolve()
            path = (directory / filename).resolve()
            return path if path.is_file() and path.is_relative_to(directory) else None

    def _work(self):
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
                if not all((directory / 'result' / name).is_file() for name in _RESULT_FILES.values()):
                    raise RuntimeError('Pipeline chưa tạo đủ báo cáo, workspace và analysis.json.')
                with self._lock:
                    if self._closing.is_set():
                        raise RuntimeError('Backend đã dừng trong khi xử lý. Hãy gửi lại tài liệu.')
                    self.update(identity, status='completed', stage='completed', error=None)
            except Exception as error:
                self.update(identity, status='failed', stage='failed', error=str(error)[:4000])
            finally:
                self._queue.task_done()

    def _run_pipeline(self, job, directory, manager):
        command = [self.python_executable, '-u', '-m', 'psai_project',
                   '--input', str(directory / ('source' + job['_suffix'])),
                   '--output', str(directory / 'result'), '--device', job['device'], '--ocr', job['ocr']]
        if job['mode'] == 'fast':
            command += ['--classifier', 'rules', '--retrieval', 'bm25', '--planner', 'evidence']
        else:
            command += ['--classifier', 'ensemble', '--retrieval', 'hybrid', '--planner', 'local']
        for flag, value in self.model_paths.items():
            if value is not None and (job['mode'] == 'full' or flag == 'resource-dir'):
                command += ['--' + flag, str(Path(value).resolve())]
        environment = dict(os.environ)
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           HF_HUB_DISABLE_TELEMETRY='1', PYTHONIOENCODING='utf-8')
        with (directory / 'pipeline.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                                  start_new_session=os.name != 'nt') as process:
                with self._lock:
                    self._process = process
                try:
                    if self._closing.is_set():
                        self._stop_process(process)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        match = re.search(r'PSAI stage: (extract|retrieve|plan)', line)
                        if match:
                            manager.update(job['id'], stage=match.group(1))
                    code = process.wait()
                finally:
                    with self._lock:
                        self._process = None
            if code:
                raise RuntimeError(f'Pipeline dừng với mã {code}. Xem nhật ký để biết nguyên nhân.')

    @staticmethod
    def _stop_process(process):
        if process.poll() is not None:
            return
        if os.name == 'nt':
            # Stop only this backend's child and its model-worker descendants.
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

    def close(self):
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


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def create_server(manager: JobManager, token: str, port: int = 8765):
    if not isinstance(token, str) or not token.strip():
        raise ValueError('A nonempty pairing token is required')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'PSAIExtension/' + __version__

        def log_message(self, *_):
            # Result capability URLs and Authorization tokens never enter logs.
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def _send(self, status, body=b'', content_type='application/json; charset=utf-8', extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            origin = self.headers.get('Origin', '')
            if _ORIGIN.fullmatch(origin):
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, status, value):
            self._send(status, json.dumps(value, ensure_ascii=False).encode('utf-8'))

        def _guard(self, auth=True):
            expected = f'127.0.0.1:{self.server.server_address[1]}'
            allowed_hosts = {expected}
            if self.server.server_address[1] == 80:
                allowed_hosts.add('127.0.0.1')
            if self.headers.get('Host') not in allowed_hosts:
                self.close_connection = True
                self._json(403, {'error': 'Host không hợp lệ. Chỉ kết nối qua 127.0.0.1.'})
                return False
            origin = self.headers.get('Origin')
            if origin is not None and not _ORIGIN.fullmatch(origin):
                self.close_connection = True
                self._json(403, {'error': 'Origin không được phép kết nối backend.'})
                return False
            if auth and not _same_secret(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.close_connection = True
                self._json(401, {'error': 'Mã ghép nối không đúng. Dán lại mã trong cửa sổ backend.'})
                return False
            return True

        def do_OPTIONS(self):
            if not self._guard(auth=False):
                return
            if (self.headers.get('Access-Control-Request-Method') not in ('GET', 'POST')
                    or urlsplit(self.path).path not in ('/api/health', '/api/jobs')
                    and not re.fullmatch(r'/api/jobs/[0-9a-f]{32}', urlsplit(self.path).path)):
                self._json(403, {'error': 'Preflight không được phép.'})
                return
            self._send(204, extra={'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                                  'Access-Control-Allow-Headers': 'Authorization, Content-Type',
                                  'Access-Control-Max-Age': '600'})

        def do_GET(self):
            path = urlsplit(self.path).path
            if not self._guard(auth=not path.startswith('/results/')):
                return
            if path == '/api/health':
                self._json(200, {'status': 'ok', 'version': __version__, 'max_upload_bytes': MAX_UPLOAD_BYTES})
            elif path == '/api/jobs':
                self._json(200, {'jobs': manager.list_jobs()})
            elif re.fullmatch(r'/api/jobs/[0-9a-f]{32}', path):
                job = manager.get_job(path.rsplit('/', 1)[1])
                self._json(200 if job else 404, {'job': job} if job else {'error': 'Không tìm thấy lượt phân tích.'})
            elif path.startswith('/results/'):
                parts = path.split('/')
                file = manager.result_file(*parts[2:]) if len(parts) == 5 else None
                if file is None:
                    self._json(404, {'error': 'Không tìm thấy kết quả hoặc liên kết không hợp lệ.'})
                    return
                is_json = file.suffix == '.json'
                extra = {'Content-Security-Policy': "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; img-src data:; connect-src 'none'; "
                         "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"}
                if is_json:
                    extra['Content-Disposition'] = 'attachment; filename="analysis.json"'
                self._send(200, file.read_bytes(),
                           'application/json; charset=utf-8' if is_json else 'text/html; charset=utf-8', extra)
            else:
                self._json(404, {'error': 'Endpoint không tồn tại.'})

        def do_POST(self):
            if not self._guard():
                return
            if urlsplit(self.path).path != '/api/jobs':
                self.close_connection = True
                self._json(404, {'error': 'Endpoint không tồn tại.'})
                return
            if self.headers.get('Transfer-Encoding'):
                self.close_connection = True
                self._json(400, {'error': 'Cần Content-Length, không hỗ trợ chunked upload.'})
                return
            if self.headers.get_content_type() != 'application/json':
                self.close_connection = True
                self._json(415, {'error': 'Cần Content-Type: application/json.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '-1'))
            except ValueError:
                length = -1
            if length < 1 or length > MAX_REQUEST_BYTES:
                self.close_connection = True
                self._json(413 if length > MAX_REQUEST_BYTES else 400,
                           {'error': 'Kích thước yêu cầu không hợp lệ hoặc tài liệu vượt giới hạn 20 MiB.'})
                return
            try:
                content = self.rfile.read(length)
                if len(content) != length:
                    raise ValueError('Tài liệu tải lên chưa đầy đủ.')
                payload = json.loads(content)
                job = manager.create_job(payload)
            except QueueFullError as error:
                self._json(429, {'error': str(error)})
            except (ValueError, UnicodeDecodeError) as error:
                self._json(400, {'error': str(error)})
            except OSError:
                self.close_connection = True
                self._json(500, {'error': 'Không lưu được tài liệu. Kiểm tra dung lượng và quyền ghi backend.'})
            else:
                self._json(202, {'job': job})

    return _Server(('127.0.0.1', port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'outputs/extension/jobs')
    parser.add_argument('--python', dest='python_executable')
    for option in ('resource-dir', 'base-model-dir', 'planner-model-dir'):
        parser.add_argument('--' + option, type=Path)
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
