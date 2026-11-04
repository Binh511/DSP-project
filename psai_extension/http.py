"""Authenticated loopback HTTP routes and capability-protected result files."""
from __future__ import annotations

from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from urllib.parse import urlsplit

from . import __version__
from .config import EXTENSION_ORIGIN_PATTERN, MAX_REQUEST_BYTES, MAX_UPLOAD_BYTES, same_secret
from .jobs import JobManager, QueueFullError


class ExtensionHTTPServer(ThreadingHTTPServer):
    """Keep the HTTP listener on loopback and give handlers their backend state."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, manager: JobManager, token: str, port: int):
        self.manager = manager
        self.pairing_token = token
        super().__init__(('127.0.0.1', port), ExtensionRequestHandler)


class ExtensionRequestHandler(BaseHTTPRequestHandler):
    server: ExtensionHTTPServer
    protocol_version = 'HTTP/1.1'
    server_version = 'PSAIExtension/' + __version__

    def log_message(self, *_: object) -> None:
        # Result capability URLs and Authorization tokens never enter logs.
        pass

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(20)

    def _send(self, status: int, body: bytes = b'',
              content_type: str = 'application/json; charset=utf-8',
              extra: Mapping[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        origin = self.headers.get('Origin', '')
        if EXTENSION_ORIGIN_PATTERN.fullmatch(origin):
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status: int, value: object) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode('utf-8'))

    def _guard(self, auth: bool = True) -> bool:
        port = self.server.server_address[1]
        allowed_hosts = {f'127.0.0.1:{port}'}
        if port == 80:
            allowed_hosts.add('127.0.0.1')
        if self.headers.get('Host') not in allowed_hosts:
            self.close_connection = True
            self._json(403, {'error': 'Host không hợp lệ. Chỉ kết nối qua 127.0.0.1.'})
            return False
        origin = self.headers.get('Origin')
        if origin is not None and not EXTENSION_ORIGIN_PATTERN.fullmatch(origin):
            self.close_connection = True
            self._json(403, {'error': 'Origin không được phép kết nối backend.'})
            return False
        if auth and not same_secret(self.headers.get('Authorization', ''),
                                    'Bearer ' + self.server.pairing_token):
            self.close_connection = True
            self._json(401, {'error': 'Mã ghép nối không đúng. Dán lại mã trong cửa sổ backend.'})
            return False
        return True

    def do_OPTIONS(self) -> None:
        if not self._guard(auth=False):
            return
        path = urlsplit(self.path).path
        valid_endpoint = (path in ('/api/health', '/api/jobs')
                          or re.fullmatch(r'/api/jobs/[0-9a-f]{32}', path))
        if self.headers.get('Access-Control-Request-Method') not in ('GET', 'POST') or not valid_endpoint:
            self._json(403, {'error': 'Preflight không được phép.'})
            return
        self._send(204, extra={'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                              'Access-Control-Allow-Headers': 'Authorization, Content-Type',
                              'Access-Control-Max-Age': '600'})

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if not self._guard(auth=not path.startswith('/results/')):
            return
        if path == '/api/health':
            self._json(200, {'status': 'ok', 'version': __version__, 'max_upload_bytes': MAX_UPLOAD_BYTES})
        elif path == '/api/jobs':
            self._json(200, {'jobs': self.server.manager.list_jobs()})
        elif re.fullmatch(r'/api/jobs/[0-9a-f]{32}', path):
            job = self.server.manager.get_job(path.rsplit('/', 1)[1])
            self._json(200 if job else 404, {'job': job} if job else {'error': 'Không tìm thấy lượt phân tích.'})
        elif path.startswith('/results/'):
            self._result(path)
        else:
            self._json(404, {'error': 'Endpoint không tồn tại.'})

    def _result(self, path: str) -> None:
        parts = path.split('/')
        file = self.server.manager.result_file(*parts[2:]) if len(parts) == 5 else None
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

    def do_POST(self) -> None:
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
            job = self.server.manager.create_job(json.loads(content))
        except QueueFullError as error:
            self._json(429, {'error': str(error)})
        except ValueError as error:
            self._json(400, {'error': str(error)})
        except OSError:
            self.close_connection = True
            self._json(500, {'error': 'Không lưu được tài liệu. Kiểm tra dung lượng và quyền ghi backend.'})
        else:
            self._json(202, {'job': job})


def create_server(manager: JobManager, token: str, port: int = 8765) -> ExtensionHTTPServer:
    """Bind an authenticated backend to IPv4 loopback; port zero is testable."""
    if not isinstance(token, str) or not token.strip():
        raise ValueError('A nonempty pairing token is required')
    return ExtensionHTTPServer(manager, token, port)
