"""Local-extension HTTP and queue integration tests without learned models."""
import base64
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from psai_extension.server import JobManager, MAX_ACTIVE_JOBS, MAX_UPLOAD_BYTES, create_server


TOKEN = 'integration-test-token'
EXTENSION_ORIGIN = 'chrome-extension://' + 'a' * 32
PRIVATE_TEXT = 'Private extension integration source 68c29f.'


def upload_payload(**overrides):
    payload = {
        'filename': 'project.txt',
        'content_base64': base64.b64encode(PRIVATE_TEXT.encode('utf-8')).decode('ascii'),
        'mode': 'fast',
        'device': 'cpu',
        'ocr': 'auto',
    }
    payload.update(overrides)
    return payload


def successful_runner(job, job_dir, manager):
    manager.update(job['id'], stage='extract')
    result_dir = job_dir / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / 'report.html').write_text('<!doctype html><p>Report fixture</p>', encoding='utf-8')
    (result_dir / 'workspace.html').write_text('<!doctype html><p>Workspace fixture</p>', encoding='utf-8')
    (result_dir / 'analysis.json').write_text('{"fixture": true}', encoding='utf-8')


class ExtensionBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager, self.server = self.start_service(self.root / 'main')

    def start_service(self, data_dir, runner=successful_runner, start_worker=True):
        manager = JobManager(data_dir, runner=runner, start_worker=start_worker)
        server = create_server(manager, token=TOKEN, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            manager.close()

        self.addCleanup(cleanup)
        return manager, server

    def request(self, method, path, payload=None, *, headers=None, server=None, token=TOKEN,
                raw_body=None):
        server = server or self.server
        request_headers = {}
        if token is not None:
            request_headers['Authorization'] = 'Bearer ' + token
        if payload is not None:
            body = json.dumps(payload).encode('utf-8')
            request_headers['Content-Type'] = 'application/json'
        else:
            body = raw_body
        request_headers.update(headers or {})
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            content = response.read()
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            if content and 'application/json' in response_headers.get('content-type', ''):
                content = json.loads(content)
            return response.status, response_headers, content
        finally:
            connection.close()

    def submit(self, *, server=None, **overrides):
        status, _, body = self.request('POST', '/api/jobs', upload_payload(**overrides), server=server)
        self.assertEqual(status, 202, body)
        self.assertIsInstance(body['job']['id'], str)
        return body['job']

    def wait_job(self, job_id, *, manager=None, status=None):
        manager = manager or self.manager
        deadline = time.monotonic() + 5
        latest = None
        while time.monotonic() < deadline:
            latest = manager.get_job(job_id)
            if latest and ((status is None and latest['status'] in ('completed', 'failed'))
                           or latest['status'] == status):
                return latest
            time.sleep(.01)
        self.fail('Job did not reach the expected state: ' + repr(latest))

    def assert_json_error(self, response, expected_status=400):
        status, headers, body = response
        self.assertEqual(status, expected_status, body)
        self.assertIn('application/json', headers.get('content-type', ''))
        self.assertIsInstance(body.get('error'), str)
        self.assertNotIn(PRIVATE_TEXT, json.dumps(body))

    def test_health_requires_token_and_reports_upload_limit(self):
        self.assert_json_error(self.request('GET', '/api/health', token=None), 401)
        status, _, body = self.request('GET', '/api/health')
        self.assertEqual(status, 200)
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['max_upload_bytes'], MAX_UPLOAD_BYTES)

    def test_upload_list_status_and_capability_results(self):
        job = self.submit()
        finished = self.wait_job(job['id'])
        self.assertEqual(finished['status'], 'completed', finished)
        status, _, body = self.request('GET', '/api/jobs')
        self.assertEqual(status, 200)
        self.assertIn(job['id'], [item['id'] for item in body['jobs']])
        status, _, body = self.request('GET', '/api/jobs/' + job['id'])
        self.assertEqual(status, 200)
        self.assertEqual(body['job']['status'], 'completed')
        self.assertEqual(set(body['job']['results']), {'report', 'workspace', 'analysis'})
        for key, suffix in (('report', 'report.html'), ('workspace', 'workspace.html'),
                            ('analysis', 'analysis.json')):
            url = body['job']['results'][key]
            self.assertTrue(url.startswith('/results/' + job['id'] + '/'), url)
            self.assertTrue(url.endswith('/' + suffix), url)
            result_status, headers, result_body = self.request('GET', url, token=None)
            self.assertEqual(result_status, 200, result_body)
            if key == 'analysis':
                self.assertEqual(result_body, {'fixture': True})
            else:
                self.assertIn('text/html', headers.get('content-type', ''))
                self.assertIn(b'fixture', result_body)

    def test_wrong_token_cannot_read_or_submit_jobs(self):
        job = self.submit()
        for path in ('/api/health', '/api/jobs', '/api/jobs/' + job['id']):
            with self.subTest(path=path):
                self.assert_json_error(self.request('GET', path, token='wrong-token'), 401)
        self.assert_json_error(
            self.request('POST', '/api/jobs', upload_payload(), token='wrong-token'), 401)

    def test_non_ascii_credentials_return_errors_without_crashing_handler(self):
        self.assert_json_error(self.request('GET', '/api/jobs', token='\u00e9'), 401)
        completed = self.wait_job(self.submit()['id'])
        self.assertIsNone(self.manager.result_file(completed['id'], '\u00e9', 'report.html'))
        self.assert_json_error(self.request('GET', '/results/' + completed['id']
                                           + '/%C3%A9/report.html', token=None), 404)

    def test_dns_rebinding_host_is_rejected_even_with_valid_token(self):
        for host in ('evil.example', '127.0.0.1:1', 'localhost:' + str(self.server.server_port)):
            with self.subTest(host=host):
                self.assert_json_error(self.request('GET', '/api/jobs', headers={'Host': host}), 403)

    def test_browser_origin_is_restricted_to_extension_origins(self):
        for origin in ('https://evil.example', 'http://127.0.0.1', 'null',
                       'chrome-extension://' + 'z' * 32, EXTENSION_ORIGIN + '.evil.example'):
            with self.subTest(origin=origin):
                self.assert_json_error(
                    self.request('GET', '/api/jobs', headers={'Origin': origin}), 403)
        status, headers, _ = self.request('GET', '/api/jobs', headers={'Origin': EXTENSION_ORIGIN})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get('access-control-allow-origin'), EXTENSION_ORIGIN)

    def test_preflight_allows_extension_without_bearer_and_blocks_web_origin(self):
        headers = {'Origin': EXTENSION_ORIGIN, 'Access-Control-Request-Method': 'POST',
                   'Access-Control-Request-Headers': 'authorization, content-type'}
        status, response_headers, _ = self.request('OPTIONS', '/api/jobs', headers=headers, token=None)
        self.assertIn(status, (200, 204))
        self.assertEqual(response_headers.get('access-control-allow-origin'), EXTENSION_ORIGIN)
        allowed = response_headers.get('access-control-allow-headers', '').lower()
        self.assertIn('authorization', allowed)
        self.assertIn('content-type', allowed)
        self.assert_json_error(self.request('OPTIONS', '/api/jobs', token=None,
                                           headers={**headers, 'Origin': 'https://evil.example'}), 403)

    def test_upload_validation_rejects_bad_content_and_path_names(self):
        invalid = [
            {'filename': 'program.exe'}, {'filename': '../project.txt'},
            {'filename': '..\\project.txt'}, {'filename': 'C:\\project.txt'},
            {'filename': '/tmp/project.txt'}, {'filename': ''},
            {'content_base64': ''}, {'content_base64': '%%%invalid-base64%%%'},
            {'content_base64': 42}, {'mode': 'invented'}, {'device': 'gpu'},
            {'ocr': 'sometimes'}, {'mode': ['fast']},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assert_json_error(self.request('POST', '/api/jobs', upload_payload(**changes)))
        self.assertEqual(self.manager.list_jobs(), [])

    def test_invalid_json_and_non_object_requests_return_json_errors(self):
        for raw in (b'{not json', b'[]', b'null'):
            with self.subTest(raw=raw):
                self.assert_json_error(self.request('POST', '/api/jobs', raw_body=raw,
                                                   headers={'Content-Type': 'application/json'}))

    def test_declared_oversize_request_is_rejected_before_body_is_read(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            connection.putrequest('POST', '/api/jobs')
            connection.putheader('Authorization', 'Bearer ' + TOKEN)
            connection.putheader('Content-Type', 'application/json')
            connection.putheader('Content-Length', str(MAX_UPLOAD_BYTES * 2 + 65536))
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            self.assertIsInstance(json.loads(response.read())['error'], str)
        finally:
            connection.close()

    def test_unknown_job_is_json_404(self):
        self.assert_json_error(self.request('GET', '/api/jobs/' + '0' * 32), 404)

    def test_results_require_capability_and_only_expose_three_exports(self):
        completed = self.wait_job(self.submit()['id'])
        report_url = completed['results']['report']
        prefix = report_url.rsplit('/', 1)[0]
        capability = prefix.rsplit('/', 1)[-1]
        invalid_paths = [
            report_url.replace('/' + capability + '/', '/wrong-capability/'),
            '/results/' + completed['id'] + '/report.html',
            prefix + '/source.txt', prefix + '/status.json', prefix + '/result/analysis.json',
            prefix + '/../status.json', prefix + '/%2e%2e/status.json',
            prefix + '/..%2fstatus.json', prefix + '/..%5cstatus.json',
            '/results/' + completed['id'] + '/' + capability + '/../../status.json',
        ]
        for path in invalid_paths:
            with self.subTest(path=path):
                status, _, body = self.request('GET', path, token=None)
                self.assertIn(status, (400, 403, 404), body)
                self.assertNotIn(PRIVATE_TEXT.encode(), json.dumps(body).encode()
                                 if isinstance(body, dict) else body)

    def test_failed_runner_does_not_publish_partial_results(self):
        def failing_runner(job, job_dir, manager):
            successful_runner(job, job_dir, manager)
            raise RuntimeError('fixture pipeline failure')
        manager, server = self.start_service(self.root / 'failure', runner=failing_runner)
        completed = self.wait_job(self.submit(server=server)['id'], manager=manager)
        self.assertEqual(completed['status'], 'failed')
        self.assertIn('fixture pipeline failure', completed['error'])
        self.assertFalse(completed.get('results'))

    def test_missing_exports_cannot_be_reported_as_completed(self):
        def incomplete_runner(job, job_dir, manager):
            (job_dir / 'result').mkdir()
            (job_dir / 'result' / 'report.html').write_text('partial', encoding='utf-8')
        manager, server = self.start_service(self.root / 'incomplete', runner=incomplete_runner)
        completed = self.wait_job(self.submit(server=server)['id'], manager=manager)
        self.assertEqual(completed['status'], 'failed')
        self.assertFalse(completed.get('results'))

    def test_jobs_execute_serially(self):
        first_entered, release_first = threading.Event(), threading.Event()
        active = []
        observed = []
        lock = threading.Lock()

        def serial_runner(job, job_dir, manager):
            with lock:
                active.append(job['id'])
                observed.append(len(active))
                first = len(observed) == 1
            if first:
                first_entered.set()
                if not release_first.wait(3):
                    raise RuntimeError('Timed out waiting for the serial-queue test')
            successful_runner(job, job_dir, manager)
            with lock:
                active.remove(job['id'])

        manager, server = self.start_service(self.root / 'serial', runner=serial_runner)
        self.addCleanup(release_first.set)
        first = self.submit(server=server)
        self.assertTrue(first_entered.wait(3))
        second = self.submit(server=server)
        try:
            self.assertEqual(manager.get_job(second['id'])['status'], 'queued')
        finally:
            release_first.set()
        self.assertEqual(self.wait_job(first['id'], manager=manager)['status'], 'completed')
        self.assertEqual(self.wait_job(second['id'], manager=manager)['status'], 'completed')
        self.assertEqual(observed, [1, 1])

    def test_restart_marks_pending_job_failed_and_preserves_completed_results(self):
        pending_dir = self.root / 'interrupted'
        manager = JobManager(pending_dir, runner=successful_runner, start_worker=False)
        pending = manager.create_job(upload_payload())
        manager.close()
        restored = JobManager(pending_dir, runner=successful_runner, start_worker=False)
        try:
            job = restored.get_job(pending['id'])
            self.assertEqual(job['status'], 'failed')
            self.assertFalse(job.get('results'))
            self.assertTrue(job.get('error'))
        finally:
            restored.close()

        completed_dir = self.root / 'persisted'
        manager = JobManager(completed_dir, runner=successful_runner)
        try:
            complete = self.wait_job(manager.create_job(upload_payload())['id'], manager=manager)
        finally:
            manager.close()
        restored = JobManager(completed_dir, runner=successful_runner, start_worker=False)
        try:
            self.assertEqual(restored.get_job(complete['id'])['status'], 'completed')
            self.assertEqual(restored.get_job(complete['id'])['results'], complete['results'])
        finally:
            restored.close()

    def test_bounded_queue_rejects_excess_upload_without_creating_another_job(self):
        manager, server = self.start_service(self.root / 'capacity', start_worker=False)
        for _ in range(MAX_ACTIVE_JOBS):
            self.submit(server=server)
        self.assert_json_error(self.request('POST', '/api/jobs', upload_payload(), server=server), 429)
        self.assertEqual(len(manager.list_jobs()), MAX_ACTIVE_JOBS)

    def test_shutdown_after_dequeue_prevents_the_queued_runner_starting(self):
        runner = Mock()
        manager = JobManager(self.root / 'dequeue-shutdown', runner=runner, start_worker=False)
        self.addCleanup(manager.close)
        job = manager.create_job(upload_payload())
        dequeued, release_dequeue = threading.Event(), threading.Event()
        original_get = manager._queue.get

        def pause_after_dequeue(*args, **kwargs):
            identity = original_get(*args, **kwargs)
            dequeued.set()
            if not release_dequeue.wait(3):
                raise RuntimeError('Timed out releasing the queue-race test')
            return identity

        with patch.object(manager._queue, 'get', side_effect=pause_after_dequeue):
            worker = threading.Thread(target=manager._work, daemon=True)
            manager._thread = worker
            worker.start()
            self.assertTrue(dequeued.wait(3))
            closer = threading.Thread(target=manager.close, daemon=True)
            closer.start()
            try:
                self.assertTrue(manager._closing.wait(3))
            finally:
                release_dequeue.set()
            closer.join(timeout=3)
            self.assertFalse(closer.is_alive())
            self.assertFalse(worker.is_alive())
        runner.assert_not_called()
        stopped = manager.get_job(job['id'])
        self.assertEqual(stopped['status'], 'failed')
        self.assertEqual(stopped['stage'], 'interrupted')
        self.assertFalse(stopped['results'])

    def test_shutdown_does_not_publish_outputs_from_an_interrupted_runner(self):
        entered = threading.Event()

        def finishing_during_shutdown(job, job_dir, manager):
            entered.set()
            if not manager._closing.wait(3):
                raise RuntimeError('Timed out waiting for shutdown')
            successful_runner(job, job_dir, manager)

        manager = JobManager(self.root / 'shutdown-output', runner=finishing_during_shutdown)
        self.addCleanup(manager.close)
        job = manager.create_job(upload_payload())
        self.assertTrue(entered.wait(3))
        manager.close()
        self.assertFalse(manager._thread.is_alive())
        stopped = manager.get_job(job['id'])
        self.assertEqual(stopped['status'], 'failed')
        self.assertFalse(stopped['results'])

    @unittest.skipUnless(os.name == 'nt', 'Exercises Windows process-tree shutdown')
    def test_shutdown_stops_its_real_pipeline_child_and_descendant(self):
        # Replace inference with a harmless sleeping process tree. Every process
        # terminated here was created by this test and belongs to this manager.
        import ctypes
        try:
            probe = subprocess.run(
                ['tasklist.exe', '/FI', f'PID eq {os.getpid()}', '/FO', 'CSV', '/NH'],
                capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self.skipTest('Windows process enumeration is unavailable in this test environment')
        if probe.returncode:
            self.skipTest('Windows process enumeration is denied; run outside the restricted sandbox')
        ready = self.root / 'child-ready.txt'
        script = (
            'import pathlib, subprocess, sys, time\n'
            'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], '
            'creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))\n'
            'pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")\n'
            'print("PSAI stage: extract", flush=True)\n'
            'time.sleep(60)\n'
        )
        actual_popen = subprocess.Popen
        spawned = []

        def launch_harmless_pipeline(command, *args, **kwargs):
            if isinstance(command, list) and 'psai_project' in command:
                process = actual_popen([sys.executable, '-u', '-c', script, str(ready)], *args, **kwargs)
                spawned.append(process)
                return process
            return actual_popen(command, *args, **kwargs)

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_bool
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.TerminateProcess.restype = ctypes.c_bool
        child_handle = None
        with patch('psai_extension.server.subprocess.Popen', side_effect=launch_harmless_pipeline):
            manager = JobManager(self.root / 'process-shutdown', python_executable=sys.executable)
            try:
                first = manager.create_job(upload_payload())
                deadline = time.monotonic() + 5
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.is_file(), 'The harmless pipeline did not start its child')
                child_pid = int(ready.read_text('ascii'))
                child_handle = kernel.OpenProcess(0x00100001, False, child_pid)
                self.assertTrue(child_handle, 'Could not observe the test child process')
                second = manager.create_job(upload_payload())
                manager.close()
                self.assertFalse(manager._thread.is_alive(), 'The pipeline thread survived shutdown')
                self.assertEqual(len(spawned), 1, 'Shutdown started a queued pipeline')
                self.assertIsNotNone(spawned[0].poll(), 'The test pipeline process survived shutdown')
                self.assertEqual(kernel.WaitForSingleObject(child_handle, 3000), 0,
                                 'The test pipeline descendant survived shutdown')
                for job_id in (first['id'], second['id']):
                    stopped = manager.get_job(job_id)
                    self.assertEqual(stopped['status'], 'failed')
                    self.assertFalse(stopped['results'])
            finally:
                # Also clean up when taskkill is blocked by a restricted test
                # environment. Handles identify only processes created above.
                for process in spawned:
                    if process.poll() is None:
                        process.terminate()
                if child_handle and kernel.WaitForSingleObject(child_handle, 0) == 258:
                    kernel.TerminateProcess(child_handle, 1)
                manager._thread.join(timeout=3)
                manager.close()
                if child_handle:
                    kernel.CloseHandle(child_handle)


if __name__ == '__main__':
    unittest.main()
