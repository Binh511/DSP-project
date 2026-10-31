"""Exercise upload -> actual local pipeline -> protected browser result links.

The default uses rules/BM25/evidence; --full runs the installed ML models. This
is functional verification with an authored example, not a model-quality test.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psai_extension.server import JobManager, create_server


def verify(output: Path, input_path: Path, full=False, timeout=900, python_executable=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new verification directory; existing evidence is preserved')
    output.mkdir(parents=True)
    token = secrets.token_urlsafe(32)
    manager = JobManager(output / 'jobs', python_executable=python_executable)
    server = create_server(manager, token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'

    def request(path, payload=None, authenticated=True):
        headers = {'Authorization': 'Bearer ' + token} if authenticated else {}
        body = None
        if payload is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(payload).encode('utf-8')
        with urlopen(Request(origin + path, data=body, headers=headers), timeout=15) as response:
            return response.read(), dict(response.headers)

    started = time.monotonic()
    try:
        try:
            request('/api/jobs', authenticated=False)
        except HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError('Unauthenticated API access was accepted')
        health = json.loads(request('/api/health')[0])
        assert health['status'] == 'ok'
        job = json.loads(request('/api/jobs', {
            'filename': input_path.name,
            'content_base64': base64.b64encode(input_path.read_bytes()).decode('ascii'),
            'mode': 'full' if full else 'fast', 'device': 'auto', 'ocr': 'auto'})[0])['job']
        last_stage = None
        while time.monotonic() - started < timeout:
            job = json.loads(request('/api/jobs/' + job['id'])[0])['job']
            if job['stage'] != last_stage:
                print('Extension API job stage:', job['stage'], flush=True)
                last_stage = job['stage']
            if job['status'] in ('completed', 'failed'):
                break
            time.sleep(.5)
        if job['status'] != 'completed':
            raise RuntimeError(str(job.get('error')) + '\n' + job.get('log_tail', ''))
        exported = {}
        for key, path in job['results'].items():
            raw, headers = request(path, authenticated=False)
            assert raw and headers.get('Cache-Control') == 'no-store'
            exported[key] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            if key == 'analysis':
                analysis = json.loads(raw)
        metadata = analysis['metadata']
        assert metadata['input_sha256'] == hashlib.sha256(input_path.read_bytes()).hexdigest()
        assert metadata['classifier'] == ('ensemble' if full else 'rules')
        assert metadata['retrieval_mode'] == ('hybrid' if full else 'bm25')
        assert metadata['planner_mode'] == ('local' if full else 'evidence')
        report_path = job['results']['report']
        try:
            request(report_path.replace(report_path.split('/')[3], 'invalid-capability'), authenticated=False)
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError('Invalid result capability was accepted')
        evidence = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'mode': job['mode'], 'input': str(input_path), 'job_id': job['id'],
                    'elapsed_seconds': round(time.monotonic() - started, 3),
                    'api_auth_checked': True, 'result_capability_checked': True,
                    'real_pipeline_executed': True, 'input_hash_verified': True,
                    'exports': exported, 'pipeline_metadata': metadata,
                    'quality_scope': 'Functional authored example; no independent accuracy claim'}
        target = output / 'acceptance.json'
        target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Extension API acceptance passed:', target, flush=True)
        return evidence
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/proposal.md')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--python', dest='python_executable')
    args = parser.parse_args()
    output = args.output or ROOT / 'outputs/extension' / (
        'verify_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_') + uuid.uuid4().hex[:8])
    verify(output, args.input, args.full, args.timeout, args.python_executable)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
