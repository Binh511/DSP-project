"""Run local PSAI stages in serial worker processes and export an atomic report.

Worker exit releases CUDA allocations before the next model is loaded. Failures
preserve requests, logs and any response in a separate diagnostic directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
WORKER_TIMEOUT_SECONDS = 1800


class ProjectStageError(RuntimeError):
    """A local worker failed; its request and response remain available."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8', newline='')


def _run_worker(stage_name: str, request: dict, stage_dir: Path) -> dict:
    jobs = Path(stage_dir) / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    request_path, response_path = jobs / f'{stage_name}.request.json', jobs / f'{stage_name}.response.json'
    stdout_path, stderr_path = jobs / f'{stage_name}.stdout.log', jobs / f'{stage_name}.stderr.log'
    _json(request_path, {**request, 'stage': stage_name})
    command = [sys.executable, '-u', '-m', 'psai_project.worker',
               '--request', str(request_path), '--response', str(response_path)]
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                       PYTHONIOENCODING='utf-8')
    started = time.monotonic()
    print(f'PSAI stage: {stage_name} (serial local worker)', flush=True)
    with stdout_path.open('w', encoding='utf-8', newline='') as stdout, \
            stderr_path.open('w', encoding='utf-8', newline='') as stderr:
        try:
            process = subprocess.run(command, cwd=ROOT, env=environment,
                                     stdout=stdout, stderr=stderr, check=False,
                                     timeout=WORKER_TIMEOUT_SECONDS,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as error:
            raise ProjectStageError(f'{stage_name} exceeded {WORKER_TIMEOUT_SECONDS}s; inspect jobs logs') from error
    response = None
    if response_path.is_file():
        try:
            response = json.loads(response_path.read_text('utf-8'))
        except (OSError, ValueError) as error:
            raise ProjectStageError(f'{stage_name} returned unreadable JSON; inspect jobs logs') from error
    if process.returncode != 0 or not isinstance(response, dict) or response.get('status') != 'completed':
        reason = response.get('error', response.get('message', 'no successful response')) if isinstance(response, dict) else 'no JSON response'
        raise ProjectStageError(f'{stage_name} failed (exit {process.returncode}): {reason}; inspect jobs logs')
    if response.get('stage') != stage_name:
        raise ProjectStageError(f'{stage_name} returned a missing or different stage identity; inspect jobs logs')
    expected_source = request.get('source_sha256')
    if (not isinstance(expected_source, str) or len(expected_source) != 64
            or any(char not in '0123456789abcdef' for char in expected_source)):
        raise ProjectStageError(f'{stage_name} request has no valid source_sha256 binding; inspect jobs logs')
    if response.get('source_sha256') != expected_source:
        raise ProjectStageError(f'{stage_name} returned a missing or different source_sha256; inspect jobs logs')
    response['worker_execution'] = {
        'stage': stage_name, 'returncode': process.returncode,
        'elapsed_seconds': round(time.monotonic() - started, 6),
        'request': request_path.relative_to(stage_dir).as_posix(),
        'response': response_path.relative_to(stage_dir).as_posix(),
        'stdout': stdout_path.relative_to(stage_dir).as_posix(),
        'stderr': stderr_path.relative_to(stage_dir).as_posix(),
        'local_only': True, 'gpu_lifecycle': 'worker_process_exited_before_next_stage'}
    return response


def _local_planner(path: Path | None) -> Path:
    if path is None:
        manifest = ROOT / 'reports/project_completion/local_planner_model.json'
        if not manifest.is_file():
            raise ValueError('Local planner is missing. Supply --planner-model-dir with an existing local model, '
                             'or use --planner evidence. This command never downloads models.')
        value = json.loads(manifest.read_text('utf-8'))
        if not isinstance(value, dict) or not isinstance(value.get('local_path'), str):
            raise ValueError('Local planner manifest must contain local_path')
        path = Path(value['local_path'])
    path = Path(path).resolve()
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError('Local planner directory is unavailable; provide --planner-model-dir or --planner evidence')
    return path


def run_project(input_path: Path, output_dir: Path, *, classifier: str = 'ensemble',
                retrieval: str = 'hybrid', planner: str = 'local', ocr: str = 'auto',
                queries: list[str] | None = None, resource_dir: Path | None = None,
                adapter_dir: Path | None = None, base_model_dir: Path | None = None,
                planner_model_dir: Path | None = None, device: str = 'auto', top_k: int = 5,
                rerank_model_dir: Path | None = None) -> Path:
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing reports are never overwritten')
    if not input_path.is_file() or input_path.suffix.lower() not in ('.pdf', '.md', '.txt'):
        raise ValueError('Input must be one existing PDF, Markdown or text file')
    if classifier not in ('ensemble', 'rules') or retrieval not in ('none', 'bm25', 'dense', 'hybrid'):
        raise ValueError('Unknown classifier or retrieval mode')
    if planner not in ('local', 'evidence') or ocr not in ('auto', 'none', 'always'):
        raise ValueError('Unknown planner or OCR mode')
    if device not in ('auto', 'cpu', 'cuda') or type(top_k) is not int or top_k < 1:
        raise ValueError('Device must be auto/cpu/cuda and top_k must be positive')
    retrieval_options = (queries, resource_dir, adapter_dir, base_model_dir, rerank_model_dir)
    if retrieval == 'none' and any(value is not None for value in retrieval_options):
        raise ValueError('Retrieval options require a retrieval mode')
    if retrieval == 'bm25' and any(value is not None for value in (base_model_dir, adapter_dir, rerank_model_dir)):
        raise ValueError('Embedding, adapter and reranker options require dense or hybrid retrieval')
    if planner == 'evidence' and planner_model_dir is not None:
        raise ValueError('A planner model is unused with --planner evidence')
    if resource_dir is not None:
        resource_dir = Path(resource_dir).resolve()
        if not resource_dir.is_dir():
            raise ValueError('Resource directory must exist locally')
    if queries is not None and (not isinstance(queries, list) or not queries
                                or any(not isinstance(query, str) or not query.strip() for query in queries)):
        raise ValueError('queries must contain at least one nonempty string')
    if planner == 'local':
        planner_model_dir = _local_planner(planner_model_dir)
    if queries is None and retrieval != 'none':
        from psai import DEFAULT_RETRIEVAL_QUERIES
        queries = list(DEFAULT_RETRIEVAL_QUERIES)
    input_before = _sha(input_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.psai-project-', dir=output_dir.parent))
    current_phase = 'source'
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'input_path': str(input_path),
                'input_sha256': input_before, 'classifier': classifier, 'retrieval_mode': retrieval,
                'planner_mode': planner, 'ocr_mode': ocr, 'stages': [],
                'production_model_changed': False, 'human_content_review_complete': False,
                'pipeline_quality_score': None, 'final_test_evaluated': False,
                'worker_policy': 'serial_processes_no_concurrent_model_allocations'}
    try:
        if input_path.suffix.lower() == '.pdf':
            from psai_ocr import extract_pdf_with_ocr
            source_metadata = extract_pdf_with_ocr(input_path, mode=ocr, cache_dir=stage / 'page_images')
            text = source_metadata.pop('text')
            for page in source_metadata.get('pages', []):
                if page.get('image_path'):
                    image = Path(page['image_path']).resolve()
                    if not image.is_relative_to(stage):
                        raise ValueError('OCR image path escaped the staged report')
                    page['image_path'] = image.relative_to(stage).as_posix()
        else:
            text = input_path.read_text('utf-8-sig')
            source_metadata = {'source_path': str(input_path), 'method': 'local_text',
                               'pdf_visual_review_complete': False, 'ocr_applied': False}
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Source contains no usable text')
        source_path = stage / 'source.txt'
        source_path.write_text(text, encoding='utf-8', newline='')
        source_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        metadata['source_text_sha256'] = source_hash
        metadata['source'] = source_metadata
        _json(stage / 'source_metadata.json', source_metadata)
        shared = {'source_path': str(source_path), 'source_sha256': source_hash,
                  'document_id': input_path.stem, 'input_path': str(input_path), 'device': device}
        current_phase = 'extract'
        extraction = _run_worker('extract', {**shared, 'classifier': classifier}, stage)
        if not isinstance(extraction.get('record'), dict):
            raise ProjectStageError('extract response has no structured record')
        metadata['stages'].append(extraction.get('worker_execution', {'stage': 'extract'}))
        current_phase = 'retrieve'
        retrieve_request = {**shared, 'retrieval': retrieval, 'queries': queries, 'top_k': top_k}
        for name, value in [('resource_dir', resource_dir), ('base_model_dir', base_model_dir),
                            ('adapter_dir', adapter_dir), ('rerank_model_dir', rerank_model_dir)]:
            if value is not None:
                retrieve_request[name] = str(Path(value).resolve())
        retrieved = _run_worker('retrieve', retrieve_request, stage)
        if not isinstance(retrieved.get('retrieval'), dict):
            raise ProjectStageError('retrieve response has no structured retrieval result')
        metadata['stages'].append(retrieved.get('worker_execution', {'stage': 'retrieve'}))
        current_phase = 'plan'
        plan_request = {**shared, 'planner': planner,
                        'record_path': str(stage / 'jobs/extract.response.json'),
                        'retrieval_path': str(stage / 'jobs/retrieve.response.json')}
        if planner_model_dir is not None:
            plan_request['planner_model_dir'] = str(planner_model_dir)
        planned = _run_worker('plan', plan_request, stage)
        if not isinstance(planned.get('plan'), dict) or not isinstance(planned.get('source_plan'), dict):
            raise ProjectStageError('plan response must contain both plan and source_plan')
        metadata['stages'].append(planned.get('worker_execution', {'stage': 'plan'}))
        current_phase = 'export'
        if _sha(source_path) != source_hash or _sha(input_path) != input_before:
            raise ValueError('Source changed during analysis; refusing to export a completed report')
        metadata['input_unchanged'] = True
        metadata['source_snapshot_unchanged'] = True
        metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
        from .report import export_project_report
        report = Path(export_project_report(stage, text, source_metadata, extraction, retrieved, planned, metadata))
        if not report.is_file() or report.resolve() != (stage / 'report.html').resolve():
            raise ProjectStageError('Report exporter did not produce the expected report.html')
        _json(stage / 'run_manifest.json', metadata)
        if output_dir.exists():
            raise FileExistsError('Output appeared during analysis; existing directory is preserved')
        stage.rename(output_dir)
        return output_dir / 'report.html'
    except Exception as error:
        failed = output_dir.with_name(output_dir.name + '.failed_' + uuid.uuid4().hex[:8])
        failure = {'status': 'failed', 'phase': current_phase, 'error_type': type(error).__name__,
                   'error': str(error), 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'requested_output': str(output_dir), 'diagnostic_dir': str(failed),
                   'report_completed': False, 'metadata': metadata}
        _json(stage / 'failure.json', failure)
        stage.rename(failed)
        raise ProjectStageError(f'{current_phase}: {error}. Diagnostics preserved at {failed}') from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--classifier', choices=('rules', 'ensemble'), default='ensemble')
    parser.add_argument('--retrieval', choices=('none', 'bm25', 'dense', 'hybrid'), default='hybrid')
    parser.add_argument('--planner', choices=('local', 'evidence'), default='local')
    parser.add_argument('--ocr', choices=('auto', 'none', 'always'), default='auto')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--query', action='append', dest='queries')
    parser.add_argument('--top-k', type=int, default=5)
    for option in ('resource-dir', 'base-model-dir', 'adapter-dir', 'rerank-model-dir', 'planner-model-dir'):
        parser.add_argument('--' + option, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_project(args.input, args.output, classifier=args.classifier, retrieval=args.retrieval,
                              planner=args.planner, ocr=args.ocr, queries=args.queries,
                              resource_dir=args.resource_dir, adapter_dir=args.adapter_dir,
                              base_model_dir=args.base_model_dir, planner_model_dir=args.planner_model_dir,
                              rerank_model_dir=args.rerank_model_dir, device=args.device, top_k=args.top_k)
    except (ValueError, OSError, ProjectStageError) as error:
        parser.exit(2, f'PSAI project: {error}\n')
    print(str(report), flush=True)
    return 0
