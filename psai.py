"""One local entry point for PSAI diagnostics and evidence-cited reports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DEFAULT_RETRIEVAL_QUERIES = [
    'project objective goal expected results mục tiêu kết quả',
    'requirements constraints technologies must support yêu cầu công nghệ',
    'activities implementation deliverables hoạt động sản phẩm',
    'budget funding timeline team resources kinh phí thời gian nhân lực',
]


class AnalysisConfigurationError(ValueError):
    """An actionable input/configuration error, suitable for concise CLI output."""


def doctor() -> dict:
    groups = {'core': ['numpy', 'pandas', 'scikit-learn', 'torch', 'transformers', 'sentencepiece'],
              'research': ['datasets', 'accelerate'], 'pdf': ['pypdf'],
              'retrieval': ['sentence-transformers', 'chromadb']}
    dependencies = {}
    for group, names in groups.items():
        dependencies[group] = {}
        for name in names:
            try:
                dependencies[group][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                dependencies[group][name] = None
    members = {}
    for name in ['final_s11', 'final_s22', 'final_s33']:
        folder = ROOT / 'models' / name
        members[name] = {file: (folder / file).is_file() for file in
                         ['model.safetensors', 'config.json', 'tokenizer.json', 'inference_config.json']}
    ensemble_missing = [name for name, files in members.items() if not all(files.values())]
    ensemble_packages = [name for name in ('numpy', 'torch', 'transformers', 'sentencepiece')
                         if dependencies['core'][name] is None]
    capabilities = {
        'markdown_text_rules': {'available': True, 'additional_packages': [],
                                'example': 'python psai.py analyze --input examples/proposal.md --output outputs/demo_rules --classifier rules'},
        'offline_bm25': {'available': (ROOT / 'psai_retrieval' / '__init__.py').is_file(),
                         'additional_packages': [],
                         'example': 'python psai.py analyze --input examples/proposal.md --output outputs/demo_bm25 --classifier rules --retrieval bm25'},
        'selectable_text_pdf': {'available': dependencies['pdf']['pypdf'] is not None,
                                'install': 'python -m pip install -r requirements-pdf.txt'},
        'production_ensemble': {'available': not ensemble_missing and not ensemble_packages,
                                'missing_model_members': ensemble_missing,
                                'missing_packages': ensemble_packages,
                                'install': 'python -m pip install -r requirements.txt',
                                'model_load_verified': False},
        'dense_retrieval': {'local_model_required': True,
                            'configuration': '--retrieval dense|hybrid --base-model-dir LOCAL_MODEL [--adapter-dir LOCAL_ADAPTER]',
                            'install': 'python -m pip install -r requirements-retrieval.txt',
                            'model_load_verified': False},
    }
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                dependencies=dependencies, production_ensemble=members,
                capabilities=capabilities,
                note='Package metadata and file checks only; no model loaded or network access. Missing optional packages do not disable Markdown analysis.')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_retrieval(payload: dict, text: str, input_path: Path,
                        resource_dir: Path | None) -> tuple[dict, dict[str, str]]:
    """Verify every retrieved quote and retain the exact cited source snapshots."""
    if not isinstance(payload, dict) or not isinstance(payload.get('hits'), list):
        raise ValueError('Retriever returned an invalid evidence payload')
    main_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    source_texts = {main_hash: text}
    sources = {main_hash: {'artifact': 'source.txt', 'origin_path': str(input_path),
                           'source_sha256': main_hash, 'source_file_sha256': _sha(input_path),
                           'role': 'document'}}
    seen_ids = set()
    for hit in payload['hits']:
        if not isinstance(hit, dict) or hit.get('verified') is not True:
            raise ValueError('Retrieved evidence must have verified source spans')
        role = hit.get('role')
        if role == 'document':
            if hit.get('doc_id') != input_path.stem:
                raise ValueError('Retrieved document evidence refers to a different document')
            source_text, digest, snapshot_id = text, main_hash, main_hash
        elif role == 'resource':
            if resource_dir is None or not hit.get('source_file'):
                raise ValueError('Retrieved resource evidence has no configured source file')
            source_path = Path(hit['source_file'])
            if not source_path.is_absolute():
                source_path = resource_dir / source_path
            source_path = source_path.resolve()
            if not source_path.is_relative_to(resource_dir) or not source_path.is_file():
                raise ValueError('Retrieved resource source is outside the configured local resource directory')
            source_text = source_path.read_text(encoding='utf-8-sig')
            digest = hashlib.sha256(source_text.encode('utf-8')).hexdigest()
            raw_digest = _sha(source_path)
            if hit.get('source_file_sha256') not in (None, raw_digest):
                raise ValueError('Retrieved resource raw-file hash differs from its source')
            snapshot_id = 'resource-' + hashlib.sha256(str(source_path).encode('utf-8')).hexdigest()[:16] + '-' + digest
            source_texts[snapshot_id] = source_text
            sources.setdefault(snapshot_id, {'artifact': f'retrieval_sources/{snapshot_id}.txt',
                                         'origin_path': str(source_path), 'source_sha256': digest,
                                         'source_file_sha256': raw_digest,
                                         'role': 'resource'})
        else:
            raise ValueError('Retrieved evidence has an unknown source role')
        start, end, quote = hit.get('char_start'), hit.get('char_end'), hit.get('quote')
        if (type(start) is not int or type(end) is not int or not isinstance(quote, str)
                or not 0 <= start < end <= len(source_text) or source_text[start:end] != quote
                or hit.get('source_sha256') != digest or hit.get('text', quote) != quote):
            raise ValueError('Retrieved evidence quote, hash or character span differs from its source')
        evidence_id, score = hit.get('evidence_id'), hit.get('score')
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in seen_ids:
            raise ValueError('Retrieved evidence IDs must be nonempty and unique')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Retrieved evidence has a nonfinite or invalid ranking score')
        seen_ids.add(evidence_id)
        hit['snapshot_id'] = snapshot_id
        hit['citation_source'] = sources[snapshot_id]['artifact']
    payload['sources'] = sources
    payload['grounding_validation'] = 'source_hash_and_spans'
    return payload, source_texts


def analyze(input_path: Path, output_dir: Path, classifier: str = 'ensemble', *,
            retrieval: str = 'none', queries: list[str] | None = None, top_k: int = 3,
            resource_dir: Path | None = None, base_model_dir: Path | None = None,
            adapter_dir: Path | None = None, device: str = 'auto',
            rerank_model_dir: Path | None = None) -> Path:
    from extraction import extract_document
    from planning import generate_plan
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if not input_path.is_file() or input_path.suffix.lower() not in ('.md', '.txt', '.pdf'):
        raise AnalysisConfigurationError('Input must be one existing .md, .txt or .pdf file')
    if output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing reports are never overwritten')
    if classifier not in ('ensemble', 'rules'):
        raise AnalysisConfigurationError('classifier must be ensemble or rules')
    if retrieval not in ('none', 'bm25', 'dense', 'hybrid'):
        raise AnalysisConfigurationError('retrieval must be none, bm25, dense or hybrid')
    if type(top_k) is not int or top_k < 1:
        raise AnalysisConfigurationError('top_k must be a positive integer')
    if device not in ('auto', 'cpu', 'cuda'):
        raise AnalysisConfigurationError('retrieval device must be auto, cpu or cuda')
    if retrieval == 'none' and any(value is not None for value in
                                   (queries, resource_dir, base_model_dir, adapter_dir, rerank_model_dir)):
        raise AnalysisConfigurationError('Retrieval options require --retrieval bm25, dense or hybrid')
    if resource_dir is not None:
        resource_dir = Path(resource_dir).resolve()
        if not resource_dir.is_dir():
            raise AnalysisConfigurationError('--resource-dir must name an existing directory of local resources')
    if retrieval != 'none':
        queries = list(DEFAULT_RETRIEVAL_QUERIES) if queries is None else queries
        if (not isinstance(queries, list) or not queries
                or any(not isinstance(query, str) or not query.strip() for query in queries)):
            raise AnalysisConfigurationError('Supply at least one nonempty --query, or omit --query for the default evidence topics')
    if classifier == 'ensemble':
        available = doctor()['capabilities']['production_ensemble']
        if not available['available']:
            raise AnalysisConfigurationError(
                'Ensemble prerequisites are missing: model members=' + ','.join(available['missing_model_members'])
                + '; packages=' + ','.join(available['missing_packages'])
                + '. Run python psai.py doctor; install requirements.txt and provide the three local checkpoints, '
                'or use --classifier rules for offline rule extraction.')
    if input_path.suffix.lower() == '.pdf':
        from pdf_preprocessing import extract_pdf
        try:
            source = extract_pdf(input_path)
        except (ValueError, RuntimeError, OSError) as error:
            raise AnalysisConfigurationError(str(error)) from error
        text = source.pop('text')
    else:
        text = input_path.read_text(encoding='utf-8-sig')
        source = dict(source_path=str(input_path), method='local_text', pdf_visual_review_complete=False)
    if not text.strip():
        raise AnalysisConfigurationError('Input contains no text')
    record = extract_document(input_path.stem, text, use_model=classifier == 'ensemble')
    retrieved, retrieval_source_texts = None, {}
    if retrieval != 'none':
        from psai_retrieval import retrieve_document
        retrieved = retrieve_document(text, input_path.stem, queries=queries, top_k=top_k,
                                      mode=retrieval, resource_dir=resource_dir,
                                      base_model_dir=base_model_dir, adapter_dir=adapter_dir,
                                      device=device, rerank_model_dir=rerank_model_dir)
        retrieved, retrieval_source_texts = _verified_retrieval(retrieved, text, input_path, resource_dir)
        record['related_context'] = retrieved['hits']
    plan = generate_plan(record, source_text=text)
    # Validate every cited span against the exact text before exporting a report.
    by_id = {}
    for items in record['fields'].values():
        for item in items:
            start, end = item['char_start'], item['char_end']
            if item.get('verified') is not True or not 0 <= start < end <= len(text) or text[start:end] != item['quote']:
                raise ValueError('Extraction returned an invalid source citation')
            by_id[item['evidence_id']] = item
    for step in plan['steps']:
        if not any(step['description'] == by_id[eid]['value'].strip() for eid in step['evidence_ids']):
            raise ValueError('Plan description differs from its cited source text')
    report = '# Báo cáo nội dung PSAI\n\n'
    report += f'Tài liệu: {input_path.name}\n\n'
    report += 'Phương pháp: ' + ('ensemble PSAI hiện tại và quy tắc trích xuất' if classifier == 'ensemble' else 'quy tắc trích xuất; chế độ kiểm tra, không đại diện kết quả classifier') + '.\n\n'
    report += 'Nội dung dưới đây trích từ văn bản đầu vào. Nhãn và lựa chọn nội dung chưa được người kiểm định; vị trí trích dẫn tính trên [source.txt](source.txt), không phải tọa độ PDF.\n\n'
    for field, items in record['fields'].items():
        report += f'## {field}\n\n'
        if not items:
            report += 'Chưa tìm thấy nội dung đủ điều kiện.\n\n'
        for item in items:
            # Quotes, including embedded Markdown, are rendered as source text.
            quote = '\n'.join('> ' + line for line in item['quote'].splitlines())
            report += quote + f'\n\nBằng chứng: {item["evidence_id"]}; ký tự [{item["char_start"]}, {item["char_end"]}).\n\n'
    report += '## Các bước lấy trực tiếp từ nguồn\n\n'
    if not plan['steps']:
        report += 'Chưa có bước đủ điều kiện để lập kế hoạch từ nguồn.\n\n'
    for step in plan['steps']:
        report += f'{step["step"]}. ' + step['description'].replace('\n', ' ') + ' — ' + ', '.join(step['evidence_ids']) + '\n'
    if retrieved is not None:
        report += '\n## Bằng chứng liên quan được truy xuất\n\n'
        report += f'Phương pháp: {retrieval}; tối đa {top_k} đoạn sau khi hợp nhất các truy vấn. Điểm xếp hạng không phải xác suất đúng.\n\n'
        report += 'Truy vấn: ' + '; '.join(retrieved.get('queries', queries)) + '.\n\n'
        if not retrieved['hits']:
            report += 'Chưa tìm thấy bằng chứng phù hợp với truy vấn.\n\n'
        for hit in retrieved['hits']:
            quote = '\n'.join('> ' + line for line in hit['quote'].splitlines())
            report += quote + f'\n\nBằng chứng: {hit["evidence_id"]}; [nguồn]({hit["citation_source"]}); '
            report += f'ký tự [{hit["char_start"]}, {hit["char_end"]}); điểm {hit["score"]:.6g}.\n\n'
    report += '\n[Bản HTML có vị trí bằng chứng](report.html) · [Kết quả JSON](analysis.json)\n'
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=_sha(input_path),
                    source_text_sha256=hashlib.sha256(text.encode()).hexdigest(), classifier=classifier,
                    production_model_changed=False, source=source,
                    code_sha256={name: _sha(ROOT / name) for name in ['psai.py', 'psai_reports.py', 'extraction.py', 'planning.py', 'ensemble_classifier.py']},
                    human_content_review_complete=False, pipeline_quality_score=None)
    metadata['report_title'] = 'Báo cáo nội dung PSAI'
    metadata['retrieval'] = {'mode': retrieval, 'queries': queries if retrieved is not None else [],
                             'top_k': top_k if retrieved is not None else None,
                             'source_spans_verified': retrieved is not None,
                             'provenance': retrieved.get('provenance') if retrieved is not None else None}
    if retrieved is not None:
        metadata['code_sha256'].update({str(path.relative_to(ROOT)): _sha(path)
                                       for path in sorted((ROOT / 'psai_retrieval').glob('*.py'))})
    if classifier == 'ensemble':
        metadata['models'] = {name: {'config_sha256': _sha(ROOT / 'models' / name / 'inference_config.json'),
                                    'weight_sha256': _sha(ROOT / 'models' / name / 'model.safetensors')}
                              for name in ['final_s11', 'final_s22', 'final_s33']}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.psai-', dir=output_dir.parent))
    try:
        from psai_reports import render_report_html
        html_retrieval = {**retrieved, 'source_texts': retrieval_source_texts} if retrieved is not None else None
        html_report = render_report_html(record, plan, text, metadata, retrieval=html_retrieval)
        (stage / 'source.txt').write_text(text, encoding='utf-8', newline='')
        (stage / 'report.md').write_text(report, encoding='utf-8')
        (stage / 'report.html').write_text(html_report, encoding='utf-8')
        if retrieved is not None:
            for digest, details in retrieved['sources'].items():
                artifact = stage / details['artifact']
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(retrieval_source_texts[digest], encoding='utf-8', newline='')
            (stage / 'retrieval.json').write_text(json.dumps(retrieved, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        analysis = {'schema_version': 1, 'document_id': input_path.stem, 'source': source,
                    'extraction': record, 'plan': plan, 'retrieval': retrieved, 'metadata': metadata}
        (stage / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        for filename, value in [('extraction.json', record), ('plan.json', plan), ('manifest.json', metadata)]:
            (stage / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        stage.rename(output_dir)
    finally:
        if stage.exists():
            if stage.resolve().parent != output_dir.parent or not stage.name.startswith('.psai-'):
                raise RuntimeError('Refusing cleanup outside the verified staging directory')
            shutil.rmtree(stage)
    return output_dir / 'report.md'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('doctor', help='Check environment and model files without loading models')
    check.add_argument('--output', type=Path)
    run = commands.add_parser('analyze', help='Extract one document and write an evidence-cited local report')
    run.add_argument('--input', required=True, type=Path)
    run.add_argument('--output', required=True, type=Path)
    run.add_argument('--classifier', choices=['ensemble', 'rules'], default='ensemble')
    run.add_argument('--retrieval', choices=['none', 'bm25', 'dense', 'hybrid'], default='none',
                     help='Optional local evidence search; bm25 works without model packages or network access')
    run.add_argument('--query', action='append', dest='queries',
                     help='Evidence query; repeat for multiple topics. Default: objectives, requirements, work and resources')
    run.add_argument('--top-k', type=int, default=3, help='Maximum unique results after combining queries (default: 3)')
    run.add_argument('--resource-dir', type=Path, help='Optional directory of local Markdown/text resources')
    run.add_argument('--base-model-dir', type=Path, help='Complete local embedding model directory for dense/hybrid search')
    run.add_argument('--adapter-dir', type=Path, help='Optional local retriever LoRA adapter; classifier selection stays unchanged')
    run.add_argument('--rerank-model-dir', type=Path, help='Optional complete local reranker directory')
    run.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Retrieval inference device (default: auto)')
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        result = json.dumps(doctor(), ensure_ascii=False, indent=2)
        print(result)
        if args.output:
            if args.output.exists():
                raise FileExistsError('Diagnostic output already exists')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding='utf-8')
    else:
        try:
            print(analyze(args.input, args.output, args.classifier, retrieval=args.retrieval,
                          queries=args.queries, top_k=args.top_k, resource_dir=args.resource_dir,
                          base_model_dir=args.base_model_dir, adapter_dir=args.adapter_dir,
                          device=args.device, rerank_model_dir=args.rerank_model_dir))
        except (AnalysisConfigurationError, FileExistsError) as error:
            parser.exit(2, f'PSAI: {error}\n')


if __name__ == '__main__':
    main()
