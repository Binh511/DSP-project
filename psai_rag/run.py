"""Run the isolated PSAI retrieval pilot; no classifier or production mutation.

Each GPU phase runs in a separate process to release the complete CUDA context.
Use ``python -m psai_rag.run --status`` to inspect the actual durable state.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'reports/rag_studies/retrieval_genq_lora_v1'
BASE = Path('C:/Users/Admin/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181')
OUTPUT = ROOT / 'models/rag/retrieval_genq_lora_v1'
PHASES = ('prepare', 'generate', 'base', 'train', 'adapter', 'report')
CONFIG = {
    'study_id': 'retrieval_genq_lora_v1', 'seed': 42,
    'scope': 'within_document_evidence_retrieval',
    'supervision': 'synthetic_queries_teacher_filtered_not_human_gold',
    'data': {'max_tokens': 254, 'train_ratio': .8, 'val_ratio': .1},
    'generation': {'seed': 42},
    'training': {'seed': 42, 'epochs': 1, 'learning_rate': 2e-5,
                 'logical_batch_size': 16, 'mini_batch_size': 1, 'max_length': 256,
                 'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': .05,
                 'output_dir': str(OUTPUT)},
    'evaluation': {'eval_batch_size': 1, 'max_length': 256, 'device': 'cuda', 'rrf_k': 60},
    'selection': {'scope': 'within_doc', 'metric': 'MRR@10',
                  'rule': 'Keep base unless adapter strictly improves pseudo-validation MRR@10; human validation required before deployment.'},
    'final_test_evaluation_enabled': False, 'production_replacement_enabled': False,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def status():
    result = {'pipeline': load(STUDY / 'run_status.json') if (STUDY / 'run_status.json').exists() else {'status': 'not_started'}}
    for key, path in [('generation', STUDY / 'generation_status.json'),
                      ('training', OUTPUT / 'training_receipt.json')]:
        if path.exists():
            receipt = load(path)
            result[key] = {name: receipt[name] for name in ('status', 'phase', 'global_step', 'planned_steps',
                           'last_step', 'counts', 'error', 'first_step_proof', 'adapter_dir',
                           'completed_candidates', 'total_candidates', 'completed_queries',
                           'total_queries', 'coverage') if name in receipt}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def check_preparation():
    manifest_path = STUDY / 'split_manifest.json'
    if not manifest_path.exists():
        return False
    manifest = load(manifest_path)
    for name, digest in manifest['artifact_sha256'].items():
        if sha(STUDY / name) != digest:
            raise ValueError('Prepared input hash changed: ' + name)
    for row in load(STUDY / 'source_manifest.json')['sources']:
        if sha(Path(row['source_file'])) != row['source_sha256']:
            raise ValueError('Source changed after preparation: ' + row['source_file'])
    return True


def review_packets():
    pairs = [json.loads(line) for line in (STUDY / 'validation_pairs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    with (STUDY / 'HUMAN_REVIEW_VALIDATION.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['query_id', 'doc_id', 'query', 'positive_id', 'positive',
                                                   'negative_id', 'negative', 'query_is_realistic',
                                                   'positive_is_relevant', 'negative_is_irrelevant', 'reviewer', 'notes'])
        writer.writeheader()
        for row in pairs:
            writer.writerow({key: row[key] for key in writer.fieldnames if key in row})
    corpus = [json.loads(line) for line in (STUDY / 'corpus.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = {}
    for row in corpus:
        if row['split'] == 'test' and row.get('token_count', 0) >= 32:
            selected.setdefault(row['doc_id'], row)
    with (STUDY / 'HUMAN_FINAL_TEST_TEMPLATE.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['doc_id', 'source_group_id', 'source_file', 'source_start_line', 'source_end_line',
                  'example_chunk_id', 'example_text', 'human_query', 'all_relevant_chunk_ids',
                  'reviewer_1', 'reviewer_2', 'agreement_notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected.values():
            writer.writerow({**{key: row.get(key, '') for key in fields if key in row},
                             'example_chunk_id': row['chunk_id'], 'example_text': row['text']})


def report():
    base, adapter = load(STUDY / 'base_metrics.json'), load(STUDY / 'adapter_metrics.json')
    training = load(OUTPUT / 'training_receipt.json')
    main = CONFIG['selection']
    before = base['metrics'][main['scope']]['dense'][main['metric']]
    after = adapter['metrics'][main['scope']]['dense'][main['metric']]
    candidate = 'adapter' if after > before else 'base'
    decision = {'created_utc': now(), 'provisional_research_candidate': candidate,
                'selection_protocol': main, 'base_score': before, 'adapter_score': after,
                'delta': after - before, 'labels_are_human_ground_truth': False,
                'production_changed': False, 'final_test_evaluated': False,
                'adapter_dir': training['adapter_dir'],
                'validation_pairs_sha256': sha(STUDY / 'validation_pairs.jsonl')}
    write_json(STUDY / 'selection_decision.json', decision)
    review_packets()
    lines = ['# Thí nghiệm retriever PSAI — GenQ + LoRA, seed 42', '',
             '**Đây là pilot dùng query và nhãn tham chiếu tự động, chưa chứng minh chất lượng với query thật.**', '',
             'Phạm vi: tìm bằng chứng trong tài liệu đã nhập. Tìm tài nguyên bên ngoài và LLM lập kế hoạch chưa được đánh giá ở đây.', '',
             f"Huấn luyện đã hoàn thành {training['global_step']}/{training['planned_steps']} bước optimizer. Adapter: `{training['adapter_dir']}`.", '',
             f"Có {base['validation_queries']} query validation; các nguồn được tách theo nhóm tài liệu trước khi sinh query.", '',
             '| Phạm vi | Phương án | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope in ['within_doc', 'global_validation']:
        for name, document, method in [('BM25', base, 'bm25'), ('BGE-M3 gốc', base, 'dense'),
                                        ('BM25 + BGE-M3 gốc (RRF)', base, 'bm25_dense_rrf'),
                                        ('BGE-M3 + LoRA', adapter, 'dense'),
                                        ('BM25 + LoRA (RRF)', adapter, 'bm25_dense_rrf')]:
            values = document['metrics'][scope][method]
            lines.append('| ' + ' | '.join([scope, name] + [f'{values[key]:.4f}' for key in
                         ('Recall@1', 'Recall@5', 'Recall@10', 'MRR@10', 'nDCG@10')]) + ' |')
    lines += ['', f"Theo giao thức đã khai báo, ứng viên nghiên cứu tạm thời: **{candidate}**. "
              f"Chênh lệch MRR@10 dense trong tài liệu: {after-before:+.4f}. Chưa thay mô hình/index đang triển khai.", '',
              '## Giới hạn cần xử lý', '',
              '- Positive là đoạn dùng để sinh query; các đoạn liên quan khác có thể chưa được gán nhãn.',
              '- Bộ sinh query và teacher thiên về tiếng Anh; chưa có bằng chứng chất lượng truy xuất tiếng Việt.',
              '- Ngưỡng teacher chỉ lọc tín hiệu yếu, không thay thế gán nhãn độc lập của người.',
              '- Kết quả truy xuất chưa đo OCR/PDF, tính đầy đủ bằng chứng hoặc độ đúng báo cáo/kế hoạch.',
              '- Final test chưa có nhãn người và chưa được chấm điểm.', '',
              '## Bằng chứng', '',
              '- [Dữ liệu, nhóm nguồn và hash](./split_manifest.json)',
              '- [Quy tắc và phiên bản sinh query](./generation_report.json)',
              '- [Baseline và dự đoán](./base_metrics.json)',
              '- [Adapter và dự đoán](./adapter_metrics.json)',
              '- [Quyết định chọn ứng viên tạm thời](./selection_decision.json)',
              '- [Phiếu người kiểm tra validation](./HUMAN_REVIEW_VALIDATION.csv)',
              '- [Mẫu xây final test độc lập](./HUMAN_FINAL_TEST_TEMPLATE.csv)', '',
              'Ưu tiên tiếp theo: kiểm tra query thực tế và relevance của người, bổ sung qrels nhiều đoạn liên quan, '
              'đánh giá ứng viên đã đóng băng trên final test độc lập, sau đó mới cân nhắc tích hợp.', '']
    (STUDY / 'KET_QUA_RAG_PILOT.md').write_text('\n'.join(lines), encoding='utf-8')


def phase(name: str, resume: bool):
    config = load(STUDY / 'run_config.json')
    if name == 'prepare':
        if not check_preparation():
            from .data import prepare_dataset
            result = prepare_dataset(ROOT, STUDY, BASE / 'tokenizer.json', seed=config['seed'], **config['data'])
            print(json.dumps(result['counts']), flush=True)
    elif name == 'generate':
        from .generate_queries import generate_pairs
        result = generate_pairs(ROOT, STUDY, load(STUDY / 'auxiliary_models.json'), config['generation'])
        print(json.dumps(result.get('counts', {}), ensure_ascii=False), flush=True)
    elif name in ('base', 'adapter'):
        from .evaluate import evaluate_retriever
        adapter_dir = Path(load(OUTPUT / 'training_receipt.json')['adapter_dir']) if name == 'adapter' else None
        result = evaluate_retriever(ROOT, STUDY, BASE, adapter_dir, config['evaluation'])
        print(json.dumps(result['metrics'], ensure_ascii=False), flush=True)
    elif name == 'train':
        from .train import train_adapter
        settings = dict(config['training'])
        if resume:
            settings['resume'] = True
        train_adapter(ROOT, STUDY, BASE, settings)
    elif name == 'report':
        report()


def run_all(resume: bool):
    STUDY.mkdir(parents=True, exist_ok=True)
    previous = load(STUDY / 'run_status.json') if (STUDY / 'run_status.json').exists() else {}
    if previous and not resume:
        raise ValueError('Study already has a run status; inspect --status and use --resume instead of starting a duplicate run.')
    config_path = STUDY / 'run_config.json'
    if config_path.exists():
        if load(config_path) != CONFIG:
            raise ValueError('Frozen run config differs; create a new study instead of overwriting it.')
    else:
        write_json(config_path, CONFIG)
    lock = STUDY / 'run.lock'
    # An exclusive lock prevents concurrent GPU jobs. Stale locks require explicit inspection/removal.
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'created_utc': now()}, stream)
    completed = list(previous.get('completed_phases', [])) if resume else []
    state = {'status': 'running', 'started_utc': now(), 'pid': os.getpid(),
             'completed_phases': completed, 'scope': CONFIG['scope'],
             'labels_are_human_ground_truth': False, 'production_changed': False}
    awake = False
    try:
        check_preparation()
        if os.name == 'nt':
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        for name in PHASES:
            if name in completed:
                continue
            state.update({'phase': name, 'phase_started_utc': now(), 'updated_utc': now()})
            write_json(STUDY / 'run_status.json', state)
            print(json.dumps({'phase': name, 'status': 'starting', 'time': now()}), flush=True)
            command = [sys.executable, '-u', '-m', 'psai_rag.run', '--phase', name]
            if resume and name == 'train' and (OUTPUT / 'latest_checkpoint.json').exists():
                command.append('--resume')
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode:
                raise RuntimeError(f'Phase {name} failed with process exit code {result.returncode}; inspect stderr.log.')
            completed.append(name)
            state.update({'completed_phases': completed, 'updated_utc': now()})
            write_json(STUDY / 'run_status.json', state)
        state.update({'status': 'completed', 'completed_utc': now()})
    except BaseException as error:
        state.update({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      'error': str(error), 'traceback': traceback.format_exc(), 'updated_utc': now()})
        raise
    finally:
        write_json(STUDY / 'run_status.json', state)
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', str(ROOT / 'models/rag/_hf_cache'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONUNBUFFERED'] = '1'
    if args.status:
        status()
    elif args.phase:
        phase(args.phase, args.resume)
    else:
        run_all(args.resume)


if __name__ == '__main__':
    main()
