"""Replay three existing classifier checkpoints on frozen validation, no training."""
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from build_classifier_report import table, export_docx
from replay_validation import replay
from study_data import ROOT, OUT, read_json, write_json, sha

DEST = OUT / 'studies/classifier_validation_replay_v1'
OLD = OUT / 'studies/classifier_stability_v1'
SOURCES = {
    42: (OUT / 'studies/user_labels_v1/experiments/baseline_original_labels', ROOT / 'models/controlled_study/user_labels_v1/baseline_original_labels'),
    123: (OLD / 'experiments/seed_123', ROOT / 'models/controlled_study/classifier_stability_v1/seed_123'),
    2026: (OLD / 'experiments/seed_2026', ROOT / 'models/controlled_study/classifier_stability_v1/seed_2026'),
}


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    files = ['classifier_validation_replay.py', 'replay_validation.py', 'final_test_integrity.py',
             'study_evaluation.py', 'study_inputs.py', 'train_human_cv.py', 'study_final_test.py']
    plan = dict(purpose='Same existing checkpoint and original frozen validation; bind one inference protocol',
                train=False, test_or_final=False, context_v2_used=False, model_selected=False,
                seeds=list(SOURCES), code_sha256={name: sha(ROOT / name) for name in files},
                historical_metric_sha256={str(seed): sha(folder / 'validation_metrics.json') for seed, (folder, _) in SOURCES.items()})
    path = DEST / 'replay_plan.json'
    if path.exists() and read_json(path) != plan:
        raise ValueError('Replay inputs/code changed; create a new replay version')
    if not path.exists():
        write_json(path, plan)


def summarize():
    comparisons, classes, proofs = [], [], []
    for seed, (folder, _) in SOURCES.items():
        run = DEST / f'seed_{seed}'
        if not (run / 'validation_metrics.json').is_file():
            continue
        before, after = read_json(folder / 'validation_metrics.json'), read_json(run / 'validation_metrics.json')
        first = pd.read_csv(folder / 'validation_predictions.csv', keep_default_na=False, float_precision='round_trip')
        second = pd.read_csv(run / 'validation_predictions.csv', keep_default_na=False, float_precision='round_trip')
        for column in ['document_id', 'text', 'label', 'true_label', 'section', 'prev_text', 'next_text', 'pos']:
            if not first[column].equals(second[column]):
                raise ValueError(f'Replay changed frozen rows/context: {seed} {column}')
        if before['model_weight_sha256'] != after['model_weight_sha256']:
            raise ValueError('Replay used different weights')
        probability_cols = [c for c in first if c.startswith('prob_')]
        comparisons.append(dict(seed=seed, original_macro_f1=before['macro_f1'], replay_macro_f1=after['macro_f1'],
                                macro_f1_delta=after['macro_f1'] - before['macro_f1'],
                                prediction_changes=int(first.predicted_label.ne(second.predicted_label).sum()),
                                max_absolute_probability_difference=float(np.abs(first[probability_cols].to_numpy() - second[probability_cols].to_numpy()).max())))
        for label, value in after['per_class'].items():
            classes.append(dict(seed=seed, label=label, **value))
        proofs.append(dict(seed=seed, validation_sha256=after['validation_sha256'], train_sha256=after['train_sha256'],
                           runtime=after['validation_inference_manifest']['runtime'],
                           code=after['validation_inference_manifest']['implementation']))
    if not comparisons:
        return
    for proof in proofs[1:]:
        if any(proof[key] != proofs[0][key] for key in ['validation_sha256', 'train_sha256', 'runtime', 'code']):
            raise ValueError('Seeds were not evaluated with the same data/runtime/implementation')
    pd.DataFrame(comparisons).to_csv(DEST / 'comparison.csv', index=False, encoding='utf-8-sig')
    class_frame = pd.DataFrame(classes)
    class_frame.to_csv(DEST / 'per_class_metrics.csv', index=False, encoding='utf-8-sig')
    values = [row['replay_macro_f1'] for row in comparisons]
    stats = dict(state='complete' if len(values) == 3 else 'pending', n=len(values), seeds=[row['seed'] for row in comparisons],
                 mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                 train_or_test_used=False, final_model_selected=False, context_v2_used=False,
                 label_review_complete=False, runtime=proofs[0]['runtime'])
    write_json(DEST / 'statistics.json', stats)
    report = '# Classifier PSAI — đánh giá lại validation theo một giao thức\n\n'
    report += 'Dùng lại đúng weights và 1.317 câu validation với nhãn/context frozen ban đầu. Không train, không dùng test/final test, không áp dụng context v2 và không thay ensemble production. Đây là validation đã dùng trong phát triển, không phải đánh giá độc lập cuối cùng.\n\n'
    report += table(['Seed', 'Macro-F1 cũ', 'Macro-F1 replay', 'Chênh lệch', 'Câu đổi dự đoán'], [[r['seed'], f"{r['original_macro_f1']:.6f}", f"{r['replay_macro_f1']:.6f}", f"{r['macro_f1_delta']:+.6f}", r['prediction_changes']] for r in comparisons])
    report += f'\n\nTrung bình {stats["mean"]:.6f}; SD mẫu: ' + (f'{stats["sample_std"]:.6f}' if stats['sample_std'] is not None else 'chưa đủ lượt') + f'; n={len(values)}.\n\n'
    report += table(['Seed', 'Lớp', 'Precision', 'Recall', 'F1', 'Support'], [[int(r['seed']), r['label'], f'{r["precision"]:.6f}', f'{r["recall"]:.6f}', f'{r["f1-score"]:.6f}', int(r['support'])] for _, r in class_frame.iterrows()])
    report += '\n\nGiao thức: saved local tokenizer; study-v1/full, text600, max_length256, longest_first, dynamic padding, batch8, model eval + inference_mode, FP32 parameters, không autocast, OTHER bias0, năm nhãn cố định. Runtime/TF32/matmul/determinism và model/tokenizer/config/code hashes được lưu cùng mỗi validation_metrics.json. Không cam kết tái lập bitwise trên môi trường khác.\n\n'
    report += 'Manifest của lượt replay giải quyết thiếu ràng buộc kỹ thuật trong điểm cũ. Điều này chưa hoàn thành phân xử nhãn và chưa tự chốt seed123. Context v2 chỉ là dữ liệu chuẩn bị riêng, chưa được train/evaluate. Cần phân xử nhãn người và xác nhận tập test độc lập trước quyết định cuối.\n\n'
    report += 'Bằng chứng: [bảng so sánh](comparison.csv), [chỉ số từng lớp](per_class_metrics.csv), [runtime và thống kê](statistics.json).\n\n'
    for seed in stats['seeds']:
        report += f'- Seed {seed}: [metrics và manifest](seed_{seed}/validation_metrics.json), [predictions](seed_{seed}/validation_predictions.csv), [confusion](seed_{seed}/validation_confusion.csv).\n'
    (DEST / 'KET_QUA_VALIDATION_THONG_NHAT.md').write_text(report, encoding='utf-8')
    export_docx(report, DEST / 'KET_QUA_VALIDATION_THONG_NHAT.docx')


def run_all():
    prepare()
    lock = DEST / 'replay.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = dict(state='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), completed=[], current=None)
    try:
        for seed, (folder, model) in SOURCES.items():
            output = DEST / f'seed_{seed}'
            state['current'] = seed
            write_json(DEST / 'status.json', state)
            if output.exists():
                raise FileExistsError(f'Replay output exists: {output}; inspect before creating another version')
            print(f'Seed {seed}: validation replay starting', flush=True)
            replay(folder, model, output)
            state['completed'].append(seed)
            summarize()
            print(f'Seed {seed}: validation replay complete', flush=True)
        state.update(state='complete', current=None)
    except BaseException as exc:
        state.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        write_json(DEST / 'status.json', state)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run_all()
