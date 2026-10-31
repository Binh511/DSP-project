"""Build a shareable evidence report from saved artifacts; no training/inference."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import shutil

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports' / 'model_evidence'
LABELS = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence, scores, details, history = {}, [], [], []

    def source(relative):
        path = ROOT / relative
        evidence[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = OUT / 'evidence' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        return path

    def add(model, dataset, n, accuracy, f1, origin, note, per_class=None):
        scores.append(dict(model=model, dataset=dataset, n=n, accuracy=accuracy,
                           macro_f1=f1, source=origin, note=note))
        if per_class:
            details.extend(dict(model=model, dataset=dataset, label=k, f1=v)
                           for k, v in per_class.items())

    for filename, dataset in [('test_predictions.csv', 'Fold0 test'),
                              ('human_holdout_predictions.csv', 'Human holdout')]:
        rel = 'runs/fair_fold0/' + filename
        df = pd.read_csv(source(rel))
        for col, model in [('baseline_prediction', 'fair_fold0_baseline'),
                           ('human_prediction', 'fair_fold0_human_once')]:
            if not df.label.isin(LABELS).all() or not df[col].isin(LABELS).all():
                raise ValueError('Invalid labels in ' + rel)
            report = classification_report(df.label, df[col], labels=LABELS,
                                           output_dict=True, zero_division=0)
            add(model, dataset, len(df), accuracy_score(df.label, df[col]),
                report['macro avg']['f1-score'], rel,
                'Recomputed from saved predictions; document split with exact-text purge. '
                'Human holdout is a small selected sample, not a random population test.',
                {lab: report[lab]['f1-score'] for lab in LABELS})
            pd.DataFrame(confusion_matrix(df.label, df[col], labels=LABELS),
                         index=LABELS, columns=LABELS).to_csv(
                OUT / f'confusion_{model}_{dataset.replace(" ", "_")}.csv', encoding='utf-8-sig')

    rel = 'models/validation_selected_f0/experiment.json'
    experiment = json.loads(source(rel).read_text(encoding='utf-8'))
    for key, value in experiment['results'].items():
        add('validation_selected_f0', 'Fold0 test' if key == 'test_fold0' else 'Human holdout',
            value['n'], value['accuracy'], value['macro_f1'], rel,
            'Saved evaluation summary; training purged 207 additional rows including context. '
            'Training data differs from fair_fold0; exploratory comparison.', value['per_label_f1'])

    rel = 'runs/human_experiment/cv_summary_targeted.json'
    summary = json.loads(source(rel).read_text(encoding='utf-8'))
    for fold in summary['test_fold_results']:
        for key, model in [('old', f'kfold_f{fold["fold"]}'),
                           ('new', f'human_targeted_f{fold["fold"]}')]:
            v = fold[key]
            if v:
                add(model, f'Historical fold{fold["fold"]}', v['n'], v['accuracy'],
                    v['macro_f1'], rel,
                    'Historical diagnostic only: old checkpoint split not proven; sentence overlap '
                    'audited. Do not treat this as a verified improvement comparison.', v['per_label_f1'])
    for key, model in [('human_holdout_old', 'kfold_f0'),
                       ('human_holdout_new', 'human_targeted_f0')]:
        v = summary.get(key)
        if v:
            add(model, 'Historical human holdout', v['n'], v['accuracy'], v['macro_f1'], rel,
                'Small selected human sample; historical saved summary.', v['per_label_f1'])

    for rel, model, n in [
        ('models/validation_selected_f0/training_state.json', 'validation_selected_f0', 1317),
        ('runs/psai_cls/checkpoint-2094/trainer_state.json', 'psai_cls training run (export mapping unknown)', None),
    ]:
        state = json.loads(source(rel).read_text(encoding='utf-8'))
        for entry in state.get('log_history', []):
            if 'eval_macro_f1' in entry:
                history.append(dict(model=model, epoch=entry['epoch'], step=entry['step'],
                                    validation_macro_f1=entry['eval_macro_f1'],
                                    validation_loss=entry.get('eval_loss'), source=rel))
        add(model, 'Validation best (selection)', n, None, state['best_metric'], rel,
            'Used to select a checkpoint; not test performance. '
            'Older training run cannot be mapped to a named model export from this log alone.')

    for rel in ['runs/fair_fold0/frozen_split.json',
                'runs/human_experiment/split_manifest.json',
                'runs/human_experiment/fold0_overlap_audit.csv',
                'runs/human_experiment/cv_fold0_targeted.csv']:
        source(rel)
    inventory = []
    for folder in sorted((ROOT / 'models').iterdir()):
        if not folder.is_dir():
            continue
        config_path = folder / 'inference_config.json'
        cfg = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        if config_path.exists():
            source(config_path.relative_to(ROOT).as_posix())
        weights = list(folder.glob('*.safetensors')) + list(folder.glob('pytorch_model*.bin'))
        inventory.append(dict(model=folder.name, path=folder.relative_to(ROOT).as_posix(),
                              has_weights=bool(weights), weight_bytes=sum(p.stat().st_size for p in weights),
                              ctx_version=cfg.get('ctx_version'), max_len=cfg.get('max_len'),
                              other_bias=cfg.get('other_bias'),
                              saved_metrics=any(s['model'] == folder.name for s in scores)))
    for name, rows in [('inventory', inventory), ('metrics', scores),
                        ('per_class_f1', details), ('validation_history', history)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    payload = dict(generated_utc=datetime.now(timezone.utc).isoformat(), inventory=inventory,
                   metrics=scores, per_class_f1=details, validation_history=history,
                   evidence_sha256=evidence,
                   limitations=['Inventory is of current model folders, not a complete training history.',
                                'Missing metrics are unknown, not zero.',
                                'Repeatedly inspected test/holdout sets support exploratory analysis; '
                                'final claims require fresh labeled independent documents.',
                                'Evidence hashes preserve the copied source files, not proof of training time '
                                'or cryptographic linkage of predictions to model weights.'])
    (OUT / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    metric_df = pd.DataFrame(scores)
    for column in ['accuracy', 'macro_f1']:
        metric_df[column] = metric_df[column].map(lambda x: f'{x:.4f}' if pd.notna(x) else '—')
    # Standalone SVG learning curve, avoiding any browser/CDN dependency.
    points = [r for r in history if r['model'] == 'validation_selected_f0']
    xy = ' '.join(f'{60+(r["epoch"]-1)*120},{230-r["validation_macro_f1"]*260}' for r in points)
    labels = ''.join(f'<text x="{60+(r["epoch"]-1)*120}" y="260">Epoch {r["epoch"]:g}</text>' for r in points)
    dots = ''.join(f'<text x="{60+(r["epoch"]-1)*120}" y="{215-r["validation_macro_f1"]*260}">{r["validation_macro_f1"]:.4f}</text>' for r in points)
    document = '''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Báo cáo bằng chứng huấn luyện PSAI</title><style>
body{font:15px system-ui;max-width:1450px;margin:40px auto;padding:0 24px;color:#172b42}
table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}th,td{border:1px solid #ccd5df;padding:8px;text-align:left}th{background:#edf3fa}h1,h2{color:#164c7d}.note{background:#fff4d6;padding:16px}svg{max-width:100%;font-size:12px} @media print{body{margin:0}table{font-size:10px}tr{break-inside:avoid}}</style>
<h1>Báo cáo bằng chứng huấn luyện mô hình PSAI</h1>'''
    document += f'<p>Thời điểm tổng hợp UTC: {html.escape(payload["generated_utc"])}. Có {len(inventory)} thư mục model.</p>'
    document += '<div class="note">Báo cáo tổng hợp các bằng chứng đã lưu, không chạy đánh giá mới. Điểm validation dùng chọn checkpoint phải tách khỏi điểm test. Các kết quả lịch sử chưa xác minh split không dùng để tuyên bố độ chính xác cuối cùng.</div>'
    document += '<h2>1. Chỉ số có nguồn bằng chứng</h2>' + metric_df.to_html(index=False, escape=True)
    document += '<h2>2. Validation theo epoch — validation_selected_f0</h2>'
    document += f'<svg viewBox="0 0 680 285" role="img" aria-label="Validation macro F1 by epoch"><path d="M40 20 V240 H650" stroke="#8393a6" fill="none"/><polyline points="{xy}" fill="none" stroke="#176abc" stroke-width="3"/>{labels}{dots}</svg>'
    document += '<h2>3. Danh sách model đang có</h2>' + pd.DataFrame(inventory).to_html(index=False, escape=True)
    document += '<h2>4. F1 từng lớp</h2>' + pd.DataFrame(details).to_html(index=False, escape=True, float_format=lambda x: f'{x:.4f}')
    document += '<h2>5. Cách diễn giải và kiểm chứng</h2><ul><li>Thí nghiệm fair_fold0: baseline 0,6088 → bổ sung nhãn thủ công một lần 0,6298 trên cùng 1.370 câu test.</li><li>Thí nghiệm mới đạt test F1 0,6014; holdout F1 0,6373. Chưa chứng minh cải thiện tổng thể. Dữ liệu train khác thí nghiệm fair_fold0.</li><li>Holdout 66 câu được chọn từ mẫu rà soát; không đại diện ngẫu nhiên cho toàn bộ tài liệu.</li><li>Không có chỉ số cho một model nghĩa là chưa tìm thấy bằng chứng đã lưu, không có nghĩa model đạt 0.</li><li>Thư mục evidence chứa bản sao nguồn. summary.json chứa SHA256 của từng nguồn. Không nạp training_args.bin bằng pickle để làm báo cáo.</li><li>Chưa có bảng kết quả đủ 5 fold để tính mean ± std; không suy diễn từ fold0.</li><li>Test đã được xem nhiều lần; cần tập tài liệu mới được gán nhãn độc lập cho kết luận cuối.</li></ul>'
    document += '<p>Có thể mở báo cáo bằng trình duyệt và chọn Print → Save as PDF. CSV mở bằng Excel. Gói này không chứa trọng số hay toàn bộ dataset.</p></html>'
    (OUT / 'bao_cao_mo_hinh.html').write_text(document, encoding='utf-8')
    archive = shutil.make_archive(str(OUT.parent / 'model_evidence'), 'zip', root_dir=OUT)
    print(f'Models: {len(inventory)}; metric rows: {len(scores)}; evidence files: {len(evidence)}')
    print(OUT / 'bao_cao_mo_hinh.html')
    print(archive)


if __name__ == '__main__':
    main()
