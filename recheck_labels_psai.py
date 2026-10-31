"""Rà soát nhãn PSAI trên kết quả dự đoán đã lưu, KHÔNG train hay sửa dataset gốc.

Chép recheck_labels_psai.py và PSAI_7_cau_de_xuat_doi_nhan.csv vào thư mục
D:\\DSP_PROJECT\\DSP_PROJECT rồi chạy:
    python recheck_labels_psai.py

Các kết quả được ghi trong runs/fair_fold0/, không sửa test_predictions.csv.
Chỉ là PHÂN TÍCH LỖI sau khi đã xem test; không phải điểm test độc lập.
"""
import csv
import re
from pathlib import Path
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
CORRECTIONS = ROOT / 'PSAI_7_cau_de_xuat_doi_nhan.csv'
PREDICTIONS = ROOT / 'runs' / 'fair_fold0' / 'test_predictions.csv'
OUTPUT = ROOT / 'runs' / 'fair_fold0' / 'test_predictions_review7.csv'
CHANGE_LOG = ROOT / 'runs' / 'fair_fold0' / 'review7_matched_rows.csv'
LABELS = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']


def read_csv(p):
    if not p.exists():
        raise SystemExit(f'KHÔNG TÌM THẤY: {p}')
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def key(doc, txt):
    return str(doc).strip(), re.sub(r'\s+', ' ', str(txt)).strip()


def show_metrics(title, rows, gold_column):
    y_true = [r[gold_column] for r in rows]
    print(f'\n{title} (N={len(rows)}):')
    for variant, pred_col in [('BASELINE', 'baseline_prediction'), ('HUMAN', 'human_prediction')]:
        y_pred = [r[pred_col] for r in rows]
        print(f'  {variant}: Accuracy={accuracy_score(y_true, y_pred):.4f} | '
              f'Macro-F1={f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0):.4f}')
        if gold_column == 'reviewed_label':
            print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0, digits=3))


def main():
    changes = read_csv(CORRECTIONS)
    predictions = read_csv(PREDICTIONS)
    if len(changes) != 7:
        raise SystemExit(f'Cần đúng 7 quyết định đã rà soát, hiện thấy {len(changes)}. Không sửa file.')
    if len(predictions) != 1370:
        raise SystemExit(f'Fold0 dự kiến có 1370 câu, hiện thấy {len(predictions)}. Dừng để kiểm tra.')
    required_pred = {'document_id', 'text', 'label', 'baseline_prediction', 'human_prediction'}
    required_correction = {'document_id', 'text', 'nhan_cu', 'nhan_moi'}
    if not required_pred.issubset(predictions[0]) or not required_correction.issubset(changes[0]):
        raise SystemExit('Tên cột CSV không giống mẫu đã dùng trước đó. Dừng để kiểm tra.')
    corrections = {}
    for r in changes:
        k = key(r['document_id'], r['text'])
        if k in corrections:
            raise SystemExit(f'Trùng (document_id, text) trong file sửa: {k}')
        if r['nhan_cu'] not in LABELS or r['nhan_moi'] not in LABELS:
            raise SystemExit(f'Nhãn không hợp lệ: {k}')
        corrections[k] = r

    matched = Counter()
    reviewed = []
    audit_log = []
    for r in predictions:
        new = dict(r)
        new['old_label'] = r['label']
        new['reviewed_label'] = r['label']
        new['reviewed_label_changed'] = 'False'
        k = key(r['document_id'], r['text'])
        if k in corrections:
            c = corrections[k]
            if r['label'] != c['nhan_cu']:
                raise SystemExit(f'Nhãn cũ không khớp! {k}: test={r["label"]}, audit={c["nhan_cu"]}. Không lưu gì.')
            new['reviewed_label'] = c['nhan_moi']
            new['reviewed_label_changed'] = 'True'
            matched[k] += 1
            audit_log.append({
                'document_id': r['document_id'], 'text': r['text'],
                'old_label': r['label'], 'reviewed_label': c['nhan_moi'],
                'baseline_prediction': r['baseline_prediction'],
                'human_prediction': r['human_prediction'],
                'ly_do': c.get('ly_do', ''),
            })
        new['baseline_correct_reviewed'] = str(new['reviewed_label'] == new['baseline_prediction'])
        new['human_correct_reviewed'] = str(new['reviewed_label'] == new['human_prediction'])
        reviewed.append(new)

    missing = [k for k in corrections if matched[k] == 0]
    if missing:
        raise SystemExit(f'Còn {len(missing)} câu chưa tìm thấy trong test. Ví dụ: {missing[:3]}. Dừng để kiểm tra.')
    print(f'Đã khớp đủ {len(matched)}/7 mục cần rà soát, ứng với {len(audit_log)} hàng của tập test.')
    for k in corrections:
        print(f'  {k[0]} | {corrections[k]["nhan_cu"]} -> {corrections[k]["nhan_moi"]} | {matched[k]} dòng')

    show_metrics('NHÃN TEST GỐC (chưa đổi)', reviewed, 'old_label')
    show_metrics('NHÃN ĐÃ RÀ SOÁT — CHỈ ĐỂ PHÂN TÍCH LỖI', reviewed, 'reviewed_label')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(reviewed[0].keys()))
        w.writeheader(); w.writerows(reviewed)
    with CHANGE_LOG.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(audit_log[0].keys()))
        w.writeheader(); w.writerows(audit_log)
    print(f'\nĐÃ LƯU: {OUTPUT}')
    print(f'ĐÃ LƯU: {CHANGE_LOG}')
    print('KHÔNG thay đổi segment_classification.csv, frozen_split.json, checkpoints, hay test_predictions.csv.')
    print('LƯU Ý: Đây là kết quả PHÂN TÍCH trên test đã được xem và rà nhãn; KHÔNG dùng làm điểm khách quan cuối cùng.')


if __name__ == '__main__':
    main()
