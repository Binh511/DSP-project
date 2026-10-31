"""Compare completed classifier runs on the SAME validation, without inference."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from build_classifier_report import export_docx, table
from study_data import OUT, read_json, write_json, sha
from train_human_cv import LABELS

STUDY = OUT / 'studies/user_labels_v1'
DEST = STUDY / 'analysis'
RUNS = ['baseline_original_labels', 'baseline', 'A0_text']


def macro_f1_from_matrices(matrices):
    matrices = np.asarray(matrices)
    tp = np.diagonal(matrices, axis1=-2, axis2=-1)
    denom = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0).mean(axis=-1)


def paired_bootstrap(frames, repeats=2000, seed=2026):
    first = next(iter(frames.values()))
    docs = sorted(first.document_id.unique())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(docs), np.ones(len(docs)) / len(docs), size=repeats)
    draws = {}
    for name, frame in frames.items():
        by_doc = np.stack([confusion_matrix(frame.loc[frame.document_id.eq(d), 'true_label'],
                                  frame.loc[frame.document_id.eq(d), 'predicted_label'], labels=LABELS) for d in docs])
        sampled = (weights @ by_doc.reshape(len(docs), -1)).reshape(repeats, 5, 5)
        draws[name] = macro_f1_from_matrices(sampled)
    results = []
    for a, b, hypothesis in [('baseline', 'baseline_original_labels', 'Effect of supplied label revision'),
                              ('baseline', 'A0_text', 'Effect of adding full context with revised labels')]:
        delta = draws[a] - draws[b]
        cm_a = confusion_matrix(frames[a].true_label, frames[a].predicted_label, labels=LABELS)
        cm_b = confusion_matrix(frames[b].true_label, frames[b].predicted_label, labels=LABELS)
        results.append(dict(a=a, b=b, hypothesis=hypothesis,
                            observed_macro_f1_delta=float(macro_f1_from_matrices(cm_a) - macro_f1_from_matrices(cm_b)),
                            document_bootstrap_95_ci=[float(x) for x in np.quantile(delta, [0.025, 0.975])],
                            documents=len(docs), repeats=repeats, seed=seed,
                            interpretation='Exploratory paired document bootstrap, conditional on these fitted models and this validation; not training-seed uncertainty or a final-test claim.'))
    return results


def analyze():
    DEST.mkdir(parents=True, exist_ok=True)
    frames, metrics, per_class, transitions, error_frames = {}, [], [], [], []
    for name in RUNS:
        folder = STUDY / 'experiments' / name
        frame = pd.read_csv(folder / 'validation_predictions.csv', keep_default_na=False, float_precision='round_trip')
        record = read_json(folder / 'validation_metrics.json')
        if record['evaluation_kind'] != 'validation_selection':
            raise ValueError('Only completed validation runs may be compared')
        if frames:
            ref = next(iter(frames.values()))
            for col in ['document_id', 'text', 'true_label', 'section', 'prev_text', 'next_text', 'pos']:
                if not frame[col].equals(ref[col]):
                    raise ValueError('Validation rows/context/order mismatch: ' + col)
        frame['item_id'] = [hashlib.sha256(f'{r.document_id}:{i}:{r.text}'.encode()).hexdigest() for i, r in enumerate(frame.itertuples())]
        frames[name] = frame
        report = classification_report(frame.true_label, frame.predicted_label, labels=LABELS, output_dict=True, zero_division=0)
        f1 = report['macro avg']['f1-score']
        if not np.isclose(f1, record['macro_f1'], atol=1e-10):
            raise ValueError('Saved score disagrees with saved predictions')
        metrics.append(dict(run=name, n=len(frame), macro_f1=f1, accuracy=accuracy_score(frame.true_label, frame.predicted_label),
                            prediction_sha256=sha(folder / 'validation_predictions.csv')))
        for label in LABELS:
            per_class.append(dict(run=name, label=label, **report[label]))
        matrix = confusion_matrix(frame.true_label, frame.predicted_label, labels=LABELS)
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(DEST / f'confusion_{name}.csv', encoding='utf-8-sig')
        errors = frame[frame.true_label.ne(frame.predicted_label)].copy()
        errors['run'] = name
        errors['transition'] = errors.true_label + ' -> ' + errors.predicted_label
        errors['priority_group'] = np.where(
            ((errors.true_label.eq('REQUIREMENT') & errors.predicted_label.eq('ACTIVITY')) |
             (errors.true_label.eq('ACTIVITY') & errors.predicted_label.eq('REQUIREMENT'))),
            'REQUIREMENT_ACTIVITY', np.where(errors.true_label.eq('EXPECTED_RESULT') | errors.predicted_label.eq('EXPECTED_RESULT'),
                                           'EXPECTED_RESULT', 'OTHER_TRANSITIONS'))
        def flags(row):
            candidates = {'text_cut_at_600_chars': len(row.text) > 600,
                          'missing_section': not row.section.strip(),
                          'encoding_replacement_character': '\ufffd' in row.text,
                          'multi_sentence_input': bool(re.search(r'[.!?]\s+[A-Z]', row.text)),
                          'modal_cue': bool(re.search(r'\b(must|shall|should|need to)\b', row.text, re.I)),
                          'markdown_table': row.text.count('|') >= 2}
            return '|'.join(k for k, v in candidates.items() if v)
        errors['candidate_flags'] = [flags(r) for r in errors.itertuples()]
        for col in ('reviewed_cause', 'reviewed_gold_label', 'review_reason'):
            errors[col] = ''
        error_frames.append(errors)
        for (truth, pred), n in errors.groupby(['true_label', 'predicted_label']).size().items():
            transitions.append(dict(run=name, true_label=truth, predicted_label=pred, count=int(n)))
    all_errors = pd.concat(error_frames, ignore_index=True)
    # Do not destroy human adjudication if this report is regenerated later.
    target = DEST / 'errors_for_review.csv'
    if target.exists():
        prior = pd.read_csv(target, keep_default_na=False)
        if any(prior[c].astype(str).str.strip().ne('').any() for c in ['reviewed_cause', 'reviewed_gold_label', 'review_reason']):
            keep = prior[['run', 'item_id', 'reviewed_cause', 'reviewed_gold_label', 'review_reason']]
            all_errors = all_errors.drop(columns=['reviewed_cause', 'reviewed_gold_label', 'review_reason']).merge(keep, on=['run', 'item_id'], how='left', validate='one_to_one').fillna('')
    all_errors.to_csv(target, index=False, encoding='utf-8-sig')
    all_errors[all_errors.priority_group.ne('OTHER_TRANSITIONS')].sort_values(['priority_group', 'confidence'], ascending=[True, False]).to_csv(DEST / 'priority_errors.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(per_class).to_csv(DEST / 'per_class_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(DEST / 'validation_comparison.csv', index=False, encoding='utf-8-sig')
    transition_frame = pd.DataFrame(transitions).sort_values('count', ascending=False)
    transition_frame.to_csv(DEST / 'error_transitions.csv', index=False, encoding='utf-8-sig')
    bootstrap = paired_bootstrap(frames)
    write_json(DEST / 'paired_document_bootstrap.json', bootstrap)
    write_json(DEST / 'analysis.json', dict(metrics=metrics, paired_comparisons=bootstrap,
               evaluation_split='validation, previously inspected', original_study=str(STUDY),
               errors=len(all_errors), priority_errors=int(all_errors.priority_group.ne('OTHER_TRANSITIONS').sum()),
               cause_status='flags are hypotheses; human root-cause adjudication not supplied',
               model_not_deployed=True, final_test_not_opened=True))
    text = '''# PHÂN TÍCH BA THÍ NGHIỆM CLASSIFIER TRÊN VALIDATION

## 1. Phạm vi

Các phương án dùng cùng 1.317 câu validation theo cùng thứ tự và ngữ cảnh nguồn. Các chỉ số được tính lại từ dự đoán đã lưu và đối chiếu với metric khi train. Chưa đánh giá final test và chưa thay model đang dùng.

Hai đối chứng khác nhau: baseline so với baseline_original_labels chỉ thay nhãn train; baseline so với A0_text giữ nhãn mới và thay đầu vào có/không có ngữ cảnh. Không gộp hai phép so sánh để quy kết nguyên nhân.

## 2. Kết quả tổng thể

'''
    text += table(['Phương án', 'Số câu', 'Macro-F1', 'Accuracy'], [[r['run'], r['n'], f"{r['macro_f1']:.4f}", f"{r['accuracy']:.4f}"] for r in metrics])
    text += '\n\n## 3. Precision/recall/F1 từng lớp\n\n'
    text += table(['Phương án', 'Nhãn', 'Precision', 'Recall', 'F1', 'Số mẫu'], [[r['run'], r['label'], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", int(r['support'])] for r in per_class])
    text += '\n\n## 4. Ma trận nhầm lẫn\n\nHàng là nhãn tham chiếu trong dataset; cột là dự đoán.\n\n'
    for name, frame in frames.items():
        matrix = confusion_matrix(frame.true_label, frame.predicted_label, labels=LABELS)
        text += '### ' + name + '\n\n' + table(['Thực tế / dự đoán'] + LABELS, [[label] + list(matrix[i]) for i, label in enumerate(LABELS)]) + '\n\n'
    text += '## 5. Chênh lệch và độ bất định theo tài liệu\n\n'
    text += table(['Đối chứng A − B', 'Chênh lệch macro-F1', 'Khoảng bootstrap 95%'], [[f"{r['a']} − {r['b']}", f"{r['observed_macro_f1_delta']:+.4f}", f"[{r['document_bootstrap_95_ci'][0]:+.4f}; {r['document_bootstrap_95_ci'][1]:+.4f}]"] for r in bootstrap])
    text += '''

Bootstrap lấy lại mẫu theo tài liệu, dùng chung các lần lấy mẫu cho hai model và giữ năm lớp khi tính macro-F1. Khoảng này chỉ phản ánh biến động theo tập tài liệu validation với các model đã fit, không thay thế biến động nhiều seed. Nếu khoảng chứa 0, chưa có bằng chứng chắc về hướng chênh lệch từ phép kiểm tra thăm dò này. Không dùng nó như kết luận cuối trên test độc lập.

## 6. Phân tích lỗi ưu tiên

'''
    wanted = transition_frame[((transition_frame.true_label.eq('REQUIREMENT') & transition_frame.predicted_label.eq('ACTIVITY')) |
                              (transition_frame.true_label.eq('ACTIVITY') & transition_frame.predicted_label.eq('REQUIREMENT')) |
                              transition_frame.true_label.eq('EXPECTED_RESULT') | transition_frame.predicted_label.eq('EXPECTED_RESULT'))]
    text += table(['Phương án', 'Nhãn dataset', 'Dự đoán', 'Số câu'], wanted[['run', 'true_label', 'predicted_label', 'count']].values.tolist())
    text += '\n\n### Ví dụ bất đồng có độ tự tin cao cần kiểm tra\n\n'
    for name in RUNS:
        examples = all_errors[all_errors.run.eq(name) & all_errors.priority_group.ne('OTHER_TRANSITIONS')].sort_values('confidence', ascending=False).head(4)
        for r in examples.itertuples():
            text += f'**{name} — {r.document_id} — {r.true_label} → {r.predicted_label}, score={r.confidence:.3f}**\n\n'
            text += r.text + '\n\nMục: ' + (r.section or '(trống)') + '\n\nCâu trước: ' + (r.prev_text or '(trống)') + '\n\nCâu sau: ' + (r.next_text or '(trống)') + '\n\nDấu hiệu cần rà soát: ' + (r.candidate_flags or 'Chưa có dấu hiệu tự động') + '\n\n'
    text += '''## 7. Đọc định tính các bất đồng tiêu biểu

Những nhận định sau là rà soát sơ bộ của trợ lý trên câu và ngữ cảnh lưu trong dataset, chưa phải phân xử nhãn bởi chuyên gia và chưa đối chiếu PDF gốc. Chúng không làm thay đổi nhãn hoặc điểm validation. Không dùng độ tự tin của model làm bằng chứng nhãn nào đúng.

| Trường hợp | Nhận định sơ bộ | Việc cần xác minh |
| --- | --- | --- |
| PSAI121: “Needs to check whether it is possible ...” — ACTIVITY thành REQUIREMENT ở model text | “Needs” gợi yêu cầu, nhưng “check whether” mô tả nhiệm vụ nghiên cứu. Đây là ranh giới chức năng câu; có khả năng model quá dựa vào từ khóa. | Đọc đoạn nhiệm vụ đầy đủ và quy tắc phân biệt yêu cầu với hoạt động kiểm tra. |
| PSAI077: “The principal outputs of Study III are therefore ...” — EXPECTED_RESULT thành OTHER ở model nhãn gốc | Câu nêu các đầu ra cụ thể, nhãn EXPECTED_RESULT có cơ sở rõ. Đây là ứng viên lỗi mô hình dù model tự tin cao. | Kiểm tra đoạn và danh sách đầu ra trong nguồn; xác nhận các mục không bị ghép sai. |
| PSAI077: “... successful only if ... Pareto improvement ...” — REQUIREMENT thành EXPECTED_RESULT ở model nhãn sửa | Câu vừa nói hiệu quả mong đợi vừa đặt tiêu chí chấp nhận bằng “only if”. REQUIREMENT có cơ sở nếu quy ước xếp tiêu chí thành công vào yêu cầu. | Cần quy tắc ưu tiên khi một câu đồng thời chứa tiêu chí và kết quả; chưa đủ cơ sở sửa nhãn. |
| PSAI077: “The resulting systems will not be positioned as authoritative translators ...” — EXPECTED_RESULT thành REQUIREMENT | Nội dung đặt giới hạn sử dụng, nên nhãn tham chiếu EXPECTED_RESULT cần xem lại. Dự đoán khác gold không tự động là lỗi model. | Người gán nhãn phân xử giữa giới hạn phạm vi, yêu cầu và kết quả; giữ nguyên gold hiện tại khi báo cáo điểm. |
| PSAI033: “A Golang SDK ... will be developed ...” và “This deliverable will create a Golang NEAR API client ...” | Hai câu gần nhau về chức năng nhưng nhãn tham chiếu lần lượt là EXPECTED_RESULT và ACTIVITY. Cách diễn đạt vừa nêu sản phẩm vừa nêu hành động dễ tạo bất đồng. | Kiểm tra ngữ cảnh milestone/deliverable và áp dụng cùng quy tắc nhãn; không kết luận nhãn sai chỉ từ hai câu. |
| PSAI028/PSAI121: “A summary of the Filecoin infrastructure ...” — EXPECTED_RESULT thành OTHER ở text-only | Đây là cụm danh từ đầu ra; ở PSAI121 câu trước là DELIVERABLES. Text riêng thiếu tín hiệu chức năng của danh sách. | So sánh dự đoán full context trên đúng item, đối chiếu tiêu đề/danh sách trong nguồn; đây là giả thuyết thiếu ngữ cảnh. |
| PSAI052: “our new mechanism ensures ... guaranteeing ...” — OTHER thành EXPECTED_RESULT | Có thể là mô tả đóng góp/kết quả đã đạt trong bài nghiên cứu, khác đầu ra dự án tương lai. Sự khác miền tài liệu có thể khiến ngôn ngữ “guaranteeing” gây nhầm. | Xem loại tài liệu, thì và vai trò đoạn; phân biệt kết quả nghiên cứu đã có với kết quả dự kiến của đề xuất. |

Đối với văn bản trích xuất: các cờ thiếu section, ký tự thay thế hoặc cắt text 600 ký tự chỉ là tín hiệu kiểm tra. Thiếu section trong CSV chưa chứng minh PDF bị trích xuất sai; câu có “heritagecritical” cũng chưa đủ xác định lỗi OCR. Vì chưa đối chiếu nguồn cho từng item, báo cáo không gán một tỷ lệ lỗi trích xuất hay tỷ lệ nguyên nhân đã xác nhận. Các cột phân xử trong bảng lỗi vẫn để trống cho người kiểm tra, tách biệt với nhận định sơ bộ trên.

## 8. Kết luận hiện tại và bước tiếp

Sửa nhãn chưa tăng macro-F1 trong seed 42; thêm full context có chênh lệch dương so với chỉ text trên nhãn mới. Mức chênh lệch nhỏ và cần đọc cùng khoảng bootstrap, phân bố lỗi và kết quả nhiều seed.

Các cột nguyên nhân lỗi, nhãn sau phân xử và lý do được để trống cho người kiểm tra. Không tự coi mọi bất đồng với nhãn dataset là lỗi model, và không suy ra nguyên nhân từ một modal cue hoặc một ví dụ. Không sửa validation để tạo điểm cao hơn rồi gọi đó là kết quả độc lập.

Bước tiếp theo là kiểm tra nhiều seed cho cấu hình nhãn gốc + full context, vì đây là cấu hình có validation macro-F1 cao nhất trong ba lượt nhưng chỉ hơn rất ít. Chỉ chốt cấu hình sau khi xem độ ổn định và các đối chứng có ngân sách giới hạn. Không chọn hay thay model production ở bước phân tích này.
'''
    (DEST / 'SO_SANH_VA_PHAN_TICH_LOI.md').write_text(text, encoding='utf-8')
    export_docx(text, DEST / 'SO_SANH_VA_PHAN_TICH_LOI.docx')
    print(DEST / 'SO_SANH_VA_PHAN_TICH_LOI.docx')
    print(bootstrap)


if __name__ == '__main__':
    analyze()
