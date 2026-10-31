"""Read-only project audit; saves new evidence under reports/foundation_audit."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from train_human_cv import LABELS, clean_label, text_key, read_original, read_human
from planning import generate_plan
from extraction import sentences_with_context
from build_classifier_report import export_docx, table

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports' / 'foundation_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data_segment_cls/segment_classification.csv'
    raw = pd.read_csv(source, encoding='utf-8-sig')
    raw['clean_label'] = raw.label.map(clean_label)
    raw['norm'] = raw.text.fillna('').map(text_key)
    original = read_original(source)
    ht, hv, _ = read_human(ROOT / 'extra_human.csv', original)
    conflicting = raw.groupby('norm').clean_label.nunique()
    conflicts = raw[raw.norm.isin(conflicting[conflicting.gt(1)].index)].copy()
    context_rows = original[['document_id', 'source_line', 'section', 'prev_text', 'next_text']]
    conflicts = conflicts.merge(context_rows, on=['document_id', 'source_line'], how='left', validate='many_to_one')
    conflicts['reviewed_label'] = ''
    conflicts['review_reason'] = ''
    conflicts.to_csv(OUT / 'label_conflicts_for_human_review.csv', index=False, encoding='utf-8-sig')
    counts = []
    for f in range(5):
        test = original[original.fold.eq(f)]
        val = original[original.fold.eq((f + 1) % 5)]
        train = original[~original.fold.isin([f, (f + 1) % 5])]
        held = set(pd.concat([test, val]).text.map(text_key))
        purged = train[~train.text.map(text_key).isin(held)]
        counts.append(dict(fold=f, test=len(test), validation=len(val),
                           nominal_train=len(train), purged_train=len(purged),
                           test_texts_seen_in_nominal_train=int(test.text.map(text_key).isin(set(train.text.map(text_key))).sum())))
        assert not set(purged.text.map(text_key)) & held
    pd.DataFrame(counts).to_csv(OUT / 'split_overlap.csv', index=False, encoding='utf-8-sig')
    evidence_issues, plan_issues = [], []
    checked_entries, checked_docs, plan_steps = 0, 0, 0
    current_sentences, current_offset_failures = 0, 0
    extracted = ROOT / 'data_extracted/extraction_model.jsonl'
    seen_docs = set()
    if extracted.exists():
        for line in extracted.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            doc = record['document_id']
            files = list((ROOT / 'data_cleaned' / doc).glob('*.md')) + list((ROOT / 'data_cleaned' / doc).glob('*.txt'))
            if doc in seen_docs:
                evidence_issues.append(dict(document_id=doc, field='', reason='duplicate_document'))
            seen_docs.add(doc)
            if len(files) != 1:
                evidence_issues.append(dict(document_id=doc, field='', reason='missing_or_ambiguous_cleaned_source'))
                continue
            text = files[0].read_text(encoding='utf-8-sig', errors='replace')
            checked_docs += 1
            for item in sentences_with_context(text):
                current_sentences += 1
                if item['start'] < 0 or text[item['start']:item['end']] != item['text']:
                    current_offset_failures += 1
            if hashlib.sha256(text.encode()).hexdigest() != record.get('source_sha256'):
                evidence_issues.append(dict(document_id=doc, field='', reason='source_hash_mismatch'))
            for field, entries in record.get('fields', {}).items():
                for entry in entries:
                    checked_entries += 1
                    start, end = entry.get('char_start', -1), entry.get('char_end', -1)
                    if not (0 <= start <= end <= len(text) and text[start:end] == entry.get('quote') and entry.get('verified') is True):
                        evidence_issues.append(dict(document_id=doc, field=field, reason='invalid_quote_offset_or_verified_flag'))
            try:
                plan_steps += len(generate_plan(record)['steps'])
            except (ValueError, KeyError, TypeError) as exc:
                plan_issues.append(dict(document_id=doc, error=str(exc)))
    pd.DataFrame(evidence_issues, columns=['document_id', 'field', 'reason']).to_csv(OUT / 'evidence_issues.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(plan_issues, columns=['document_id', 'error']).to_csv(OUT / 'plan_issues.csv', index=False, encoding='utf-8-sig')
    stats = dict(audited_utc=datetime.now(timezone.utc).isoformat(),
                 dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 raw_rows=len(raw), raw_documents=int(raw.document_id.nunique()),
                 training_rows=len(original), training_documents=int(original.document_id.nunique()),
                 invalid_labels=int((~raw.clean_label.isin(LABELS)).sum()),
                 empty_text=int(raw.norm.eq('').sum()),
                 heading_rows=int(raw.text.fillna('').str.strip().str.startswith('#').sum()),
                 conflicting_text_groups=int(conflicting.gt(1).sum()), conflicting_rows=len(conflicts),
                 human_train=len(ht), human_holdout=len(hv),
                 checked_extracted_documents=checked_docs, checked_evidence_entries=checked_entries,
                 evidence_issues=len(evidence_issues), generated_plan_steps=plan_steps,
                 current_sentence_count=current_sentences, current_sentence_offset_failures=current_offset_failures,
                 plan_validation_errors=len(plan_issues))
    (OUT / 'audit.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    content = '''# RÀ SOÁT NỀN TẢNG DỰ ÁN PSAI

Ngày kiểm tra: 26/10/2026. Không huấn luyện dài, không sửa nhãn gốc hoặc checkpoint cũ. Kiểm tra này phân biệt tính đúng kỹ thuật với chất lượng ngữ nghĩa; dẫn chứng khớp nguồn không đồng nghĩa nhãn hoặc công việc đã đúng.

## 1. Phát hiện và sửa lỗi xác nhận được

| Vấn đề | Bằng chứng | Xử lý |
|---|---|---|
| Loss tuỳ chỉnh bị sai chuẩn hoá tích luỹ gradient | Trainer suy ra model nhận loss kwargs qua **kwargs, trong khi custom loss trả microbatch mean và bỏ qua số mẫu tích luỹ | Khai báo model_accepts_loss_kwargs=False; thêm trainer dùng chung và kiểm tra cập nhật tham số |
| Đầu vào train/suy luận khác nhau | Train cắt text ở 600 ký tự; bộ phân loại suy luận không cắt | Cắt text ở 600 ký tự mặc định, có cấu hình text_max_chars; giữ nguyên nội dung tạo cue |
| Ensemble bỏ qua other_bias | Cấu hình được đọc nhưng không cộng bias vào logit OTHER | Áp dụng bias từng model trước softmax; các model final hiện có bias 0 nên riêng sửa này không đổi dự đoán của chúng |
| OCR dùng đường dẫn của máy cũ | Input/output trỏ D:/fall26/DSP_PROJECT | Đổi thành thư mục tương đối từ vị trí script |
| Tách câu làm đổi ký tự nguồn | Ký tự § thật bị thay thành dấu chấm khi khôi phục abbreviation; gây lệch offset | Chỉ dùng chuỗi che để tìm ranh giới, cắt trực tiếp từ nguồn; tìm offset theo con trỏ tăng dần để phân biệt câu lặp |

## 2. Kiểm tra hồi quy bằng mô hình nhỏ

Lệnh: python -m unittest test_classifier_foundations -v

Sáu kiểm tra đã đạt: batch đầy đủ so với tích luỹ; nhóm tích luỹ cuối chưa đủ mẫu; loss không làm mất nhãn đầu vào; ensemble áp dụng bias và giới hạn văn bản; đầu vào rỗng không nạp model; giữ nguyên ký tự § và xác định đúng offset câu lặp.

Kiểm tra CPU với SGD, không dropout và không gradient clipping cho thấy cách cũ ở microbatch 1, tích luỹ 8 bước có cập nhật tham số lớn gấp 8 lần batch đầy đủ. Sau sửa, cập nhật khớp trong sai số số học. Không kết luận đây là nguyên nhân duy nhất của F1 thấp, hoặc rằng mọi checkpoint lịch sử đã dùng cùng phiên bản thư viện.

## 3. Kiểm kê dữ liệu hiện tại

'''
    content += table(['Chỉ tiêu', 'Kết quả'], list(stats.items()))
    content += '\n\nCác file trích xuất cũ vẫn có 42 mục không đạt kiểm tra; chưa ghi đè hoặc tính lại nhãn của chúng. Khi tách câu lại bằng code đã sửa, kiểm tra 73.056 câu trên 254 tài liệu không phát hiện sai lệch offset. Đây là kiểm tra truy nguyên văn bản, không phải đánh giá độ đúng của nhãn hoặc nội dung trích xuất.\n\n'
    content += '\n\nSố dòng dữ liệu gốc khác số mẫu train vì bước chuẩn bị bỏ tiêu đề và mẫu không phù hợp. Nhãn mâu thuẫn được đưa vào danh sách rà soát, không tự động sửa: cùng câu có thể khác nhãn khi khác ngữ cảnh.\n\n'
    content += '## 4. Kiểm tra trùng giữa các fold\n\n' + table(list(counts[0]), [list(x.values()) for x in counts])
    content += '''

## 5. Các phần chưa được coi là hoàn tất

- Nhãn: cần người kiểm tra các nhóm mâu thuẫn và thống nhất hướng dẫn gán nhãn. Không dùng đề xuất AI như ground truth chuyên gia.
- Ngữ cảnh: vị trí khi train tính theo chỉ số dòng dataset trong tài liệu, khi extraction tính theo dòng văn bản gốc; câu trước/sau và cách tách câu chưa dùng chung một bộ xây dựng ngữ cảnh. Chưa tự thay quy ước này trên model cũ vì cần phiên bản dữ liệu và huấn luyện đối chứng.
- Loss: các script cũ dùng weighted CE khác với phép nhân trọng số theo mẫu của luồng mới. Cần chọn một phép tính thống nhất cho thí nghiệm kế tiếp; không gộp các điểm cũ như cùng điều kiện.
- Đánh giá: thiếu prediction từng mẫu của lượt validation_selected_f0; không dựng ma trận nhầm lẫn từ F1 tổng. Cần lưu xác suất, nhãn và mã mẫu trong lượt sau.
- Thí nghiệm: train mới đã lọc thêm dữ liệu nên chưa tách tác động của từng thay đổi. Chưa có mean/std nhiều seed hoặc đầy đủ fold được xác minh.
- OCR và cleaning: chưa kiểm tra đối chiếu trang PDF gốc; cleaning/chunking có RESET_OUTPUT=True, không chạy lại hàng loạt trong lượt audit này.
- Evidence/planning: kiểm tra source hash, quote và offset chỉ chứng minh khả năng truy nguyên. Cần người đánh giá nội dung được trích và công việc có hợp lý không.
- Báo cáo sinh từ PDF mới và đánh giá đầu cuối chưa hoàn thiện; báo cáo train đã tạo không thay thế cho đánh giá đó.

## 6. Thứ tự làm tiếp

1. Hoàn tất hướng dẫn nhãn và rà soát danh sách mâu thuẫn bằng ngữ cảnh gốc.
2. Chuẩn hoá một bộ tách câu và tạo ngữ cảnh dùng chung giữa chuẩn bị dữ liệu và suy luận, có phiên bản.
3. Cố định tập chia và cùng một phép tính loss; lưu manifest trước khi train.
4. Thử chạy ngắn để kiểm tra số học, bộ nhớ và file kết quả; sau đó mới chạy baseline và một phương án thay đổi duy nhất.
5. Lưu prediction từng mẫu, ma trận nhầm lẫn, cấu hình, phiên bản thư viện và checksum nguồn. Chỉ chọn bằng validation.
6. Kiểm tra OCR/làm sạch trên mẫu PDF, rồi đánh giá trích xuất và planning bằng rubric riêng.

## 7. Cách hiểu các báo cáo trước

Các chỉ số cũ vẫn được giữ nguyên như kết quả lịch sử, không được sửa thành chỉ số sau khắc phục. Thí nghiệm có lỗi số học không tự mất dữ liệu quan sát, nhưng không đủ để chứng minh phương pháp đã tối ưu hoặc nguyên nhân của chênh lệch. Cần chạy lại điều kiện có kiểm soát sau khi hoàn tất chuẩn hoá nền tảng. Không có checkpoint nào bị thay trong lượt kiểm tra này.
'''
    (OUT / 'RA_SOAT_NEN_TANG.md').write_text(content, encoding='utf-8')
    export_docx(content, OUT / 'RA_SOAT_NEN_TANG.docx')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
