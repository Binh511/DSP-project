"""Prepare a blinded manual review packet; never invent reviewer labels."""
from pathlib import Path

import pandas as pd

from build_classifier_report import export_docx, table
from study_data import OUT, sha, write_json


def prepare():
    source = OUT / 'studies/user_labels_v1/analysis/priority_errors.csv'
    destination = OUT / 'studies/user_labels_v1/manual_adjudication'
    destination.mkdir(parents=True, exist_ok=True)
    errors = pd.read_csv(source, keep_default_na=False)
    # Two functional strata, limited per document; reproducible first batch.
    selected, seen, counts = [], set(), {}
    for group in ['REQUIREMENT_ACTIVITY', 'EXPECTED_RESULT']:
        pool = errors[errors.priority_group.eq(group)].sort_values(['confidence', 'item_id'], ascending=[False, True])
        n = 0
        for row in pool.itertuples():
            if row.item_id in seen or counts.get(row.document_id, 0) >= 4:
                continue
            selected.append(row.item_id)
            seen.add(row.item_id)
            counts[row.document_id] = counts.get(row.document_id, 0) + 1
            n += 1
            if n == 15:
                break
    rows = errors.drop_duplicates('item_id').set_index('item_id').loc[selected].reset_index()
    columns = ['item_id', 'document_id', 'text', 'section', 'prev_text', 'next_text']
    sheet = rows[columns].copy()
    fields = ['reviewer', 'reviewed_label', 'cause', 'source_checked', 'source_page', 'reason']
    for field in fields:
        sheet[field] = ''
    target = destination / 'PHIEU_PHAN_XU.csv'
    if target.exists():
        previous = pd.read_csv(target, keep_default_na=False).set_index('item_id')
        for field in fields:
            sheet[field] = sheet.item_id.map(previous[field]).fillna('')
    sheet.to_csv(target, index=False, encoding='utf-8-sig')
    errors[errors.item_id.isin(selected)].to_csv(destination / 'coordinator_key.csv', index=False, encoding='utf-8-sig')
    intro = '''# PHIẾU KIỂM ĐỊNH NHÃN CLASSIFIER

Đợt đầu: tối đa 30 câu ưu tiên, chọn từ hai nhóm bất đồng REQUIREMENT–ACTIVITY và EXPECTED_RESULT, tối đa 4 câu/tài liệu. Đây là mẫu có chủ đích để phân xử lỗi, không đại diện ngẫu nhiên cho toàn bộ dữ liệu; không suy ra tỷ lệ nhãn sai toàn tập hoặc Cohen's Kappa từ phiếu này.

Phiếu không hiển thị nhãn dataset và dự đoán model nhằm giảm gợi ý. Người kiểm tra đọc câu cùng ngữ cảnh, ghi nhãn và lý do trước khi xem bảng đáp án điều phối. Nhận định của trợ lý chưa phải kết quả kiểm định thủ công.

Quy tắc cần thống nhất trước khi gán: OBJECTIVE là mục tiêu; REQUIREMENT là điều kiện, ràng buộc hoặc tiêu chí bắt buộc; ACTIVITY là công việc/phương pháp thực hiện; EXPECTED_RESULT là sản phẩm hoặc kết quả dự kiến; OTHER là nội dung còn lại. Với câu chứa nhiều chức năng, cần ghi quy tắc ưu tiên và lý do thay vì chỉ dựa vào will, need hoặc deliverable. Đọc cả thì và vai trò đoạn để phân biệt kết quả nghiên cứu đã có với kết quả dự kiến.

Ghi cause bằng MODEL, CONTEXT, EXTRACTION, REFERENCE_LABEL, MIXED hoặc UNRESOLVED. Có thể để UNRESOLVED nếu chưa đủ bằng chứng. Chỉ xác nhận EXTRACTION sau khi đối chiếu PDF gốc; ghi trang và phần mất/ghép sai. Nhãn mới chỉ ghi vào phiếu, chưa thay validation dùng cho các số liệu đã công bố.

Đợt này chưa thay ensemble PSAI. Model cuối cùng chưa chọn. Cấu hình nhãn gốc + full context đạt validation macro-F1 0,6439 cao nhất trong ba lượt mới ở seed 42; chưa có phép so sánh đủ điều kiện để kết luận tốt hơn ensemble đang dùng.

'''
    text = intro
    for i, row in enumerate(sheet.itertuples(), 1):
        text += f'## Câu {i} — {row.document_id}\n\nMã: {row.item_id}\n\n'
        text += f'**Câu cần gán:** {row.text}\n\nMục: {row.section or "(trống)"}\n\n'
        text += f'Câu trước: {row.prev_text or "(trống)"}\n\nCâu sau: {row.next_text or "(trống)"}\n\n'
        text += table(['Thông tin phân xử', 'Kết quả'], [
            ['Người kiểm tra', row.reviewer], ['Nhãn lựa chọn', row.reviewed_label],
            ['Nguyên nhân', row.cause], ['Đã đối chiếu PDF/trang', row.source_checked + ' ' + row.source_page],
            ['Lý do/quy tắc áp dụng', row.reason]]) + '\n\n'
    (destination / 'PHIEU_PHAN_XU.md').write_text(text, encoding='utf-8')
    export_docx(text, destination / 'PHIEU_PHAN_XU.docx')
    write_json(destination / 'packet_manifest.json', dict(
        source_sha256=sha(source), items=len(sheet), selected_ids=selected,
        selection='15 per transition group where possible; confidence descending; cap 4/document; unique items',
        blinded_to_predictions=True, human_review_complete=False,
        purpose='targeted qualitative adjudication, not representative label accuracy estimate',
        validation_labels_modified=False))
    print(destination / 'PHIEU_PHAN_XU.docx')
    print(f'{len(sheet)} unique items; human fields preserved; no labels generated')


if __name__ == '__main__':
    prepare()
