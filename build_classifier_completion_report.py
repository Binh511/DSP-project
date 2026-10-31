"""Current classifier-only handoff, generated from saved validation and manifests."""
from pathlib import Path
import os

import pandas as pd

from build_classifier_report import table, export_docx
from study_data import ROOT, OUT, read_json, write_json, sha

DEST = ROOT / 'reports/classifier'
REPLAY = OUT / 'studies/classifier_validation_replay_v1'
CONTEXT = OUT / 'studies/classifier_context_v2_1'
REVIEW = OUT / 'studies/classifier_stability_v1/human_review'


def link(path):
    return f'[{path.name}]({Path(os.path.relpath(path, DEST)).as_posix()})'


def main():
    checks = read_json(DEST / 'verification.json')
    statistics = read_json(REPLAY / 'statistics.json')
    comparison = pd.read_csv(REPLAY / 'comparison.csv')
    classes = pd.read_csv(REPLAY / 'per_class_metrics.csv')
    preparation = read_json(CONTEXT / 'preparation_manifest.json')
    review = read_json(REVIEW / 'imports/blank_sheet_status_v1/validation_report.json')
    assert checks['passed'] and statistics['state'] == 'complete'
    text = '# Hoàn thiện nhánh classifier PSAI\n\n'
    text += '## Phạm vi và trạng thái\n\n'
    text += 'Đợt này chỉ xử lý classifier: dữ liệu/context, kiểm định nhãn, inference validation và bằng chứng bàn giao. Không triển khai RAG, không train thêm model, không dùng test/final test hoặc thay ensemble production. Ba checkpoint nghiên cứu đã được đánh giá lại trên đúng validation frozen. Dữ liệu context mới là ứng viên chuẩn bị riêng, chưa được train/evaluate và chưa áp dụng vào các điểm dưới đây.\n\n'
    text += '## 1. Đã đánh giá lại ba checkpoint theo cùng giao thức\n\n'
    text += table(['Seed', 'Macro-F1 cũ', 'Macro-F1 replay', 'Câu đổi dự đoán', 'Max Δprob tuyệt đối'], [[int(r['seed']), f'{r["original_macro_f1"]:.6f}', f'{r["replay_macro_f1"]:.6f}', int(r['prediction_changes']), f'{r["max_absolute_probability_difference"]:.8f}'] for _, r in comparison.iterrows()])
    text += f'\n\nTrung bình **{statistics["mean"]:.6f} ± {statistics["sample_std"]:.6f}** (SD mẫu, n=3). Ba lượt replay khớp toàn bộ dự đoán và xác suất cũ trong môi trường hiện tại; đây là bằng chứng tái lập một lượt trên môi trường đã ghi, không phải cam kết bitwise trên mọi máy. Vẫn dùng 1.317 câu validation đã phục vụ phát triển, nhãn gốc và context cũ.\n\n'
    text += table(['Seed', 'Nhãn', 'Precision', 'Recall', 'F1', 'Support'], [[int(r['seed']), r['label'], f'{r["precision"]:.4f}', f'{r["recall"]:.4f}', f'{r["f1-score"]:.4f}', int(r['support'])] for _, r in classes.iterrows()])
    text += '\n\nĐã lưu cùng kết quả: hash weights, tokenizer, model/inference config, provenance, code encoder/evaluation; batch8, input study-v1/full, text600, max_length256, OTHER bias0, FP32 parameters, không autocast; GPU/CUDA/cuDNN/TF32/matmul/determinism. Điểm Trainer BF16 lịch sử vẫn được giữ riêng, không trộn với điểm FP32 để xếp hạng. Seed123 là ứng viên validation tốt nhất, **chưa chốt model**.\n\n'
    text += 'Bằng chứng: ' + link(REPLAY / 'KET_QUA_VALIDATION_THONG_NHAT.md') + ', ' + link(REPLAY / 'comparison.csv') + ', ' + link(REPLAY / 'statistics.json') + '.\n\n'
    text += '## 2. Đã tạo dữ liệu context sửa Section, giữ nguyên nhãn\n\n'
    rows = []
    for split, value in preparation['splits'].items():
        rows.append([split, value['rows'], value['section_blank_before'], value['section_blank_after'], value['blank_section_recovered'], value['existing_nonblank_section_corrected'], value['unresolved_kept_old_section']])
    text += table(['Split', 'Câu', 'Section trống cũ', 'Trống mới', 'Khôi phục trống', 'Cập nhật không trống', 'Chưa xác định'], rows)
    text += '\n\nVersion hiện tại: **classifier_context_v2_1**. Chỉ thay Section khi text chuẩn hóa khớp đúng một nguồn Markdown và một đường dẫn heading không rỗng. Nhãn, row order, doc/text/prev/next/pos được giữ nguyên. Các trường hợp mơ hồ giữ Section cũ và có danh sách 110 câu để người kiểm tra. Có snapshot chính xác của script chuẩn bị và hash nguồn/đầu ra.\n\n'
    text += 'Parser đã kiểm tra cấp heading thực, code fences và ranh giới # hợp lệ; không lấy shebang hoặc #define làm tiêu đề. Rà lại parser làm khác một Section train so với bản chuẩn bị v2 trước đó; validation không đổi. Bản v2 cũ được giữ làm lịch sử và thiếu snapshot script cũ đã được ghi rõ; dùng v2_1 làm ứng viên hiện tại.\n\n'
    text += '**Chưa có F1 cho context v2_1.** Việc giảm Section trống chỉ là sửa biểu diễn có bằng chứng từ Markdown, chưa chứng minh tăng chất lượng classifier hoặc đúng cấu trúc PDF. Không dùng kết quả replay context cũ để tuyên bố context mới tốt hơn.\n\n'
    text += 'Bằng chứng: ' + link(CONTEXT / 'README.md') + ', ' + link(CONTEXT / 'preparation_manifest.json') + ', ' + link(CONTEXT / 'section_unresolved_for_review.csv') + ', ' + link(CONTEXT / 'PARSER_REVIEW.md') + '.\n\n'
    text += '## 3. Đã có công cụ kiểm định và nhập phiếu nhãn\n\n'
    text += table(['Nhóm', 'Cần phân xử', 'Đã có nhãn trong phiếu kiểm tra', 'Còn chờ'], [[name, review[key]['assigned'], review[key]['completed'], review[key]['pending']] for name, key in [('Đề xuất AI trong train', 'training'), ('Bất đồng validation ưu tiên', 'validation')]])
    text += '\n\nPhiếu hiện có trong project chưa có nhãn mới được nhập. Công cụ kiểm tra đủ ID, text/context và hash tính lại; ràng buộc nhãn gốc/AI với nguồn phân tích, kiểm tra reviewer/lý do và khai báo vị trí nguồn. Chấp nhận đổi thứ tự hàng, từ chối đổi nội dung hoặc nhãn không hợp lệ. Kết quả tách train/validation; overlay train chỉ là PROPOSED_NOT_APPLIED. Không tự sửa validation, không coi AI là ground truth hoặc tự chứng nhận người gán/dữ liệu độc lập.\n\n'
    text += 'Mở phiếu: ' + link(REVIEW / 'PHAN_XU_91_CAU.html') + '. Trạng thái bản trống: ' + link(REVIEW / 'imports/blank_sheet_status_v1/VALIDATION_REPORT.md') + '.\n\n'
    text += 'Khi có bản xuất CSV, dùng thư mục kết quả mới:\n\n```powershell\npython classifier_adjudication.py --input <file_CSV_da_dien> --output <thu_muc_moi>\n```\n\nThêm `--complete` khi đủ 91 nhãn/reviewer/lý do; chưa đầy đủ vẫn có thể nhập như bản đang làm. Chạy importer không tự áp dụng thay đổi vào dữ liệu frozen hoặc chốt model.\n\n'
    text += '## 4. Kiểm chứng kỹ thuật và giới hạn\n\n'
    text += f'**{checks["tests_run"]}/{checks["tests_run"]} kiểm tra của phần classifier đạt** trong đợt này. Bao gồm gradient/loss trên dữ liệu tổng hợp nhỏ, ablation encoder, toàn vẹn nhãn, parser/context, importer, metric và final gate trên dữ liệu giả lập. Không có huấn luyện thí nghiệm trên dữ liệu PSAI. Ba lượt inference replay dùng checkpoint thật; không mở final test.\n\n'
    text += 'Bằng chứng kiểm tra: ' + link(DEST / 'verification.json') + '. Chưa có bước freeze-selection hoặc final-evaluation thật; ensemble final_s11/final_s22/final_s33 giữ nguyên.\n\n'
    text += '## 5. Việc còn lại để chốt classifier\n\n1. Người kiểm tra hoàn thành phân xử nhãn và các Section chưa rõ, ghi lý do/nguồn; quyết định những thay đổi train nào được chấp nhận.\n2. Xác nhận version nhãn/context và kế hoạch đánh giá tiếp theo. Không tự train chỉ vì đã chuẩn bị xong dữ liệu context mới.\n3. Chốt checkpoint bằng validation theo một giao thức sau khi nhãn được rà soát; lưu trọn bộ artifact/hash và quyết định.\n4. Chuẩn bị tập final độc lập thực sự, có nhãn người. Tám ứng viên hiện tại có dấu vết phát triển nên chưa đủ điều kiện dùng để khẳng định đánh giá độc lập.\n\n'
    text += 'Trạng thái hiện tại: **code và giao thức kiểm chứng đã tiến thêm; nhãn người và lựa chọn model cuối vẫn chưa hoàn tất**.\n'
    md = DEST / 'HOAN_THIEN_CLASSIFIER.md'
    md.write_text(text, encoding='utf-8')
    export_docx(text, DEST / 'HOAN_THIEN_CLASSIFIER.docx')
    files = ['classifier_context_v2.py', 'classifier_adjudication.py', 'classifier_validation_replay.py',
             'test_classifier_context_v2.py', 'test_classifier_adjudication.py', 'replay_validation.py']
    write_json(DEST / 'completion_status.json', dict(scope='classifier', research_training_performed=False,
                validation_replay_completed=True, context_version='classifier_context_v2_1', context_evaluated=False,
                label_review_complete=False, final_model_selected=False, production_replaced=False,
                code_sha256={name: sha(ROOT / name) for name in files}))
    print(md)


if __name__ == '__main__':
    main()
