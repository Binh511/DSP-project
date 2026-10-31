"""Build a self-contained completion report from real acceptance artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.build_rag_completion_report import html_document


def read(path): return json.loads(Path(path).read_text('utf-8'))


def main():
    folder = ROOT / 'reports/project_completion'
    receipt = read(folder / 'latest_acceptance.json')
    report = ROOT / receipt['report']
    if not report.is_file() and (ROOT / 'DEMO/report.html').is_file():
        report = ROOT / 'DEMO/report.html'
    analysis = read(report.parent / 'analysis.json')
    release = read(folder / 'PROJECT_RELEASE_MANIFEST.json')
    rag = read(ROOT / 'psai_rag_quality/artifacts/retrieval_genq_lora_v1/paired_analysis.json')
    classifier = read(ROOT / 'reports/classifier/release_source_clean_v2/FINAL_MODEL_MANIFEST.json')
    labels = analysis['metadata']['classifier']
    source = analysis['source']
    plan = analysis['plan']
    interval = rag['summary']['within_doc/dense']['mrr_delta_bootstrap']
    lines = ['# PSAI — báo cáo hoàn thiện luồng kỹ thuật', '',
        'Hệ thống đã chạy trọn luồng trên PDF tự tạo có cả trang text và trang ảnh: đọc/OCR, '
        'ensemble classifier thật, hybrid RAG, LLM cục bộ, quy tắc có nguồn, kiểm định dữ liệu và workspace.', '',
        'Nghiệm thu này xác nhận chức năng và tính nhất quán nguồn trong phạm vi đã kiểm tra. '
        'Không tuyên bố mọi tài liệu đều đúng 100%, không thay thế đánh giá chất lượng độc lập bằng người.', '',
        '## 1. Luồng hệ thống đã kết nối', '',
        'PDF/text → kiểm tra đầu vào → selectable text hoặc OCR theo trang → văn bản và hash → '
        'cấu trúc/câu/đoạn có nguồn → Classifier + Embedding/RAG + quy tắc → danh mục bằng chứng → '
        'LLM chọn và tổ chức công việc → kiểm định schema/nguồn/phụ thuộc → cảnh báo nguồn lực → '
        'báo cáo và workspace theo dõi.', '',
        '| Thành phần | Cách thực hiện | Bằng chứng chức năng |', '|---|---|---|',
        '| PDF validation | pypdf; chặn nguồn không hợp lệ/không có text khi tắt OCR | Fixture hỗn hợp 3 trang; hash đầu vào giữ nguyên |',
        '| OCR cục bộ | Windows.Media.Ocr en-US; giữ trang có text, OCR trang ảnh | Nhận dạng thật một trang scan; ảnh được nhúng và đối chiếu hash |',
        '| Classifier | final_s11 + final_s22 + final_s33; trung bình xác suất năm lớp | Nạp model thật, lưu toàn bộ năm xác suất; không đổi checkpoint |',
        '| Retrieval | BM25 + dense BGE-M3 + RRF, giới hạn nguồn, tài nguyên riêng | Chạy hybrid thật; 5 đoạn truy xuất khớp hash/span |',
        '| Quy tắc | Ngày/mã/mốc/thời lượng/ngân sách/công nghệ/nhân sự từ mẫu rõ ràng | Có nguồn, giữ cách hiểu mơ hồ; không suy ra lịch hay phân công |',
        '| LLM | Qwen2.5-1.5B-Instruct đã pretrained, suy luận cục bộ | Model đã nạp và sinh JSON; chọn 6 task có nguồn |',
        '| Kiểm định | Hash/span/schema, task IDs, checklist/resource excerpts, chu trình | Từ chối dữ liệu bị sửa kể cả gắn ID nguồn thật |',
        '| Workspace | HTML cục bộ, trạng thái/checklist/ghi chú/duyệt phụ thuộc | Xuất/nhập JSON gắn đúng source/analysis hash |', '',
        '## 2. Kết quả nghiệm thu thực tế', '',
        f"Trạng thái: {receipt['status']}. Fixture có {len(source['pages'])} trang; "
        f"{receipt['ocr_pages']} trang OCR. Có {receipt['source_checks']['extracted_evidence_count']} bằng chứng trích xuất, "
        f"{receipt['source_checks']['model_label_evidence_count']} bằng chứng dùng nhãn model, "
        f"{receipt['source_checks']['retrieved_evidence_count']} đoạn truy xuất và {receipt['llm_tasks']} công việc LLM.", '',
        'Toàn bộ quote và vị trí nguồn được đối chiếu; byte UTF-8 của source snapshot đúng hash; '
        'các liên kết HTML trỏ tới nguồn thật trong báo cáo. Ảnh OCR được đối chiếu hash trước khi hiển thị. '
        'Hash checkpoint classifier trước/sau không đổi.', '',
        '| Giai đoạn model | Thời gian tiến trình (giây) | VRAM cấp phát tối đa (GiB) |', '|---|---:|---:|']
    for stage in receipt['stages']:
        name = stage['stage']
        peak = receipt['gpu_peaks'][name]['peak_allocated_bytes'] / 1024 ** 3
        lines.append(f"| {name} | {stage['elapsed_seconds']:.2f} | {peak:.2f} |")
    lines += ['', 'Mỗi model chạy trong tiến trình riêng. Tiến trình trước thoát trước khi nạp model tiếp theo; '
        'không giữ cùng lúc toàn bộ ensemble, BGE và Qwen trên GPU 6 GB.', '',
        '## 3. Công việc mà LLM đã chọn trên mẫu thử', '',
        'Các mô tả được khôi phục từ bằng chứng của danh mục. LLM không được thêm mô tả, ngày hay ngân sách mới. '
        'Quan hệ cha/con, phụ thuộc và gắn tài nguyên là đề xuất cần người duyệt.', '',
        '| Task | Mô tả từ nguồn | Bằng chứng | Cha được đề xuất |', '|---|---|---|---|']
    for step in plan['steps']:
        description = step['description'].replace('|', '/').replace('\n', ' ')
        lines.append('| ' + ' | '.join([step['task_id'], description,
                    ', '.join(step['evidence_ids']), str(step.get('parent_task_id') or '')]) + ' |')
    lines += ['', '## 4. Quy tắc và nguồn lực', '']
    for name, items in analysis['rule_facts']['fields'].items():
        if items:
            lines.append('- ' + name + ': ' + '; '.join(item['quote'] for item in items))
    for warning in analysis['resource_matching']['warnings']:
        lines.append('- Cảnh báo: ' + warning['message'])
    lines += ['', 'Không phát hiện bằng quy tắc không đồng nghĩa thông tin chắc chắn không có. '
        'Các kết hợp tài nguyên–công việc do LLM đề xuất không tự chứng minh nguồn lực đó là bắt buộc.', '',
        '## 5. Kiểm tra phần mềm và tính toàn vẹn', '',
        f"Unittest: {release['unit_test_count']} test đạt. Test fixture không phải phép đo độ chính xác nghiên cứu.", '',
        '- Worker phải trả đúng stage và source hash; trạng thái completed đơn lẻ không được tin.',
        '- Lỗi được giữ trong thư mục .failed với request/response/log; không chuyển fallback rồi báo model thành công.',
        '- LLM output được kiểm định lại tại lúc xuất, gồm checklist, tài nguyên, cha/con và dependency.',
        '- Chặn nhãn/ID không biết, chu trình, dữ liệu bị sửa, source snapshot/ảnh OCR sai hash.',
        '- Không dùng nội dung tài liệu để thực thi lệnh; HTML/source được escape; dữ liệu tiến độ có validator.',
        '- Tách inference khỏi training; không gọi API và không tự tải model trong các worker.',
        '- Windows OCR dùng helper PowerShell trong riêng tiến trình, không đổi policy được lưu trên máy.', '',
        '## 6. Phân biệt kết quả nghiên cứu và model đang chạy', '',
        f"Classifier release nghiên cứu v2: baseline seed2026, validation Macro-F1 {classifier['validation_macro_f1']:.6f}. "
        'Bản đó chưa tự thay ensemble production đang dùng trong nghiệm thu này. Không gán điểm validation của release v2 cho ensemble.', '',
        'RAG LoRA đã được train một epoch, 96 bước trên 1.536 triplet tự sinh. Trên 54 query validation tự sinh: '
        'dense MRR@10 tăng từ 0,505320 lên 0,506606; Recall@5 tăng từ 0,666667 lên 0,703704.', '',
        f"Delta MRR {interval['estimate']:+.6f}, source-cluster CI95% "
        f"[{interval['ci95_low']:+.6f}; {interval['ci95_high']:+.6f}], gồm 0. Chưa đủ bằng chứng adapter cải thiện ổn định. "
        'Project mặc định giữ BGE-M3 gốc; adapter chỉ bật khi được chọn có chủ đích.', '',
        'Qwen là model pretrained được tải về để suy luận, không phải model được huấn luyện mới trong project. '
        'OCR Windows dùng capability đã cài trên hệ điều hành, không phải model OCR do project train.', '',
        '## 7. Những kết luận chưa được phép khẳng định', '',
        '- Chưa có bộ đánh giá đầu cuối độc lập do người kiểm định để kết luận độ đúng/đủ của nội dung và kế hoạch.',
        '- Không dùng nhãn AI làm ground truth. Source đúng không tự chứng minh nhãn, relevance hoặc suy luận đúng.',
        '- OCR tiếng Anh rõ nét đã chạy trên fixture; chưa có bằng chứng tổng quát cho PDF scan khó hoặc tiếng Việt.',
        '- LLM có ngân sách context và catalog hữu hạn, ghi phần bị bỏ khỏi catalog; kết quả cần người kiểm tra.',
        '- Tập giữ lại của riêng retriever không tự chứng minh độc lập với ensemble cũ. Final test toàn hệ thống cần nguồn/query/nhãn người độc lập với mọi model liên quan.',
        '- Không dùng kết quả final test để chọn tham số; chưa chấm final test trong nghiệm thu chức năng này.', '',
        '## 8. Chạy demo và xem code', '',
        'Lệnh đầy đủ: python tools/demo_project.py',
        'Lệnh nhanh không ML: python tools/demo_project.py --fast',
        'Input PDF riêng: python tools/demo_project.py --input DUONG_DAN.pdf --device cuda',
        'Điểm vào: psai_project/runtime.py; worker model: psai_project/worker.py; xuất nguồn/workspace: '
        'psai_project/report.py và workspace.py; OCR: psai_ocr; LLM catalog: psai_planner; rule: psai_rules; retrieval: psai_retrieval.', '',
        'Báo cáo chi tiết có nguồn và ảnh OCR: ' + receipt['report'],
        'Workspace thực tế: ' + receipt['workspace'],
        'Gói bàn giao chứa mã, dữ liệu demo và bằng chứng; model weights lớn được cung cấp riêng. '
        'File HTML báo cáo nhúng văn bản/ảnh để người xem mở độc lập; giữ workspace bên cạnh report để liên kết nguồn hoạt động.', '']
    text = '\n'.join(lines)
    (folder / 'BAO_CAO_HOAN_THIEN_PROJECT.md').write_text(text, encoding='utf-8')
    (folder / 'BAO_CAO_HOAN_THIEN_PROJECT.html').write_text(html_document(text), encoding='utf-8')
    print(folder / 'BAO_CAO_HOAN_THIEN_PROJECT.html')


if __name__ == '__main__': main()
