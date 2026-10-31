# PSAI — chạy toàn bộ luồng cục bộ

Luồng đã chạy thật: **PDF/text → text/OCR → ensemble classifier → hybrid RAG → quy tắc có nguồn → Qwen cục bộ → kiểm định schema/nguồn/phụ thuộc → báo cáo và workspace**.

Đây là hoàn thiện luồng chức năng trong phạm vi đã nghiệm thu trên máy hiện tại; không phải cam kết độ chính xác 100% trên mọi tài liệu. Nhãn, relevance và kế hoạch vẫn cần người kiểm định để có kết luận nghiên cứu độc lập.

## Một lệnh demo

Trong thư mục gốc project:

```powershell
python tools/demo_project.py
```

Lệnh mặc định chạy ba model classifier production, BGE-M3 và LLM Qwen đã tải cục bộ. Các bước model chạy trong tiến trình riêng, tuần tự, phù hợp GPU 6 GB trên máy đã kiểm tra. Lệnh in giai đoạn đang chạy và file báo cáo/workspace sau khi hoàn tất. Không train thêm và không gọi API trong suy luận.

Input riêng:

```powershell
python tools/demo_project.py --input "D:/tai_lieu/de_xuat.pdf" --device cuda
```

Kiểm tra nhanh không dùng learned model:

```powershell
python tools/demo_project.py --fast
```

`--fast` được ghi rõ là rules + BM25 + kế hoạch theo nguồn; không được báo như kết quả classifier/LLM thật. Nếu model trả dữ liệu sai schema, chương trình báo lỗi và giữ chẩn đoán; không âm thầm báo thành công bằng fallback.

## Output cần mở

Mỗi lượt tạo một thư mục mới:

- `report.html`: nội dung, công việc, checklist, cảnh báo nguồn lực, văn bản nguồn và ảnh OCR được nhúng để mở độc lập.
- `workspace.html`: cập nhật trạng thái công việc/checklist, ghi chú, duyệt ứng viên phụ thuộc và xuất/nhập JSON tiến độ.
- `analysis.json`: dữ liệu tổng hợp; `classifier_predictions.json` có đủ năm xác suất.
- `jobs/`: request, response và log từng model; có hash nguồn và provenance.

Giữ report và workspace cùng thư mục để liên kết nguồn hoạt động. Tiến độ trong workspace là tiến độ công việc của tài liệu đề xuất, không phải phần trăm độ chính xác model hoặc mức hoàn thiện mã nguồn. Trạng thái người dùng không trở thành nhãn chuẩn nghiên cứu. Xuất JSON để lưu chắc chắn trước khi đóng trình duyệt; dữ liệu tiến độ gắn với đúng bản analysis/source hash.

## OCR và giới hạn input

PDF có text giữ nguyên nội dung đọc được. PDF scan được OCR bằng Windows.Media.Ocr en-US sẵn có khi thiếu backend RapidOCR/PDFium. Không cần tải phụ thuộc cho backend Windows đã kiểm tra. [Chi tiết OCR](psai_ocr/README.md) ghi rõ nguồn ảnh, box chữ, hash và confidence không có sẵn trong API. Helper dùng ExecutionPolicy Bypass cho riêng tiến trình PowerShell chạy script cục bộ đã kiểm tra; không đổi policy được lưu trên máy.

Chưa công bố chất lượng OCR tiếng Việt. `auto` chỉ OCR trang có ít text; tài liệu vừa có text vừa có nội dung ảnh cần đối chiếu hoặc dùng `--ocr always`. Kiểm tra chức năng trên một mẫu rõ nét không đại diện độ chính xác thực tế.

LLM nhận tối đa 3.072 token đầu vào và 1.024 token đầu ra, catalog mặc định 12 công việc. Các mục ngoài catalog được thống kê; prompt quá giới hạn báo lỗi, không cắt âm thầm. Qwen chỉ chọn/sắp xếp task có nguồn; quan hệ cha/con, phụ thuộc và gắn tài nguyên là gợi ý cần duyệt. Không tự thêm ngày, ngân sách hoặc cam kết mới.

## Kiểm tra, model và khoa học

```powershell
.venv/psai_runtime/Scripts/python.exe -m unittest discover -p "test_*.py"
python tools/finalize_project_release.py --verify-only --models
.venv/psai_runtime/Scripts/python.exe tools/verify_complete_project.py
```

294 test đã đạt; nghiệm thu PDF tự tạo có hai trang selectable và một trang scan đã đi qua toàn bộ model thật. Xem [bằng chứng gần nhất](reports/project_completion/latest_acceptance.json). Đây là kiểm tra chức năng, không phải final test.

Trên máy khác cần cung cấp ba thư mục `models/final_s11`, `models/final_s22`, `models/final_s33` đầy đủ; base BGE-M3 và Qwen đầy đủ. Dùng `--base-model-dir`/`PSAI_BGE_MODEL_DIR` cho BGE, `--planner-model-dir` cho Qwen. Môi trường đã kiểm tra dùng Python 3.13, PyTorch CUDA và các phiên bản được ghi trong manifest. Chưa chứng nhận cài mới trên mọi hệ điều hành.

Ensemble production và release nghiên cứu classifier v2 là hai đối tượng khác nhau. Qwen chỉ được tích hợp để suy luận, không phải model do project huấn luyện. [Nhánh classifier](README_CLASSIFIER.md), [nhánh RAG](README_RAG.md) có số liệu riêng; mức tăng MRR của adapter chưa có bằng chứng ổn định.

Final test toàn hệ thống phải dùng tài liệu/query và nhãn người độc lập với mọi model đang dùng. Tập nguồn giữ lại của study retriever không tự chứng minh độc lập với ensemble cũ. Không lấy nhãn AI làm ground truth, không dùng final test chọn tham số.
