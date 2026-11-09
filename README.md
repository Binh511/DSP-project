# PSAI — phân tích tài liệu đề xuất dự án

PSAI đọc tài liệu PDF, Markdown hoặc TXT, trích xuất nội dung, truy hồi bằng chứng và tạo kế hoạch công việc có dẫn nguồn. Kết quả gồm báo cáo HTML, workspace theo dõi công việc và JSON để kiểm tra hoặc xử lý tiếp.

Kho mã chứa ứng dụng, extension Chrome/Edge, kiểm thử, công cụ huấn luyện/đánh giá và đầu vào mẫu. Trọng số model, dữ liệu nghiên cứu, báo cáo sinh ra và môi trường ảo được chuẩn bị riêng.

## Chạy nhanh

Cần Python 3.11 trở lên. Với Markdown/TXT, chế độ fast chỉ dùng thư viện chuẩn Python:

```sh
git clone https://github.com/Binh511/DSP-project.git
cd DSP-project
python tools/demo_project.py --fast --input examples/PSAI_DEMO_01_MINER_MONITORING.md
python tools/demo_project.py --fast --input examples/PSAI_DEMO_02_LAB_MAINTENANCE.md
```

Hai đầu vào mẫu mô tả [giám sát thiết bị miner](examples/PSAI_DEMO_01_MINER_MONITORING.md) và [bảo trì phòng thí nghiệm](examples/PSAI_DEMO_02_LAB_MAINTENANCE.md). Lệnh `python tools/demo_project.py --fast` dùng [proposal.md](examples/proposal.md), một fixture kiểm tra luồng cơ bản. Thay `--input` bằng tài liệu UTF-8 của bạn; trên máy dùng lệnh `python3`, thay tên lệnh tương ứng.

Mỗi lượt tạo một thư mục kết quả mới và in đường dẫn ra terminal. Mở `report.html` để đối chiếu bằng chứng và `workspace.html` để theo dõi công việc. Thư mục còn có `analysis.json`, bản văn bản nguồn, metadata và nhật ký từng bước. Nếu dùng `--output`, chọn một đường dẫn chưa tồn tại.

## Chế độ xử lý

| Chế độ | Phân loại | Truy hồi | Lập kế hoạch |
|---|---|---|---|
| Fast | Quy tắc | BM25 | Công việc theo thứ tự nguồn |
| Full | Ensemble ba classifier | Hybrid BM25 và BGE-M3 | Qwen cục bộ, giao thức planner v2 |

Full cần [dependency và model cục bộ](docs/FULL_RUNTIME_SETUP.md). Runtime thực hiện suy luận offline, không huấn luyện lại hoặc tự tải model. PDF có lớp chữ cần `pypdf`; PDF quét cần thêm [OCR](psai_ocr/README.md).

Pipeline full mặc định dùng `task_policy=preserve_source_catalog`: model đề xuất thứ tự công việc, phần mềm giữ thêm các công việc đủ điều kiện còn lại trong catalog theo thứ tự nguồn. Mỗi công việc ghi rõ nguồn lựa chọn; các liên kết checklist/nguồn lực vẫn cần kiểm tra. Catalog có giới hạn nên chính sách này không bảo đảm đã tìm đủ mọi công việc. Xem [giao thức planner](psai_planner/README.md) để phân biệt cấu hình pipeline với API planner trực tiếp.

## Extension và cấu trúc mã

Giao diện Chrome/Edge kết nối backend Python cục bộ bằng token ghép nối:

```sh
python -m psai_extension
```

Làm theo [hướng dẫn extension](README_EXTENSION.md) để cài giao diện và chọn fast/full.

| Thành phần | Mã nguồn hoặc tài liệu |
|---|---|
| Điều phối pipeline, worker, báo cáo và workspace | [psai_project](psai_project/), [bản đồ module](docs/REFACTOR_GUIDE.md) |
| Backend extension | [psai_extension](psai_extension/) |
| Giao diện extension | [extension/psai](extension/psai/) |
| OCR PDF | [psai_ocr](psai_ocr/README.md) |
| Truy hồi | [psai_retrieval](psai_retrieval/README.md), [README RAG](README_RAG.md) |
| Planner cục bộ | [psai_planner](psai_planner/README.md) |
| Phân loại và công cụ nghiên cứu | [README classifier](README_CLASSIFIER.md) |
| Đánh giá chất lượng | [quy trình đánh giá](docs/QUALITY_EVALUATION.md) |

## Kiểm tra

Chạy từ thư mục gốc dự án:

```sh
python -m unittest discover -s tests -p "test_project_*.py"
python -m unittest discover -s tests -p "test_extension_*.py"
node --test extension/psai/tests/*.test.mjs
python tools/verify_clean_clone.py
```

Kiểm thử JavaScript cần Node.js 22 trở lên. Công cụ clean-clone kiểm tra luồng CLI và API fast trong bản sao nguồn với môi trường Python trống. Các kiểm tra này xác nhận chức năng; đánh giá độ chính xác model cần dữ liệu và nhãn đối chiếu riêng.

## Giới hạn

Hash, đoạn trích và vị trí ký tự cho phép kiểm tra xuất xứ, nhưng không chứng minh nhãn đúng, kế hoạch đầy đủ hoặc quan hệ phụ thuộc đúng. Công việc, checklist, nguồn lực và liên kết do hệ thống đề xuất cần được người sử dụng rà soát.

Nhãn do AI tạo không phải nhãn người đã xác nhận. Các script huấn luyện và đánh giá cần dữ liệu, checkpoint và cấu hình thí nghiệm tương ứng, không đi kèm bản clone. Không dùng kết quả fast để suy ra chất lượng classifier hoặc LLM. OCR hiện giới hạn tiếng Anh; chất lượng tiếng Việt chưa được chứng nhận.
