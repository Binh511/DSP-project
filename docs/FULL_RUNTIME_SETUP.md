# Thiết lập runtime đầy đủ

Chế độ full chạy ensemble classifier, truy hồi hybrid với BGE-M3 và planner Qwen cục bộ. Cần Python 3.11 trở lên, dependency tương thích với thiết bị và các model đã có trên máy. Runtime suy luận offline, không huấn luyện hoặc tự tải model.

Nếu chỉ cần kiểm tra Markdown/TXT, dùng [chế độ fast](../README.md) với thư viện chuẩn Python.

## Môi trường Python

Tại gốc dự án, trên Windows:

```powershell
python -m venv .venv/psai_runtime
.venv/psai_runtime/Scripts/python.exe -m pip install -r requirements-project.txt
```

Trên Linux/macOS:

```sh
python3 -m venv .venv/psai_runtime
.venv/psai_runtime/bin/python -m pip install -r requirements-project.txt
```

[requirements-project.txt](../requirements-project.txt) tập hợp dependency runtime. Chọn bản PyTorch CPU/CUDA phù hợp với thiết bị. Việc cài dependency hoặc chuẩn bị model có thể cần mạng; đó là bước thiết lập tách khỏi quá trình phân tích.

Trình demo trên Windows tự ưu tiên `.venv/psai_runtime/Scripts/python.exe` khi có sẵn. Trên Linux/macOS, gọi rõ `.venv/psai_runtime/bin/python` trong các lệnh để dùng đúng môi trường.

## Model cục bộ

| Thành phần | Vị trí/cấu hình | Nội dung cần giữ |
|---|---|---|
| Classifier seed 11 | `models/final_s11/` | Weights, config, tokenizer, `inference_config.json` |
| Classifier seed 22 | `models/final_s22/` | Export đầy đủ cùng cấu trúc |
| Classifier seed 33 | `models/final_s33/` | Export đầy đủ cùng cấu trúc |
| BGE-M3 | `--base-model-dir` | Snapshot đầy đủ, tokenizer và cấu hình pooling/normalize |
| Qwen2.5-1.5B-Instruct | `--planner-model-dir` | Weights, config, tokenizer và generation config |

Ba export classifier phải nằm đúng vị trí trên. Checkpoint thí nghiệm khác không tự thay thế ensemble này.

BGE-M3 dùng revision `5617a9f61b028005a4858fdac845db406aefb181`; giữ thư mục snapshot mang tên revision cùng cấu hình tương ứng. Qwen dùng revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Chuẩn bị các thư mục model đầy đủ trước khi chạy, rồi truyền đường dẫn bằng hai cờ trên.

Kho mã không kèm trọng số, dữ liệu nghiên cứu, báo cáo sinh ra hoặc môi trường ảo. Không dựa vào manifest chứa đường dẫn tuyệt đối của máy khác. Việc tái lập huấn luyện/đánh giá còn cần dữ liệu, nhãn và cấu hình của thí nghiệm; runtime full không yêu cầu huấn luyện lại.

## Chạy pipeline

Kiểm tra tài nguyên classifier:

```powershell
.venv/psai_runtime/Scripts/python.exe psai.py doctor
```

Thay hai đường dẫn model sau bằng thư mục thực tế trên máy:

```powershell
.venv/psai_runtime/Scripts/python.exe tools/demo_project.py --input examples/PSAI_DEMO_01_MINER_MONITORING.md --device auto --base-model-dir "D:/PSAI_MODELS/BGE-M3/snapshots/5617a9f61b028005a4858fdac845db406aefb181" --planner-model-dir "D:/PSAI_MODELS/Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
```

Trên Linux/macOS, dùng cùng tham số với `.venv/psai_runtime/bin/python` và đường dẫn model tương ứng. Có thể đổi đầu vào sang [mẫu bảo trì phòng thí nghiệm](../examples/PSAI_DEMO_02_LAB_MAINTENANCE.md). Với PDF, xem [dependency và giới hạn OCR](../psai_ocr/README.md).

Các bước model chạy tuần tự trong worker riêng. Mỗi lượt tạo thư mục mới chứa `report.html`, `workspace.html`, `analysis.json`, văn bản nguồn, metadata và log. Lỗi được lưu trong thư mục chẩn đoán riêng. Nếu dùng `--output`, chọn đường dẫn chưa tồn tại.

Planner v2 trong pipeline mặc định giữ các công việc đủ điều kiện trong catalog bằng `preserve_source_catalog`. Raw response và nguồn lựa chọn được lưu để kiểm tra. Các liên kết vẫn là đề xuất cần rà soát; xem [giao thức planner](../psai_planner/README.md).

## Dùng extension

Khởi động backend với cùng model:

```powershell
.venv/psai_runtime/Scripts/python.exe -m psai_extension --base-model-dir "D:/PSAI_MODELS/BGE-M3/snapshots/5617a9f61b028005a4858fdac845db406aefb181" --planner-model-dir "D:/PSAI_MODELS/Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
```

Trên Linux/macOS, dùng Python trong `.venv/psai_runtime/bin/python`. Backend in URL loopback và token ghép nối mới mỗi lần khởi động. Giữ terminal mở, kết nối giao diện theo [README_EXTENSION](../README_EXTENSION.md), rồi chọn **Đầy đủ**.

Khi thiếu dependency hoặc model, kiểm tra thông báo và log của bước lỗi. `doctor` kiểm tra cấu hình/tài nguyên; chạy fast thành công chỉ xác nhận luồng fast, chưa xác nhận suy luận full hoặc chất lượng đầu ra.
