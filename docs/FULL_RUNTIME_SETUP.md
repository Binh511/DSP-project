# Chạy đầy đủ model trên máy nhận

Demo fast trong [README](../README.md) chạy từ bản clone với Python chuẩn. Full dùng ensemble classifier, BGE-M3 và Qwen; cần chuẩn bị các model dưới đây. Runtime không huấn luyện lại hoặc tự tải model trong quá trình phân tích.

## Chuẩn bị môi trường

Tại gốc dự án, tạo môi trường riêng:

```powershell
python -m venv .venv/psai_runtime
.venv/psai_runtime/Scripts/python.exe -m pip install -r requirements-project.txt
```

Trên macOS/Linux, đường dẫn Python là `.venv/psai_runtime/bin/python`. Các phiên bản trong [requirements RAG](../requirements-rag.txt) ghi nhận môi trường Windows đã dùng; chọn Torch CPU/CUDA phù hợp máy nhận. Cài dependency cần mạng; runtime dùng model đã có trên máy.

Môi trường đã nghiệm thu dùng Python 3.13 và NVIDIA RTX 4050 6 GB. Chưa chứng nhận tốc độ/mức dùng RAM cho mọi máy khác. CPU có thể chậm. Xem [OCR](../psai_ocr/README.md) nếu cần PDF scan.

## Những model cần chuyển riêng

| Model | Vị trí trên máy nhận | Thành phần |
|---|---|---|
| Classifier seed 11 | `models/final_s11/` | Weights, config, tokenizer, `inference_config.json` |
| Classifier seed 22 | `models/final_s22/` | Export đầy đủ cùng cấu trúc |
| Classifier seed 33 | `models/final_s33/` | Export đầy đủ cùng cấu trúc |
| BGE-M3 base | Snapshot cục bộ, truyền qua `--base-model-dir` | Weights, tokenizer, pooling/normalize config, đúng revision |
| Qwen2.5-1.5B-Instruct | Model cục bộ, truyền qua `--planner-model-dir` | Weights, config, tokenizer, generation config |

Các đường dẫn trên là tài nguyên sau khi chép model, không có sẵn trong clone. **Ba export classifier production do tác giả cung cấp.** Classifier nghiên cứu v2 và các model k-fold lịch sử không thay thế ba export này. Không dùng `git lfs pull` để coi model lịch sử là production mặc định.

Tác giả chia sẻ các thư mục export nguyên vẹn qua ổ đĩa hoặc kênh chuyển file thống nhất với người nhận. Gói model phải tách khỏi tài liệu/nhãn riêng. Bản GitHub không chứa weights hoặc đường link tải do tác giả chưa cung cấp.

BGE-M3 dùng revision `5617a9f61b028005a4858fdac845db406aefb181`; giữ tên thư mục snapshot revision và cấu hình pooling/normalize. Qwen đã nghiệm thu dùng revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. [Manifest](../reports/project_completion/PROJECT_RELEASE_MANIFEST.json) và [biên bản chạy](../reports/project_completion/latest_acceptance.json) có hash để đối chiếu.

Nếu muốn tự tải Qwen base từ Hugging Face, sau khi cài dependency, chạy [công cụ setup](../tools/setup_project_planner_model.py). Đây là thao tác tải model riêng, cần mạng:

```powershell
.venv/psai_runtime/Scripts/python.exe tools/setup_project_planner_model.py
```

Công cụ tạo manifest đường dẫn của máy nhận. Không sao chép `local_planner_model.json` chứa đường dẫn tuyệt đối của máy tác giả. BGE-M3 có thể được chuẩn bị từ snapshot đã kiểm tra của tác giả; runtime không tự tải thay.

## Chạy và kiểm tra

Kiểm tra tài nguyên classifier:

```powershell
.venv/psai_runtime/Scripts/python.exe psai.py doctor
```

Doctor không thay thế việc chạy full. Thay hai đường dẫn ví dụ dưới đây bằng thư mục model thật:

```powershell
.venv/psai_runtime/Scripts/python.exe tools/demo_project.py --input examples/proposal.md --device auto --base-model-dir "D:/PSAI_MODELS/BGE-M3/snapshots/5617a9f61b028005a4858fdac845db406aefb181" --planner-model-dir "D:/PSAI_MODELS/Qwen"
```

Lệnh in các bước extract/retrieve/plan và đường dẫn báo cáo/workspace. Mỗi lượt tạo output mới. Khi có lỗi, giữ log để xem nguyên nhân; không ghi nhận full đạt chỉ vì fast chạy được.

Để dùng extension với cùng tài nguyên:

```powershell
.venv/psai_runtime/Scripts/python.exe -m psai_extension --base-model-dir "D:/PSAI_MODELS/BGE-M3/snapshots/5617a9f61b028005a4858fdac845db406aefb181" --planner-model-dir "D:/PSAI_MODELS/Qwen"
```

Ghép nối theo [hướng dẫn extension](../README_EXTENSION.md), sau đó chọn **Đầy đủ**. Backend chạy từng model tuần tự trước khi chuyển bước tiếp theo.

## Khi review và thuyết trình

Ghi phiên bản Python/Torch, thiết bị, commit và chế độ demo. Đối chiếu nguồn → bằng chứng → công việc, không chỉ hash/schema. Xem [hướng dẫn review](REVIEW_GUIDE.md) và [kịch bản thuyết trình](PRESENTATION_GUIDE.md).

Các báo cáo/manifest nghiên cứu mô tả những lần chạy đã lưu. Chạy lại study hoặc các công cụ `finalize_*` lịch sử cần đúng dữ liệu, nhãn, checkpoint và artifact của study. Bản clone cùng demo fast không cung cấp các tài nguyên này.
