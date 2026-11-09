# OCR PDF cục bộ

Mô-đun đọc lớp chữ PDF bằng `pypdf` và nhận dạng trang quét khi được yêu cầu. OCR chạy cục bộ, không gọi API hoặc tự tải model. Văn bản nhận dạng vẫn có thể sai và cần đối chiếu với trang gốc.

## Dependency và backend

PDF có lớp chữ chỉ cần [requirements-pdf.txt](../requirements-pdf.txt). PDF quét cần Pillow và một backend:

- `windows`: dùng `Windows.Data.Pdf` và `Windows.Media.Ocr` có sẵn trên Windows, với gói ngôn ngữ tiếng Anh `en-US` đã cài.
- `rapidocr`: dùng PDFium để render và RapidOCR/ONNX Runtime cho OCR CPU; cài dependency trong [requirements.txt](requirements.txt), chuẩn bị đủ model ONNX cục bộ.
- `auto`: ưu tiên RapidOCR khi đủ dependency/model; chuyển sang Windows native khi thiếu dependency. Lỗi render hoặc nhận dạng thực tế được báo ra.

Bản hiện tại chỉ nhận `--language en`. Chưa chứng nhận chất lượng OCR tiếng Việt, bảng, bố cục nhiều cột hoặc ảnh mờ.

Ví dụ cài dependency trong môi trường riêng:

```powershell
.venv/psai_runtime/Scripts/python.exe -m pip install -r requirements-pdf.txt
.venv/psai_runtime/Scripts/python.exe -m pip install Pillow
```

Nếu chọn RapidOCR, cài thêm:

```powershell
.venv/psai_runtime/Scripts/python.exe -m pip install -r psai_ocr/requirements.txt
```

RapidOCR cần đúng một model cho mỗi thành phần `det`, `cls`, `rec` trong thư mục model của package hoặc `--model-dir`. Thiếu model sẽ báo lỗi trước suy luận. Cài dependency là bước thiết lập riêng, có thể cần mạng.

## Chạy CLI

```powershell
.venv/psai_runtime/Scripts/python.exe -m psai_ocr --doctor
.venv/psai_runtime/Scripts/python.exe -m psai_ocr --input "D:/documents/input.pdf" --output outputs/ocr_trial/extraction.json --mode auto --backend windows --page-image-dir outputs/ocr_trial/pages
```

Thay đường dẫn PDF bằng tệp thực tế; chọn đường dẫn output mới mỗi lần. Trên Linux/macOS, dùng `.venv/psai_runtime/bin/python` và backend `rapidocr` với model cục bộ. `--doctor` chỉ kiểm tra dependency/khả năng backend; không thực hiện OCR.

| Chế độ | Hành vi |
|---|---|
| `none` | Chỉ đọc lớp chữ, không nạp engine OCR hoặc renderer |
| `auto` | OCR trang có ít hơn 32 ký tự không phải khoảng trắng; giữ lớp chữ nếu OCR thu được ít chữ hơn |
| `always` | OCR mọi trang; giữ lớp chữ và cảnh báo nếu không nhận dạng được chữ |

Trang có nhiều chữ selectable nhưng còn nội dung nằm trong ảnh có thể bị bỏ qua trong `auto`; cần rà soát và chọn `always` khi phù hợp. Hệ thống chọn văn bản theo trang, không nối cơ học hai bản text có thể trùng nhau.

## API và dữ liệu nguồn

```python
from psai_ocr import extract_pdf_with_ocr

result = extract_pdf_with_ocr(
    pdf_path, mode="auto", backend="auto", cache_dir=page_image_dir
)
text = result["text"]
pages = result["pages"]
```

Kết quả giữ hash PDF/text, phương pháp từng trang, span ký tự và thông tin ảnh trang khi render. Tọa độ box tính bằng pixel trên ảnh, không phải tọa độ PDF; box dòng và box từ là hai trường riêng.

Span OCR chỉ trỏ vào toàn văn hợp nhất khi text OCR được chọn. Nếu giữ lớp chữ, span OCR ứng viên có `span_space=page_ocr_candidate`. Windows OCR không cung cấp confidence hoặc weights qua API; kết quả ghi `confidence=null` cùng thông tin hệ điều hành/ngôn ngữ. RapidOCR ghi hash các model cục bộ.

`pdf_visual_review_complete` giữ giá trị `false`; confidence hoặc hash không thay cho việc đọc lại trang gốc. Kiểm tra tên, số liệu, thứ tự đọc và bảng trước khi dùng văn bản cho phân tích. Unit test với engine giả kiểm tra luồng và provenance, không đo chất lượng nhận dạng. Có thể chạy kiểm tra chức năng Windows bằng `python -m psai_ocr.verify_native`; đánh giá OCR cần transcript đối chiếu được xác nhận riêng.
