# OCR cục bộ, có nguồn theo trang

Mô-đun này bổ sung đường xử lý PDF scan mà không gọi API, không sinh caption và không tự tải model. Tài liệu có text sẵn vẫn giữ text từ `pypdf`. OCR là bước nhận dạng có thể sai, chưa phải văn bản đã được người kiểm định.

## Tình trạng kiểm chứng

- Đã có kiểm thử bằng engine và renderer giả để kiểm tra lựa chọn trang, bảo toàn text, nguồn, offset, hash và lỗi phụ thuộc.
- Đã chạy OCR thật bằng `Windows.Media.Ocr` và render bằng `Windows.Data.Pdf` có sẵn trên Windows; không cài hoặc tải gì. PDF tự tạo có một trang selectable và một trang scan: trang đầu giữ nguyên, trang sau nhận dạng được 5 dòng/24 từ. CER/WER chuẩn hóa đều bằng 0 trên riêng ví dụ tiếng Anh rõ nét này. Đây là smoke test chức năng, không phải độ chính xác tổng quát hoặc final test.
- Biên bản thực chạy: [smoke_receipt.json](../outputs/ocr_verification/20261029T033512Z_f4ktz5s1/smoke_receipt.json); [kết quả trích xuất](../outputs/ocr_verification/20261029T033512Z_f4ktz5s1/extraction.json). Lặp kiểm tra bằng `python -m psai_ocr.verify_native`.
- Chưa có `pypdfium2` và `rapidocr`; đường chạy RapidOCR chưa được kiểm chứng thực trên máy. Yêu cầu kiểm tra phiên bản qua mạng đã bị từ chối và không được thử lại; Windows native là giải pháp dùng khả năng có sẵn, không phải thử lại lệnh mạng đó.
- Không lấy số test bằng engine giả làm bằng chứng chất lượng nhận dạng chữ.
- Phiên bản đầu giới hạn `en` (tiếng Anh); Windows đang có language pack `en-US`. Chưa công bố hỗ trợ tiếng Việt; cần model tương ứng và kiểm chứng riêng trước khi mở lựa chọn đó.

## Chạy ngay trên máy Windows hiện tại

```powershell
.venv\psai_runtime\Scripts\python.exe -m psai_ocr --doctor
.venv\psai_runtime\Scripts\python.exe -m psai_ocr --input examples/scanned.pdf --output outputs/ocr_trial/extraction.json --mode auto --backend windows --page-image-dir outputs/ocr_trial/pages
```

Thay đường dẫn PDF ví dụ bằng tài liệu thật. `--backend auto` ưu tiên RapidOCR nếu đủ phụ thuộc local, nếu thiếu thì dùng Windows native; chỉ chuyển backend khi thiếu phụ thuộc, không che lỗi render/nhận dạng thực tế. `--backend rapidocr` hoặc `windows` chọn rõ một engine.

Helper [windows_native.ps1](./windows_native.ps1) được Python gọi bằng **danh sách argument**, không ghép đường dẫn vào code PowerShell, không mở GUI. Do máy chặn script `.ps1` theo mặc định, chỉ **process con** chạy script local này nhận `-NoProfile -NonInteractive -ExecutionPolicy Bypass`. Không gọi `Set-ExecutionPolicy`, không thay policy lưu trên máy, không cần admin. Nếu AppLocker/Group Policy chặn helper thì báo lỗi và dừng; không tìm cách vượt chặn đó.

Windows không cung cấp confidence hoặc file model weights qua API: kết quả ghi `confidence=null`, OS build, language pack và nguồn model là khả năng OCR đã cài trong Windows. Không bịa SHA256 model hoặc confidence. Hash của PDF, text và ảnh trang vẫn được ghi đầy đủ.

## Cài khi được cho phép

Cài tùy chọn RapidOCR riêng trong môi trường runtime khi được người dùng cho phép, không thay thư viện classifier. Windows native hiện tại không cần bước này:

```powershell
.venv\psai_runtime\Scripts\python.exe -m pip install -r psai_ocr/requirements.txt
.venv\psai_runtime\Scripts\python.exe -m psai_ocr --doctor
```

Chỉ dùng wheel RapidOCR có đủ model ONNX local cho detection, orientation và recognition; hoặc cung cấp `--model-dir` có đúng một file chứa thành phần `_det`, `_cls`, `_rec` tương ứng (chấp nhận hậu tố `_infer.onnx`). Thiếu file sẽ báo lỗi rõ ràng trước suy luận, không tự download. Engine ghi SHA256 model vào kết quả.

## Chạy

```powershell
.venv\psai_runtime\Scripts\python.exe -m psai_ocr --input examples/scanned.pdf --output outputs/ocr_trial/extraction.json --mode auto --page-image-dir outputs/ocr_trial/pages
```

Đường dẫn PDF trên là ví dụ: thay bằng tài liệu thực tế. Output đã có sẽ không bị ghi đè.

- `none`: chỉ dùng selectable text, không tải engine/renderer.
- `auto`: OCR trang có ít hơn 32 ký tự không phải khoảng trắng. Giữ text cũ nếu OCR cho ít text hơn; ghi nhận lần OCR thử. Chưa xử lý tốt trường hợp một trang có cả nhiều text selectable và một ảnh chứa nội dung bổ sung. Với trang đó, dùng `always` sau khi xác định cần OCR.
- `always`: OCR toàn bộ trang; nếu không có chữ nhận dạng được thì giữ selectable text để tránh làm mất dữ liệu, đồng thời phát cảnh báo.

Chế độ auto/always là lựa chọn rõ ràng của người chạy. Không trộn text selectable và text OCR bằng phép nối cơ học vì có thể lặp nội dung. Không suy đoán bảng, hình hoặc phần chữ không nhìn rõ.

## API tích hợp

```python
from psai_ocr import extract_pdf_with_ocr

result = extract_pdf_with_ocr(pdf_path, mode='auto', backend='auto', cache_dir=page_image_dir)
text = result['text']
pages = result['pages']
```

Kết quả gồm PDF SHA256, text SHA256, phương pháp từng trang, số dòng, char span `[start,end)`, text gốc SHA256, thời gian, PNG SHA256 và kích thước khi render. Box là tọa độ pixel trên ảnh trang, không phải tọa độ PDF. Box từng dòng và box từng từ được ghi riêng; không đặt tên box dòng thành box từ.

Các OCR span chỉ trỏ tới text hợp nhất nếu text OCR được chọn. Nếu giữ selectable text, span OCR có `span_space=page_ocr_candidate`, tránh nhầm với nguồn dùng cho RAG. `pdf_visual_review_complete` luôn `false`; ứng dụng không được tự chuyển thành `true` chỉ vì có confidence cao.

## Nghiệm thu sau khi cài phụ thuộc

1. Tạo hoặc chọn PDF chỉ chứa ảnh có nội dung tiếng Anh biết trước.
2. Kiểm tra `none` báo không có text; `auto` thực sự nhận dạng được các từ và số quan trọng.
3. Đối chiếu text với ảnh gốc, đo CER/WER trên transcript do người xác nhận; không dùng chính output OCR làm transcript chuẩn.
4. Chạy PDF trộn một trang selectable và một trang scan, xác nhận chỉ trang scan bị OCR và char span/nguồn không đổi.
5. Kiểm tra báo cáo cuối giữ cảnh báo OCR và dẫn đúng trang. Bảng, bố cục nhiều cột, ảnh mờ và chữ nghiêng cần tập kiểm tra riêng.

## Nguồn kỹ thuật

- [RapidOCR chính thức](https://github.com/RapidAI/RapidOCR): package hợp nhất, CPU ONNX và model local.
- [Hướng dẫn cài RapidOCR](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/install/).
- [API render của pypdfium2](https://pypdfium2.readthedocs.io/en/stable/python_api.html).
- [Windows OCR API](https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr.ocrengine).
- [Windows PDF API](https://learn.microsoft.com/en-us/uwp/api/windows.data.pdf.pdfdocument).
