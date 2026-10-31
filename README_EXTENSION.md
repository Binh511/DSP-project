# PSAI Extension cho Chrome và Microsoft Edge

Bản demo v0.1.0 dùng giao diện Manifest V3 và backend Python chạy cục bộ. Bạn chọn một tệp PDF, Markdown hoặc TXT, gửi phân tích, theo dõi trạng thái rồi mở/tải báo cáo. Bản này cài bằng chế độ nhà phát triển; chưa phát hành lên Chrome Web Store hoặc Microsoft Edge Add-ons.

## 1. Cài giao diện extension

1. Với bản clone GitHub, chọn trực tiếp thư mục [extension/psai](./extension/psai/). Nếu muốn ZIP để chuyển máy, chạy `python tools/package_extension.py` trước; công cụ tạo `dist/extension/PSAI_EXTENSION_CHROME_EDGE_v0.1.0.zip`. Giải nén ZIP vào thư mục cố định và chọn thư mục chứa trực tiếp `manifest.json`.
2. Mở `chrome://extensions` trong Chrome hoặc `edge://extensions` trong Edge.
3. Bật **Developer mode / Chế độ nhà phát triển**.
4. Chọn **Load unpacked / Tải tiện ích đã giải nén**, rồi chọn thư mục ở bước 1.
5. Ghim PSAI vào thanh công cụ, bấm biểu tượng để mở giao diện.

Không chọn tệp ZIP trong hộp thoại Load unpacked. Giữ nguyên thư mục đã giải nén trong suốt thời gian sử dụng. Quy trình của Chrome có trong [hướng dẫn chính thức](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked).

## 2. Chạy backend trên Windows

Mở PowerShell tại thư mục gốc dự án, chạy:

```powershell
.\tools\start_extension_backend.ps1
```

[Trình khởi động](./tools/start_extension_backend.ps1) ưu tiên Python của môi trường `.venv/psai_runtime`, sau đó `.venv`, rồi lệnh `python` trong PATH. Giữ cửa sổ PowerShell mở trong khi dùng extension.

Nếu máy chặn việc chạy script, có thể dùng Python trực tiếp trong cửa sổ đó:

```powershell
.\.venv\psai_runtime\Scripts\python.exe -u -m psai_extension --port 8765
```

Backend in địa chỉ `http://127.0.0.1:8765` và token ghép nối ngẫu nhiên. Trong extension, nhập địa chỉ backend, dán token vào trường ghép nối rồi kết nối. Sau mỗi lần khởi động backend, token đổi và cần ghép nối lại. Không cần tài khoản hoặc API key bên ngoài.

### Chọn Python, model và tài liệu tham khảo

```powershell
.\tools\start_extension_backend.ps1 -Python '.\.venv\psai_runtime\Scripts\python.exe' `
  -BaseModelDir 'D:\PSAI_MODELS\bge-m3-snapshot' `
  -PlannerModelDir 'D:\PSAI_MODELS\planner-snapshot' `
  -ResourceDir 'D:\PSAI_RESOURCES'
```

Thay các đường dẫn ví dụ bằng thư mục có thật trên máy. `BaseModelDir` cần snapshot BGE-M3 đúng revision mà pipeline chấp nhận; `PlannerModelDir` cần model planner tương thích đã tải cục bộ. Các tham số tùy chọn còn có `-Port 8765` và `-DataDir 'D:\PSAI_JOBS'`. Nếu đổi cổng, cập nhật địa chỉ trong extension.

## 3. Phân tích tài liệu

1. Kết nối backend bằng token.
2. Chọn tệp `.pdf`, `.md` hoặc `.txt`, tối đa **20 MiB**. Có thể thử [proposal.md](./examples/proposal.md).
3. Chọn **Nhanh (fast)** để kiểm tra luồng mà không tải model phân loại, embedding hoặc LLM: dùng luật trích xuất, truy hồi từ khóa BM25 và lập kế hoạch theo bằng chứng. Kết quả fast không phải kết quả ensemble hoặc planner LLM. PDF quét vẫn cần OCR.
4. Chọn **Đầy đủ (full)** để chạy ensemble phân loại, truy hồi hybrid và planner cục bộ. Chế độ này cần đủ dependency, trọng số và tài nguyên máy; không tự tải model.
5. Gửi phân tích, chờ trạng thái hoàn tất rồi mở hoặc tải báo cáo. Hãy kiểm tra nội dung và bằng chứng trong báo cáo trước khi sử dụng.

Markdown/TXT ở chế độ fast chỉ cần Python chuẩn. PDF cần `pypdf`; PDF quét còn cần OCR và Pillow. Windows OCR dùng khả năng OCR sẵn có của Windows và gói ngôn ngữ đã cài. Chi tiết trong [hướng dẫn OCR](./psai_ocr/README.md) khi dùng toàn bộ dự án; bản backend đóng gói có [danh sách dependency OCR](./psai_ocr/requirements.txt).

Các job và báo cáo được lưu mặc định trong `outputs/extension/jobs`; nhật ký pipeline nằm trong từng thư mục job và phần cuối nhật ký hiển thị trong giao diện khi cần kiểm tra lỗi. Backend nhận tối đa bốn job đang chờ/chạy và xử lý tuần tự để tránh tải nhiều model cùng lúc. Đóng tab extension không xóa kết quả đã lưu; đóng backend sẽ ngắt kết nối. Khởi động lại với cùng thư mục dữ liệu để xem lại job đã hoàn tất; job bị gián đoạn cần gửi lại. Dừng backend bằng `Ctrl+C`.

## 4. Dùng gói backend trên máy khác

Gói `dist/extension/PSAI_EXTENSION_BACKEND_v0.1.0.zip` được tạo bằng `python tools/package_extension.py`, chứa mã nguồn backend, pipeline cần thiết, extension, tệp dependency, tài liệu mẫu và trình khởi động. Gói không kèm Python, môi trường ảo, trọng số model, checkpoint, tập dữ liệu hay tài liệu cá nhân. Đây là gói bàn giao mã nguồn, cần chuẩn bị môi trường cục bộ trước khi chạy full.

Giải nén vào thư mục mới. Với Python 3.11 trở lên đã cài, tạo môi trường để chạy fast:

```powershell
python -m venv .venv\psai_runtime
.\tools\start_extension_backend.ps1
```

Để đọc PDF có lớp chữ:

```powershell
.\.venv\psai_runtime\Scripts\python.exe -m pip install -r requirements-pdf.txt
```

Để chuẩn bị full, cài [dependency của pipeline](./requirements-project.txt), chọn bản Torch CPU/CUDA phù hợp với máy, và cung cấp các model cục bộ:

```powershell
.\.venv\psai_runtime\Scripts\python.exe -m pip install -r requirements-project.txt
```

Sao chép riêng ba thư mục classifier `models/final_s11`, `models/final_s22`, `models/final_s33` từ bản model đã được nghiệm thu vào đúng vị trí trong dự án đích. Mỗi thư mục phải giữ nguyên trọng số, tokenizer, cấu hình và `inference_config.json`. Truyền đường dẫn embedding và planner qua `-BaseModelDir` và `-PlannerModelDir`; không sao chép manifest chứa đường dẫn tuyệt đối của máy cũ. Các lệnh cài dependency cần mạng; quá trình phân tích dùng model đã có trên máy.

## 5. Quyền và dữ liệu

Giao diện gửi tệp bạn chủ động chọn tới backend tại địa chỉ loopback; backend chỉ lắng nghe `127.0.0.1`. Quyền truy cập host loopback cho phép extension gọi API Python theo [cơ chế host permissions của Chrome](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests). Token dùng để ghép nối với phiên backend hiện tại; giữ token trong máy. Tệp đầu vào, log và báo cáo tồn tại trong thư mục dữ liệu cho đến khi bạn chủ động xóa.

## 6. Kiểm tra luồng và xử lý lỗi

[Công cụ kiểm tra demo](./tools/verify_extension_demo.py) tự chạy backend trên cổng tạm, gửi tài liệu mẫu qua API thật, kiểm tra xác thực, đợi pipeline fast hoàn tất rồi kiểm tra liên kết báo cáo và hash đầu vào:

```powershell
python tools\verify_extension_demo.py
python tools\verify_extension_demo.py --full
```

Kết quả kiểm tra nằm trong thư mục mới bên dưới `outputs/extension/verify_*`, gồm `acceptance.json` và bằng chứng từng job. Đây là kiểm tra chức năng trên tài liệu mẫu, không phải đánh giá độ chính xác mô hình. Có thể chỉ định Python chạy pipeline bằng `--python`.

Kiểm tra extension được nạp thật trên Edge với Node.js 22 trở lên, không cần npm:

```powershell
node tools/verify_extension_browser.mjs
```

Công cụ dùng hồ sơ Edge tạm trong thư mục output, chạy luồng fast, kiểm tra mở workspace, lưu kết nối/lịch sử và giao diện ở chiều rộng 390 px; sau đó đóng các tiến trình do công cụ tạo. Mặc định tìm Edge ở `C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe`; có thể đặt biến môi trường `PSAI_EDGE_PATH` nếu Edge được cài ở vị trí khác. Một số sandbox chặn tiến trình GPU của Edge nên cần chạy kiểm tra trình duyệt trong môi trường cho phép.

- **Không kết nối được:** kiểm tra backend đang chạy, địa chỉ/cổng trùng với thông tin in trên console và token của phiên hiện tại. Dùng `127.0.0.1` trong địa chỉ.
- **Cổng đã được sử dụng:** khởi động với `-Port 8766` và cập nhật địa chỉ trong extension.
- **Thiếu model/dependency khi chạy full:** dùng fast để kiểm tra giao diện; chuẩn bị model và Python đúng môi trường để chạy full. Xem thông báo lỗi và log của job.
- **Không đọc được PDF quét:** kiểm tra dependency OCR, ngôn ngữ Windows OCR và bản PDF có thể đọc được. Có thể thử TXT/Markdown để xác minh luồng cơ bản.
- **Vừa cập nhật extension:** mở trang quản lý extension, bấm Reload, sau đó mở lại giao diện.

## 7. Đóng gói lại

Đóng gói từ gốc dự án:

```powershell
python tools\package_extension.py
```

[Công cụ đóng gói](./tools/package_extension.py) tạo hai ZIP trong `dist/extension` và `package_receipt.json` với SHA-256 từng tệp và từng ZIP, đồng thời kiểm tra CRC. Đây là output tạo cục bộ, không có sẵn trong clone. ZIP extension đặt manifest ở gốc. Công cụ chỉ lấy mã nguồn trong danh sách cho phép; không lấy model, môi trường, secrets hoặc kết quả cũ. Chạy lại với nguồn giống hệt cho ra cùng bytes; nguồn thay đổi sẽ không ghi đè bản phát hành cũ. Dùng thư mục đầu ra mới khi tạo bản khác:

```powershell
python tools\package_extension.py --output-dir dist\extension-next
```
