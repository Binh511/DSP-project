# Cấu trúc mã ứng dụng sau refactor v0.2.0

Luồng sử dụng vẫn là tài liệu → extraction → retrieval → planning → report/workspace. API, định dạng job và chế độ fast/full được giữ tương thích. Bản refactor tập trung vào phân chia trách nhiệm và khả năng kiểm tra; không huấn luyện model hoặc thay ngưỡng/quy tắc nghiên cứu.

## Backend extension

| Thành phần | Trách nhiệm |
|---|---|
| [config.py](../psai_extension/config.py) | Giới hạn, đường dẫn và hằng số giao thức |
| [uploads.py](../psai_extension/uploads.py) | Kiểm tra tên file, nội dung và các tùy chọn upload |
| [jobs.py](../psai_extension/jobs.py) | Lưu trạng thái, hàng đợi, khôi phục và công bố kết quả |
| [pipeline.py](../psai_extension/pipeline.py) | Chạy pipeline và quản lý tiến trình con |
| [http.py](../psai_extension/http.py) | Định tuyến HTTP, xác thực và phục vụ kết quả |
| [cli.py](../psai_extension/cli.py) | Tham số khởi động và vòng đời server |
| [server.py](../psai_extension/server.py) | Điểm import tương thích cho script/test hiện có |

Chạy backend bằng `python -m psai_extension`. Mã ghép nối vẫn thay đổi sau mỗi lần khởi động. Các job hoàn thành được giữ trên đĩa; job bị gián đoạn không tự chạy lại.

## Pipeline Python

[execution.py](../psai_project/execution.py) tập trung cách chọn Python, biến môi trường offline, tham số đường dẫn và lệnh chạy fast/full. [runtime.py](../psai_project/runtime.py) điều phối các worker và việc xuất kết quả. [worker.py](../psai_project/worker.py) tiếp tục thực hiện từng bước model trong tiến trình riêng.

Những script demo dùng chung cách tạo lệnh, giúp CLI và extension giữ cùng lựa chọn chế độ. Các báo cáo và dữ liệu nghiên cứu đã lưu mô tả phiên bản/lần chạy của chúng; refactor không cập nhật lại số liệu lịch sử.

## Frontend extension

| Thành phần | Trách nhiệm |
|---|---|
| [app.mjs](../extension/psai/app.mjs) | Khởi tạo ứng dụng |
| [controller.mjs](../extension/psai/controller.mjs) | Kết nối, polling và vòng đời tác vụ |
| [api-client.mjs](../extension/psai/api-client.mjs) | HTTP, token, timeout và chuẩn hóa phản hồi |
| [view.mjs](../extension/psai/view.mjs) | Hiển thị trạng thái, lịch sử, kết quả và chọn file |
| [core.mjs](../extension/psai/core.mjs) | Các hàm kiểm tra dữ liệu thuần, không phụ thuộc DOM |
| [app.html](../extension/psai/app.html), [app.css](../extension/psai/app.css) | Cấu trúc và kiểu hiển thị được định dạng để đọc/sửa |

## Kiểm tra và xuất bản cục bộ

Các lệnh unit test áp dụng trên bộ mã nguồn dự án. Gói backend phân phối chỉ kèm các công cụ smoke test cần thiết để kiểm tra luồng chạy, không kèm toàn bộ test và artifact nghiên cứu.

```sh
python -m unittest discover -s tests -p "test_extension_*.py"
python -m unittest discover -s tests -p "test_project_*.py"
node --test extension/psai/tests/*.test.mjs
python tools/verify_clean_clone.py
python tools/package_extension.py
```

[Công cụ đóng gói](../tools/package_extension.py) kiểm tra phiên bản backend/manifest, module Python và tài sản frontend trước khi tạo ZIP. Mỗi phiên bản xuất vào thư mục riêng, kèm SHA-256, CRC và thư mục extension đã giải nén sẵn để Load unpacked. Gói backend chỉ chứa source cần thiết; model và dữ liệu riêng được cung cấp theo [hướng dẫn full](FULL_RUNTIME_SETUP.md).
