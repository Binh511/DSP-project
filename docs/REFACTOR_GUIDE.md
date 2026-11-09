# Cấu trúc mã ứng dụng

Luồng chính: tài liệu → văn bản nguồn → trích xuất → truy hồi → lập kế hoạch → báo cáo/workspace. CLI và extension dùng chung cấu hình fast/full.

## Pipeline Python

| Module | Trách nhiệm |
|---|---|
| [tools/demo_project.py](../tools/demo_project.py) | Điểm vào demo, chọn chế độ và tạo thư mục kết quả mới |
| [execution.py](../psai_project/execution.py) | Chọn Python, dựng tham số lệnh và môi trường offline |
| [runtime.py](../psai_project/runtime.py) | Đọc nguồn, chạy worker tuần tự, kiểm tra hash và công bố kết quả |
| [worker.py](../psai_project/worker.py) | Thực hiện từng bước extract, retrieve và plan trong tiến trình riêng |
| [report.py](../psai_project/report.py) | Xuất HTML và dữ liệu phân tích có dẫn nguồn |
| [workspace.py](../psai_project/workspace.py) | Tạo workspace theo dõi trạng thái công việc |

Fast dùng quy tắc, BM25 và kế hoạch theo thứ tự nguồn. Full dùng ensemble ba classifier, truy hồi hybrid và planner cục bộ. Worker thoát trước khi bước model tiếp theo bắt đầu; log và phản hồi từng bước được giữ để kiểm tra lỗi.

[Planner v2](../psai_planner/README.md) xây ngữ cảnh rà soát từ toàn văn và dùng danh mục ID cho phản hồi model. Pipeline full mặc định bật `preserve_source_catalog`: thứ tự model chọn được giữ trước, các công việc còn lại trong catalog được nối theo thứ tự nguồn. Trường `selection_origin` phân biệt công việc model chọn và công việc được phần mềm giữ lại. Chính sách này không tự thêm liên kết hoặc chứng nhận mức bao phủ; API planner trực tiếp vẫn mặc định `select_only`.

Kết quả được dựng trong thư mục tạm rồi công bố vào đường dẫn mới sau khi kiểm tra. Nội dung nguồn và hash được đối chiếu trước khi xuất báo cáo. Lượt lỗi giữ dữ liệu chẩn đoán riêng, không công bố như một báo cáo hoàn tất.

## Backend extension

| Module | Trách nhiệm |
|---|---|
| [config.py](../psai_extension/config.py) | Giới hạn, đường dẫn và hằng số giao thức |
| [uploads.py](../psai_extension/uploads.py) | Kiểm tra tệp và tùy chọn đầu vào |
| [jobs.py](../psai_extension/jobs.py) | Hàng đợi, trạng thái, lưu và khôi phục lịch sử job |
| [pipeline.py](../psai_extension/pipeline.py) | Khởi chạy và theo dõi tiến trình pipeline |
| [http.py](../psai_extension/http.py) | HTTP, xác thực và phục vụ kết quả |
| [cli.py](../psai_extension/cli.py) | Tham số khởi động và vòng đời server |
| [server.py](../psai_extension/server.py) | Điểm import tương thích |

Backend chạy bằng `python -m psai_extension`, lắng nghe trên loopback và tạo token ghép nối mới mỗi phiên. Job hoàn tất được giữ trên đĩa; job bị gián đoạn cần gửi lại. Chi tiết sử dụng ở [hướng dẫn extension](../README_EXTENSION.md).

## Giao diện extension

| Module | Trách nhiệm |
|---|---|
| [app.mjs](../extension/psai/app.mjs) | Khởi tạo |
| [controller.mjs](../extension/psai/controller.mjs) | Kết nối, polling và vòng đời tác vụ |
| [api-client.mjs](../extension/psai/api-client.mjs) | HTTP, token, timeout và chuẩn hóa phản hồi |
| [view.mjs](../extension/psai/view.mjs) | Chọn tệp, trạng thái, lịch sử và kết quả |
| [core.mjs](../extension/psai/core.mjs) | Hàm kiểm tra dữ liệu độc lập với DOM |
| [app.html](../extension/psai/app.html), [app.css](../extension/psai/app.css) | Cấu trúc và kiểu hiển thị |

## Kiểm tra và đóng gói

```sh
python -m unittest discover -s tests -p "test_project_*.py"
python -m unittest discover -s tests -p "test_extension_*.py"
node --test extension/psai/tests/*.test.mjs
python tools/verify_clean_clone.py
python tools/package_extension.py
```

Kiểm thử JavaScript cần Node.js 22 trở lên. [Công cụ đóng gói](../tools/package_extension.py) kiểm tra phiên bản và thành phần cần thiết, rồi tạo ZIP backend cùng extension có checksum. Trọng số model, dữ liệu, báo cáo sinh ra, môi trường ảo và metadata Git không được đóng trong gói backend. Chuẩn bị model theo [thiết lập full](FULL_RUNTIME_SETUP.md).
