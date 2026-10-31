# Demo để đọc và thuyết trình

Đầu vào của hai bản demo là [proposal.md](../../examples/proposal.md), một ví dụ tự tạo. Các file HTML nhúng nguồn và có thể mở độc lập sau khi clone. Giữ report và workspace cùng thư mục để liên kết qua lại hoạt động.

| Bản demo | Đầu ra | Phạm vi |
|---|---|---|
| Fast | [Báo cáo](fast/report.html), [workspace](fast/workspace.html), [JSON](fast/analysis.json) | Quy tắc + BM25 + kế hoạch theo bằng chứng; chạy được không cần model |
| Full | [Báo cáo](full/report.html), [workspace](full/workspace.html), [JSON](full/analysis.json) | Ensemble production + hybrid BGE-M3 + Qwen trên máy tác giả |

**Xem HTML:** GitHub hiển thị mã HTML; clone hoặc tải file về rồi mở bằng trình duyệt để xem giao diện. Có thể xem trước [ảnh giao diện extension](../images/extension-demo.png) trực tiếp trên GitHub.

Các bản full và fast đã đi qua API tải tài liệu thực tế. [Biên bản fast](fast/acceptance.json) và [biên bản full](full/acceptance.json) ghi hash, chế độ, thời gian chạy và phạm vi kiểm tra. Thông tin đường dẫn trong JSON phản ánh máy tác giả khi tạo bằng chứng; không phải đường dẫn bắt buộc của máy nhận. Thư mục này chứa kết quả gọn để review; chạy lại sẽ tạo đủ request/response/log mới.

Full hoàn tất khoảng 96 giây trên môi trường đã ghi nhận. Thời gian đó không dự đoán hiệu năng trên máy khác. Đây là kiểm tra chức năng trên một ví dụ tự tạo, không đo accuracy độc lập, không xác nhận nhãn người hoặc độ đầy đủ của kế hoạch.

## Tái tạo fast từ clone

```sh
python tools/demo_project.py --fast
python tools/verify_clean_clone.py
```

Lệnh đầu in đường dẫn output mới. Lệnh thứ hai tự dựng venv trống và kiểm tra CLI/API thật. Bản mới có timestamp và đường dẫn riêng; nội dung provenance phản ánh máy chạy.

## Tái tạo full

Chuẩn bị dependency và model theo [hướng dẫn full](../FULL_RUNTIME_SETUP.md), rồi chạy `python tools/demo_project.py` với đường dẫn model thích hợp. Không trình bày fast như kết quả của model thật. [Hướng dẫn thuyết trình](../PRESENTATION_GUIDE.md) có dàn ý slide và câu hỏi phản biện.
