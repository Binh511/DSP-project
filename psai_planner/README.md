# Lập kế hoạch bằng LLM cục bộ có kiểm định nguồn

Gói này là bước suy luận tùy chọn sau Classifier và RAG. Không huấn luyện lại
Classifier hoặc retriever, không gọi API và không tự tải model. Model causal-LM
được cung cấp qua một thư mục Hugging Face đầy đủ trên máy.

```python
from psai_planner import plan_project

plan = plan_project(extraction_record, source_text,
                    retrieval=verified_retrieval_with_source_texts,
                    model_dir=local_model_folder,
                    config={"device": "cuda"})
```

LLM chọn và sắp xếp `task_id` từ danh mục các hoạt động/sản phẩm đã đủ điều kiện
trong quy trình lập kế hoạch hiện có. Tên công việc được khôi phục từ nguồn,
không lấy lời viết lại của LLM làm thông tin đã được chứng minh. Có thể tổ chức
công việc con qua `parent_task_id`. Checklist chỉ chứa đoạn trích nguyên văn
trong bằng chứng của chính công việc. Tài nguyên chỉ lấy từ bằng chứng tài
nguyên/công nghệ của dự án; việc gắn một tài nguyên vào công việc vẫn là gợi ý
cần người kiểm tra. Tài liệu bên ngoài chỉ là tài liệu tham khảo, không biến
thành yêu cầu dự án. Quan hệ phụ thuộc đều là ứng viên gợi ý cần kiểm tra.

Kiểm định bao gồm SHA-256 toàn văn, ID bằng chứng, vị trí ký tự, đoạn trích,
schema JSON, ID trùng, tham chiếu không biết và chu trình phụ thuộc/công việc
cha. Không chấp nhận trường ngày, thời lượng, ngân sách hoặc mô tả mới do LLM
tự thêm. Các con số/ngày đã xuất hiện trong đoạn nguồn chỉ được giữ nguyên như
đoạn trích, không được suy ra thành lịch hoặc ngân sách đã xác nhận.

Giới hạn mặc định: ngữ cảnh tổng 4.096 token, tối đa 3.072 token đầu vào và
1.024 token đầu ra; 12 công việc, 12 bằng chứng tài nguyên và 6 tham khảo.
Danh mục vượt số lượng có thống kê phần chưa đưa vào. Prompt quá dài báo lỗi
và yêu cầu giảm giới hạn danh mục; không cắt văn bản âm thầm. Kết quả lưu prompt,
phản hồi thô, danh mục nguồn, hash giao thức/code/model và tham số suy luận.
Suy luận greedy (`do_sample=False`); kernel GPU vẫn có thể khác biệt giữa môi
trường nên hash và phiên bản thư viện được ghi lại.

Nếu không có công việc đủ điều kiện, gói trả `status="abstained"` và không tải
LLM. Nếu LLM trả JSON sai hoặc nội dung không hợp lệ, gói báo lỗi rõ ràng, kèm
`raw_response`, `messages` và `provenance` trên ngoại lệ để lưu bằng chứng lỗi;
không tự chuyển sang quy tắc rồi báo là LLM đã chạy thành công.

**Giới hạn:** kiểm định hash/span chứng minh nguồn gốc câu chữ, không chứng
minh nhãn phân loại đúng, quan hệ suy luận đúng hoặc kế hoạch đủ mọi công việc.
Phân rã công việc, sự cần thiết của tài nguyên và quan hệ phụ thuộc vẫn cần
người duyệt. Kết quả là bản đề xuất có nguồn, không tự nhận đã đạt đánh giá
chất lượng đầu cuối.

Kiểm tra giao thức không dùng GPU:

```powershell
python -m unittest psai_planner.test_planner -v
```
