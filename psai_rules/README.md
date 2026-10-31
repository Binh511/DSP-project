# Nhánh quy tắc: dữ kiện nguyên văn có vị trí nguồn

```python
from psai_rules import extract_facts, resource_warnings

facts = extract_facts(document_id, source_text)
warnings = resource_warnings(extraction_record, validated_plan, facts)
```

Quy tắc nhận diện ngày ISO hợp lệ, ngày dạng ngày/tháng/năm, mã định danh có
nhãn, mốc có nhãn, thời lượng có số và đơn vị, số tiền có mã USD/EUR/VND hoặc
ký hiệu $, tên công nghệ trong danh mục cố định và số lượng nhân sự có đơn vị.
Mỗi dữ kiện giữ nguyên `quote`, vị trí ký tự Unicode, hash SHA-256 toàn văn và
ID ổn định theo nội dung/vị trí nguồn. Các kết quả trùng cùng trường và vị trí
được loại trùng; cùng từ xuất hiện ở hai vị trí vẫn là hai bằng chứng riêng.

Đầu ra có `value` chuẩn hóa và `normalization_policy`. Ngày `03/04/2027` giữ hai
cách hiểu, không tự chọn tháng/ngày. `$` không tự quy thành USD. Số tiền `1,500`
hoặc `1.500` giữ khả năng số thập phân và phân cách hàng nghìn. Không tự diễn
giải số tiền viết triệu/tỷ, khoảng tiền hoặc giá trị âm thành một số dương
khác. Chỉ ngày hợp lệ theo lịch mới được nhận diện; không suy đoán múi giờ.

`verified=True` chỉ xác nhận đoạn trích khớp văn bản, không xác nhận tài liệu
đúng hoặc dữ kiện là yêu cầu/nguồn lực bắt buộc. Tên công nghệ và số lượng nhân
sự không chứng minh khả dụng, phân công hoặc mức đủ của nguồn lực. Thời lượng
không tự chuyển thành lịch bắt đầu/kết thúc hoặc quan hệ phụ thuộc.

Cảnh báo thiếu thông tin dùng `status="not_detected"`: quy tắc chưa nhận diện,
không phải khẳng định tài liệu không có thông tin đó. Những dữ kiện nhận diện
được và liên kết tài nguyên/phụ thuộc do LLM gợi ý vẫn cần người kiểm tra.

Chạy kiểm tra:

```powershell
python -m unittest psai_rules.test_facts -v
```
