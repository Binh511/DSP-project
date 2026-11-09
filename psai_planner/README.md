# Planner cục bộ và ngữ cảnh rà soát nguồn

## Protocol v2 mặc định

Không train lại model. V2 giữ task nguồn đủ điều kiện, đồng thời cho planner đọc
ngữ cảnh nguyên văn từ toàn tài liệu, kể cả thông tin classifier đã bỏ sót.

```python
from psai_planner import build_review_context, validate_review_context, plan_project

context = build_review_context(record, source_text, retrieval)
validate_review_context(context, record, source_text, retrieval)
plan = plan_project(record, source_text, retrieval=retrieval,
                    model_dir=local_model_folder, config={"device": "cuda"})
```

`build_review_context` phân nhóm câu theo dấu hiệu từ vựng/tiêu đề: yêu cầu,
nghiệm thu, phụ thuộc, rủi ro, nhân sự, chỉ số và quyết định còn mở. Quote, hash,
ID và vị trí ký tự được giữ nguyên; `verified` chỉ xác nhận xuất xứ. Nhóm và liên
kết vẫn cần người kiểm tra, không phải nhãn người đã được xác nhận.

`task_catalog` dùng ID T001... theo cùng thứ tự candidate. `task_links` là gợi ý
đối chiếu dựa vào ít nhất hai từ chung, có `shared_terms`,
`status=lexical_candidate`, `needs_review=true`; chưa khẳng định mức bao phủ.
`coverage_status=not_assessed`. `unlinked_evidence_ids` chỉ nghĩa là chưa có gợi ý
theo heuristic. Ngữ cảnh giữ tối đa 512 đoạn và bốn liên kết/đoạn, có số đếm bỏ qua.
`validate_review_context` dựng lại từ nguồn để phát hiện sửa quote, ID, liên kết
hoặc giả trạng thái đã duyệt. FAST và FULL dùng được cùng lớp ngữ cảnh này.

V2 cho phép checklist trích nguyên văn yêu cầu/nghiệm thu ở câu khác hoặc đoạn
tài liệu retrieval chọn. Nhân sự nguyên văn có thể trở thành nguồn lực đề xuất,
không phải phân công hoặc cam kết sẵn sàng. Ứng viên phụ thuộc bắt buộc dẫn chứng
từ đoạn có dấu hiệu phụ thuộc; `basis=source_cited_llm_candidate` chưa xác nhận
ánh xạ đúng. “Trước phê duyệt” không tự đồng nghĩa “trước bắt đầu”.

Hit `role=document` chỉ vào prompt khi khớp document ID, hash và span của nguồn
chính. Hit resource ngoài vẫn là tham khảo, không trở thành yêu cầu dự án. Prompt
dùng một quote cho mỗi ID với danh sách vai trò sử dụng, tránh lặp metadata.
Catalog kiểm toán giữ thông tin đầy đủ.

Phản hồi v2 mặc định dùng `response_format=candidate_ids_v1`: LLM chỉ trả
`{"task_ids":["T001"],"link_ids":["L001"]}`. `catalog.link_candidates` chứa các cặp
task–evidence đủ ít nhất hai từ chung, thuộc đúng nhóm checklist/nguồn lực và không
trùng/đè lên chính câu task. Đây vẫn chỉ là ứng viên từ vựng; model phải xem nghĩa
để lựa chọn. Tối đa 32 ứng viên, chọn luân phiên giữa các task; phần bỏ qua được
đếm. LLM không được tự ghép lại task với evidence ngoài danh mục ứng viên.

Phần mềm khôi phục nguyên văn evidence được chọn, gắn trạng thái đề xuất và
`needs_review=true`. Hierarchy không được sinh trong định dạng mặc định;
`parent_task_id` được khôi phục thành null. Phụ thuộc cần người xem các ràng buộc
nguồn; định dạng mặc định chưa tự ánh xạ các câu trước-phê-duyệt thành cạnh giữa
task. Prompt chỉ chứa quote được một ứng viên dùng, số quote ngoài prompt được
ghi rõ; catalog kiểm toán và phần rà soát vẫn giữ ngữ cảnh đầy đủ.

Model có thể trả danh sách rỗng khi không có liên kết rõ ràng. Không ép tạo liên
kết để làm đẹp số đếm. `response_format=links_v1` giữ schema ID theo từng loại liên
kết; `response_format=expanded_v1` giữ schema v2 mở rộng trước đó để replay những
lượt sử dụng schema đó. Protocol v1 có hành vi riêng như phần dưới.

Nếu phản hồi mặc định có `task_ids` nhưng thiếu riêng `link_ids`, phần mềm coi là
model abstain khỏi việc chọn liên kết, để toàn bộ liên kết rỗng và thêm cảnh báo
`Model omitted link_ids; no link was inferred`. Raw response vẫn giữ nguyên.
Không tự chọn ứng viên thay model. Thiếu `task_ids`, trường lạ, ID lạ, link trùng
hoặc link đến task chưa chọn vẫn bị từ chối.

Ứng dụng có thể bật `task_policy=preserve_source_catalog` cho định dạng mặc định:
task ID model trả về là gợi ý thứ tự; phần mềm nối thêm các task còn lại trong
catalog theo thứ tự nguồn, không để model làm mất công việc đã trích được. Mỗi
step ghi `selection_origin=llm_selection` hoặc `source_catalog_preserved`, cùng
danh sách ID được giữ lại và cảnh báo số lượng. Raw response vẫn nguyên vẹn.
Model trả danh sách task rỗng thì `llm_task_selection_status=abstained`, dù các
task nguồn vẫn được giữ ở trạng thái đề xuất. Liên kết được model chọn có thể
trỏ tới task được giữ theo chính sách này; không tự chọn thêm link. Chính sách
chỉ giữ các task trong catalog có giới hạn, không khẳng định đã giữ mọi việc
ngoài giới hạn. API mặc định `task_policy=select_only` vẫn giữ hành vi chọn task
như trước; protocol v1 không thay đổi.

Ngân sách mặc định vẫn là 3.072 token vào và 1.024 token ra. Context nguồn chọn
cân bằng nhóm, tối đa 12 đoạn/1.800 ký tự; retrieval thêm tối đa 600 ký tự. Nguồn
lực/tham khảo có ngân sách ký tự riêng. Đoạn không vừa bị bỏ nguyên đoạn và được
đếm trong `catalog_omitted_counts`; không cắt câu âm thầm. Ngữ cảnh đầy đủ vẫn
nằm trong catalog để người đọc rà soát. Tokenizer kiểm tra token thực; quá giới
hạn hoặc JSON bị cắt sẽ báo lỗi rõ ràng.

Kiểm tra không cần ML:

```powershell
python -m unittest psai_planner.test_planner psai_planner.test_context -v
```

Tests gồm thông tin bị extraction bỏ sót, checklist khác câu, nguồn khác/hash/span
sai, ID trùng, phụ thuộc thiếu dẫn chứng, giả đã duyệt và replay v1. Có fixture lĩnh
vực khác; production không có trường hợp riêng theo tên proposal miner.

## Protocol v1 để replay kết quả cũ

Truyền `config={"protocol_version": "psai-local-evidence-catalog-v1"}` để dựng lại
catalog cũ. `build_prompt`/`validate_response` hiểu catalog thiếu protocol là v1.
Provenance cũ cần truyền v1 tường minh vào `build_catalog`, không dùng mặc định v2.
Phần mô tả dưới đây ghi lại hành vi v1, đặc biệt giới hạn checklist cùng câu và
ứng viên phụ thuộc suy luận không có dẫn chứng.

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
