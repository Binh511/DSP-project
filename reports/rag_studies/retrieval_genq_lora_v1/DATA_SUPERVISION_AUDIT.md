# Kiểm tra dữ liệu và giám sát tổng hợp cho RAG

Ngày kiểm tra: 28/10/2026. Phạm vi: dữ liệu đã tạo cho thử nghiệm `retrieval_genq_lora_v1`; kiểm tra trên CPU, không chạy mô hình, không thay đổi mã nguồn hoặc dữ liệu dùng huấn luyện.

**Kết luận:** cấu trúc dữ liệu, tham chiếu đoạn nguồn và phân hoạch đều vượt qua các kiểm tra tự động. Bộ dữ liệu có **1.536 triplet train trên 178 nhóm nguồn** và **54 truy vấn validation trên 22 nhóm nguồn**. Đây là **giám sát tổng hợp**, chưa có nhãn mức độ liên quan do người đánh giá. Kiểm tra mẫu phát hiện truy vấn sai nghĩa và nguồn lỗi mã hóa vẫn lọt qua bộ lọc teacher; vì vậy kết quả validation chỉ đo khả năng khớp các nhãn giả của thử nghiệm.

## 1. Nguồn, khử trùng và phân hoạch

Nguồn kiểm tra: [source_manifest.json](./source_manifest.json), [split_manifest.json](./split_manifest.json), [corpus.jsonl](./corpus.jsonl).

| Hạng mục | Số lượng thực tế |
|---|---:|
| Tài liệu Markdown ban đầu | 254 |
| Nguồn khác nhau sau chuẩn hóa toàn văn | 242 |
| Nhóm nguồn sau gộp bản trùng và gần trùng | 223 |
| Tài liệu chuẩn được giữ lại | 223 |
| Tài liệu bí danh không tạo thêm bản sao đoạn | 31 |
| Cặp gần trùng vượt ngưỡng đã khai báo | 19 |
| Đoạn trùng trong cùng nhóm bị bỏ | 113 |
| Hash đoạn dùng chung giữa các nhóm bị loại | 92 |
| Số bản ghi đoạn bị loại do dùng chung giữa nhóm | 214 |
| Đoạn còn lại trong corpus | 18.702 |
| Đoạn ứng viên tạo truy vấn | 1.443 |

Gộp nguồn trước khi chia tập bằng hash toàn văn Unicode NFC, chữ thường và gộp khoảng trắng; sau đó gộp các cặp có Jaccard trên 5-gram từ ít nhất 0,85 và tỷ lệ độ dài từ ít nhất 0,8. Đây là heuristic ngưỡng cao, chưa chứng minh mọi bản gần trùng theo ngữ nghĩa đã được phát hiện. Mỗi nhóm chỉ giữ tài liệu chuẩn đầu tiên theo mã tài liệu. Đoạn có cùng nội dung chuẩn hóa giữa các nhóm bị loại khỏi mọi phân hoạch.

| Phân hoạch | Nhóm nguồn | Đoạn corpus | Ứng viên tạo truy vấn | Cặp sau lọc và giới hạn |
|---|---:|---:|---:|---:|
| Train | 178 | 14.328 | 1.399 | 1.536 |
| Validation | 22 | 2.352 | 44 | 54 |
| Test | 23 | 2.022 | 0 | 0 |

Nhóm nguồn giữa ba tập không giao nhau. Corpus tổng chứa đoạn test chưa gán nhãn; tạo truy vấn, chọn negative và đánh giá validation phải lọc theo phân hoạch. Kiểm tra này chỉ đối chiếu nguồn và cấu trúc của các đoạn test, không đọc nhãn test hay tính điểm truy hồi test.

## 2. Các kiểm tra tự động đã chạy trên dữ liệu thật

Đã gọi `train.validate_training_data(train_rows, split_manifest, corpus)` và `evaluate._load_validation_inputs(corpus_path, validation_pairs_path)` mà không khởi tạo Torch hay mô hình GPU. Cả hai hoàn tất thành công.

| Kiểm tra | Kết quả |
|---|---:|
| Hash byte của tệp nguồn khớp manifest | 254/254 |
| Hash nội dung đoạn corpus khớp | 18.702/18.702 |
| Hash nguồn của đoạn corpus khớp nguồn hiện tại | 18.702/18.702 |
| Nội dung thân đoạn nằm trong khối dòng nguồn khai báo, sau chuẩn hóa khoảng trắng | 18.702/18.702 |
| Tham chiếu positive/negative đúng ID, nguyên văn đoạn corpus, tài liệu, nhóm nguồn và phân hoạch | 3.180/3.180 |
| Tham chiếu known-positive cùng tài liệu và phân hoạch | 1.594/1.594 |
| Truy vấn có 4–40 từ | 1.590/1.590 |
| Margin teacher đạt ngưỡng 0,5 | 1.590/1.590 |
| Margin khớp phép trừ hai logit đã lưu | 1.590/1.590 |
| Truy vấn chuẩn hóa trùng trong train | 0 |
| Truy vấn chuẩn hóa trùng trong validation | 0 |
| Truy vấn chuẩn hóa giao train–validation | 0 |
| Nội dung đoạn corpus chuẩn hóa trùng nhau | 0 |

Các số trên xác nhận tính toàn vẹn và liên kết với nguồn. Chúng **không xác nhận** positive thực sự trả lời truy vấn, negative thực sự không liên quan, hoặc truy vấn đúng ngữ nghĩa. Phạm vi dòng trong corpus bao quanh khối nguồn; đoạn có thể thêm tiền tố tiêu đề và cắt theo token nên không được diễn đạt là một trích dẫn nguyên văn đầy đủ theo offset ký tự.

## 3. Nguồn giám sát và mức lọc

Mô hình tạo truy vấn: `doc2query/msmarco-t5-small-v1`, revision `eb04306040c87ff35caa4c87078dfd94a6b7b64e`.

Teacher: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.

Thông số đã khai báo trước: seed 42; hai truy vấn mỗi ứng viên; đầu vào T5 tối đa 256 token; sinh tối đa 40 token, tối thiểu 4 token; top-p 0,95, top-k 50; cấm lặp 3-gram. Teacher đọc tối đa 384 token. BM25 tìm tối đa hai negative trong **cùng tài liệu và nhóm nguồn**, bỏ positive đã biết và đoạn trùng nội dung. Giữ negative khó nhất trong các negative có margin ít nhất 0,5. Không đổi ngưỡng dựa trên kết quả kiểm tra này.

| Bước | Số lượng |
|---|---:|
| Ứng viên đã tạo truy vấn | 1.443 |
| Chuỗi truy vấn đã sinh | 2.886 |
| Bị loại do ngoài 4–40 từ | 355 |
| Bản trùng dư trong cùng tài liệu/phân hoạch bị bỏ | 28 |
| Bản truy vấn trùng giữa các tài liệu bị bỏ toàn bộ | 32 |
| Bị loại do truy vấn sao chép nguyên đoạn hoặc đoạn liên tiếp dài ≥80% nguồn | 0 |
| Bị loại do truy vấn trùng giữa các phân hoạch | 0 |
| Truy vấn duy nhất trước chọn negative | 2.471 |
| Có negative cùng tài liệu | 2.471 |
| Negative ứng viên không đạt margin | 1.486 |
| Truy vấn không còn negative đạt margin | 504 |
| Cặp được teacher chấp nhận trước giới hạn số lượng | 1.967 |
| Cặp train sau giới hạn phân bổ luân phiên giữa nhóm | 1.536 |
| Cặp validation sau lọc | 54 |

Sau giới hạn, dữ liệu vẫn phủ **178/178 nhóm train và 22/22 nhóm validation** có ứng viên; không có nhóm bị bỏ hoàn toàn vì giới hạn số lượng. Tập train đạt yêu cầu tối thiểu 64 cặp và validation đạt yêu cầu tối thiểu 16 cặp. Việc phủ nhóm không đồng nghĩa phủ mọi chủ đề hoặc mọi phần của mỗi tài liệu.

**Hạn chế của teacher:** margin chỉ là chênh lệch điểm giữa hai đoạn. Cả positive lẫn negative có thể kém liên quan mà vẫn đạt margin. Có **499/1.536 cặp train (32,5%)** và **19/54 cặp validation (35,2%)** với logit positive âm. Logit không phải xác suất đã hiệu chỉnh; dấu âm không tự chứng minh nhãn sai, nhưng cùng với các mẫu dưới đây cho thấy cần người kiểm tra ngữ nghĩa. Margin nhỏ nhất là 0,5014 ở train và 0,5151 ở validation.

## 4. Đọc thăm dò tám mẫu: cần người đánh giá tiếp

Mẫu được chọn có chủ đích: logit positive thấp, cờ rủi ro ngôn ngữ, và hai trường hợp validation có logit cao. Đây không phải mẫu ngẫu nhiên, không phải đánh giá nhãn độc lập, không cho phép ước lượng tỷ lệ nhãn đúng/sai của toàn tập. Mã truy vấn dưới đây có thể tra trực tiếp trong [train_pairs.jsonl](./train_pairs.jsonl) hoặc [validation_pairs.jsonl](./validation_pairs.jsonl).

| Tập / nguồn / query_id | Truy vấn đã sinh | Quan sát thăm dò từ positive | Margin |
|---|---|---|---:|
| Train / PSAI226 / `genq-80a0f60e7accf4ce9c691ee0` | what city is piner muncer | Đoạn là địa chỉ pháp lý tại Pune; tên trong truy vấn bị méo và không xuất hiện trong đoạn. Cần kiểm tra hoặc loại ở phiên bản dữ liệu sau. | 0,5055 |
| Train / PSAI214 / `genq-8f310e45fd804e68e44ca0b6` | what is an emp project | Nguồn nói xây dựng cộng đồng và “IMP”; truy vấn đổi thành “emp”, quan hệ hỏi–đáp không rõ. | 0,5864 |
| Train / PSAI049 / `genq-0aee63ec1d1b0d7816eabcac` | what is the number of mnmn | Đoạn nguồn có ký tự lỗi mã hóa/giải mã, không đọc hiểu đáng tin cậy; margin cao vẫn không bảo đảm dữ liệu dùng được. Cần rà soát PDF/OCR. | 4,2025 |
| Train / PSAI107 / `genq-0d4fc6ea4ba5b567122bea00` | what does a yong ying wilian stand for in china | Nguồn mô tả công ty Beijing Gong Yi Lian và có tên tiếng Trung; truy vấn phiên âm méo, cần người đọc hiểu nguồn đánh giá. | 2,0378 |
| Validation / PSAI139 / `genq-4b295fd2c8ceee1e64f455f5` | where do I get my chris | Positive là hồ sơ thành tích và kinh nghiệm lập trình; không thấy nội dung trả lời truy vấn này trong toàn đoạn positive. | 0,5151 |
| Validation / PSAI102 / `genq-b892e9bdc56020bf34cd1a3a` | does ascriptx show a limited definition | Positive chứa danh mục tài liệu tham khảo dưới tiêu đề “Limitations”; câu hỏi không có liên kết nội dung rõ ràng. | 0,9107 |
| Validation / PSAI101 / `genq-b647ef4a773fa28a0520e5b9` | what is a symbolic approach to control task oriented dialog systems | Positive bàn tích hợp quy tắc và dialog acts vào mô hình; có liên hệ chủ đề trực tiếp, nhưng mức độ trả lời vẫn cần người chấm. | 0,9139 |
| Validation / PSAI080 / `genq-ea0e2ba1bf44590eb56cb213` | what region has the highest brain predictivity | Positive nêu các vùng superior/middle temporal cortices và ATL; có dữ kiện tương ứng với câu hỏi, chưa được xác nhận bằng nhãn người. | 15,2796 |

Không sửa hoặc loại lại các mẫu này trong lượt đang chạy: thay đổi nhãn sau khi xem kết quả sẽ làm thay đổi giao thức. Nếu cải thiện chất lượng nguồn, query hoặc teacher filter, cần phiên bản dữ liệu/thử nghiệm mới với manifest mới.

## 5. Phạm vi ngôn ngữ và diễn giải kết quả

Hai mô hình phụ được huấn luyện cho MS MARCO thiên về tiếng Anh. Dữ liệu hiện không có nhãn ngôn ngữ đã xác minh. Heuristic phát hiện **7 cặp train** có ký tự cần lưu ý: 6 cặp có chữ ngoài Latin, 1 cặp có ký tự dấu thường gặp trong tiếng Việt; validation có 0 cặp bị gắn cờ. Những cờ này có thể xuất phát từ tên riêng, đa ngôn ngữ hoặc lỗi mã hóa, không phải phân loại ngôn ngữ đáng tin cậy. Không có cờ cũng không chứng minh nguồn là tiếng Anh đúng chuẩn.

**Chưa có cơ sở khẳng định chất lượng truy hồi tiếng Việt.** Cần tập câu hỏi tự nhiên tiếng Việt và các đánh giá relevance do người đọc hiểu nguồn xác nhận. Chỉ số trên validation tổng hợp cũng chưa chứng minh chất lượng sinh kế hoạch, mức hữu ích với người dùng, hay chất lượng pipeline PDF/OCR.

Một khác biệt về miền truy vấn cần đánh giá riêng: luồng RAG hiện có ghép các giá trị trích xuất theo trường thành chuỗi truy vấn dài, còn GenQ ở đây tạo câu hỏi ngắn theo phong cách tìm kiếm MS MARCO. Cải thiện trên câu hỏi tổng hợp ngắn không bảo đảm cải thiện cho chuỗi mục tiêu/yêu cầu/hoạt động/tài nguyên mà ứng dụng đang gửi vào bộ truy hồi. Nên ưu tiên thu thập truy vấn thực tế từ luồng ứng dụng, bao gồm truy vấn tiếng Việt, rồi nhờ người chấm mức liên quan của các đoạn nguồn.

Thứ tự cải thiện cho **phiên bản thử nghiệm tiếp theo**: sửa trích xuất/giải mã nguồn trước đối với tài liệu lỗi như PSAI049; lập tập truy vấn thật và relevance do người xác nhận; sau đó mới đánh giá thêm một ngưỡng điểm positive tuyệt đối kết hợp margin. Ngưỡng điểm tuyệt đối cũng là heuristic cần hiệu chỉnh trên dữ liệu con người, không tự trở thành nhãn thật. Giữ nguyên giao thức pilot hiện tại và dùng các chỉ số của nó như chẩn đoán giám sát yếu; chưa thay thế bộ truy hồi BGE-M3 nền trong ứng dụng bằng adapter chỉ dựa vào 54 truy vấn validation tổng hợp này.

## 6. Dấu vết phiên bản dữ liệu đã kiểm tra

- Fingerprint tạo giám sát: `91aea4c2d685c76d47f4e7ffb5d81830b524119ce53b7739ae890fa8accdb3b8`.
- SHA256 [corpus.jsonl](./corpus.jsonl): `2e62a33dacc5973467b1e404d51b65d51d7106c09e785ec13f11b7a260448f80`.
- SHA256 [train_pairs.jsonl](./train_pairs.jsonl): `dcb8f7519d47e29b8ac94286483af3faf9fa87a80d8529ae710b4ad8661b0398`.
- SHA256 [validation_pairs.jsonl](./validation_pairs.jsonl): `5a591a2c19aa5a758ae46e88b66b2b36ce2ed754bbf892dee41ca1bb0c411a84`.

Chi tiết số lượng và cấu hình: [generation_report.json](./generation_report.json). Các nhận xét thăm dò trong báo cáo này không thay thế quy trình đánh giá relevance độc lập do con người thực hiện.
