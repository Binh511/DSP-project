# Nhánh RAG PSAI

Nhánh này tìm đoạn bằng chứng trong tài liệu dự án và kho tài nguyên Markdown/text cục bộ. Runtime có BM25, BGE-M3 dense, hybrid/RRF, reranker tùy chọn và báo cáo HTML nhúng nguồn. Default dense giữ BGE-M3 gốc; adapter LoRA là ứng viên nghiên cứu bật riêng vì chưa đủ bằng chứng cải thiện ổn định.

## Hai báo cáo chính

- [Báo cáo RAG — HTML độc lập](reports/rag_release_v1/BAO_CAO_RAG.html): số liệu, CI, ví dụ lỗi, phần đã làm và giới hạn nằm trực tiếp trong file.
- [Demo toàn pipeline dùng model thật — HTML có nguồn](docs/demo/full/report.html): ví dụ dùng đề xuất mẫu do dự án soạn, chạy qua extension API với production ensemble, hybrid retrieval mặc định và local LLM. Bấm dẫn chứng để đối chiếu nguồn trong báo cáo. Đây là demo chức năng hiện hành; không phải lượt đánh giá LoRA/reranker của study RAG hoặc phép đo chất lượng độc lập.

Các báo cáo ở trên được đưa vào bản GitHub; kết quả mới của người nhận sẽ được tạo cục bộ. Kết quả demo RAG lịch sử tại `outputs/rag_ready/release_demo_v1_1/` không nằm trong bản clone.

Phạm vi đã kiểm tra là phần truy xuất của nhánh RAG. LLM lập kế hoạch, OCR cho PDF scan và đánh giá nội dung cuối bằng người chưa được chứng nhận bởi các kết quả này. Classifier và index Chroma production được bảo toàn.

## Demo nhanh

Mở PowerShell tại gốc project:

```powershell
python tools/demo_rag.py
```

Lệnh in tiến trình và đường dẫn HTML. Demo dùng rules để trích nội dung và BM25 để truy xuất; không train, không tải model, không gọi API. Mỗi lượt có output mới.

Model thật với GPU, sau khi chuẩn bị môi trường và BGE-M3 theo [hướng dẫn full](docs/FULL_RUNTIME_SETUP.md). Thay đường dẫn ví dụ bằng snapshot thật trên máy nhận:

```powershell
python tools/demo_rag.py --mode hybrid --device cuda --base-model-dir "D:/PSAI-models/BGE-M3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
```

Adapter LoRA và reranker là tùy chọn nghiên cứu cần model/artifact được cung cấp riêng. Launcher `--use-adapter` kiểm tra release adapter; `--rerank` còn tham chiếu manifest model phụ của study. Bản GitHub không chứa các tài nguyên này. Khi đã có export phù hợp, có thể truyền `--adapter-dir` và `--rerank-model-dir` trực tiếp cho [CLI](psai.py) hoặc [runtime retrieval](psai_retrieval/README.md); không bật các cờ này chỉ để tái hiện số liệu đẹp hơn.

Đầu vào và query riêng:

```powershell
python tools/demo_rag.py --input "D:/tai_lieu/de_xuat.pdf" --query "security requirements" --top-k 5
```

PDF của demo RAG riêng cần selectable text; PDF scan/không có text bị từ chối với yêu cầu OCR. Luồng OCR tích hợp ở [demo toàn dự án](README_PROJECT.md). Launcher dùng môi trường riêng `.venv/psai_runtime` nếu người nhận đã tạo môi trường này; thư mục môi trường không có trong GitHub. `--resource-dir` thêm tài nguyên cục bộ; tài nguyên có nguồn/role riêng và không tự tạo thêm việc trong kế hoạch.

## Index lưu trên đĩa

```powershell
python -m psai_retrieval index --input examples/proposal.md --output outputs/my_rag_index --mode hybrid --device cuda
python -m psai_retrieval search --index outputs/my_rag_index --query "API credentials" --mode hybrid --device cuda --top-k 3
```

Với nhiều tài liệu, `--document-id` giới hạn nguồn; `--include-resources` mới cho tài nguyên tham gia search. Khi đổi nguồn, encoder, tokenizer, adapter hoặc code chia đoạn, tạo index mới. Không đổi một index BM25 thành dense bằng cách đổi cờ. Chi tiết ở [API runtime](psai_retrieval/README.md).

Quote giữ nguyên char spans. Heading được thêm riêng vào embedding theo quy ước study; tổng input tối đa 256 token. Runtime giữ khoảng trắng nguồn và có cửa sổ riêng nên điểm study chưa chứng minh chất lượng runtime.

## Kết quả nghiên cứu

Train: 1.536 triplet, 178 nhóm nguồn, seed42, một epoch, 96 bước, khoảng 12 phút 40 giây. Validation: 54 query tự sinh trên 22 nhóm nguồn.

| Dense trong cùng tài liệu | BGE-M3 gốc | LoRA |
|---|---:|---:|
| MRR@10 | 0,5053 | 0,5066 |
| Recall@5 | 0,6667 | 0,7037 |
| Recall@10 | 0,8333 | 0,8333 |

Delta MRR +0,001286; source-cluster CI95% [-0,001667; +0,004365], gồm 0; 5 query tăng/2 giảm/47 hòa. Chưa đủ bằng chứng cải thiện ổn định. Query/positive là nhãn tự động. Đọc [báo cáo](reports/rag_release_v1/BAO_CAO_RAG.md) và [audit dữ liệu](reports/rag_studies/retrieval_genq_lora_v1/DATA_SUPERVISION_AUDIT.md).

## Phần cần người kiểm định

Phiếu ưu tiên của study tại `psai_rag_quality/artifacts/retrieval_genq_lora_v1/priority_review_packet/REVIEW.html` gồm 22 query/80 cặp query–đoạn. Packet chứa dữ liệu/nhãn nghiên cứu, được cung cấp riêng nếu có quyền chia sẻ; không có trong bản clone. Hai người chấm độc lập, xuất CSV riêng; phần chưa biết giữ là chưa biết. Đây là tập chọn để chẩn đoán, không đại diện toàn validation. Bộ đầy đủ có 54 query/718 cặp.

Khi đã nhận packet và hai tệp CSV hợp lệ, chạy importer:

```powershell
python -m psai_rag_quality import-review --packet psai_rag_quality/artifacts/retrieval_genq_lora_v1/priority_review_packet --rater-a A.csv --rater-b B.csv --output outputs/review_import
```

Importer kiểm tra nội dung, nguồn, nhãn và hai người độc lập; không điền nhãn trống. Mẫu final test tại `reports/rag_studies/retrieval_genq_lora_v1/HUMAN_FINAL_TEST_TEMPLATE.csv` là artifact study cần cung cấp riêng, giúp chuẩn bị query thật trên nguồn giữ lại. Không dùng final test để chọn model/tham số. Người chỉ nhận mã vẫn có thể xem giao thức trong [psai_rag_quality](psai_rag_quality/) và góp ý theo [hướng dẫn review](docs/REVIEW_GUIDE.md).

## Kiểm tra và chuyển máy

Kiểm tra runtime BM25 không cần model trên bản clone:

```powershell
python -m unittest psai_retrieval.test_runtime -v
```

Các công cụ [finalize_rag_release.py](tools/finalize_rag_release.py) và [verify_rag_pdf_demo.py](tools/verify_rag_pdf_demo.py) phục vụ release/study gốc; chạy lại cần đúng bộ dữ liệu, checkpoint và artifact mà chúng tham chiếu. `--verify-only` kiểm tra toàn vẹn, không thay thế đánh giá relevance và không thể đạt đầy đủ chỉ với bản clone mã nguồn.

193 test đã đạt ở lần nghiệm thu RAG được lưu; [biên bản](reports/rag_release_v1/verification.json) phân biệt kiểm tra chức năng với chất lượng nghiên cứu. Đây không phải số test đã chạy lại tự động khi người nhận clone mã. [Requirements RAG](requirements-rag.txt) ghi phiên bản quan sát trên Windows/RTX4050; chưa tuyên bố đã cài thành công trên mọi máy mới.

Export adapter tại `models/rag/release_v1/retriever_adapter` chứa weights khoảng 3 MB, config và hash của base; tác giả cần cung cấp riêng nếu người nhận muốn nghiên cứu LoRA. Weights không được đưa vào GitHub. Máy nhận cần BGE-M3 base đầy đủ cùng revision `5617a9f61b028005a4858fdac845db406aefb181`, tokenizer và config pooling/normalize. Adapter không thay thế base. Truyền `--base-model-dir` hoặc `PSAI_BGE_MODEL_DIR`; folder base giữ tên snapshot revision. Xây lại index trên nguồn của máy nhận để có đường dẫn/hash đúng.

Mã study [psai_rag](psai_rag/) đã đóng băng; runtime mới ở [psai_retrieval](psai_retrieval/). Thiết kế tham khảo [BGE-M3](https://arxiv.org/abs/2402.03216), [GPL](https://arxiv.org/abs/2112.07577) và [GradCache](https://arxiv.org/abs/2101.06983); pilot GenQ + teacher filter + contrastive loss không tái hiện đầy đủ GPL.
