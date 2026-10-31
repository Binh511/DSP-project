# Báo cáo hoàn thiện nhánh RAG — PSAI

Bản runtime đã nối vào luồng phân tích tài liệu và xuất báo cáo có nguồn. Lượt train LoRA đã hoàn thành; chưa có bằng chứng đủ mạnh để khẳng định tốt hơn BGE-M3 gốc trên truy vấn thực tế.

## 1. Phần đã thực hiện

- Chuẩn bị dữ liệu theo nhóm tài liệu trùng/gần trùng, giữ hash nguồn và tách train/validation/test trước sinh query.
- Sinh query bằng T5-small, lấy negative BM25 trong cùng tài liệu, lọc bằng cross-encoder; toàn bộ là nhãn tự động.
- Train BGE-M3 dense + LoRA rank8, một epoch, seed42; CachedMNRL batch16/mini1, BF16 encoder và logits FP32.
- Chạy baseline BM25, dense và RRF; đánh giá base/adapter trên cùng validation và phân tích sai số theo nhóm nguồn.
- Hoàn thiện runtime index/search: nhiều truy vấn, giới hạn tài liệu, tài nguyên tham khảo riêng, hybrid/RRF và reranker tùy chọn.
- Giữ quote nguyên văn và char spans; heading được thêm vào embedding riêng; kiểm tra hash model/tokenizer/adapter/index và nguồn.
- Tích hợp Markdown/text/PDF có text → trích nội dung → truy xuất → báo cáo HTML/Markdown/JSON. HTML nhúng nguồn để mở độc lập.

Luồng: PDF/text → văn bản + hash → chia đoạn theo cấu trúc → BM25/dense → RRF/rerank tùy chọn → hợp nhất → đối chiếu nguồn → nội dung và báo cáo có dẫn chứng. Luồng LLM lập kế hoạch không được tính là đã hoàn thành trong nhánh RAG này.

## 2. Dữ liệu và lượt huấn luyện

254 tài liệu ban đầu, 223 nhóm nguồn; 18702 đoạn theo tokenizer BGE-M3.

| Phần | Nhóm nguồn | Đoạn | Cặp/query sau lọc |
|---|---:|---:|---:|
| train | 178 | 14328 | 1536 |
| validation | 22 | 2352 | 54 |
| test | 23 | 2022 | 0 |

Train hoàn thành 96/96 bước; 786,432 tham số trainable; 759.9 giây. Checkpoint có commit/hash và bằng chứng gradient khác 0, trọng số được cập nhật.

## 3. Kết quả validation thực tế

54 query tự sinh trên 22 nhóm nguồn. Positive là đoạn dùng để sinh query; không phải nhãn relevance do người kiểm định. Các số dưới đây thuộc dữ liệu study đã đóng băng, không phải điểm chất lượng end-to-end của runtime.

| Phạm vi | Cách truy xuất | Phiên bản | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---:|---:|---:|---:|---:|
| global_validation | bm25 | base | 0.2778 | 0.4630 | 0.5370 | 0.3413 | 0.3872 |
| global_validation | bm25_dense_rrf | base | 0.2963 | 0.5556 | 0.7037 | 0.4190 | 0.4863 |
| global_validation | bm25_dense_rrf | adapter | 0.2963 | 0.5926 | 0.7037 | 0.4252 | 0.4915 |
| global_validation | dense | base | 0.2963 | 0.6111 | 0.7222 | 0.4329 | 0.5023 |
| global_validation | dense | adapter | 0.2963 | 0.6111 | 0.7037 | 0.4312 | 0.4972 |
| within_doc | bm25 | base | 0.2963 | 0.5370 | 0.6296 | 0.3819 | 0.4406 |
| within_doc | bm25_dense_rrf | base | 0.3333 | 0.6481 | 0.7778 | 0.4616 | 0.5367 |
| within_doc | bm25_dense_rrf | adapter | 0.3333 | 0.6481 | 0.7593 | 0.4639 | 0.5346 |
| within_doc | dense | base | 0.3704 | 0.6667 | 0.8333 | 0.5053 | 0.5831 |
| within_doc | dense | adapter | 0.3704 | 0.7037 | 0.8333 | 0.5066 | 0.5844 |

## 4. Độ chắc chắn của mức cải thiện

Bootstrap ghép cặp 10.000 lần, lấy mẫu lại toàn nhóm nguồn để giữ các query cùng tài liệu đi cùng nhau. Khoảng tin cậy chỉ phản ánh biến thiên lấy mẫu trên tập tự sinh này, chưa bao gồm lỗi nhãn hoặc biến thiên seed train.

| Phạm vi/phương pháp | Delta MRR | CI95% thấp | CI95% cao | Thắng/thua/hòa |
|---|---:|---:|---:|---|
| global_validation/bm25 | +0.000000 | +0.000000 | +0.000000 | 0/0/54 |
| global_validation/bm25_dense_rrf | +0.006217 | -0.004157 | +0.025490 | 2/3/49 |
| global_validation/dense | -0.001646 | -0.005769 | +0.000556 | 1/1/52 |
| within_doc/bm25 | +0.000000 | +0.000000 | +0.000000 | 0/0/54 |
| within_doc/bm25_dense_rrf | +0.002234 | -0.005892 | +0.014348 | 3/2/49 |
| within_doc/dense | +0.001286 | -0.001667 | +0.004365 | 5/2/47 |

Tiêu chí chính dense/within_doc tăng +0.001286, nhưng CI95% [-0.001667; +0.004365] chứa 0. Không đủ bằng chứng kết luận cải thiện ổn định; cũng không chứng minh hai mô hình tương đương. Default runtime dense giữ base; adapter được bật có chủ đích để nghiên cứu.

## 5. Các query đổi thứ hạng

Đây là thay đổi theo nhãn tự động hiện có. Cần người kiểm tra xem query có hợp lý và positive có thật sự liên quan.

### PSAI042 — win

what is thesis procedure

Query ID: genq-175037e46e59adce980d43df; thứ hạng positive: 9 → 8; delta MRR +0.013889.

Đoạn input được gắn positive tự động: PSAI042-RAG00063; nguồn bao quanh dòng 177–177. Text dưới có prefix heading của study; không phải quote PDF có char spans.

> # Thesis Proposal: People-Centric Natural Language Processing > 3.1 Completed work > 3.1.1 TACL 2014
> 
> The motivation for this part of my thesis comes from completed work on the unsupervised induction of life events in biographies, forthcoming in Transactions of the ACL (Bamman and Smith, 2014). In this work, we developed a set of probabilistic graphical models for inferring latent event classes like BORN, GRADUATES HIGH SCHOOL, and BECOMES CITIZEN from timestamped text, leveraging the correlation structure of events in individual's lives to help guide inference (as a logistic normal prior). Fi…

### PSAI252 — win

how to find a deadlock

Query ID: genq-629730697381f5a5492c2ce9; thứ hạng positive: 10 → 9; delta MRR +0.011111.

Đoạn input được gắn positive tự động: PSAI252-RAG00032; nguồn bao quanh dòng 93–93. Text dưới có prefix heading của study; không phải quote PDF có char spans.

> # Avoiding Rust Deadlocks via Visualizing Lifetime > Development Status :open_book:
> 
> We applied the bug detection component of the prototype to Substrate, Polkadot, and ink!. We found four previously unknown deadlocks. We reported all the detected bugs and all of them were fixed based on our reporting [PR-1, PR-2, PR-3].

### PSAI102 — loss

what is a llmd tool

Query ID: genq-a4ffbfe8c1d13d032ae2126f; thứ hạng positive: 8 → 9; delta MRR -0.013889.

Đoạn input được gắn positive tự động: PSAI102-RAG00076; nguồn bao quanh dòng 205–205. Text dưới có prefix heading của study; không phải quote PDF có char spans.

> # 4.1 Related Work > 4.2 Preliminary Work
> 
> Methods ProRefine adaptively improves prompts for a task-performing LLM (LLMtask) by using textual feedback from a judge (LLMfeedback) and an optimizer (LLMoptimizer). This workflow (Figure 4), motivated by the teacher-student framework (Torrey and Taylor, 2013) where a teacher agent guides a student agent to perform a task by providing feedback at intermediate steps, but implemented via LLM interactions without pre-training, represents a novel approach to adaptive agentic reasoning. ProRefine involves interactions between three LLMs:

### PSAI215 — win

which polkadot ecosystem is polkardo

Query ID: genq-a837bd2333cc77a85b0e137c; thứ hạng positive: 8 → 7; delta MRR +0.017857.

Đoạn input được gắn positive tự động: PSAI215-RAG00011; nguồn bao quanh dòng 29–32. Text dưới có prefix heading của study; không phải quote PDF có char spans.

> # BD Wallet > Project Overview :page_facing_up: > Overview
> 
> - 1. Launch a special version of Polkadot ecosystem project Rating, in order to let more users know about Polkadot and other Polkadot related projects.
> - 2. Put the service of DOT Staking online to help the construction of Polkadot Node.
> - 3. Support DOT, KSM and other polkadot ecosystem hot projects token recharge and withdrawal.
> - 4. Support Polkadot ecosystem DAPP access in the future, making more users to use Polkadot Eco's DAPP easily.

### PSAI102 — win

does ascriptx show a limited definition

Query ID: genq-b892e9bdc56020bf34cd1a3a; thứ hạng positive: 7 → 5; delta MRR +0.057143.

Đoạn input được gắn positive tự động: PSAI102-RAG00127; nguồn bao quanh dòng 268–268. Text dưới có prefix heading của study; không phải quote PDF có char spans.

> # Limitations
> 
> coding. arXiv preprint. ArXiv:2503.01422 [cs]. Yingxu Wang, Siwei Liu, Jinyuan Fang, and Zaiqiao Meng. 2025b. Evoagentx: An automated framework for evolving agentic workflows. Preprint, arXiv:2507.03616. Tharindu Weerasooriya, Sujan Dutta, Tharindu Ranasinghe, Marcos Zampieri, Christopher Homan, and Ashiqur KhudaBukhsh. 2023. Vicarious offense and noise audit of offensive speech classifiers: Unifying human and machine disagreement on what is offensive. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 11648–11668, Singapore. Associa…

### PSAI215 — win

what does the polkadot ecosystem do

Query ID: genq-ba03119c93da7912af846727; thứ hạng positive: 6 → 5; delta MRR +0.033333.

Đoạn input được gắn positive tự động: PSAI215-RAG00011; nguồn bao quanh dòng 29–32. Text dưới có prefix heading của study; không phải quote PDF có char spans.

> # BD Wallet > Project Overview :page_facing_up: > Overview
> 
> - 1. Launch a special version of Polkadot ecosystem project Rating, in order to let more users know about Polkadot and other Polkadot related projects.
> - 2. Put the service of DOT Staking online to help the construction of Polkadot Node.
> - 3. Support DOT, KSM and other polkadot ecosystem hot projects token recharge and withdrawal.
> - 4. Support Polkadot ecosystem DAPP access in the future, making more users to use Polkadot Eco's DAPP easily.

### PSAI054 — loss

what is the most common problem in acm programming software

Query ID: genq-c63c0184f7102f8244408b84; thứ hạng positive: 4 → 5; delta MRR -0.050000.

Đoạn input được gắn positive tự động: PSAI054-RAG00202; nguồn bao quanh dòng 432–432. Text dưới có prefix heading của study; không phải quote PDF có char spans.

> # References
> 
> 10.1145/3589505. 2.3, 2.4.2 [18] Arthur Charguéraud and François Pottier. Verifying the Correctness and Amortized Complexity of a Union-Find Implementation in Separation Logic with Time Credits. Journal of Automated Reasoning, 62(3):331–365, March 2019. ISSN 1573-0670. doi: 10.1007/ s10817-017-9431-7. 1.1 [19] Krishnendu Chatterjee, Hongfei Fu, and Amir Kafshdar Goharshady. Non-polynomial Worst-Case Analysis of Recursive Programs. ACM Transactions on Programming Languages and Systems, 41(4):20:1–20:52, October 2019. ISSN 0164-0925. doi: 10.1145/3339984. 1.1 [20] Ezgi Çiçek, Gille…

## 6. Giới hạn dữ liệu và điều kiện chốt

- Teacher margin không chứng minh positive liên quan: 499/1.536 cặp train và 19/54 query validation có positive-logit âm; logit này không phải xác suất và không tự là nhãn sai.
- Có query méo nghĩa và văn bản nguồn lỗi; lỗi nguồn đã được ghi tại DATA_SUPERVISION_AUDIT.md. Không tự sửa nhãn hoặc gọi các suy đoán này là ground truth.
- Bộ sinh query/teacher hướng tiếng Anh; chưa có đánh giá truy xuất tiếng Việt do người kiểm định.
- Runtime giữ nguyên khoảng trắng/char spans còn study có chuẩn hóa block và cửa sổ riêng. Dùng cùng quy ước prefix heading chưa biến điểm study thành chứng nhận chất lượng runtime.
- Chưa đánh giá cuối trên tập query thực độc lập. Không dùng final test để chọn tham số; không thay ensemble classifier/Chroma production.

## 7. Kiểm tra chức năng và bàn giao

Unittest discovery đã kiểm tra: 193 test. Các test fixture kiểm tra code, không phải phép đánh giá chất lượng nghiên cứu.
- Demo PDF tự tạo hai trang: kiểm tra đủ câu mẫu, hash/page spans, toàn bộ quote và link HTML; PDF trắng yêu cầu OCR.
- Demo model thật: hybrid + LoRA + MiniLM reranker trả về bằng chứng từ tài liệu và tài nguyên riêng, có kiểm tra nguồn.
- Báo cáo HTML chứa văn bản nguồn, mở không cần project gốc. Nguồn tài nguyên trùng text vẫn giữ riêng origin và snapshot.
- Index mới độc lập với Chroma cũ; thay nguồn/model/tokenizer/adapter hoặc artifact sai hash thì bị chặn.

## 8. Phần cần người hoàn tất

Packet ưu tiên có 22 query và 80 cặp query–đoạn, được chọn để chẩn đoán; không đại diện toàn benchmark. Hai người chấm độc lập và phân xử bất đồng; phần chưa rõ giữ là chưa biết.
Sau đó cần query thực tế và relevance trên tài liệu giữ lại để đánh giá mô hình đã đóng băng. Đây là công việc còn thiếu về bằng chứng nghiên cứu, không thể hoàn tất trung thực bằng cách tự sinh thêm nhãn AI.

## 9. Mở tài liệu và chạy demo

Chạy tại gốc project: python tools/demo_rag.py
Model thật: python tools/demo_rag.py --mode hybrid --device cuda
Adapter thử nghiệm: python tools/demo_rag.py --mode hybrid --use-adapter --rerank --device cuda
Kiểm tra release: python tools/finalize_rag_release.py --verify-only

Các bằng chứng máy đọc: [paired analysis](../../psai_rag_quality/artifacts/retrieval_genq_lora_v1/paired_analysis.json), [review ưu tiên](../../psai_rag_quality/artifacts/retrieval_genq_lora_v1/priority_review_packet/REVIEW.html), [kết quả train](../rag_studies/retrieval_genq_lora_v1/KET_QUA_RAG_PILOT.md), [manifest release](./RELEASE_MANIFEST.json).
