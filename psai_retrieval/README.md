# Truy xuất bằng chứng cục bộ

Package runtime này độc lập với classifier đã đóng băng, study RAG v1 và Chroma cũ. BM25 mặc định chạy bằng thư viện chuẩn Python, không import Torch/tokenizer/model hoặc gọi mạng.

```python
from psai_retrieval import retrieve_document

result = retrieve_document(
    text, "proposal", queries=["security requirements", "deployment schedule"],
    top_k=3, mode="bm25", resource_dir=None,
)
```

Mỗi phần tử `result['hits']` có `evidence_id`, `doc_id`, `role`, `quote`, `char_start`, `char_end`, `source_sha256` và `query_matches`. `quote == text[char_start:char_end]` trên đúng nguồn được nhận diện. Heading chỉ là metadata tìm kiếm, không được thêm vào quote. `verified=true` xác minh nguồn và vị trí; `relevance_verified=false` cho biết độ liên quan vẫn do thuật toán ước lượng.

Tài nguyên trong `resource_dir` là các file Markdown/text cục bộ có nguồn riêng, được đánh dấu `role=resource`. Không tải tài nguyên Internet. Hash `source_sha256` tính trên text đã đọc UTF-8-sig với universal newlines; `source_file_sha256` tính trên byte file gốc.

Tạo index mới từ một file hoặc thư mục nhiều tài liệu:

```powershell
python -m psai_retrieval index --input examples/proposal.md --output outputs/retrieval_index
python -m psai_retrieval search --index outputs/retrieval_index --query "security requirements" --top-k 3
```

Với kho tài liệu nhiều file, thêm `--document-id` để giới hạn tài liệu. Kho tài nguyên chỉ tham gia search khi có `--include-resources`. Index lưu text nguồn, đoạn trích, char spans và SHA256 manifest; từ chối file nguồn thay đổi, dữ liệu index sai hash hoặc nguồn/đoạn không khớp. Chọn thư mục index mới; không ghi đè index đã có.

`--mode dense` hoặc `--mode hybrid` nạp BGE-M3 nền từ snapshot pin cục bộ; không tải model. Khi tạo dense index, input gồm prefix heading theo quy ước study và đoạn nguyên văn, tổng tối đa 254 token chưa tính special tokens. `quote`/char spans chỉ chứa nguồn nguyên văn; prefix có trường riêng `embedding_prefix`/`embedding_text`. BM25 index dùng cửa sổ ký tự và không tự chuyển thành dense index. Dense inference dùng CLS BF16 rồi cast và normalize FP32, tối đa 256 token. Cần tạo index dense mới trước khi search dense/hybrid.

`--base-model-dir` hoặc `PSAI_BGE_MODEL_DIR` chỉ rõ snapshot cục bộ; base phải là normalized-CLS BGE-M3 pin revision. `--adapter-dir` là tùy chọn có chủ đích: adapter LoRA feature extraction phải khai báo đúng base snapshot. Export chuyển máy có `base_provenance.json` để kiểm tra hash file base, không phụ thuộc đường dẫn cũ. Index adapter cần truyền lại đúng adapter khi search; không tự lấy checkpoint pilot hoặc thay model production. Hash model/tokenizer/adapter query phải khớp index. Giao thức số học lúc tạo index và lúc encode query đều được ghi trong provenance.

`--rerank-model-dir` bật MiniLM-L6 MS MARCO cross-encoder đã pin cục bộ, chạy CPU sau khi dense model đã được giải phóng. Điểm teacher là logit chưa calibration và hướng tiếng Anh; không coi đó là nhãn người. Hybrid và hợp nhất nhiều query dùng RRF `k=60`.

Kiểm tra CPU bằng fixture/fake vectors:

```powershell
python -m unittest psai_retrieval.test_runtime -v
```
