# Truy xu?t b?ng ch?ng

[psai_retrieval](psai_retrieval/README.md) h? tr? BM25, dense BGE-M3 v? hybrid RRF. BM25 d?ng th? vi?n chu?n Python; dense/hybrid c?n model c?c b?. Reranker v? adapter LoRA l? t?y ch?n, kh?ng t? k?ch ho?t.

```sh
python tools/demo_rag.py --mode bm25 --input examples/PSAI_DEMO_01_MINER_MONITORING.md
python -m psai_retrieval index --input examples/proposal.md --output outputs/retrieval_index
python -m psai_retrieval search --index outputs/retrieval_index --query "security requirements" --top-k 3
```

Index l?u hash, c?c ?o?n nguy?n v?n v? v? tr? ngu?n. T?i li?u ngu?n thay ??i th? c?n t?o index m?i. T?i nguy?n ngo?i ???c ph?n bi?t v?i t?i li?u d? ?n; ?o?n tham kh?o kh?ng t? tr? th?nh y?u c?u b?t bu?c.

Full setup, model revision v? dependency: [FULL_RUNTIME_SETUP](docs/FULL_RUNTIME_SETUP.md). API v? c?c t?y ch?n: [README package](psai_retrieval/README.md).

M? chu?n b? d? li?u, sinh query, train LoRA v? ch?m retrieval n?m trong [psai_rag](psai_rag/). D? li?u v? model hu?n luy?n kh?ng k?m Git. C?c script study c? c?u h?nh th? nghi?m ri?ng c?n ki?m tra tr??c khi ch?y l?i. Nh?n query/positive t? sinh ph?i ???c ph?n bi?t v?i relevance do ng??i x?c nh?n; ?i?m retrieval kh?ng thay th? ch?t l??ng danh s?ch c?ng vi?c.
