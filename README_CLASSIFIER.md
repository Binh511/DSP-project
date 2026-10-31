# Classifier PSAI — bản hiện hành v2

Bản classifier 2.0-source-clean chọn **baseline seed 2026**, sau khi hoàn tất hai cấu hình × ba seed trên train đã loại trùng toàn nguồn. Validation Macro-F1 **0.632077**, accuracy **0.760061**, n=1.317. Đây là điểm validation phát triển, không phải final test độc lập.

## Tài liệu để báo cáo và demo

- [Báo cáo Word đầy đủ](reports/classifier/release_source_clean_v2/BAO_CAO_CLASSIFIER.docx): số liệu, P/R/F1 của cả năm lớp, sáu ma trận và ví dụ lỗi nằm trực tiếp trong báo cáo.
- [Báo cáo text](reports/classifier/release_source_clean_v2/BAO_CAO_CLASSIFIER.txt).
- [Hướng dẫn demo và show code](reports/classifier/HUONG_DAN_DEMO_CHO_THAY.md).
- [Manifest model đã đóng băng](reports/classifier/release_source_clean_v2/FINAL_MODEL_MANIFEST.json), [quyết định lựa chọn](reports/classifier/release_source_clean_v2/final_selection.json), [model card](reports/classifier/release_source_clean_v2/model_card.md).
- [Phân tích lỗi sáu lượt](reports/classifier/release_source_clean_v2/PHAN_TICH_LOI.md), [kiểm tra artifact độc lập](reports/classifier/release_source_clean_v2/audit_results.json).

## Chạy sau khi clone

Để thử luồng ứng dụng mà chưa cần model, chạy fast theo [README](README.md). Để chạy production ensemble với đủ model, làm theo [hướng dẫn full trên máy nhận](docs/FULL_RUNTIME_SETUP.md). Production ensemble và classifier nghiên cứu v2 dưới đây là hai đối tượng khác nhau; điểm validation của v2 không đại diện ensemble.

### Suy luận classifier nghiên cứu v2 với bộ bàn giao riêng

Lệnh demo nghiên cứu sau cần export v2, thư viện ML và các artifact mà bước kiểm tra release tham chiếu. Các model, dữ liệu/nhãn gốc và artifact study không có sẵn trong bản clone GitHub. Đây không phải lệnh kiểm tra nhanh một bản clone chỉ có mã nguồn:

```powershell
python tools/demo_classifier_final.py
```

Đầu vào mặc định: [classifier_teacher_live.json](examples/classifier_teacher_live.json). Lệnh kiểm tra hash release rồi dùng checkpoint baseline seed 2026 để suy luận CPU; in tiến trình, nhãn, năm xác suất và file JSON đầu ra. Launcher và manifest nghiên cứu ghi nhận môi trường máy tác giả; người nhận cần kiểm tra đường dẫn và cung cấp đầy đủ các tài nguyên tương ứng trước khi chạy.

```powershell
python tools/finalize_classifier_release.py --verify-only
```

Lệnh trên chỉ kiểm tra toàn vẹn, không train hoặc chấm test; cần các tài nguyên mà manifest đã đóng băng tham chiếu. Export v2 cần được tác giả cung cấp riêng và đặt tại `models/controlled_study/classifier_source_clean_context_study_v1/baseline_seed_2026/`, gồm weights/tokenizer/config/inference config. Đường dẫn đó là vị trí cần tạo sau khi nhận model, không phải thư mục có sẵn trên GitHub. Chỉ có mã nguồn và báo cáo không đủ để chạy lại toàn bộ study hoặc kiểm tra mọi hash dữ liệu gốc.

## Kết quả đối chứng

| Cấu hình | Số seed | Mean Macro-F1 | SD mẫu |
| --- | --- | --- | --- |
| Baseline context cũ | 3 | 0,615912 | 0,014231 |
| Context Section khôi phục | 3 | 0,388637 | 0,213019 |

Chọn cấu hình theo mean đủ ba seed, sau đó chọn checkpoint export tốt nhất trong cấu hình đó. Context mới có một lượt tốt nhưng không ổn định; bản v2 giữ context baseline. Train3.942/147 ID; validation1.317/29 ID. Encoder study-v1/full, ctx_version3, text600 ký tự, max256 tokens toàn cặp, OTHER bias0, argmax năm lớp; không áp dụng confidence cutoff.

## Code chính

| Chức năng | Nguồn |
| --- | --- |
| Loại nguồn trùng | [classifier_source_split.py](classifier_source_split.py#L97) |
| Context đối chứng | [classifier_context_v2.py](classifier_context_v2.py#L54) |
| Encode input | [study_inputs.py](study_inputs.py#L12) |
| Weighted loss | [classifier_training.py](classifier_training.py#L18) |
| Train/evaluate study | [classifier_context_experiment.py](classifier_context_experiment.py#L96) |
| Inference | [classifier_predict.py](classifier_predict.py#L81) |
| Freeze/verify v2 | [finalize_classifier_release.py](tools/finalize_classifier_release.py) |

Các nguồn đã được băm trong study/release được giữ nguyên. Các script lịch sử không còn gọi trực tiếp nằm trong [archive/classifier_legacy](archive/classifier_legacy/), với [biên bản dọn](reports/classifier/CLASSIFIER_CLEANUP_20261027.json). Checkpoint tốt nhất/cuối, bản export, dữ liệu, nhãn và bằng chứng nghiên cứu được bảo toàn trên máy tác giả. Bản GitHub chia sẻ mã và báo cáo đã chọn; không tải lên weights, dữ liệu hoặc nhãn gốc.

## Phạm vi kết luận và lịch sử

Nhãn gốc chưa được phân xử độc lập hoàn chỉnh; 57 đề xuất AI không áp dụng vào nghiên cứu mới. Full-source raw/normalized hashes đã tách, không phải chứng nhận hết mọi near-duplicate. Chưa chấm final test vì gate nguồn/nhãn người chưa đạt; không tuyên bố tốt hơn ensemble production hoặc đã đánh giá chất lượng PDF/RAG/toàn hệ thống.

Ensemble production final_s11 + final_s22 + final_s33 giữ nguyên. Bản v1 seed123 với validation0,654426 là kết quả lịch sử trước purge nguồn, được giữ tại [manifest v1](reports/classifier/FINAL_MODEL_MANIFEST.json) và [selection v1](reports/research_review/final_selection.json). Không gán điểm v1 cho checkpoint v2. Công cụ build_classifier_handoff.py là builder v1; builder hiện hành là tools/build_classifier_release_report.py và tools/package_classifier_v2.py.
