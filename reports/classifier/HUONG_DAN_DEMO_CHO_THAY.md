# Demo và show code classifier PSAI bản v2

## 1. Giới thiệu

“Em fine-tune mDeBERTa cho năm vai trò câu: OBJECTIVE, REQUIREMENT, ACTIVITY, EXPECTED_RESULT, OTHER. Em đã hoàn thành các đối chứng dữ liệu/context và ba seed, rồi chọn checkpoint bằng validation. Sau đây em chạy suy luận thật bằng bản đã đóng băng.”

## 2. Chạy demo

Mở Terminal PowerShell trong VS Code:

```powershell
Set-Location "D:\DSP_PROJECT\DSP_PROJECT"
python tools/demo_classifier_final.py
```

Chỉ sao chép lệnh, không sao chép dấu nhắc PS ...> hoặc >>. Nếu đang ở >>, nhấn Ctrl+C rồi chạy lại lệnh một dòng. Chương trình kiểm tra hash, in phiên bản và SELECTED RUN baseline_seed_2026, sau đó nạp model CPU. Màn hình in tiến trình và cuối cùng DEMO COMPLETE cùng đường dẫn OUTPUT. Thời gian tùy tải máy.

Mở [classifier_teacher_live.json](../../examples/classifier_teacher_live.json), sửa text/context, Ctrl+S và chạy lại để thử câu khác. Đầu vào gồm text, section, prev_text, next_text, pos và id. ID chỉ ghép kết quả, không đưa vào encoder. Đây là classifier nhận văn bản; lệnh không đọc PDF.

Demo đã kiểm tra với câu “The system must allow users to upload PDF files.”, Section “System Requirements”: dự đoán **REQUIREMENT**, confidence **0.9375**. Đây là điểm softmax của câu, không phải accuracy của model. Mở JSON tại OUTPUT để xem đủ xác suất năm lớp.

## 3. Sáu tab code nên mở

| Thầy hỏi | Code | Câu giải thích |
| --- | --- | --- |
| Vì sao loại202 câu? | [prepare](../../classifier_source_split.py:97) | Đối chiếu toàn nguồn, bỏ tài liệu train trùng validation; giữ nguyên validation. |
| Section khôi phục thế nào? | [parse_source](../../classifier_context_v2.py:54), [revise_rows](../../classifier_context_v2.py:146) | Đọc heading Markdown, ghép vị trí câu, giữ cũ khi không xác định. Đây là phương án đối chứng; model v2 chọn context baseline. |
| Model nhận gì? | [context_for](../../study_inputs.py:12), [encode_items](../../study_inputs.py:33) | Ghép Section/vị trí/câu lân cận;600 ký tự câu,256 tokens toàn cặp. |
| Loss huấn luyện? | [compute_loss](../../classifier_training.py:18) | CE theo mẫu × trọng số lớp; smoothing0,1; mean microbatch và accumulation. |
| Chạy thí nghiệm/chọn model? | [run](../../classifier_context_experiment.py:96), [TrainingArguments](../../classifier_context_experiment.py:143), [summarize](../../classifier_context_experiment.py:179) | Hai context × ba seed; checkpoint theo validation, so sánh export FP32; chọn cấu hình theo mean rồi chọn checkpoint trong cấu hình đó. |
| Dự đoán nhãn? | [predict](../../classifier_predict.py:81) | Load checkpoint, eval/inference_mode, softmax, argmax và xuất JSON. |

Mở code train để giải thích; không nhấn Run các script train trong buổi demo. [Năm nhãn và model nền](../../train_human_cv.py:26) cho thấy đây là fine-tune pretrained model, không pretrain từ đầu.

## 4. Số liệu của đúng model demo

Macro-F1 validation **0.632077**, accuracy **0.760061**, n=1.317; train3.942 câu. Baseline ba seed có mean0,615912 và SD mẫu0,014231. [Metrics](../research_review/studies/classifier_source_clean_context_study_v1/experiments/baseline_seed_2026/validation_metrics.json), [confusion](../research_review/studies/classifier_source_clean_context_study_v1/experiments/baseline_seed_2026/validation_confusion.csv).

Giải thích Macro-F1: lấy trung bình F1 của năm lớp để mỗi lớp có trọng số ngang nhau khi báo cáo. Accuracy là tỷ lệ đúng toàn bộ câu. Đây là validation, không phải final test độc lập.

## 5. Nếu hỏi lịch sử thí nghiệm

Có năm lượt đối chứng ban đầu (baseline_original_labels, baseline, A0_text, seed123, seed2026) và sáu lượt sau purge toàn nguồn. [Driver ban đầu](../../study_experiments.py:95), [hai seed bổ sung](../../classifier_followup.py:22). Không đếm mọi lựa chọn trong plan là đã chạy.

Điểm0,654426 thuộc v1 seed123 trước khi loại nguồn trùng. V2 dùng context baseline và seed2026 trên train3.942 câu, điểm0,632077. Hai điều kiện train khác nhau; không quy chênh lệch hoàn toàn cho purge và không gán điểm v1 cho v2. Khôi phục Section làm metadata đầy đủ hơn, nhưng đối chứng không cho thấy lợi ích ổn định nên không chọn phương án đó.

## 6. Bằng chứng và kiểm tra

- [Báo cáo Word](release_source_clean_v2/BAO_CAO_CLASSIFIER.docx) có đầy đủ bảng số liệu và ví dụ lỗi để đọc trực tiếp.
- [Manifest v2](release_source_clean_v2/FINAL_MODEL_MANIFEST.json), [quyết định chọn](release_source_clean_v2/final_selection.json), [model card](release_source_clean_v2/model_card.md).
- [Phân tích lỗi](release_source_clean_v2/PHAN_TICH_LOI.md): có trường hợp model sai, không dùng demo để khẳng định model luôn đúng.

```powershell
python tools/finalize_classifier_release.py --verify-only
```

Lệnh verify chỉ kiểm tra toàn vẹn. Ensemble production được giữ nguyên; không có điểm test độc lập hoặc chứng nhận nhãn chuyên gia trong báo cáo này.
