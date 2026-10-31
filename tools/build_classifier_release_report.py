"""Build self-contained teacher documents for the frozen classifier v2.

Reads saved evidence; does not train, score test data or alter production.
"""
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_classifier_report import export_docx, table
from tools.finalize_classifier_release import LABELS, RELEASE_REL, STUDY_REL, read, sha, verify_release


def plain(markdown):
    value = re.sub(r'\[([^\]]+)\]\(([^\n)]+)\)', r'\1', markdown)
    value = re.sub(r'^#{1,6}\s+', '', value, flags=re.MULTILINE)
    value = re.sub(r'^```[^\n]*\n?', '', value, flags=re.MULTILINE)
    return value.replace('**', '').replace('`', '')


def runtime_minutes(study, name):
    for line in (study / (name + '.stdout.log')).read_text(encoding='utf-8', errors='replace').splitlines():
        if 'train_runtime' in line:
            return float(ast.literal_eval(line)['train_runtime']) / 60
    return None


def main():
    manifest = verify_release()
    release = ROOT / RELEASE_REL
    study = ROOT / STUDY_REL
    audit = read(release / 'audit_results.json')
    errors = read(release / 'error_analysis.json')
    prep = read(ROOT / 'reports/research_review/studies/classifier_source_clean_v1/preparation_manifest.json')
    runs = audit['run_results']
    selected = next(row for row in runs if row['run'] == manifest['selected_run'])
    paired_deltas = read(study / 'summary.json')['paired_seed_deltas']
    demo_evidence = read(release / 'demo_verification.json')
    demo_inputs = read(release / 'demo_input.json')
    demo_predictions = read(release / 'demo_predictions.json')
    software_checks = read(release / 'software_verification.json')
    if software_checks['exit_code'] != 0:
        raise ValueError('Release software checks did not pass')
    for relative, digest in software_checks['source_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Software checks are stale after a source change: ' + relative)
    if demo_evidence['selected_run'] != manifest['selected_run'] or demo_evidence['exit_code'] != 0:
        raise ValueError('Live demo does not match the selected classifier release')
    now = datetime.now().astimezone()
    old = []
    for name, folder, description in [
        ('baseline_original_labels', 'user_labels_v1', 'Nhãn gốc; full context; seed 42'),
        ('baseline', 'user_labels_v1', 'Nhãn AI đề xuất; full context; seed 42'),
        ('A0_text', 'user_labels_v1', 'Nhãn AI đề xuất; text-only; seed 42'),
        ('seed_123', 'classifier_stability_v1', 'Nhãn gốc; full context; seed 123'),
        ('seed_2026', 'classifier_stability_v1', 'Nhãn gốc; full context; seed 2026')]:
        metric = read(ROOT / f'reports/research_review/studies/{folder}/experiments/{name}/validation_metrics.json')
        old.append([name, description, f"{metric['macro_f1']:.6f}", f"{metric['accuracy']:.6f}"])
    old_summary = read(ROOT / 'reports/research_review/studies/classifier_stability_v1/analysis/summary.json')
    report = [f'''# BÁO CÁO XÂY DỰNG VÀ THỰC NGHIỆM CLASSIFIER PSAI

Phiên bản bàn giao: 2.0-source-clean. Ngày tổng hợp: {now:%d/%m/%Y %H:%M} (UTC+7).

## 1. Kết quả và sản phẩm của nhánh classifier

Nhánh classifier đã hoàn tất quy trình chuẩn bị dữ liệu, tạo ngữ cảnh, fine-tune, đánh giá validation, lưu checkpoint/tokenizer và suy luận. Đã hoàn thành sáu lượt đối chứng trên train được loại trùng toàn nguồn; mỗi cấu hình có đủ ba seed 123, 42 và 2026. Mô hình được chọn cho bản bàn giao này là baseline seed 2026 với Validation Macro-F1 {selected['macro_f1']:.6f}, Accuracy {selected['accuracy']:.6f} trên 1.317 câu.

Việc chọn tuân theo quy tắc: so sánh Macro-F1 trung bình của đủ ba seed để chọn cấu hình, sau đó chọn checkpoint xuất ra có điểm validation cao nhất trong cấu hình đó. Baseline đạt trung bình 0,615912 và độ lệch chuẩn mẫu 0,014231; context khôi phục Section đạt 0,388637 và 0,213019. Kết quả quan sát không ủng hộ sử dụng context mới cho bản bàn giao này.

Bản v2 có manifest đóng băng trọng số, tokenizer, cấu hình, encoder, phiên bản dữ liệu và bằng chứng validation. Bản v1 được giữ để truy vết; ensemble final_s11 + final_s22 + final_s33 của ứng dụng không bị thay thế. Các điểm trong báo cáo là validation đã dùng trong phát triển, không phải kết quả trên final test độc lập.

## 2. Bài toán và phạm vi chức năng

Classifier phân loại mỗi câu thành một trong năm vai trò:

{table(['Nhãn', 'Ý nghĩa'], [['OBJECTIVE', 'Mục tiêu muốn đạt của dự án'], ['REQUIREMENT', 'Yêu cầu, ràng buộc hoặc khả năng cần đáp ứng'], ['ACTIVITY', 'Công việc hoặc hoạt động dự kiến thực hiện'], ['EXPECTED_RESULT', 'Kết quả, sản phẩm hoặc trạng thái kỳ vọng'], ['OTHER', 'Nội dung còn lại']])}

Luồng xử lý của nhánh:

```text
Câu + Section + vị trí + câu trước/sau
    -> tạo context -> tokenizer -> mDeBERTa
    -> logits năm lớp -> softmax -> argmax
    -> nhãn, confidence và xác suất năm lớp trong JSON
```

Mô hình nền microsoft/mdeberta-v3-base được pretrain sẵn; công việc thực hiện là fine-tune cho năm nhãn cùng quy trình dữ liệu, đánh giá và suy luận. Classifier không trực tiếp sinh báo cáo. Các chỉ số phân loại câu không chứng nhận chất lượng trích xuất PDF, OCR, RAG hoặc toàn bộ luồng sinh báo cáo.

## 3. Dữ liệu và kiểm soát trùng nguồn

Khi rà soát nguồn Markdown, phát hiện ba ID train PSAI069, PSAI086, PSAI100 có toàn văn trùng với validation PSAI035. Nhóm train chứa 202 câu; tài liệu validation liên quan chứa hai câu. Khác ID tài liệu chưa bảo đảm khác nguồn. Phát hiện này xác định hạn chế của split ban đầu, không tự đo được mức độ làm tăng hoặc giảm F1.

Đã loại nguyên các tài liệu train thuộc nhóm trùng, giữ nguyên nhãn và thứ tự validation. Train giảm từ 4.144 xuống 3.942 câu, từ 150 xuống 147 ID tài liệu; validation giữ 1.317 câu thuộc 29 ID. Hash toàn nguồn thô và sau chuẩn hóa của 179 nguồn đã được đối chiếu, không còn nhóm trùng xuyên hai tập theo các phép kiểm tra này. Chưa chứng nhận hết mọi near-duplicate hoặc ảnh hưởng từ lịch sử phát triển.

{table(['Nhãn', 'Train ban đầu', 'Train sau loại nguồn trùng', 'Validation'], [[label, prep['class_counts']['training_before'][label], prep['class_counts']['training_after'][label], prep['class_counts']['validation_unchanged'][label]] for label in LABELS])}

Hai cấu hình mới dùng đúng cùng câu, nhãn, vị trí, câu lân cận và thứ tự dữ liệu; chỉ Section khác nhau. Nhãn tham chiếu gốc được giữ nguyên. Có 57 thay đổi nhãn AI có hiệu lực trong thí nghiệm lịch sử, nhưng không có thay đổi nào được áp dụng vào sáu lượt của nghiên cứu mới; các đề xuất đó không được xem là ground truth.

## 4. Thiết kế đầu vào và huấn luyện

Context full gồm Section, Pos, dấu Bullet, ba Cue từ câu và Previous/Next. Cue là đặc trưng đầu vào; quyết định nhãn đến từ model. Text lấy tối đa 600 ký tự; Section lấy 120 ký tự cuối; Previous lấy 150 ký tự cuối, Next lấy 150 ký tự đầu. Tokenizer nhận cặp text/context với truncation longest_first và giới hạn 256 tokens cho toàn cặp.

Phương án context mới khôi phục Section từ heading Markdown, nhận biết vùng code fence; trường hợp không xác định giữ Section cũ và ghi để kiểm tra. Trong dữ liệu chuẩn bị trước purge, validation thiếu Section giảm 1.045 xuống 26; train từ 2.151 xuống 33. Đó là số đo độ đầy đủ metadata, chưa phải tỷ lệ đúng được người/PDF xác nhận và không tự chứng minh F1 tăng.

{table(['Tham số', 'Giá trị'], [['Model', 'microsoft/mdeberta-v3-base; đầu phân loại năm nhãn'], ['Train / validation', '3.942 / 1.317 câu'], ['Epochs / optimizer updates', '5 / 1.235'], ['Learning rate', '3e-5'], ['Microbatch / accumulation / batch hiệu dụng', '1 / 16 / 16'], ['Optimizer / scheduler', 'AdamW torch / linear'], ['Warmup / weight decay / max grad norm', '124 bước / 0,01 / 1,0'], ['Max length / text limit', '256 tokens cho cặp / 600 ký tự câu'], ['Label smoothing', '0,1'], ['Tham số model / train autocast', 'FP32 / BF16; gradient checkpointing'], ['Seeds', '123, 42, 2026'], ['Chọn checkpoint trong một lượt', 'Validation Macro-F1 mỗi epoch trong Trainer BF16'], ['So sánh các lượt export', 'Cùng giao thức FP32, batch evaluation 8; OTHER bias 0']])}

Loss được tính theo mẫu: cross-entropy có label smoothing 0,1, nhân trọng số lớp tham chiếu rồi lấy trung bình microbatch. Trọng số w[c] = N_train / (5 × số mẫu lớp c). Custom Trainer đặt model_accepts_loss_kwargs=False để giữ cơ chế chia loss theo accumulation.

```python
per_sample = cross_entropy(logits, labels,
    label_smoothing=0.1, reduction='none')
loss = (per_sample * class_weights[labels]).mean()
```

Macro-F1 là trung bình F1 của năm lớp; mỗi lớp có trọng số ngang nhau khi báo cáo. Accuracy đo tỷ lệ đúng trên toàn bộ câu. Do OTHER chiếm 843/1.317 câu validation, accuracy riêng không thể hiện đầy đủ chất lượng ở các lớp ít mẫu.

Điểm Trainer BF16 và điểm đánh giá export FP32 được ghi riêng. Với checkpoint được chọn, Trainer best Macro-F1 là {audit['selection']['selected_trainer_bf16_best_macro_f1']:.6f}, còn FP32 từ dự đoán đã lưu là {selected['macro_f1']:.6f}. Toàn bộ bảng so sánh dưới đây dùng điểm FP32; không trộn hai giao thức để xếp hạng.

## 5. Kết quả sáu lượt đối chứng hoàn tất

{table(['Lượt', 'Seed', 'Cấu hình', 'Macro-F1', 'Accuracy', 'Train runtime (phút)'], [[r['run'], r['seed'], r['variant'], f"{r['macro_f1']:.6f}", f"{r['accuracy']:.6f}", f"{runtime_minutes(study, r['run']):.1f}"] for r in runs])}

Train runtime là số Trainer ghi, không phải tổng thời gian từ lúc mở chương trình đến hoàn tất mọi bước nạp model/export. Không suy ra thời gian PDF/OCR hoặc toàn hệ thống từ số này.

{table(['Cấu hình', 'Số seed', 'Mean Macro-F1', 'SD mẫu'], [[g['variant'], g['n'], f"{g['macro_f1_mean']:.6f}", f"{g['macro_f1_sample_sd']:.6f}"] for g in audit['configuration_statistics']])}

{table(['Seed', 'Baseline', 'Context mới', 'Context − baseline'], [[d['seed'], f"{d['baseline_f1']:.6f}", f"{d['context_f1']:.6f}", f"{d['delta']:+.6f}"] for d in paired_deltas])}

Context mới có lợi ở seed 42 nhưng giảm mạnh ở seed 123 và 2026. Không bỏ hai lượt thấp khỏi thống kê và không chọn cấu hình chỉ vì một lượt đạt điểm cao. Nghiên cứu này gồm ba seed trên một validation cố định, không phải ba tập validation độc lập. SD phản ánh độ biến động theo seed trong điều kiện đã đo, không phải khoảng tin cậy cho mọi tài liệu mới.

## 6. Precision, recall và F1 của cả năm lớp

{table(['Lượt', 'Nhãn', 'Precision', 'Recall', 'F1', 'Support'], [[r['run'], label, f"{r['per_class'][label]['precision']:.6f}", f"{r['per_class'][label]['recall']:.6f}", f"{r['per_class'][label]['f1-score']:.6f}", r['per_class'][label]['support']] for r in runs for label in LABELS])}

## 7. Phân tích lỗi có bằng chứng

Context seed 123 và 2026 không dự đoán câu nào là REQUIREMENT. Context seed 2026 nhận đúng 0/97 câu EXPECTED_RESULT và chỉ đưa tám câu vào lớp này. Đây là hiện tượng quan sát từ dự đoán và confusion đã lưu, chưa phải kết luận về cơ chế gây lỗi.

Số câu baseline đúng/context sai lần lượt là 313, 81 và 410 cho seed 123, 42 và 2026; chiều ngược lại là 131, 99 và 105. Có 263 câu giữ nguyên Section ở hai cấu hình, nhưng trong nhóm đó vẫn có lỗi hồi quy 79, 14 và 96 câu. Vì trọng số model được huấn luyện trên toàn tập khác context, không thể quy mọi lỗi ở validation trực tiếp cho Section riêng của câu đó.

Điểm thấp đã xuất hiện trong validation theo epoch của Trainer ở hai seed yếu; chưa có bằng chứng cho rằng suy giảm chỉ do export hoặc độ chính xác số. Lỗi mô hình, tác động của context trong huấn luyện, chất lượng extraction và nhãn tham chiếu là các khả năng cần phân biệt; report không tự xác nhận lỗi PDF hay đổi nhãn khi chưa kiểm tra nguồn/người.

Các ví dụ dưới đây dùng nhãn tham chiếu frozen hiện có. B là baseline, C là context mới:
''']
    for case in errors['cases']:
        if case['case_id'] not in {'E01', 'E03', 'E05', 'E09', 'E14', 'E15'}:
            continue
        rows = [[seed, case['predictions'][seed]['baseline']['label'], case['predictions'][seed]['context']['label']]
                for seed in ['123', '42', '2026']]
        report.append(f"""### {case['case_id']} — {case['document_id']}, validation row {case['validation_row_1based']}

Câu: {case['full_text']}

Section cũ: {case['old_section'] or '(trống)'}

Section mới: {case['new_section'] or '(trống)'}

Câu trước: {case['prev_text'] or '(không có)'}

Câu sau: {case['next_text'] or '(không có)'}

Nhãn tham chiếu: {case['gold_frozen_reference_label']}

{table(['Seed', 'B dự đoán', 'C dự đoán'], rows)}

{case['evidence_note_vi']}

Lỗi extraction/PDF, nguyên nhân trực tiếp từ context và khả năng nhãn tham chiếu sai đều chưa được xác minh cho ví dụ này; cần người kiểm tra trước khi sửa nhãn.
""")
    report.append(f'''## 8. Model được đóng băng cho bản bàn giao

Cấu hình được chọn là baseline context cũ trên train đã loại nguồn trùng, seed 2026. Model giữ đúng encoder study-v1/full, ctx_version=3; không thay sang Section v2.1. Đầu ra dùng argmax của năm xác suất, OTHER bias 0; không áp dụng một ngưỡng lọc confidence trong CLI classifier. Confidence softmax chưa được hiệu chuẩn nên không gọi nó là xác suất đúng đã được chứng nhận.

{table(['Trường', 'Giá trị'], [['Release', manifest['release_version']], ['Selected run', manifest['selected_run']], ['Checkpoint', manifest['model_path']], ['Macro-F1 validation', f"{selected['macro_f1']:.6f}"], ['Accuracy validation', f"{selected['accuracy']:.6f}"], ['Weight SHA256', manifest['model_sha256']], ['Train fingerprint', manifest['dataset']['train_sha256']], ['Validation fingerprint', manifest['dataset']['validation_sha256']], ['Điểm chốt', 'Model export sau checkpoint tốt nhất theo giao thức Trainer; step 1235'], ['Production ensemble', 'Giữ nguyên final_s11 + final_s22 + final_s33']])}

Checkpoint export có model.safetensors, tokenizer.json, tokenizer_config.json, config.json, inference_config.json và training_provenance.json. Manifest đối chiếu hash các file này, mã inference/encoder và dữ liệu/validation đã dùng. Công cụ verify từ chối file bị thay đổi hoặc đường dẫn vượt khỏi project.

## 9. Demo và code đã thực hiện

Chạy tại thư mục project:

```powershell
python tools/demo_classifier_final.py
```

Lệnh kiểm tra manifest v2 rồi chạy suy luận thật bằng CPU với checkpoint baseline seed 2026. Trên máy này, lệnh dùng môi trường CPU riêng trên ổ D; log ghi tên model, đường dẫn đầu vào và file JSON đầu ra. Để đổi câu, sửa examples/classifier_teacher_live.json, lưu và chạy lại.

Demo đã chạy thành công với câu: {demo_inputs[0]['text']}

Section: {demo_inputs[0].get('section', '')}. Nhãn dự đoán: {demo_predictions[0]['label']}; confidence {demo_predictions[0]['confidence']:.6f}. Xác suất năm lớp:

{table(['Nhãn', 'Xác suất demo'], [[label, f"{demo_predictions[0]['probabilities'][label]:.6f}"] for label in LABELS])}

Đây là suy luận thật trên câu minh họa, không phải kết quả benchmark. Confidence của một câu không phải accuracy toàn mô hình. Dữ liệu đầu vào/đầu ra của demo và hash được lưu cùng gói bàn giao để đối chiếu.

{table(['Chức năng', 'Code để giải thích'], [['Tách nguồn và loại trùng train', 'classifier_source_split.py: prepare'], ['Khôi phục Section đối chứng', 'classifier_context_v2.py: parse_source, revise_rows'], ['Tạo input', 'study_inputs.py: context_for, encode_items'], ['Loss', 'classifier_training.py: WeightedClassificationTrainer.compute_loss'], ['Train và metric', 'classifier_context_experiment.py: run, compute_metrics, summarize'], ['Suy luận', 'classifier_predict.py: predict'], ['Demo bản đã chốt', 'tools/demo_classifier_final.py'], ['Đóng băng/kiểm tra release', 'tools/finalize_classifier_release.py']])}

Các file train được mở để giải thích khi bảo vệ; demo không khởi động train lại. Demo là kiểm tra chức năng suy luận, không phải tập đánh giá chất lượng.

## 10. Kết quả lịch sử được giữ để đối chiếu

Năm lượt dưới đây thuộc điều kiện train ban đầu 4.144 câu, trước khi phát hiện trùng toàn nguồn. Chúng được giữ làm bằng chứng quá trình, không trộn vào bảng sáu lượt mới hoặc dùng điểm cao nhất để thay model v2.

{table(['Lượt lịch sử', 'Điều kiện', 'Validation Macro-F1', 'Accuracy'], old)}

Ba seed nhãn gốc/full context ban đầu đạt mean {old_summary['mean']:.6f}, SD mẫu {old_summary['sample_std']:.6f}; seed 123 được chọn cho v1 với Macro-F1 0,654426. Điểm v1 và điểm v2 có điều kiện train khác nhau; chênh lệch không đo riêng tác động của việc loại nguồn trùng. Các đề xuất sửa nhãn AI trong giai đoạn trước không chứng minh nhãn chuẩn hơn và không được tự coi là ground truth.

## 11. Kiểm chứng và giới hạn kết luận

Kiểm tra độc lập các artifact đã lưu đạt 26/26 nhóm kiểm tra và toàn bộ kiểm tra từng run: metrics/confusion tính lại khớp predictions, nhãn/text/document/order được ghép đúng, dữ liệu khớp phiên bản chuẩn bị, source hash và code snapshot khớp, model được chọn khớp hash đã ghi. Việc xuất báo cáo chỉ đọc lại bằng chứng, không train thêm hoặc chấm test.

Đã chạy {software_checks['targeted_tests_passed']}/{software_checks['targeted_tests_run']} kiểm tra phần mềm liên quan đến chọn release, toàn vẹn file, demo, ghép cặp thí nghiệm và loại nguồn trùng. Đây là nhóm kiểm tra có phạm vi cụ thể, không phải toàn bộ test suite của dự án. Demo CPU của đúng checkpoint v2 hoàn tất thành công và xuất xác suất hợp lệ.

Tập final test chưa được chấm: các gate xác minh nguồn và nhãn người hiện chưa đạt. Tám tài liệu ứng viên cũ có dấu vết phát triển; gói nguồn mới một tài liệu/80 câu vẫn chưa có nhãn người trong phiếu tham chiếu. Vì vậy không có điểm final test hoặc Cohen's Kappa đủ điều kiện để báo cáo. Những giới hạn này không được thay bằng số F1 validation.

Không có phép so sánh độc lập đủ điều kiện với ensemble production, và chưa đánh giá toàn luồng PDF đến báo cáo trong phạm vi classifier. Nhãn gốc còn trường hợp mơ hồ; phần lỗi trình bày theo nhãn tham chiếu hiện có. Kết luận của bản bàn giao là cấu hình baseline ổn định hơn trong nghiên cứu đã thực hiện, không phải chứng nhận tổng quát cho mọi nguồn tài liệu.

## 12. Toàn bộ quá trình thực hiện và hoàn thiện

Quá trình dưới đây tổng hợp theo thứ tự công việc và các hồ sơ có thể đối chiếu. Hai giai đoạn được báo cáo có 11 lượt controlled hoàn tất: năm lượt theo dữ liệu ban đầu và sáu lượt sau loại trùng toàn nguồn. Con số này không đại diện cho mọi lần train trong toàn bộ lịch sử project; các export lịch sử thiếu provenance đầy đủ không được tự cộng thành thí nghiệm độc lập đã kiểm chứng.

{table(['Giai đoạn', 'Công việc đã thực hiện', 'Kết quả và giới hạn'], [
['1. Xác định bài toán và kiểm kê', 'Giới hạn phần thực hiện ở classifier năm nhãn; rà soát code, checkpoint và kết quả đã lưu.', 'Kiểm kê ban đầu có 27 thư mục export chứa weights: 19 lịch sử, 5 controlled và 3 smoke. Số thư mục không phải số lần train độc lập; smoke không tham gia chọn model.'],
['2. Chuẩn bị dữ liệu ban đầu', 'Chuẩn bị câu, nhãn, document_id, Section, vị trí và câu lân cận; cố định train/validation cho các đối chứng.', 'Train 4.144 câu, validation 1.317 câu. Có kiểm tra trùng câu/ngữ cảnh đã biết, nhưng về sau phát hiện vẫn còn trùng toàn nguồn.'],
['3. Xây dựng đầu vào và loss', 'Tạo context full, tokenize text/context; cấu hình fine-tune mDeBERTa với class weights, label smoothing và gradient accumulation.', 'Lưu quy ước microbatch mean và kiểm tra normalization; không quy mọi khác biệt F1 lịch sử cho một lỗi loss.'],
['4. Chạy năm đối chứng ban đầu', 'So sánh nhãn gốc/nhãn AI đề xuất, full context/text-only và ba seed cho nhãn gốc/full context.', 'Điểm và điều kiện ở mục10. V1 seed123 đạt validation0,654426; đây là kết quả trước khi xử lý trùng toàn nguồn.'],
['5. Phân tích độ ổn định và nhãn', 'Đối chiếu predictions, P/R/F1 và confusion của ba seed ban đầu; lập danh sách câu cần kiểm tra.', 'Có165 câu seed2026 sai trong khi ít nhất một seed khác đúng;73 câu hai seed kia cùng đúng.68 đề xuất nhãn AI tạo57 thay đổi train có hiệu lực; chưa phải ground truth độc lập.'],
['6. Kiểm tra và khôi phục Section', 'Phát hiện nhiều trường Section trống; xây parser heading Markdown có nhận biết code fence, ghi các trường hợp chưa xác định.', 'Validation trống1.045→26, train2.151→33 trong dữ liệu chuẩn bị trước purge; còn110 trường hợp cần xác nhận. Metadata đầy đủ hơn chưa chứng minh tăng F1.'],
['7. Rà soát trùng toàn nguồn', 'So raw/normalized SHA256, phát hiện ba ID train cùng nguồn với PSAI035 validation; ưu tiên giữ validation.', 'Loại202 câu, còn3.942 train; giữ1.317 validation. Kiểm tra179 nguồn và lưu manifest, snapshot/hash. Không tuyên bố loại hết mọi near-duplicate.'],
['8. Chạy sáu đối chứng mới', 'Train lại cả context cũ và Section khôi phục trên đúng cùng dữ liệu/nhãn, mỗi cấu hình ba seed.', 'Đủ6/6 lượt, năm epoch/lượt. Mean và SD, điểm từng lớp và sáu confusion được công bố cả khi kết quả suy giảm. Không mở thêm thí nghiệm để săn điểm cao.'],
['9. Thống nhất cách so sánh', 'Giữ riêng Trainer BF16 và đánh giá export FP32; đối chiếu lại predictions với metrics/confusion và provenance.', 'Baseline mean0,615912±0,014231; context mới0,388637±0,213019. Dấu ± là SD mẫu theo seed, không phải khoảng tin cậy trên nguồn mới.'],
['10. Chọn và đóng băng v2', 'Chọn cấu hình theo mean ba seed rồi chọn checkpoint export tốt nhất trong cấu hình; lưu model/tokenizer/config/data/code hashes.', 'Chọn baseline seed2026: Macro-F1 0,632077, accuracy0,760061. Bảo toàn v1 và ensemble production; không dùng final test để chọn tham số.'],
['11. Hoàn thiện demo suy luận', 'Tạo CLI đọc JSON/JSONL, kiểm tra input/nhãn, trả probabilities; thêm tiến trình hiển thị, log và bắt lỗi tiến trình con.', 'Khi gặp lỗi native liên quan thư viện NVIDIA, dùng môi trường torch2.14.1+cpu và torchvision0.29.1+cpu riêng trên D. Đây là cách tránh lỗi runtime, không phải sửa driver GPU.'],
['12. Kiểm thử và bàn giao', 'Kiểm tra19 test phần mềm có phạm vi,26 nhóm kiểm tra artifact; chạy demo thật; đóng ZIP với hash/CRC và kiểm tra sau giải nén.', 'Demo v2 trả REQUIREMENT/confidence0,937519 cho ví dụ đã lưu. ZIP chứa code và bằng chứng; weights cần chuyển riêng. Demo minh họa không phải benchmark.'],
['13. Dọn và tổ chức project', 'Dọn cache tải gói pip, xóa ba checkpoint trung gian dư của các lượt đã xong và lưu trữ15 script cũ.', 'Xóa khoảng8,60GiB cache pip trên C; dung lượng trống đo được tăng1,90→10,51GiB. Ba checkpoint giải phóng9,39GiB trên D. Giữ export, checkpoint tốt nhất/cuối, dữ liệu, nhãn và báo cáo.']])}

Nội dung hoàn tất gồm phần mềm classifier, đối chứng và lựa chọn bằng validation, toàn vẹn artifact, demo và hồ sơ bàn giao. Những việc không được xem là đã hoàn tất gồm phán xử nhãn độc lập, Cohen's Kappa đủ điều kiện và đánh giá final test có nguồn/nhãn người được xác nhận. Vì vậy báo cáo mô tả sản phẩm và kết quả đã thực hiện, không tự chuyển các giới hạn đó thành chứng nhận chất lượng cuối cùng.

## Phụ lục: Ma trận nhầm lẫn của sáu lượt

Hàng là nhãn tham chiếu, cột là nhãn dự đoán. Các bảng dưới đây được tính lại từ validation predictions đã lưu.
''')
    for run in runs:
        report.append('### ' + run['run'] + '\n\n' + table(['Gold / Pred'] + LABELS,
                     [[label] + run['confusion_matrix'][i] for i, label in enumerate(LABELS)]) + '\n')
    report.append('''## Nguồn bằng chứng trong gói bàn giao

- FINAL_MODEL_MANIFEST.json và final_selection.json: danh tính model, dữ liệu, cấu hình và hash.
- audit_results.json: kiểm tra độc lập sáu lượt.
- PHAN_TICH_LOI.md và error_analysis.json: 15 câu đối chiếu đủ sáu dự đoán.
- Nghiên cứu classifier_source_clean_context_study_v1: plan, metrics, predictions, confusion và provenance.
- Dữ liệu classifier_source_clean_v1: train/validation đã chuẩn bị, các dòng bị loại và manifest nguồn.

Các đường dẫn/tên file phục vụ đối chiếu khi có mã nguồn. Toàn bộ số liệu chính, ma trận, cấu hình và ví dụ lỗi đã nằm trực tiếp trong báo cáo để giảng viên đọc độc lập.
''')
    text = '\n\n'.join(report)
    report_path = release / 'BAO_CAO_CLASSIFIER.md'
    report_path.write_text(text, encoding='utf-8')
    report_path.with_suffix('.txt').write_text(plain(text), encoding='utf-8-sig')
    export_docx(text, report_path.with_suffix('.docx'))
    card = f'''# Classifier PSAI v2 — model card

Model: {manifest['model_path']}. Base: microsoft/mdeberta-v3-base.

Task: five-way sentence classification with text/context. Labels: {', '.join(LABELS)}.

Selected by configuration mean over three validation seeds, then best exported checkpoint in that configuration. Selected baseline seed 2026; validation Macro-F1 {selected['macro_f1']:.6f}, accuracy {selected['accuracy']:.6f}, n=1317. Baseline mean 0.615912 ± 0.014231 sample SD on one development split. These are not independent-test results.

Train: 3942 rows/147 IDs after removing 202 rows from three full-source duplicate IDs. Validation: 1317 rows/29 IDs. Original labels retained; no proposed AI changes applied. Raw/normalized full-source matches removed; unknown near-duplicate/exposure risks not certified absent.

Inputs: text600 characters, full study-v1 context, ctx_version3, 256 tokens total paired input. Inference: FP32, five-way softmax argmax, OTHER bias0, no confidence cutoff, no calibrated correctness probability. Training:5epochs,1235updates,LR3e-5,batch1×GA16,weighted CE,smoothing0.1,BF16 train.

Weight SHA256: {manifest['model_sha256']}

Model/tokenizer/code/data bindings: [FINAL_MODEL_MANIFEST.json](FINAL_MODEL_MANIFEST.json). Selected configuration and recorded runtime: [final_selection.json](final_selection.json).

No superiority claim over deployed ensemble. No final-test score, expert-label certification, PDF-extraction or report-generation quality claim. V1 and production ensemble preserved.

Run: `python tools/demo_classifier_final.py` from the project root. See [DEMO.md](DEMO.md).
'''
    (release / 'model_card.md').write_text(card, encoding='utf-8')
    demo = '''# Demo classifier PSAI bản đã chốt

1. Trong VS Code mở Terminal PowerShell và vào thư mục project.
2. Mở examples/classifier_teacher_live.json; sửa câu/context nếu cần và lưu.
3. Chạy lệnh sau:

```powershell
Set-Location "D:\\DSP_PROJECT\\DSP_PROJECT"
python tools/demo_classifier_final.py
```

Lệnh kiểm tra hash của bản v2 rồi suy luận thật bằng CPU. Model được in phải là baseline_seed_2026. Khi xong, chương trình in DEMO COMPLETE và đường dẫn OUTPUT để mở JSON. Confidence là điểm softmax, không phải accuracy của toàn model.

Các số để trả lời thầy: Macro-F1 validation 0,632077; accuracy 0,760061; 1.317 câu. Điểm 0,654426 thuộc v1 lịch sử trước khi loại nguồn trùng, không phải model đang demo.

Giải thích code theo thứ tự: classifier_source_split.prepare → study_inputs.context_for/encode_items → classifier_training.compute_loss → classifier_context_experiment.run → classifier_predict.predict. Model nền và năm nhãn ở train_human_cv.py. Mở code train để giải thích, không chạy train trong buổi demo.

Trên máy khác cần chép riêng export model đầy đủ vào vị trí ghi trong manifest và thiết lập thư viện phù hợp. ZIP mã/bằng chứng không chứa weights dung lượng lớn hoặc môi trường .venv. CPU demo trên máy hiện tại dùng torch2.14.1+cpu và torchvision0.29.1+cpu trong môi trường riêng.
'''
    (release / 'DEMO.md').write_text(demo, encoding='utf-8')
    (release / 'DEMO.txt').write_text(plain(demo), encoding='utf-8-sig')
    # Preserve serving/training source bytes for this release without moving them.
    for relative, digest in manifest['artifact_sha256'].items():
        if Path(relative).suffix == '.py':
            destination = release / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha(destination) != digest:
                raise ValueError('Release source snapshot already differs: ' + relative)
            if not destination.exists():
                shutil.copyfile(ROOT / relative, destination)
    pointer = dict(release_version=manifest['release_version'], manifest=(RELEASE_REL / 'FINAL_MODEL_MANIFEST.json').as_posix(),
                   manifest_sha256=sha(release / 'FINAL_MODEL_MANIFEST.json'), selected_run=manifest['selected_run'],
                   model_path=manifest['model_path'], report=(RELEASE_REL / 'BAO_CAO_CLASSIFIER.docx').as_posix(),
                   production_ensemble_replaced=False)
    (ROOT / 'reports/classifier/CURRENT_CLASSIFIER_RELEASE.json').write_text(json.dumps(pointer, indent=2), encoding='utf-8')
    current_readme = f'''# Classifier PSAI — bản hiện hành v2

Bản classifier 2.0-source-clean chọn **baseline seed 2026**, sau khi hoàn tất hai cấu hình × ba seed trên train đã loại trùng toàn nguồn. Validation Macro-F1 **{selected['macro_f1']:.6f}**, accuracy **{selected['accuracy']:.6f}**, n=1.317. Đây là điểm validation phát triển, không phải final test độc lập.

## Tài liệu để báo cáo và demo

- [Báo cáo Word đầy đủ](reports/classifier/release_source_clean_v2/BAO_CAO_CLASSIFIER.docx): số liệu, P/R/F1 của cả năm lớp, sáu ma trận và ví dụ lỗi nằm trực tiếp trong báo cáo.
- [Báo cáo text](reports/classifier/release_source_clean_v2/BAO_CAO_CLASSIFIER.txt).
- [Hướng dẫn demo và show code](reports/classifier/HUONG_DAN_DEMO_CHO_THAY.md).
- [Manifest model đã đóng băng](reports/classifier/release_source_clean_v2/FINAL_MODEL_MANIFEST.json), [quyết định lựa chọn](reports/classifier/release_source_clean_v2/final_selection.json), [model card](reports/classifier/release_source_clean_v2/model_card.md).
- [Phân tích lỗi sáu lượt](reports/classifier/release_source_clean_v2/PHAN_TICH_LOI.md), [kiểm tra artifact độc lập](reports/classifier/release_source_clean_v2/audit_results.json).

## Chạy suy luận thật trên máy này

```powershell
python tools/demo_classifier_final.py
```

Đầu vào mặc định: [classifier_teacher_live.json](examples/classifier_teacher_live.json). Lệnh kiểm tra hash release rồi dùng checkpoint baseline seed 2026 để suy luận CPU; in tiến trình, nhãn, năm xác suất và file JSON đầu ra. Môi trường CPU riêng của máy nằm trên ổ D; không sửa môi trường CUDA train.

```powershell
python tools/finalize_classifier_release.py --verify-only
```

Lệnh trên chỉ kiểm tra toàn vẹn, không train hoặc chấm test. [Đường dẫn model](models/controlled_study/classifier_source_clean_context_study_v1/baseline_seed_2026/) có đủ weights/tokenizer/config. Người nhận mã cần chép riêng export đầy đủ này; ZIP mã/bằng chứng không chứa weights hoặc .venv.

## Kết quả đối chứng

| Cấu hình | Số seed | Mean Macro-F1 | SD mẫu |
| --- | --- | --- | --- |
| Baseline context cũ | 3 | 0,615912 | 0,014231 |
| Context Section khôi phục | 3 | 0,388637 | 0,213019 |

Chọn cấu hình theo mean đủ ba seed, sau đó chọn checkpoint export tốt nhất trong cấu hình đó. Context mới có một lượt tốt nhưng không ổn định; bản v2 giữ context baseline. Train3.942/147 ID; validation1.317/29 ID. Encoder study-v1/full, ctx_version3, text600 ký tự, max256 tokens toàn cặp, OTHER bias0, argmax năm lớp; không áp dụng confidence cutoff.

## Code chính

| Chức năng | Nguồn |
| --- | --- |
| Loại nguồn trùng | [classifier_source_split.py](classifier_source_split.py:97) |
| Context đối chứng | [classifier_context_v2.py](classifier_context_v2.py:54) |
| Encode input | [study_inputs.py](study_inputs.py:12) |
| Weighted loss | [classifier_training.py](classifier_training.py:18) |
| Train/evaluate study | [classifier_context_experiment.py](classifier_context_experiment.py:96) |
| Inference | [classifier_predict.py](classifier_predict.py:81) |
| Freeze/verify v2 | [finalize_classifier_release.py](tools/finalize_classifier_release.py) |

Các nguồn đã được băm trong study/release được giữ nguyên. Các script lịch sử không còn gọi trực tiếp nằm trong [archive/classifier_legacy](archive/classifier_legacy/), với [biên bản dọn](reports/classifier/CLASSIFIER_CLEANUP_20261027.json). Checkpoint tốt nhất/cuối, bản export, dữ liệu, nhãn và bằng chứng nghiên cứu được bảo toàn.

## Phạm vi kết luận và lịch sử

Nhãn gốc chưa được phân xử độc lập hoàn chỉnh; 57 đề xuất AI không áp dụng vào nghiên cứu mới. Full-source raw/normalized hashes đã tách, không phải chứng nhận hết mọi near-duplicate. Chưa chấm final test vì gate nguồn/nhãn người chưa đạt; không tuyên bố tốt hơn ensemble production hoặc đã đánh giá chất lượng PDF/RAG/toàn hệ thống.

Ensemble production final_s11 + final_s22 + final_s33 giữ nguyên. Bản v1 seed123 với validation0,654426 là kết quả lịch sử trước purge nguồn, được giữ tại [manifest v1](reports/classifier/FINAL_MODEL_MANIFEST.json) và [selection v1](reports/research_review/final_selection.json). Không gán điểm v1 cho checkpoint v2. Công cụ build_classifier_handoff.py là builder v1; builder hiện hành là tools/build_classifier_release_report.py và tools/package_classifier_v2.py.
'''
    (ROOT / 'README_CLASSIFIER.md').write_text(current_readme, encoding='utf-8')
    current_guide = f'''# Demo và show code classifier PSAI bản v2

## 1. Giới thiệu

“Em fine-tune mDeBERTa cho năm vai trò câu: OBJECTIVE, REQUIREMENT, ACTIVITY, EXPECTED_RESULT, OTHER. Em đã hoàn thành các đối chứng dữ liệu/context và ba seed, rồi chọn checkpoint bằng validation. Sau đây em chạy suy luận thật bằng bản đã đóng băng.”

## 2. Chạy demo

Mở Terminal PowerShell trong VS Code:

```powershell
Set-Location "D:\\DSP_PROJECT\\DSP_PROJECT"
python tools/demo_classifier_final.py
```

Chỉ sao chép lệnh, không sao chép dấu nhắc PS ...> hoặc >>. Nếu đang ở >>, nhấn Ctrl+C rồi chạy lại lệnh một dòng. Chương trình kiểm tra hash, in phiên bản và SELECTED RUN baseline_seed_2026, sau đó nạp model CPU. Màn hình in tiến trình và cuối cùng DEMO COMPLETE cùng đường dẫn OUTPUT. Thời gian tùy tải máy.

Mở [classifier_teacher_live.json](../../examples/classifier_teacher_live.json), sửa text/context, Ctrl+S và chạy lại để thử câu khác. Đầu vào gồm text, section, prev_text, next_text, pos và id. ID chỉ ghép kết quả, không đưa vào encoder. Đây là classifier nhận văn bản; lệnh không đọc PDF.

Demo đã kiểm tra với câu “{demo_inputs[0]['text']}”, Section “{demo_inputs[0].get('section', '')}”: dự đoán **{demo_predictions[0]['label']}**, confidence **{demo_predictions[0]['confidence']:.4f}**. Đây là điểm softmax của câu, không phải accuracy của model. Mở JSON tại OUTPUT để xem đủ xác suất năm lớp.

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

Macro-F1 validation **{selected['macro_f1']:.6f}**, accuracy **{selected['accuracy']:.6f}**, n=1.317; train3.942 câu. Baseline ba seed có mean0,615912 và SD mẫu0,014231. [Metrics](../research_review/studies/classifier_source_clean_context_study_v1/experiments/baseline_seed_2026/validation_metrics.json), [confusion](../research_review/studies/classifier_source_clean_context_study_v1/experiments/baseline_seed_2026/validation_confusion.csv).

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
'''
    guide_path = ROOT / 'reports/classifier/HUONG_DAN_DEMO_CHO_THAY.md'
    guide_path.write_text(current_guide, encoding='utf-8')
    guide_path.with_suffix('.txt').write_text(plain(current_guide), encoding='utf-8-sig')
    for suffix in ['.md', '.txt']:
        shutil.copyfile(report_path.with_suffix(suffix), ROOT / ('reports/classifier/BAO_CAO_CLASSIFIER_GUI_THAY' + suffix))
    print(json.dumps(dict(report=report_path.as_posix(), docx=report_path.with_suffix('.docx').as_posix(),
                         selected_run=manifest['selected_run'], new_training_started=False)))


if __name__ == '__main__':
    main()
