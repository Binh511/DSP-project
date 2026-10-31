"""Export a self-contained Vietnamese teacher report from saved evidence only.

Standard library only: no training, model loading, test evaluation, or GPU work.
"""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/classifier'
STUDIES = ROOT / 'reports/research_review/studies'
STUDY = STUDIES / 'classifier_source_clean_context_study_v1'
LABELS = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']


def read_json(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.1)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def check_saved_evaluation(folder):
    """Recount saved predictions; never invoke a model or read final-test data."""
    result = read_json(folder / 'validation_metrics.json')
    if result.get('evaluation_kind') != 'validation_selection':
        raise ValueError(f'Unexpected evaluation role: {folder}')
    with (folder / 'validation_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = [[0] * 5 for _ in LABELS]
    for row in rows:
        counts[LABELS.index(row['true_label'])][LABELS.index(row['predicted_label'])] += 1
    with (folder / 'validation_confusion.csv').open(encoding='utf-8-sig', newline='') as stream:
        saved = list(csv.reader(stream))
    assert saved[0][1:] == LABELS
    assert [row[0] for row in saved[1:]] == LABELS
    assert [[int(value) for value in row[1:]] for row in saved[1:]] == counts
    f1s = []
    for index, label in enumerate(LABELS):
        true_count = sum(counts[index])
        predicted_count = sum(row[index] for row in counts)
        tp = counts[index][index]
        precision = tp / predicted_count if predicted_count else 0.
        recall = tp / true_count if true_count else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        expected = result['per_class'][label]
        for key, value in [('precision', precision), ('recall', recall), ('f1-score', f1), ('support', true_count)]:
            assert abs(float(expected[key]) - value) < 1e-9, (folder, label, key)
        f1s.append(f1)
    assert len(rows) == result['n'] == 1317
    assert abs(sum(f1s) / 5 - result['macro_f1']) < 1e-9
    assert abs(sum(counts[i][i] for i in range(5)) / len(rows) - result['accuracy']) < 1e-9
    return result, counts


def main():
    now = datetime.now(ZoneInfo('Asia/Bangkok'))
    queue = read_json(STUDY / 'queue_status.json')
    source_path = REPORTS / 'BAO_CAO_TONG_KET_CLASSIFIER.md'
    original = source_path.read_text(encoding='utf-8-sig')
    evidence = [source_path, STUDY / 'queue_status.json', STUDY / 'experiment_plan.json',
                STUDIES / 'classifier_source_clean_v1/preparation_manifest.json',
                STUDIES / 'classifier_context_v2_1/preparation_manifest.json',
                ROOT / 'reports/research_review/final_selection.json']
    old = {}
    for seed in [42, 123, 2026]:
        folder = STUDIES / f'classifier_validation_replay_v1/seed_{seed}'
        old[seed], _ = check_saved_evaluation(folder)
        evidence.extend(folder / filename for filename in
                        ['validation_metrics.json', 'validation_predictions.csv', 'validation_confusion.csv'])
    for name in ['baseline', 'A0_text']:
        folder = STUDIES / f'user_labels_v1/experiments/{name}'
        check_saved_evaluation(folder)
        evidence.extend(folder / filename for filename in
                        ['validation_metrics.json', 'validation_predictions.csv', 'validation_confusion.csv'])
    mean = statistics.mean(row['macro_f1'] for row in old.values())
    sd = statistics.stdev(row['macro_f1'] for row in old.values())
    assert f'{mean:.6f}' in original and f'{sd:.6f}' in original

    intro = f'''# BÁO CÁO XÂY DỰNG VÀ THỰC NGHIỆM CLASSIFIER PSAI

Bản thuyết minh gửi giảng viên. Số liệu chốt tại {now:%H:%M, ngày %d/%m/%Y} (UTC+7).

## 1. Mục tiêu, sản phẩm và phạm vi

Nhánh classifier phân loại câu trong tài liệu đề xuất dự án thành năm vai trò: OBJECTIVE (mục tiêu), REQUIREMENT (yêu cầu hoặc ràng buộc), ACTIVITY (hoạt động dự kiến thực hiện), EXPECTED_RESULT (kết quả hoặc sản phẩm kỳ vọng), OTHER (nội dung còn lại). Đầu vào gồm câu và ngữ cảnh; đầu ra gồm nhãn dự đoán và phân phối xác suất của năm lớp. Độ tin cậy softmax chưa được kiểm định hiệu chuẩn nên không diễn giải trực tiếp thành xác suất đúng thực tế.

Luồng của riêng nhánh này:
Văn bản và cấu trúc tài liệu -> câu + Section + câu trước/sau + vị trí -> tokenizer -> mDeBERTa -> xác suất năm lớp -> nhãn dự đoán -> bảng đánh giá và dữ liệu phục vụ ứng dụng.

Đã thực hiện chuẩn bị dữ liệu, fine-tune mDeBERTa, năm lượt đối chứng hoàn chỉnh theo giao thức ban đầu, phân tích ba seed, thống nhất inference FP32, đóng băng bản bàn giao v1, xây dựng công cụ kiểm tra nguồn/nhãn và demo. Năm lượt đối chứng gồm ba seed của nhãn gốc/full context, một lượt nhãn AI đề xuất/full context và một lượt nhãn AI đề xuất/text-only. Không đồng nhất năm lượt này với năm phương pháp độc lập.

Bản v1 lưu checkpoint seed 123 với Validation Macro-F1 0,654426. Sau rà soát đã phát hiện trùng toàn nguồn giữa train và validation, vì vậy v1 và các điểm ban đầu được giữ làm kết quả theo điều kiện lịch sử, không chứng nhận là đánh giá trên nguồn độc lập. Đợt kiểm chứng mới so sánh hai cách tạo context trên cùng train đã loại nguồn trùng; kết quả hoàn tất đến thời điểm xuất báo cáo được trình bày ở mục 11.

Phạm vi báo cáo là classifier. Chất lượng trích xuất PDF, sinh báo cáo và RAG cần bộ đánh giá riêng; F1 của classifier không chứng minh chất lượng toàn hệ thống. Ensemble production final_s11 + final_s22 + final_s33 được giữ nguyên.

'''
    body = original[original.index('## 2.'):]
    stale = ('Đã chuẩn bị context v2.1 từ Markdown nguồn, chỉ sửa Section khi vị trí/heading đủ rõ. '
             'Đây là dữ liệu chuẩn bị riêng, chưa train/evaluate; không dùng F1 ở trên để tuyên bố context mới tốt hơn.')
    assert stale in body
    body = body.replace(stale, 'Đã chuẩn bị context v2.1 từ Markdown nguồn, chỉ sửa Section khi khớp vị trí/heading theo quy tắc. Các con số dưới đây đo mức khôi phục trường Section trong dữ liệu chuẩn bị, chưa phải tỷ lệ đúng đã được người kiểm tra hoặc PDF xác nhận. Đợt train kiểm chứng context này đang được thực hiện trên train đã loại nguồn trùng; xem mục 11. Không dùng điểm lịch sử ở mục 4 để kết luận context mới tốt hơn.')
    body = body.replace('**56/56 kiểm tra classifier đạt**.', '**Hồ sơ bàn giao v1 ghi nhận 56/56 kiểm tra classifier đạt**. Đây là kết quả kiểm tra đã lưu ở thời điểm bàn giao v1, không phải toàn bộ test suite được chạy lại khi xuất bản text này.')
    body = body.replace('Bản 1.0 được chốt trên bộ nhãn tham chiếu gốc và context đã train, giữ nguyên khi tái lập validation.', 'Bản 1.0 đã được đóng băng trên bộ nhãn tham chiếu gốc và context đã train để lưu bằng chứng tái lập. Bản này có hạn chế trùng toàn nguồn nêu ở mục 1; ứng viên của đợt loại nguồn trùng chưa được chọn khi nghiên cứu mới còn chưa hoàn tất.')
    qstart = body.index('## 10.')
    qend = body.index('## Nguồn bằng chứng', qstart)
    oral = '''## 10. Nội dung có thể trình bày khi bảo vệ

"Em phụ trách nhánh classifier, phân loại câu thành năm vai trò bằng mDeBERTa và ngữ cảnh tài liệu. Em đã chuẩn bị dữ liệu, xây dựng hàm loss có trọng số lớp và label smoothing, thực hiện các đối chứng nhãn/context, đánh giá precision/recall/F1 từng lớp và ma trận nhầm lẫn. Với nhãn gốc/full context, ba seed đạt Macro-F1 trung bình 0,6340 và độ lệch chuẩn mẫu 0,0268 trên cùng validation. Em đã phân tích 165 trường hợp seed 2026 sai mà ít nhất một seed khác đúng. Qua rà soát em phát hiện ba ID train trùng toàn nguồn với một ID validation, nên đã loại 202 câu train và chạy lại đối chứng có kiểm soát. Em trình bày cả kết quả tốt và kết quả suy giảm, không coi đề xuất nhãn AI là ground truth. Bản bàn giao v1 và ensemble production được bảo toàn; chưa tuyên bố đã có final test độc lập hoặc model mới tốt hơn ensemble."

'''
    body = body[:qstart] + oral + body[qend:]

    new_rows, completed, matrices = [], {}, {}
    for name in queue['planned']:
        folder = STUDY / 'experiments' / name
        status = folder / 'run_status.json'
        if status.is_file() and read_json(status).get('state') == 'complete':
            metrics, matrix = check_saved_evaluation(folder)
            completed[name], matrices[name] = metrics, matrix
            new_rows.append([name, 'Hoàn tất', f"{metrics['macro_f1']:.6f}", f"{metrics['accuracy']:.6f}"])
            evidence.extend(folder / filename for filename in
                            ['run_status.json', 'validation_metrics.json', 'validation_predictions.csv', 'validation_confusion.csv', 'provenance.json'])
        else:
            state = 'Đang chạy' if name == queue.get('current') and queue['state'] == 'running' else 'Chưa có kết quả hoàn tất'
            new_rows.append([name, state, 'Chưa có', 'Chưa có'])
    data = read_json(STUDIES / 'classifier_source_clean_v1/preparation_manifest.json')
    followup = f'''## 11. Kiểm chứng sau khi loại nguồn trùng và khôi phục Section

11.1. Vì sao cần một đợt huấn luyện khác?

Đối chiếu SHA256 toàn văn Markdown phát hiện PSAI069, PSAI086 và PSAI100 trong train cùng nguồn với PSAI035 trong validation. Nhóm train chứa 202 câu; tài liệu validation liên quan có hai câu. Khác ID không bảo đảm khác nguồn. Phát hiện này xác định hạn chế của split ban đầu, chưa đo được mức độ làm tăng hoặc giảm F1.

Đã loại toàn bộ 202 câu của ba ID train, giữ nguyên nhãn và thứ tự 1.317 câu validation. Train còn {data['train_rows']:,} câu và {data['train_documents']} ID tài liệu; validation có {data['validation_documents']} ID. Kiểm tra hash toàn nguồn thô và sau chuẩn hóa không còn nhóm trùng xuyên train/validation theo hai phép kiểm tra này; chưa chứng nhận hết mọi near-duplicate hoặc ảnh hưởng từ lịch sử phát triển.

Không thể chỉ xóa các dòng CSV rồi dùng checkpoint cũ để xem như đã loại ảnh hưởng nguồn trùng: checkpoint đã học từ dữ liệu cũ. Vì vậy cả hai cấu hình đều được train lại trên cùng tập 3.942 câu. Hai nhánh giữ nguyên nhãn, text, hàng dữ liệu, vị trí và câu trước/sau; chỉ cách lấy Section khác nhau.

{table(['Nhãn', 'Train ban đầu', 'Train đã loại nguồn trùng', 'Validation giữ nguyên'], [[label, data['class_counts']['training_before'][label], data['class_counts']['training_after'][label], data['class_counts']['validation_unchanged'][label]] for label in LABELS])}
11.2. Giao thức của đợt kiểm chứng

Hai cấu hình: baseline dùng context cũ; context dùng Section v2.1 khôi phục từ Markdown. Mỗi cấu hình chạy seed 123, 42 và 2026. Giữ learning rate 3e-5, năm epoch, microbatch 1, accumulation 16, max length 256, BF16 và hàm loss như cũ. Với tập train mới, mỗi lượt có 1.235 optimizer updates và 124 warmup steps. Các con số 1.295 updates/130 warmup ở mục 3 thuộc tập train 4.144 câu trước đây.

Mỗi epoch chọn checkpoint bằng validation trong Trainer; sau export đánh giá lại bằng FP32 theo cùng giao thức. Bảng dưới dùng điểm FP32 cuối cùng đã lưu. Nghiên cứu dự kiến chọn cấu hình bằng Macro-F1 trung bình của đủ ba seed, sau đó xác định checkpoint ứng viên trong cấu hình đó. Không bỏ seed thấp và không dùng test/final test để chọn tham số.

11.3. Kết quả hoàn tất tại thời điểm xuất báo cáo

{table(['Lượt', 'Trạng thái', 'Validation Macro-F1', 'Accuracy'], new_rows)}
Đã có {len(completed)}/6 lượt hoàn tất. Những ô chưa có là chưa có kết quả cuối, không phải điểm bằng 0. Đây là bản chụp số liệu tại thời điểm ghi ở đầu báo cáo.

'''
    base, context = completed.get('baseline_seed_123'), completed.get('context_seed_123')
    if base and context:
        delta = context['macro_f1'] - base['macro_f1']
        followup += f'''Riêng cặp seed 123: context v2.1 đạt {context['macro_f1']:.6f}, baseline đạt {base['macro_f1']:.6f}; chênh lệch Macro-F1 là {delta:+.6f}, tương đương {delta * 100:+.2f} điểm phần trăm. Đây là kết quả giảm ở seed này, cần được báo cáo đầy đủ. Không dùng kết quả đó để tuyên bố sửa Section đã cải thiện classifier.

{table(['Cấu hình seed 123', 'Nhãn', 'Precision', 'Recall', 'F1', 'Support'], [[variant, label, f"{metrics['per_class'][label]['precision']:.6f}", f"{metrics['per_class'][label]['recall']:.6f}", f"{metrics['per_class'][label]['f1-score']:.6f}", int(metrics['per_class'][label]['support'])] for variant, metrics in [('baseline', base), ('context v2.1', context)] for label in LABELS])}
'''
        for name in ['baseline_seed_123', 'context_seed_123']:
            followup += f"Ma trận nhầm lẫn {name} (hàng tham chiếu, cột dự đoán):\n\n"
            followup += table(['Gold / Pred'] + LABELS, [[label] + matrices[name][index] for index, label in enumerate(LABELS)]) + '\n'
        if sum(row[1] for row in matrices['context_seed_123']) == 0:
            followup += 'Context v2.1 ở seed 123 không dự đoán câu nào là REQUIREMENT: 89 câu tham chiếu của lớp này bị chuyển sang các lớp khác. Điều này là lỗi quan sát được theo nhãn tham chiếu hiện có; nguyên nhân chưa được xác định. Không quy ngay cho mất Section, PDF extraction, lỗi code hoặc nhãn sai khi chưa có kiểm tra riêng.\n\n'
    if len(completed) < 6:
        followup += 'Nghiên cứu mới chưa có đủ ba seed mỗi cấu hình, nên chưa công bố mean/SD hoặc chốt ứng viên mới. Không so điểm baseline mới với model cũ rồi quy toàn bộ chênh lệch cho việc loại nguồn trùng vì điều kiện train đã thay đổi. Bản v1 tiếp tục được lưu để tái lập; kết quả đợt mới chưa thay production.\n\n'
    else:
        summary = read_json(STUDY / 'summary.json')
        followup += table(['Cấu hình', 'Số seed', 'Mean Macro-F1', 'SD mẫu'], [[g['variant'], g['n'], f"{g['mean']:.6f}", f"{g['sample_std']:.6f}"] for g in summary['groups']]) + '\n'
        followup += 'Khuyến nghị lựa chọn từ nghiên cứu này cần đọc cùng giới hạn nhãn, nguồn và validation đã được sử dụng trong phát triển; không tự thay production.\n\n'
    followup += '''11.4. Những phần cần hoàn tất để có đánh giá cuối độc lập

- Hoàn tất nghiên cứu đã định trước và báo cáo toàn bộ seed, kể cả kết quả thấp.
- Người kiểm định phân xử nhãn còn tranh cãi, nhất là REQUIREMENT/ACTIVITY và EXPECTED_RESULT. 57 thay đổi AI đề xuất vẫn là giả thuyết cần kiểm tra. Chưa có bằng chứng hai người gán nhãn độc lập và Cohen's Kappa đủ điều kiện.
- Xác minh nguồn chưa dùng trong phát triển, hoàn thiện nhãn tham chiếu của final test và đóng băng ứng viên trước khi đánh giá. Tám tài liệu ứng viên cũ đã có dấu vết sử dụng trong phát triển nên chưa đủ điều kiện gọi là final test độc lập.
- Báo cáo cuối phải phân biệt lỗi quan sát được theo nhãn tham chiếu với nguyên nhân đã kiểm chứng. Số câu thiếu Section đo độ đầy đủ metadata, không tự chứng minh đúng/sai của model.

Các công cụ kiểm tra đã có giúp thực hiện các bước này, nhưng không thay thế quyết định của người gán nhãn hay biến tập phát triển thành tập kiểm tra độc lập.

## 12. Các đoạn code và demo phục vụ bảo vệ

Các tệp sau có trong mã nguồn dự án; số liệu và ma trận cần đọc đã được đưa trực tiếp vào báo cáo này:

- classifier_source_split.py: tìm nhóm cùng toàn nguồn, ưu tiên validation và loại cả tài liệu train có nguồn trùng; xuất dữ liệu và manifest.
- study_inputs.py: context_for tạo Section/Pos/Bullet/Cue/Previous/Next; encode_items ghép câu/context và tokenize. Giới hạn 600 ký tự áp dụng cho câu trước tokenize; 256 tokens áp dụng cho cả cặp đầu vào.
- classifier_training.py: WeightedClassificationTrainer.compute_loss tính cross-entropy theo mẫu, label smoothing 0,1, nhân trọng số của lớp tham chiếu rồi lấy trung bình microbatch. model_accepts_loss_kwargs=False giữ cơ chế chia loss theo số bước accumulation của Trainer.
- classifier_context_experiment.py: set_seed, nạp model/tokenizer, tạo dataset, cấu hình train, chọn checkpoint, export và đánh giá validation. Đây là driver của đợt kiểm chứng mới đã lưu hash mã nguồn.
- study_evaluation.py: tính Macro-F1, precision/recall/F1 từng lớp, lưu predictions/errors/confusion và giao thức inference.
- final_test_integrity.py: ghi/bảo vệ thông tin model, tokenizer, mã nguồn và cấu hình inference để đối chiếu khi đánh giá cuối.
- classifier_predict.py: API và lệnh suy luận checkpoint chỉ định, nhận JSON/JSONL và trả nhãn/xác suất.
- classifier_demo.py: demo kết quả đã lưu hoặc suy luận thật; chế độ được in rõ khi chạy.

Công thức loss theo microbatch B:
L = (1 / |B|) * sum_i [ w[y_i] * CrossEntropy(logits_i, y_i, smoothing=0.1) ].
Trọng số lớp w[c] = N_train / (5 * số mẫu train của lớp c).
Macro-F1 = trung bình F1 của năm lớp, để mỗi lớp có trọng số ngang nhau khi báo cáo. Accuracy là tỷ lệ dự đoán đúng trên toàn bộ câu; không thay Macro-F1 khi dữ liệu lệch lớp.

Lệnh minh họa kết quả đã lưu, không tải model và không dùng GPU:
python classifier_demo.py

Lệnh suy luận thật bằng CPU cho ví dụ mới:
python classifier_demo.py --live --cpu --input examples/classifier_teacher_live.json

Lệnh suy luận CPU vẫn dùng RAM và thời gian tính toán. Trong khi train đang chạy, ưu tiên demo kết quả đã lưu. File demo có in rõ SAVED RESULTS hoặc LIVE INFERENCE; phát lại kết quả đã lưu không được trình bày như vừa chạy model trực tiếp.

'''
    inputs = read_json(ROOT / 'examples/classifier_input.json')
    predictions = read_json(REPORTS / 'classifier_demo_predictions.json')
    assert len(inputs) == len(predictions) == 4
    followup += 'Kết quả demo đã lưu của checkpoint v1 seed 123 trên bốn câu tự biên soạn:\n\n'
    for item, prediction in zip(inputs, predictions):
        assert item['id'] == prediction['id']
        followup += f"Câu: {item['text']}\nDự đoán: {prediction['label']}; confidence: {prediction['confidence']:.6f}.\n\n"
    followup += 'Các câu demo không phải tập kiểm định. Câu nói về final deliverable hiện được dự đoán ACTIVITY; báo cáo giữ nguyên kết quả thật để minh họa giới hạn, không sửa đầu ra thành nhãn kỳ vọng.\n\n'
    followup += 'Bản text này chỉ đọc số liệu đã lưu và đếm lại validation predictions để kiểm tra các bảng. Không train, suy luận mới, tính điểm test/final test hoặc thay model trong lúc xuất báo cáo.\n\n'

    marker = body.index('## Nguồn bằng chứng')
    markdown = intro + body[:marker] + followup + body[marker:]
    markdown += '\nBằng chứng của đợt kiểm chứng mới:\n\n'
    for filename in ['experiment_plan.json', 'queue_status.json', 'completed_experiments.csv', 'summary.json']:
        markdown += f'- {STUDY.relative_to(ROOT).as_posix()}/{filename}\n'
    markdown += '\nCác tệp được ghi tên để đối chiếu trong bộ mã nguồn; giảng viên có thể đọc toàn bộ bảng kết quả và nhận xét ngay trong bản text, không cần mở đường dẫn trên máy của sinh viên.\n'
    text = re.sub(r'\[([^\]]+)\]\(([^\n)]+)\)', r'\1', markdown)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```[^\n]*\n?', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('`', '')
    assert '\ufffd' not in text
    assert 'chưa train/evaluate' not in text
    assert 'Không train thêm và không thay ensemble production' not in text
    output = REPORTS / 'BAO_CAO_CLASSIFIER_GUI_THAY.txt'
    output.write_text(text, encoding='utf-8-sig')
    (REPORTS / 'BAO_CAO_CLASSIFIER_GUI_THAY.md').write_text(markdown, encoding='utf-8')
    manifest = dict(created_local=now.isoformat(), report=output.name,
                    report_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    checked_validation_prediction_sets=5 + len(completed),
                    new_completed_runs=len(completed), expected_new_runs=6,
                    old_three_seed_mean=mean, old_three_seed_sample_sd=sd,
                    training_started=False, model_inference_performed=False,
                    test_or_final_used=False, production_modified=False,
                    inputs=[dict(path=p.relative_to(ROOT).as_posix(),
                                 sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in evidence])
    (REPORTS / 'teacher_text_export_verification.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(report=output.as_posix(), bytes=output.stat().st_size,
                         checked_validation_sets=manifest['checked_validation_prediction_sets'],
                         new_completed_runs=len(completed), created_local=now.isoformat())))


if __name__ == '__main__':
    main()
