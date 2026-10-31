"""Assemble the classifier deadline handoff from existing, traceable artifacts."""
from pathlib import Path
import hashlib
import json
import os
import zipfile

import pandas as pd

from build_classifier_report import export_docx, table
from study_data import ROOT, OUT, read_json, write_json

DEST = ROOT / 'reports/classifier'
REPLAY = OUT / 'studies/classifier_validation_replay_v1'
CONTROL = OUT / 'studies/user_labels_v1'
STABILITY = OUT / 'studies/classifier_stability_v1'
CONTEXT = OUT / 'studies/classifier_context_v2_1'
MODEL = ROOT / 'models/controlled_study/classifier_stability_v1/seed_123'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def link(path):
    return f'[{path.name}]({Path(os.path.relpath(path, DEST)).as_posix()})'


def number(value):
    return f'{value:.6f}' if isinstance(value, (int, float)) else 'chưa xác định'


def main():
    history = read_json(DEST / 'handoff_source/history_inventory.json')
    checked = read_json(DEST / 'release_verification.json')
    stats = read_json(REPLAY / 'statistics.json')
    prep = read_json(CONTEXT / 'preparation_manifest.json')
    metrics = read_json(REPLAY / 'seed_123/validation_metrics.json')
    provenance = read_json(MODEL / 'training_provenance.json')
    training = pd.read_csv(CONTROL / 'experiments/baseline_original_labels/training_rows.csv', keep_default_na=False)
    validation = pd.read_csv(CONTROL / 'experiments/baseline_original_labels/validation_rows.csv', keep_default_na=False)
    original = pd.read_csv(ROOT / 'data_segment_cls/segment_classification.csv', keep_default_na=False)
    labels = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']
    counts = history['counts']
    selection_path = OUT / 'final_selection.json'
    selected = selection_path.is_file()
    selection = read_json(selection_path) if selected else None
    if selected:
        assert Path(selection['model_dir']).resolve() == MODEL.resolve()
        assert selection['model_sha256'] == metrics['model_weight_sha256']
    candidate = dict(status='VALIDATION_CANDIDATE_FOR_DEMO_AND_HANDOFF; not research-final selection',
        candidate_model=str(MODEL.relative_to(ROOT)).replace('\\', '/'), selection_basis='highest Macro-F1 among the five completed controlled validation exports',
        validation_macro_f1=metrics['macro_f1'], seed=123, labels=labels,
        model_files={name: sha(MODEL / name) for name in metrics['validation_inference_manifest']['model_files']},
        input_protocol=metrics['config'], context_dataset='fixed_dataset_v3; context v2.1 NOT used by these weights',
        code_sha256={name: sha(ROOT / name) for name in ['classifier_predict.py', 'study_inputs.py']},
        validation_evidence_sha256=sha(REPLAY / 'seed_123/validation_metrics.json'),
        train_sha256=metrics['train_sha256'], validation_sha256=metrics['validation_sha256'],
        production_ensemble_replaced=False, human_label_review_complete=False, independent_final_evaluation_complete=False,
        final_selection_file_created=False, limitations=['Validation was used in development.', 'No superiority claim over the old production ensemble.', 'Original-label/reference ambiguities remain.'])
    assert candidate['model_files'] == metrics['validation_inference_manifest']['model_files']
    if selected:
        candidate.update(status='FINAL_CLASSIFIER_RELEASE_V1_VALIDATION_SELECTED',
                         release_version='1.0', final_selection_file_created=True,
                         model_path=candidate['candidate_model'],
                         frozen_selection_path=str(selection_path.relative_to(ROOT)).replace('\\', '/'),
                         frozen_selection_sha256=sha(selection_path),
                         label_policy='original reference labels, unchanged; not certified expert adjudicated',
                         excluded_from_release=['AI label proposals not approved', 'context v2.1 not train/evaluated'],
                         release_scope='classifier model/software submission; independent test remains unmeasured')
        manifest_path = DEST / 'FINAL_MODEL_MANIFEST.json'
    else:
        manifest_path = DEST / 'CANDIDATE_MODEL_MANIFEST.json'
    source_followup = ROOT / 'reports/research_review/studies/classifier_source_clean_v1/preparation_manifest.json'
    if source_followup.exists():
        followup = read_json(source_followup)
        candidate['additional_evaluation_limit'] = dict(
            issue='Original split had a full-source duplicate group crossing train/validation despite no exact target/context sentence overlap.',
            affected_training_documents=followup['removed_train_documents'],
            removed_rows_for_new_study=followup['removed_train_rows'],
            original_release_kept_for_traceability=True,
            new_source_disjoint_study='reports/research_review/studies/classifier_source_clean_context_study_v1',
            source_audit_manifest_sha256=sha(source_followup))
    write_json(manifest_path, candidate)
    controlled = [r for r in history['score_records'] if r['classification'] == 'validation_export_replay_for_selection']
    text = '# BÁO CÁO TỔNG KẾT NHÁNH CLASSIFIER PSAI\n\n'
    text += '## 1. Mục tiêu và phạm vi đã hoàn thành\n\n'
    text += 'Phần được thực hiện là phân loại câu trong tài liệu đề xuất dự án thành năm vai trò: OBJECTIVE, REQUIREMENT, ACTIVITY, EXPECTED_RESULT và OTHER. Đầu vào gồm câu và context; đầu ra gồm nhãn, độ tin cậy và xác suất năm lớp. Phần mềm bàn giao có lệnh suy luận độc lập, checkpoint chỉ định rõ, công cụ train/evaluation đã lưu, dữ liệu/split/hash, kiểm tra hồi quy và báo cáo số liệu. RAG và đánh giá toàn hệ thống không nằm trong phạm vi báo cáo này.\n\n'
    if selected:
        text += '**Kết quả chốt phiên bản 1.0:** model cuối của nhánh classifier là **mDeBERTa seed 123, nhãn gốc + context study-v1/full**, chọn bằng validation Macro-F1 cao nhất trong năm lượt controlled: **0,654426**. Checkpoint, tokenizer, cấu hình, dataset/split hash và giao thức inference đã được đóng băng trước đánh giá cuối. Không train thêm và không thay ensemble production. Đây là báo cáo tổng kết kết quả bản bàn giao; giới hạn nhãn tham chiếu và chưa có test độc lập được trình bày ở mục 9, không được gọi là đã kiểm định xong.\n\n'
    else:
        text += '**Trạng thái bàn giao:** bản phần mềm chạy được; seed 123 là ứng viên validation, chưa đóng băng model.\n\n'
    if source_followup.exists():
        text += '**Bổ sung kiểm tra nguồn:** phát hiện ba ID train PSAI069/PSAI086/PSAI100 trỏ tới cùng toàn bộ Markdown với validation PSAI035 (hai câu validation). Dù không còn trùng câu/neighbor chính xác sau purge cũ, tách theo ID chưa đủ bảo đảm tách theo nguồn. Bản 1.0 và điểm đã ghi vẫn được giữ làm bằng chứng theo điều kiện ban đầu, không xem là đánh giá độc lập. Dataset mới loại202 câu train của nhóm này, giữ nguyên1.317 câu validation; cả baseline và context mới phải train lại trên cùng3.942 câu để so sánh công bằng. Xem ' + link(source_followup) + '.\n\n'
    text += '## 2. Dữ liệu, nhãn và split\n\n'
    text += table(['Nguồn/giai đoạn', 'Câu/dòng', 'Tài liệu', 'Diễn giải'], [
        ['CSV gốc hiện có', len(original), original.document_id.nunique(), 'Có cả dòng heading/không dùng làm câu; không đồng nhất với số mẫu train'],
        ['Corpus gốc sau chuẩn bị theo manifest', 7169, 130, 'Split theo nhóm tài liệu; heading/empty được xử lý'],
        ['Train controlled nhãn gốc', len(training), training.document_id.nunique(), 'Sau purge và bổ sung extra_human; doc count gồm nguồn bổ sung'],
        ['Validation controlled', len(validation), validation.document_id.nunique(), 'Cố định cho cả 5 lượt controlled'],
        ['Đề xuất AI sửa nhãn', 57, 'thuộc train', '57 thay đổi có hiệu lực từ 68 đề xuất; chưa phải ground truth']])
    text += '\n\n' + table(['Nhãn', 'Train', 'Validation'], [[label, int(training.label.eq(label).sum()), int(validation.label.eq(label).sum())] for label in labels])
    text += '\n\nDùng StratifiedGroupKFold seed42 và nhóm các tài liệu trùng nội dung; các lượt controlled giữ fold và thứ tự dữ liệu cố định. Loại train trùng text với các phần held-out và kiểm tra câu lân cận. Chỉ thay nhãn của fold train trong biến thể AI overlay; không sửa validation. Các biện pháp này kiểm soát trùng khớp đã biết, chưa chứng minh mọi near-duplicate hoặc nguồn lịch sử đều độc lập.\n\n'
    text += '## 3. Mô hình và cấu hình huấn luyện\n\n'
    text += 'Mô hình nền là microsoft/mdeberta-v3-base, đầu phân loại năm nhãn, fine-tune bằng cross-entropy có class weights và label smoothing. Context study-v1/full gồm Section, vị trí, bullet/cue và phần câu trước/sau. Đây là phân loại câu/đoạn, không phải BIO token tagging.\n\n'
    args = provenance['effective_training_arguments']
    text += table(['Tham số', 'Giá trị'], [
        ['Epochs / updates', '5 / 1295'], ['Learning rate', '3e-5'], ['Microbatch / accumulation / batch hiệu dụng', '1 / 16 / 16'],
        ['Optimizer / scheduler', 'AdamW torch / linear'], ['Warmup / weight decay / max grad norm', '130 steps / 0,01 / 1,0'],
        ['Text / input max length', '600 ký tự / 256 tokens'], ['Previous / next / Section', '150 / 150 / 120 ký tự'],
        ['Label smoothing', '0,1'], ['Class weights', ', '.join(f'{v:.6f}' for v in args['class_weights'])],
        ['Parameter dtype / train autocast', 'FP32 / BF16'], ['Checkpoint selection trong Trainer', 'Validation macro-F1 mỗi epoch; lưu checkpoint tốt nhất'],
        ['Seeds kiểm tra ổn định', '42, 123, 2026']])
    text += '\n\nĐã sửa normalization gradient trong custom Trainer để accumulation tương đương microbatch loss trung bình; có test dữ liệu tổng hợp. Lỗi cũ không được xem là nguyên nhân duy nhất của mọi khác biệt F1 lịch sử. Điểm khi Trainer dùng BF16 và điểm export FP32 được giữ riêng; không dùng chúng thay thế lẫn nhau.\n\n'
    text += '## 4. Năm lượt controlled đã chạy hoàn chỉnh\n\n'
    rows = []
    for record in controlled:
        result = record['data']
        cfg = result['config']
        rows.append([record['model'], cfg['seed'], 'gốc' if not cfg.get('use_label_overlay', True) else 'AI đề xuất', cfg['context_mode'], number(result['macro_f1']), number(result['accuracy'])])
    text += table(['Lượt', 'Seed', 'Nhãn train', 'Context', 'Validation Macro-F1', 'Accuracy'], rows)
    text += '\n\nỞ seed42, sửa nhãn AI chưa tăng macro-F1: 0,642605 so với 0,643892 của nhãn gốc. Với nhãn AI, full context đạt 0,642605 so với text-only 0,629482. Bootstrap thăm dò theo tài liệu của cả hai phép so sánh đều chứa 0; chưa đủ kết luận chắc chắn về hướng cải thiện. Chỉ cấu hình nhãn gốc + full context được chạy thêm seed123 và2026, không suy ra mọi cấu hình đều ổn định.\n\n'
    text += table(['Seed nhãn gốc/full', 'Validation Macro-F1'], [[r['seed'], number(r['macro_f1'])] for r in read_json(STABILITY / 'stability_with_reference.json')['runs']])
    text += f'\n\nTrung bình **{stats["mean"]:.6f} ± {stats["sample_std"]:.6f}** (SD mẫu, n=3 trên một split). Không bỏ seed2026 thấp khỏi thống kê. Ba checkpoint đã được replay theo cùng FP32/encoder/batch/runtime: dự đoán và xác suất trùng toàn bộ số liệu đã lưu. Replay là inference, không phải lượt train mới.\n\n'
    text += '## 5. Precision/recall/F1 và confusion matrix\n\n'
    per_class = []
    for record in controlled:
        for label, value in record['data']['per_class'].items():
            per_class.append([record['model'], label, number(value['precision']), number(value['recall']), number(value['f1-score']), int(value['support'])])
    text += table(['Lượt', 'Lớp', 'Precision', 'Recall', 'F1', 'Support'], per_class)
    text += '\n\nMa trận theo ba seed nhãn gốc/full, hàng là tham chiếu và cột là dự đoán:\n\n'
    for seed in [42, 123, 2026]:
        matrix = pd.read_csv(REPLAY / f'seed_{seed}/validation_confusion.csv', index_col=0).loc[labels, labels]
        text += f'### Seed {seed}\n\n' + table(['Gold / Pred', *labels], [[label, *matrix.loc[label].tolist()] for label in labels]) + '\n\n'
    text += '## 6. Phân tích lỗi và các sửa đổi đã làm\n\n'
    text += 'Seed2026 có 165 câu sai trong khi seed42 hoặc123 đúng; 73 câu cả hai seed kia đúng. REQUIREMENT giảm mạnh nhất, từ F1 0,594286 ở seed42 xuống 0,450867. EXPECTED_RESULT thấp ở cả ba seed, khoảng 0,46–0,52. OTHER bị gán sang nhãn chức năng 200 câu ở seed2026, so với142 ở seed42. Không quy mọi sai lệch thành lỗi model vì nhãn tham chiếu còn mơ hồ.\n\n'
    regressions = pd.read_csv(STABILITY / 'analysis/seed_2026_regressions.csv', keep_default_na=False)
    for index in [217, 1211, 805]:
        row = regressions[regressions.validation_row.eq(index)].iloc[0]
        text += f'**{row.document_id}, validation row {index}:** {row.text}\n\n'
        text += table(['Tham chiếu', '42', '123', '2026'], [[row.true_label, row.prediction_42, row.prediction_123, row.prediction_2026]]) + '\n\n'
    text += 'Câu nêu thuộc tính có thể lẫn với hoạt động; cụm đầu ra dưới Deliverables có thể bị gán ACTIVITY; câu “should report” có gold EXPECTED_RESULT cần người phân xử theo guideline. Các ví dụ trên là bất đồng cần kiểm tra, không phải nhãn chuẩn mới do trợ lý quyết định.\n\n'
    text += 'Đã chuẩn bị context v2.1 từ Markdown nguồn, chỉ sửa Section khi vị trí/heading đủ rõ. Đây là dữ liệu chuẩn bị riêng, chưa train/evaluate; không dùng F1 ở trên để tuyên bố context mới tốt hơn.\n\n'
    text += table(['Split', 'Section trống trước', 'Trống sau', 'Chưa xác định'], [[split, value['section_blank_before'], value['section_blank_after'], value['unresolved_kept_old_section']] for split, value in prep['splits'].items()])
    text += '\n\nCó phiếu 91 câu và importer kiểm tra ID/context/hash/reviewer/lý do; kết quả chỉ là đề xuất overlay, không tự áp dụng. Có81 nhãn annotation hiện tại khác CSV frozen, và110 context cần người xác nhận. Các con số này là danh sách cần xử lý, không tự đổi thành “đã sửa đúng”.\n\n'
    text += '## 7. Kiểm kê model và kết quả lịch sử\n\n'
    text += f'Tại thời điểm kiểm kê lưu trong hồ sơ có **{counts["exported_weight_bearing_directories"]} thư mục export chứa weights**: {counts["historical_root_exports"]} lịch sử, {counts["completed_controlled_full_trials"]} lượt controlled đầy đủ, {counts["smoke_export_directories"]} smoke. Không đồng nhất số thư mục với số phương pháp hoặc số lượt train độc lập; lịch sử độc lập thực sự chưa xác định khi thiếu provenance. Smoke/invalid không tham gia xếp hạng.\n\n'
    text += table(['Tên export', 'Đường dẫn', 'Phân loại', 'Context / maxlen'], [[r['name'], r['path'], r['category'], str(r['inference_config'].get('ctx_version', '?')) + ' / ' + str(r['inference_config'].get('max_len', '?'))] for r in history['inventory']])
    text += '\n\n### Các điểm lưu từ lịch sử — không dùng chọn model hiện tại\n\n'
    historical_rows = []
    for record in history['score_records']:
        if record['classification'] == 'validation_export_replay_for_selection':
            continue
        result = record['data']
        historical_rows.append([record['model'], result.get('dataset', result.get('evaluation_kind', 'diagnostic')), result.get('n', '?'), number(result.get('macro_f1')), record['classification']])
    text += table(['Model', 'Dataset/loại điểm', 'N', 'Macro-F1 đã lưu', 'Giới hạn'], historical_rows)
    text += '\n\nCác điểm lịch sử trên được chép từ summary/diagnostic có sẵn, không tính lại test trong đợt bàn giao. Điểm0,852137 của kfold_f0 chưa có provenance đủ để chứng nhận test độc lập; không dùng nó làm độ chính xác cuối cùng. Điểm Fold0/human holdout đã từng được xem nhiều lần, dữ liệu/cách purge có thể khác các lượt controlled. Điểm thiếu là chưa biết, không phải0.\n\n'
    text += '## 8. Bản phần mềm bàn giao và cách chạy\n\n'
    text += 'Đã thêm classifier_predict.py cho checkpoint study-v1 được chỉ định tường minh. Đầu vào JSON array hoặc JSONL gồm text và context; đầu ra có ID, nhãn, confidence và xác suất năm lớp. Không tự lựa chọn model, không đổi production.\n\n'
    text += '```powershell\npython classifier_predict.py --model-dir models/controlled_study/classifier_stability_v1/seed_123 --input examples/classifier_input.json --output outputs/classifier_predictions_new.json\n```\n\n'
    text += f'**{checked["tests_run"]}/{checked["tests_run"]} kiểm tra classifier đạt**. Đã chạy thực tế seed123 trên bốn câu ví dụ tự biên soạn, xuất xác suất hợp lệ; đây là smoke chức năng, không phải tập đo độ chính xác. Checkpoint/tokenizer/config/hash và giao thức inference được ghi trong manifest ứng viên.\n\n'
    text += 'Gói ZIP kèm mã, dữ liệu train/validation và bằng chứng báo cáo; weights dung lượng lớn được giữ riêng ở thư mục checkpoint đã chỉ định. Người nhận cần chép checkpoint hoặc dùng đường dẫn có sẵn; ZIP nhỏ không tự chứa weights. Lệnh train không tự chạy khi nhận gói.\n\n'
    text += '## 9. Giới hạn của kết quả và hướng phát triển\n\n'
    text += 'Bản 1.0 được chốt trên bộ nhãn tham chiếu gốc và context đã train, giữ nguyên khi tái lập validation. Các đề xuất nhãn AI và context mới không được tự nhập vào model cuối. Kiểm định nhãn người và final test độc lập chưa hoàn thành, nên số0,654426 là validation Macro-F1, không phải độ chính xác trên một test mới. Tám ứng viên test có dấu vết phát triển trước nên không được dùng để tuyên bố test độc lập. Ensemble cũ có đường train trên toàn corpus gốc trong source; chưa có phép so sánh độc lập công bằng với model mới.\n\n'
    text += 'Hướng phát triển sau bản bàn giao: phân xử nhãn/context theo nguồn, phát hành version dữ liệu mới nếu được chấp nhận và nghiên cứu tác động trong thí nghiệm mới; chuẩn bị tập test thực sự độc lập rồi đánh giá model đã đóng băng. Không dùng kết quả final test để chọn tiếp tham số cho bản1.0. Báo cáo bao quát artifact tìm thấy, không khẳng định khôi phục những lượt train không còn file/log.\n\n'
    text += '## 10. Cách trả lời khi thầy hỏi đã làm gì\n\n'
    text += '“Em xây dựng bộ phân loại năm vai trò câu bằng mDeBERTa, chuẩn bị dữ liệu theo tài liệu, kiểm soát trùng lặp và lưu provenance. Em có năm lượt controlled hoàn chỉnh: đối chứng nhãn/context và ba seed cho cấu hình nhãn gốc/full context. Em báo cáo đủ từng lớp, confusion và mean/SD, phân tích165 bất đồng khi seed2026 giảm. Em sửa cơ chế loss/accumulation, thêm kiểm tra toàn vẹn và inference API, replay validation để tái lập. Model bản bàn giao được chọn là seed123 theo validation Macro-F1 0,6544; em đã đóng băng model, tokenizer và giao thức. Em không khẳng định tốt hơn ensemble hoặc đã qua test độc lập; các hạn chế nhãn và dữ liệu cuối được ghi rõ.”\n\n'
    text += '## Nguồn bằng chứng\n\n'
    for path in [DEST / 'handoff_source/history_inventory.json', manifest_path, DEST / 'release_verification.json', REPLAY / 'KET_QUA_VALIDATION_THONG_NHAT.md', CONTEXT / 'README.md', STABILITY / 'human_review/PHAN_XU_91_CAU.html']:
        text += '- ' + link(path) + '\n'
    path = DEST / 'BAO_CAO_TONG_KET_CLASSIFIER.md'
    path.write_text(text, encoding='utf-8')
    export_docx(text, path.with_suffix('.docx'))
    package(history)
    print(path.with_suffix('.docx'))


def package(history):
    names = ['README_CLASSIFIER.md', 'classifier_predict.py', 'classifier_demo.py', 'study_inputs.py', 'classifier_training.py',
             'study_evaluation.py', 'study_experiments.py', 'study_data.py', 'train_human_cv.py', 'fair_fold0.py',
             'reviewed_study.py', 'classifier_followup.py', 'replay_validation.py', 'classifier_validation_replay.py',
             'classifier_adjudication.py', 'classifier_context_v2.py', 'study_final_test.py', 'final_test_integrity.py',
             'classifier_source_split.py', 'classifier_context_experiment.py', 'classifier_context_review.py',
             'classifier_external_packet.py',
             'extraction.py', 'ensemble_classifier.py', 'planning.py', 'study_pipeline.py', 'extraction_rules_backup.py',
             'sentence_classifier.py', 'psai.py', 'pdf_preprocessing.py', 'ocr.py', 'run_extraction.py', 'analyze_classifier_seeds.py',
             'classifier_analysis.py', 'classifier_review_workbench.py', 'build_classifier_report.py',
             'build_classifier_handoff.py', 'requirements.txt', 'requirements-research.txt', 'examples/classifier_input.json',
             'examples/classifier_teacher_live.json',
             'extra_human.csv', 'data_segment_cls/segment_classification.csv', 'runs/fair_fold0/frozen_split.json']
    paths = {ROOT / name for name in names}
    if (OUT / 'final_selection.json').is_file():
        paths.add(OUT / 'final_selection.json')
    paths.update(ROOT.glob('test_classifier*.py'))
    paths.update([ROOT / 'test_research_review.py', ROOT / 'test_project_runtime.py', ROOT / 'test_pipeline_quality.py'])
    for folder in [DEST, REPLAY, CONTEXT, STABILITY / 'analysis', STABILITY / 'human_review',
                   CONTROL / 'analysis', CONTROL / 'source_snapshot',
                   OUT / 'studies/classifier_source_clean_v1',
                   OUT / 'studies/classifier_context_review_v1']:
        paths.update(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.py', '.json', '.csv', '.md', '.docx', '.html'})
    for study in [CONTROL, STABILITY]:
        paths.update(p for p in study.glob('*') if p.is_file() and p.suffix in {'.json', '.csv'})
        for name in ['baseline', 'baseline_original_labels', 'A0_text', 'seed_123', 'seed_2026']:
            folder = study / 'experiments' / name
            if folder.is_dir():
                paths.update(p for p in folder.glob('*') if p.is_file() and p.suffix in {'.json', '.csv'})
                paths.update((folder / 'source_snapshot').glob('*.py'))
    for name in history['source_registry']:
        path = ROOT / name
        if path.is_file() and path.suffix in {'.json', '.csv', '.md', '.py'}:
            paths.add(path)
    historical_only = {'package_verification.json', 'CANDIDATE_MODEL_MANIFEST.json',
                       'HOAN_THIEN_CLASSIFIER.md', 'HOAN_THIEN_CLASSIFIER.docx'}
    paths = sorted(p for p in paths if p.is_file() and 'checkpoints' not in p.parts and p.name not in historical_only)
    archive = DEST / 'PSAI_CLASSIFIER_CODE_VA_BANG_CHUNG.zip'
    manifest = {}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            stream.writestr(name, data)
        stream.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
        stream.writestr('START_HERE.txt', 'Final classifier release v1.0: validation-selected seed123. Read README_CLASSIFIER.md and reports/classifier/FINAL_MODEL_MANIFEST.json. Model weights are NOT included; copy the recorded checkpoint separately or pass its existing local path to classifier_predict.py. No automatic training or production replacement. Results are validation-based; expert label adjudication and independent final testing are limitations, not completed evaluations.\n')
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        assert not any(name.endswith(('.safetensors', '.bin', '.pt', '.pth')) for name in stream.namelist())
        assert all(hashlib.sha256(stream.read(name)).hexdigest() == value for name, value in manifest.items())
    write_json(DEST / 'package_verification.json', dict(files=len(manifest), bytes=archive.stat().st_size,
                sha256=sha(archive), zip_integrity_passed=True, weights_included=False))


if __name__ == '__main__':
    main()
