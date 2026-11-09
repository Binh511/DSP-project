"""Summarize actual pilot artifacts without promoting AI labels to human gold."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psai_project.study_metrics import classification_metrics


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def percent(value):
    return 'không xác định' if value is None else f'{value * 100:.2f}%'


def summarize(study):
    study = Path(study)
    protocol, attempts = read(study / 'protocol_v1.json'), read(study / 'runs_v1/attempts.json')
    primary, agreement = read(study / 'references/rater_b.json'), read(study / 'annotation_agreement.json')
    classifier = read(study / 'production_classifier_diagnostic.json')
    scores = {mode: read(study / ('scored_' + mode) / 'score.json') for mode in ['fast', 'full']}
    if attempts['status'] != 'completed' or any(value['status'] != 'complete' for value in scores.values()):
        raise ValueError('Do not summarize incomplete attempts/reviews as complete evaluation')
    truth = [item['label'] for doc in primary['documents'] for item in doc['sentence_references']]
    baseline = classification_metrics(truth, ['OTHER'] * len(truth))
    ablation = read(study / 'document_retrieval_ablation.json')
    ocr = read(study / 'ocr/en_us_vi_diagnostic_v1/diagnostic.json')
    retrieval = read(study / 'retrieval_probe_v1/diagnostic.json')
    retrieval_summary = read(study / 'retrieval_probe_v1/summary.json')
    metrics = {mode: {'task_matching': value['metrics']['task_matching'],
                     'accepted_target_task_support': value['metrics']['evidence_correctness'],
                     'literal_source_matches': value['literal_source_matches'],
                     'dependency_correctness': value['metrics']['dependency_correctness'],
                     'returned_hit_relevance': value['retrieval_relevance']}
               for mode, value in scores.items()}
    loaded = {}
    for attempt in attempts['attempts']:
        if attempt['mode'] == 'full':
            analysis = read(study / 'runs_v1/full' / attempt['document_id'] / 'analysis.json')
            loaded[attempt['document_id']] = analysis['plan']['provenance']['model_loaded']
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Three technical-design proposals; distribution-shift pilot with AI references, not representative human final testing',
        'human_ground_truth': False, 'independent_final_test_status': 'NOT_SCORED',
        'sources': read(study / 'sources/source_manifest.json')['sources'],
        'fixed_primary_reference': {'reviewer': 'AI_B', 'sentence_count': len(truth),
            'task_count': sum(len(doc['gold_tasks']) for doc in primary['documents']),
            'file_sha256': hashlib.sha256((study / 'references/rater_b.json').read_bytes()).hexdigest()},
        'AI_rater_agreement': agreement['aggregate'],
        'pipeline': {'attempts': len(attempts['attempts']), 'successful': sum(item['returncode'] == 0 for item in attempts['attempts']),
                     'wall_seconds': [{'document': item['document_id'], 'mode': item['mode'], 'seconds': item['elapsed_wall_seconds']}
                                     for item in attempts['attempts']], 'llm_loaded_by_source': loaded,
                     'production_weights_unchanged': attempts['production_weights_unchanged'],
                     'external_weights_unchanged': attempts['external_weights_unchanged']},
        'task_evaluation': metrics,
        'production_classifier': {'raw': classifier['raw'], 'thresholded': classifier['thresholded'],
                                  'always_OTHER_baseline': baseline, 'source': 'cached actual ensemble stage probabilities'},
        'document_retrieval_ablation': ablation,
        'targeted_retrieval_probe': retrieval,
        'OCR': {'engine_language': ocr['engine_language'], 'vietnamese_support_claimed': False,
                'metrics': ocr['aggregate_by_language'], 'fixtures': 6, 'human_reference': False},
        'limitations': protocol['limitations'] + [
            'Primary task target is atomic accepted core-design obligations. Tutorial/user-code changes and existing behaviour can be source-faithful but outside this target.',
            'RST line wrapping and Markdown-oriented splitting create fragments; no reflow or parameter tuning was done after observing outcomes.',
            'No predicted dependency edges: dependency correctness is undefined, not 100%.',
            'No inference repeated for the retrieval-prompt ablation; exact input equivalence/replay is reported separately.',
            'The three public documents may have appeared in pretraining; only local known-development overlap was screened.']}
    for name in ['summary.json', 'README.md']:
        if (study / name).exists():
            raise FileExistsError('Existing study summaries are preserved: ' + name)
    (study / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    task_rows = []
    for mode in ['fast', 'full']:
        row = metrics[mode]['task_matching']['micro']
        task_rows.append(f'| {mode.upper()} | {row["tp"]} | {row["fp"]} | {row["fn"]} | '
                         f'{percent(row["precision"])} | {percent(row["recall"])} | {percent(row["f1"])} |')
    lines = ['# Kết quả thực nghiệm PSAI trên nguồn ngoài', '',
        'Giao thức được khóa trước dự đoán; primary reference B giữ nguyên sau khi xem kết quả. '
        'Người gán nhãn và người chấm là **AI**, không giả tên người, không gọi là final test độc lập.', '',
        '## Dữ liệu và khả năng chạy', '',
        'Ba tài liệu đầy đủ: Python PEP604, Rust RFC2046, Vue RFC0041, pin theo commit, giữ hash và ghi tình trạng giấy phép từng nguồn (Vue chưa xác định). '
        'Đã rà trùng với nguồn phát triển hiện biết; chưa loại trừ lịch sử không có hồ sơ hoặc pretraining. '
        'Đây là đề xuất thiết kế kỹ thuật ngoài miền proposal ban đầu, không đại diện mọi tài liệu dự án.', '',
        f'- Hoàn tất {summary["pipeline"]["successful"]}/{summary["pipeline"]["attempts"]} lượt yêu cầu Fast/Full.',
        f'- Hai lượt gán nhãn chỉ đọc nguồn: {len(truth)} đơn vị splitter; mức đồng thuận '
        f'{percent(agreement["aggregate"]["agreement"])}; kappa {agreement["aggregate"]["cohen_kappa"]:.4f}. Đây không phải độ tin cậy giữa hai người.',
        '- Primary B có 32 công việc, A có 28; khác biệt tách/gộp là một giới hạn của nhãn tham chiếu.',
        '- Ba lượt Full đều hoàn tất trích xuất và truy xuất; Qwen thực nạp ở Python và Vue. '
        'Rust không nạp Qwen vì catalog không có công việc đủ điều kiện.',
        '- Weights production, BGE-M3 và Qwen giữ nguyên; không train hoặc đổi ngưỡng để tăng điểm.', '',
        '## Độ đúng và đủ công việc theo reference B', '',
        '| Cấu hình | TP | FP | FN | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|---:|',
        *task_rows, '',
        'Rubric ghép một-một nhắm vào nghĩa vụ thay đổi core được chấp nhận. Câu hỏi tiêu đề, '
        'phản đối, hành vi đang tồn tại và ví dụ tutorial không tự trở thành core task. '
        'Công việc tutorial có thể là hướng dẫn có thật nhưng vẫn nằm ngoài target đã khóa. '
        'Các số thấp cho thấy bỏ sót nhiều trong phạm vi này; không dùng chúng làm accuracy tổng quát của sản phẩm.', '',
        'Fast có 6/6 và Full có 3/3 trích dẫn công việc khớp nguồn, nhưng hỗ trợ đúng **target task** '
        'theo rubric chỉ lần lượt 1/6 và 1/3. Không được lấy quote match thay Task F1.',
        'Không có cạnh phụ thuộc dự đoán; phép đo correctness phụ thuộc trả null, chưa đo được recall phụ thuộc.', '',
        '## Classifier production', '',
        f'Đã chấm xác suất thật của ensemble final_s11/s22/s33 trên {len(truth)} đơn vị nguồn. '
        f'Argmax: accuracy {percent(classifier["raw"]["accuracy"])}, Macro-F1 {classifier["raw"]["macro_f1_fixed_labels"]:.4f}. '
        f'Sau ngưỡng cũ: accuracy {percent(classifier["thresholded"]["accuracy"])}, Macro-F1 '
        f'{classifier["thresholded"]["macro_f1_fixed_labels"]:.4f}. '
        f'Baseline luôn OTHER có accuracy {percent(baseline["accuracy"])}, Macro-F1 {baseline["macro_f1_fixed_labels"]:.4f}.',
        'Các điểm này là diagnostic theo nhãn AI, thuộc ensemble production; không thay số validation của classifier nghiên cứu v2.', '',
        '## Retrieval và OCR', '',
        'So sánh cùng extraction cho thấy bỏ các document hits không đổi catalog hoặc prompt ở cả ba nguồn. '
        'Đã xác thực lại response cũ khi có response; đây là kiểm tra tương đương đầu vào, không phải lần inference thứ hai. '
        'Cấu hình không có resource corpus chưa chứng minh RAG cải thiện danh sách task.',
        'Sáu câu hỏi truy xuất do AI viết khi chỉ đọc nguồn được chấm riêng bằng BM25/dense/hybrid. '
        'Xem [retrieval probe](retrieval_probe_v1/diagnostic.json) cho rank/coverage và các đoạn reference có thể vượt ranh giới chunk. '
        'Positive spans chưa phải qrels đầy đủ; không gọi chỉ số coverage này là Recall@k được người chứng nhận.', '',
        '| Retrieval, cùng 6 câu hỏi và 8 đoạn tham chiếu | Phủ đoạn top 5 | RR đoạn đầu tiên trung bình |',
        '|---|---:|---:|',
        *[f'| {mode.upper()} | {percent(retrieval_summary["modes"][mode]["whole_known_span_micro_coverage_at_5"])} | '
          f'{retrieval_summary["modes"][mode]["first_known_span_mean_reciprocal_rank"]:.4f} |'
          for mode in ['bm25', 'dense', 'hybrid']], '',
        '| OCR hiện tại, 3 trang mỗi ngôn ngữ | CER giữ dấu | WER giữ dấu |', '|---|---:|---:|',
        f'| Tiếng Việt, engine en-US | {percent(ocr["aggregate_by_language"]["vi"]["diacritics_preserved"]["micro_cer"])} | '
        f'{percent(ocr["aggregate_by_language"]["vi"]["diacritics_preserved"]["micro_wer"])} |',
        f'| Tiếng Anh, engine en-US | {percent(ocr["aggregate_by_language"]["en"]["diacritics_preserved"]["micro_cer"])} | '
        f'{percent(ocr["aggregate_by_language"]["en"]["diacritics_preserved"]["micro_wer"])} |', '',
        'Bỏ dấu tiếng Việt được báo cáo riêng để phân tích lỗi, không thay metric giữ dấu. '
        'Sáu trang là ảnh tự tạo có transcript cố định, gồm văn xuôi/số liệu/bảng. '
        'Engine hiện chỉ en-US, yêu cầu language=vi vẫn bị từ chối; chưa có bằng chứng hỗ trợ OCR tiếng Việt.', '',
        '## Hồ sơ để kiểm tra lại', '',
        '- [Protocol trước inference](protocol_v1.json), [nguồn/license/hash](sources/source_manifest.json).',
        '- [Reference B](references/rater_b.json), [reference A](references/rater_a.json), [bất đồng](annotation_agreement.json).',
        '- [Quyết định Fast](reviews/fast_decisions.json), [quyết định Full](reviews/full_decisions.json).',
        '- [Điểm Fast](scored_fast/score.json), [điểm Full](scored_full/score.json), [so sánh ghép cặp](paired_comparison.json).',
        '- [Classifier thực](production_classifier_diagnostic.json), [ablation cấu trúc](document_retrieval_ablation.json).',
        '- [OCR thật và giới hạn](ocr/en_us_vi_diagnostic_v1/diagnostic.json), [tổng hợp có cấu trúc](summary.json).', '',
        '**Trạng thái:** đã hoàn tất thực nghiệm và rà soát AI của pilot này; final test có nhãn người độc lập vẫn NOT_SCORED. '
        'Không chỉnh manifest nghiên cứu cũ, không tự đổi AI thành human trong metadata.', '']
    (study / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    arguments = parser.parse_args()
    value = summarize(arguments.study)
    print(json.dumps({'successful_runs': value['pipeline']['successful'], 'human_ground_truth': False,
                      'independent_final_test_status': value['independent_final_test_status']}, indent=2))
