"""Read-only analysis of existing validation outputs. No training or test access."""
import hashlib
import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from build_classifier_report import table
from study_data import ROOT, OUT, read_json, write_json, sha
from study_inputs import context_for
from train_human_cv import LABELS

V1 = OUT / 'studies/user_labels_v1'
V2 = OUT / 'studies/classifier_stability_v1'
DEST = V2 / 'analysis'
SOURCES = {42: (V1, 'baseline_original_labels', 'user_labels_v1'),
           123: (V2, 'seed_123', 'classifier_stability_v1'),
           2026: (V2, 'seed_2026', 'classifier_stability_v1')}


def link(path):
    import os
    return '[' + path.name + '](' + Path(os.path.relpath(path, DEST)).as_posix() + ')'


def csv(frame, name):
    frame.to_csv(DEST / name, index=False, encoding='utf-8-sig')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    evidence, frames, records, states, configurations = [], {}, {}, {}, {}
    metrics, histories, transitions, integrity, matrices, log_checks = [], [], [], [], {}, []
    def record(path):
        evidence.append(dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), sha256=sha(path)))
    for seed, (study, name, model_group) in SOURCES.items():
        folder = study / 'experiments' / name
        model = ROOT / 'models/controlled_study' / model_group / name
        paths = [folder / f'validation_{suffix}' for suffix in ['predictions.csv', 'metrics.json', 'confusion.csv']]
        paths += [folder / 'provenance.json', model / 'training_state.json', model / 'inference_config.json',
                  study / f'{name}.stdout.log', study / f'{name}.stderr.log']
        for path in paths:
            record(path)
        stdout = paths[-2].read_text(encoding='utf-8', errors='replace')
        stderr = paths[-1].read_text(encoding='utf-8', errors='replace')
        stdout_events = []
        for line in stdout.splitlines():
            if line.startswith('{'):
                event = ast.literal_eval(line)
                if 'eval_macro_f1' in event:
                    stdout_events.append(event)
        log_checks.append(dict(seed=seed, stdout_eval_events=len(stdout_events),
                               traceback_in_stderr='Traceback (most recent call last)' in stderr,
                               warning_lines=[line.strip() for line in stderr.splitlines() if 'warning' in line.lower()][:12]))
        frame = pd.read_csv(paths[0], keep_default_na=False, float_precision='round_trip')
        result, provenance = read_json(paths[1]), read_json(folder / 'provenance.json')
        state = read_json(model / 'training_state.json')
        state_events = [event for event in state['log_history'] if 'eval_macro_f1' in event]
        assert len(stdout_events) == len(state_events) == 5
        for logged, stored_event in zip(stdout_events, state_events):
            # stdout uses rounded strings; training_state preserves full precision.
            assert abs(float(logged['eval_macro_f1']) - stored_event['eval_macro_f1']) < 0.0001
        actual_training = pd.read_csv(folder / 'training_rows.csv', keep_default_na=False, float_precision='round_trip')
        if seed == 42:
            reference_training = actual_training
        else:
            pd.testing.assert_frame_equal(actual_training, reference_training)
        assert result['evaluation_kind'] == 'validation_selection'
        assert result['config']['seed'] == seed and not result['config']['use_label_overlay']
        if frames:
            first = frames[42]
            for col in ['document_id', 'text', 'label', 'true_label', 'section', 'prev_text', 'next_text', 'pos']:
                assert frame[col].equals(first[col]), (seed, col)
            assert result['train_sha256'] == records[42]['train_sha256']
            assert result['validation_sha256'] == records[42]['validation_sha256']
            assert {k: v for k, v in result['config'].items() if k != 'seed'} == {k: v for k, v in records[42]['config'].items() if k != 'seed'}
        report = classification_report(frame.true_label, frame.predicted_label, labels=LABELS, output_dict=True, zero_division=0)
        matrix = confusion_matrix(frame.true_label, frame.predicted_label, labels=LABELS)
        stored = pd.read_csv(paths[2], index_col=0).loc[LABELS, LABELS].to_numpy()
        np.testing.assert_array_equal(matrix, stored)
        np.testing.assert_allclose(report['macro avg']['f1-score'], result['macro_f1'], rtol=0, atol=1e-12)
        probs = frame[['prob_' + label for label in LABELS]].to_numpy()
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
        assert np.array_equal(np.array(LABELS)[probs.argmax(axis=1)], frame.predicted_label)
        for label in LABELS:
            for key in ['precision', 'recall', 'f1-score', 'support']:
                np.testing.assert_allclose(report[label][key], result['per_class'][label][key], atol=1e-12)
            metrics.append(dict(seed=seed, label=label, **report[label]))
        weight_hash = sha(model / 'model.safetensors')
        assert weight_hash == result['model_weight_sha256']
        integrity.append(dict(seed=seed, rows=len(frame), documents=frame.document_id.nunique(),
                              train_sha256=result['train_sha256'], validation_sha256=result['validation_sha256'],
                              weight_sha256=weight_hash, tokenizer_backend_sha256=provenance['tokenizer_backend_sha256'],
                              recomputed_metrics_match=True, confusion_matches=True, probabilities_finite=True,
                              best_step=state['best_global_step'], total_steps=state['global_step']))
        for event in state['log_history']:
            histories.append(dict(seed=seed, **event))
        for i, gold in enumerate(LABELS):
            for j, pred in enumerate(LABELS):
                if i != j:
                    transitions.append(dict(seed=seed, true_label=gold, predicted_label=pred, count=int(matrix[i, j])))
        frames[seed], records[seed], states[seed], matrices[seed] = frame, result, state, matrix
        configurations[seed] = read_json(model / 'inference_config.json')
    assert len({r['tokenizer_backend_sha256'] for r in integrity}) == 1
    combined = frames[42][['document_id', 'text', 'section', 'prev_text', 'next_text', 'pos', 'true_label']].copy()
    combined['validation_row'] = combined.index
    combined['item_id'] = [hashlib.sha256(f'{r.document_id}:{i}:{r.text}'.encode()).hexdigest() for i, r in enumerate(combined.itertuples())]
    combined['encoded_context'] = [context_for(row, configurations[42]) for row in combined.to_dict('records')]
    for seed, frame in frames.items():
        combined[f'prediction_{seed}'] = frame.predicted_label
        combined[f'confidence_{seed}'] = frame.confidence
        combined[f'correct_{seed}'] = frame.predicted_label.eq(frame.true_label)
    # Input feature limitations are observable; their causal effect is not established.
    combined['text_cut_600_chars'] = combined.text.str.len().gt(600)
    combined['missing_section'] = combined.section.str.strip().eq('')
    combined['previous_cut_150_chars'] = combined.prev_text.str.strip().str.len().gt(150)
    combined['next_cut_150_chars'] = combined.next_text.str.strip().str.len().gt(150)
    combined['replacement_character'] = combined[['text', 'prev_text', 'next_text']].apply(lambda row: any('\ufffd' in value for value in row), axis=1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/controlled_study/user_labels_v1/baseline_original_labels', local_files_only=True)
    raw = tokenizer(combined.text.str.slice(0, 600).tolist(), combined.encoded_context.tolist(), truncation=False)
    combined['tokens_before_maxlen'] = [len(ids) for ids in raw['input_ids']]
    combined['token_pair_over_256'] = combined.tokens_before_maxlen.gt(256)
    loss = combined[~combined.correct_2026 & (combined.correct_42 | combined.correct_123)].copy()
    loss['model_evidence'] = 'Same input/reference; seed-dependent prediction disagreement is verified. Semantic truth still requires label review.'
    loss['context_evidence'] = loss.apply(lambda row: '; '.join(k for k in ['missing_section', 'text_cut_600_chars', 'previous_cut_150_chars', 'next_cut_150_chars', 'token_pair_over_256'] if row[k]) or 'No automated limitation flag', axis=1)
    loss['pdf_extraction_status'] = 'UNVERIFIED: PDF not inspected; source/page check required'
    loss['reference_label_status'] = 'UNVERIFIED: human adjudication required; disagreement is not proof gold is wrong'
    csv(combined, 'all_validation_seed_predictions.csv')
    csv(loss, 'seed_2026_regressions.csv')
    classes = pd.DataFrame(metrics)
    csv(classes, 'per_class_comparison.csv')
    trans = pd.DataFrame(transitions)
    csv(trans, 'confusion_pairs.csv')
    csv(pd.DataFrame(histories), 'training_history.csv')
    by_label = loss.groupby('true_label').size().reindex(LABELS, fill_value=0)
    documents = loss.groupby('document_id').size().sort_values(ascending=False).rename('regressions').reset_index()
    csv(documents, 'regressions_by_document.csv')
    original = pd.read_csv(V1 / 'training_original_labels.csv', keep_default_na=False, float_precision='round_trip')
    revised = pd.read_csv(V1 / 'training_revised_labels.csv', keep_default_na=False, float_precision='round_trip')
    for col in original.columns.difference(['label']):
        assert original[col].equals(revised[col])
    changed = original.label.ne(revised.label)
    proposals = original[changed].copy()
    proposals['proposed_label'] = revised.loc[changed, 'label']
    proposals.rename(columns={'label': 'original_label'}, inplace=True)
    overlay = pd.read_csv(V1 / 'label_overlay.csv', keep_default_na=False)
    keys = ['document_id', 'text', 'original_label']
    proposals = proposals.merge(overlay[keys + ['item_id', 'annotation', 'reason', 'source_line']], on=keys, how='left', validate='one_to_one')
    assert len(proposals) == 57 and proposals.item_id.notna().all()
    assert proposals.proposed_label.equals(proposals.annotation)
    assert not set(proposals.document_id) & set(combined.document_id)
    proposals['validation_prediction_for_same_item'] = 'NOT_AVAILABLE: training item, not in validation'
    proposals['human_confirmed'] = False
    csv(proposals, 'ai_57_effective_proposals.csv')
    proposal_pairs = proposals.groupby(['original_label', 'proposed_label']).size().rename('ai_proposals_training').reset_index()
    wide = trans.pivot(index=['true_label', 'predicted_label'], columns='seed', values='count').reset_index()
    overlap = proposal_pairs.merge(wide, left_on=['original_label', 'proposed_label'], right_on=['true_label', 'predicted_label'], how='left')
    csv(overlap, 'proposal_vs_validation_transition_counts.csv')
    for name in ['dataset_revision.json', 'training_original_labels.csv', 'training_revised_labels.csv', 'label_overlay.csv']:
        record(V1 / name)
    # Exhaustive companion Markdown: all regressions, full stored context and actual encoded context.
    appendix = '# Toàn bộ câu seed 2026 sai nhưng seed 42 hoặc 123 đúng theo nhãn validation\n\n'
    appendix += 'Đúng/sai ở đây là đối chiếu nhãn tham chiếu hiện có, chưa phải xác minh nội dung bởi chuyên gia.\n\n'
    for row in loss.itertuples():
        appendix += f'## Validation row {row.validation_row} — {row.document_id}\n\n'
        appendix += f'Mã: {row.item_id}\n\nCâu: {row.text}\n\n'
        appendix += f'Section: {row.section or "(trống)"}\n\nPrevious: {row.prev_text or "(trống)"}\n\nNext: {row.next_text or "(trống)"}\n\nPos: {row.pos}\n\nContext thực sự tạo trước tokenization: {row.encoded_context}\n\n'
        appendix += table(['Tham chiếu', 'Seed 42', 'Seed 123', 'Seed 2026'], [[row.true_label, row.prediction_42, row.prediction_123, row.prediction_2026]])
        appendix += f'\n\nHạn chế đầu vào quan sát được: {row.context_evidence}. PDF/nhãn: cần người kiểm tra.\n\n'
    (DEST / 'CAC_CAU_SEED_2026_SAI.md').write_text(appendix, encoding='utf-8')
    summary = [dict(seed=s, macro_f1=records[s]['macro_f1'], accuracy=records[s]['accuracy'],
                    errors=int((~combined[f'correct_{s}']).sum()), trainer_best=states[s]['best_metric'], best_step=states[s]['best_global_step']) for s in SOURCES]
    values = [r['macro_f1'] for r in summary]
    text = '# Phân tích 3 seed classifier PSAI — nhãn gốc + full context\n\n'
    text += '## 1. Phạm vi và tính toàn vẹn\n\nPhân tích lại file có sẵn; không train, không chạy inference model, không mở test/final test, không thay checkpoint production. Chỉ tokenizer cục bộ được dùng để đếm độ dài đầu vào. Validation đã từng được xem, nên đây là phân tích phát triển, không phải kết quả test độc lập.\n\n'
    text += f'Cùng {len(combined)} câu/{combined.document_id.nunique()} tài liệu; dữ liệu train, validation, thứ tự câu, nhãn và context khớp giữa ba seed. Cấu hình ngoài seed bằng nhau; tokenizer backend hash bằng nhau. Đã tính lại metric từ predictions, so với per_class trong metrics JSON và toàn bộ confusion CSV; tất cả khớp. Hash checkpoint thực tế khớp metadata.\n\n'
    text += table(['Seed', 'Macro-F1 FP32 đã lưu', 'Accuracy', 'Câu sai', 'Best F1 trong Trainer', 'Best step'], [[r['seed'], f"{r['macro_f1']:.6f}", f"{r['accuracy']:.6f}", r['errors'], f"{r['trainer_best']:.6f}", r['best_step']] for r in summary])
    text += f'\n\nMacro-F1 trung bình **{np.mean(values):.6f}**, độ lệch chuẩn mẫu (ddof=1) **{np.std(values, ddof=1):.6f}**, n=3 seed trên cùng một split. Không phải CI hoặc biến thiên giữa các split. Seed 123 cao nhất trong nhóm này; chưa chứng minh tốt hơn ensemble final_s11 + final_s22 + final_s33. Ensemble đang dùng chưa thay; model cuối cùng chưa chốt.\n\n'
    text += '## 2. Precision, recall, F1 của 5 nhãn\n\n'
    text += table(['Seed', 'Nhãn', 'Precision', 'Recall', 'F1', 'Support'], [[int(row['seed']), row['label'], f'{row["precision"]:.6f}', f'{row["recall"]:.6f}', f'{row["f1-score"]:.6f}', int(row['support'])] for _, row in classes.iterrows()])
    deltas = []
    for label in LABELS:
        scores = [records[s]['per_class'][label]['f1-score'] for s in SOURCES]
        deltas.append([label, *[f'{v:.6f}' for v in scores], f'{scores[2]-scores[0]:+.6f}', f'{scores[2]-scores[1]:+.6f}'])
    text += '\n\n' + table(['Nhãn', 'F1 42', 'F1 123', 'F1 2026', 'Δ 2026−42', 'Δ 2026−123'], deltas)
    text += '\n\nMỗi lớp đóng góp 1/5 vào macro-F1; số lượng OTHER ảnh hưởng accuracy nhiều hơn. Xem độ giảm F1 từng lớp thay vì chỉ số câu sai.\n\n'
    delta_macro = records[42]['macro_f1'] - records[2026]['macro_f1']
    requirement_contribution = (records[42]['per_class']['REQUIREMENT']['f1-score'] - records[2026]['per_class']['REQUIREMENT']['f1-score']) / 5
    text += f'So với seed 42, F1 REQUIREMENT giảm 0,594286 → 0,450867; đóng góp {requirement_contribution:.6f} vào mức giảm macro-F1 {delta_macro:.6f} (khoảng {requirement_contribution / delta_macro:.1%} theo phân rã số học, không phải tỷ lệ nguyên nhân). So với seed 123, seed 2026 thấp hơn ở cả năm lớp. OTHER recall giảm 0,831554 → 0,762752 so với seed 42; số OTHER bị gán nhãn chức năng tăng từ 142 lên 200.\n\n'
    text += '## 3. Ma trận nhầm lẫn và các cặp ưu tiên\n\nHàng là nhãn tham chiếu, cột là dự đoán.\n\n'
    for seed in SOURCES:
        text += f'### Seed {seed}\n\n' + table(['Gold / Pred', *LABELS], [[label, *matrices[seed][i].tolist()] for i, label in enumerate(LABELS)]) + '\n\n'
    pairs = wide.copy()
    pairs['total'] = pairs[[42, 123, 2026]].sum(axis=1)
    priority = ((pairs.true_label.eq('REQUIREMENT') & pairs.predicted_label.eq('ACTIVITY')) |
                (pairs.true_label.eq('ACTIVITY') & pairs.predicted_label.eq('REQUIREMENT')) |
                pairs.true_label.eq('EXPECTED_RESULT') | pairs.predicted_label.eq('EXPECTED_RESULT'))
    text += table(['Gold', 'Pred', '42', '123', '2026'], pairs.loc[priority].sort_values('total', ascending=False)[['true_label', 'predicted_label', 42, 123, 2026]].values.tolist())
    text += '\n\nCác cặp còn lại có đầy đủ trong ' + link(DEST / 'confusion_pairs.csv') + '.\n\n'
    text += 'REQUIREMENT→ACTIVITY: 16/24/24; chiều ngược ACTIVITY→REQUIREMENT: 12/7/6 (theo seed 42/123/2026). Vì vậy không thể nói seed 2026 giảm chỉ do cặp này tăng: so với 123, chiều REQUIREMENT→ACTIVITY không tăng. Seed 2026 còn mất REQUIREMENT đúng sang OTHER, EXPECTED_RESULT và OBJECTIVE. EXPECTED_RESULT có số đúng 50/51/50, nhưng OTHER→EXPECTED_RESULT tăng 45/38/53, làm giảm precision; đây là thay đổi cần xem bên cạnh recall.\n\n'
    text += '## 4. Các câu seed 2026 thua seed khác\n\n'
    text += f'Có **{len(loss)} câu** seed 2026 sai nhưng ít nhất một seed kia đúng; **{int((loss.correct_42 & loss.correct_123).sum())} câu** cả 42 và 123 đều đúng. Ngược lại, seed 2026 đúng nhưng ít nhất một seed kia sai ở **{int((combined.correct_2026 & (~combined.correct_42 | ~combined.correct_123)).sum())} câu**; tránh chỉ nhìn các câu thua.\n\n'
    text += table(['Gold', 'Số câu regression'], [[label, int(by_label[label])] for label in LABELS])
    text += '\n\n' + table(['Tài liệu', 'Số regression'], documents.head(12).values.tolist())
    text += '\n\nToàn bộ câu, document_id, text, section, previous, next, pos, context thực sự đưa vào tokenizer, gold và ba dự đoán: ' + link(DEST / 'CAC_CAU_SEED_2026_SAI.md') + '; dữ liệu lọc được: ' + link(DEST / 'seed_2026_regressions.csv') + '.\n\n'
    text += '## 5. Training logs và giới hạn suy luận nguyên nhân\n\n'
    epoch_rows = []
    for seed in SOURCES:
        for event in states[seed]['log_history']:
            if 'eval_macro_f1' in event:
                epoch_rows.append([seed, event['epoch'], event['step'], f"{event['eval_macro_f1']:.6f}", f"{event['eval_loss']:.6f}"])
    text += table(['Seed', 'Epoch', 'Step', 'Trainer val macro-F1', 'Eval loss'], epoch_rows)
    text += '\n\nTrainer chọn checkpoint bằng validation macro-F1 trong lúc train. Bảng đầu báo cáo dùng dự đoán sau export qua predict_local FP32, batch 8, other_bias=0. Provenance ghi train/Trainer dùng BF16, parameters FP32; do đó đây là hai đường đánh giá có khác biệt độ chính xác tính toán. Các giá trị thực tế không hoàn toàn trùng. Chưa lưu logits Trainer để đối chiếu từng câu, nên chưa thể quy toàn bộ chênh lệch cho BF16, padding hay nguyên nhân cụ thể. Không lấy điểm FP32 của checkpoint chưa được đánh giá để khẳng định checkpoint khác tốt hơn.\n\n'
    text += 'Seed 2026 tăng F1 Trainer qua các epoch nhưng eval loss tăng từ epoch 4 sang 5. Đây là bất đồng giữa hai chỉ số, không đủ chứng minh lỗi phần mềm hay overfitting là nguyên nhân duy nhất. Seed 123 học nhanh hơn ở các epoch đầu theo log; khác seed thay đổi khởi tạo head, dropout và thứ tự dữ liệu theo code. Chưa tách từng tác động bằng đối chứng, nên chỉ xác nhận có biến động theo seed.\n\n'
    text += '## 6. Phân nhóm nguyên nhân theo mức bằng chứng\n\n'
    flags = ['missing_section', 'text_cut_600_chars', 'previous_cut_150_chars', 'next_cut_150_chars', 'token_pair_over_256', 'replacement_character']
    text += table(['Dấu hiệu quan sát', 'Toàn validation', 'Trong regression 2026'], [[flag, int(combined[flag].sum()), int(loss[flag].sum())] for flag in flags])
    text += '\n\n' + table(['Nhóm', 'Có bằng chứng gì', 'Điều chưa được xác nhận / hành động'], [
        ['Model/seed', 'Input và gold giống nhau nhưng dự đoán khác; ma trận và lịch sử học khác.', 'Xác nhận bất đồng theo seed, chưa xác nhận gold đúng hay lỗi do một cơ chế tối ưu cụ thể.'],
        ['Context', 'Đếm trực tiếp section trống, giới hạn text600, previous/next150, token pair >256 theo tokenizer cục bộ.', 'Đây là hạn chế biểu diễn có thật, chưa phải quan hệ nhân quả. Cần người đọc đoạn nguồn; không chạy ablation mới trong đợt này.'],
        ['PDF extraction', 'Chỉ có CSV/context đã lưu; có thể có dấu hiệu ký tự bất thường.', 'CHƯA XÁC MINH: không đọc PDF gốc trong phân tích này. Không gán lỗi extraction chỉ vì section trống. Cần trang, đoạn gốc và span đối chiếu.'],
        ['Nhãn tham chiếu mơ hồ', 'Câu có thể đồng thời nêu hoạt động/đầu ra/điều kiện; 57 đề xuất AI chỉ là ý kiến cần xem.', 'CHƯA XÁC MINH: người gán nhãn phân xử theo guideline và nguồn. Không thay validation hoặc xem đồng thuận model là ground truth.']])
    text += '\n\nCác nhóm không loại trừ nhau; báo cáo không bịa tỷ lệ nguyên nhân đã xác nhận. Bảng câu regression đánh dấu riêng evidence model/context và trạng thái chưa xác minh PDF/gold.\n\n'
    text += '### Các trường hợp có nội dung để người kiểm tra phân xử\n\nCác nhận xét dưới đây là giả thuyết có trích câu/context, không phải nhãn chuyên gia đã xác nhận. Validation row dùng chỉ số bắt đầu từ 0 trong CSV.\n\n'
    examples = [
        (217, 'Model / ranh giới REQUIREMENT–ACTIVITY', 'Cụm “immutable version tag” mô tả thuộc tính/tính năng; seed 42 và 123 đồng ý REQUIREMENT, 2026 chọn ACTIVITY. Cần guideline xác nhận đây là đặc tả, không phải hành động. Section trống không đủ kết luận lỗi context.'),
        (84, 'Context + ranh giới REQUIREMENT–ACTIVITY', 'Câu mô tả password/encryption trong mục tạo shared folder, vừa có tính năng vừa có thao tác. Previous bị cắt 150 ký tự theo protocol; cần xem đầy đủ đoạn trước khi kết luận mất context là nguyên nhân.'),
        (1211, 'Ứng viên lỗi model về EXPECTED_RESULT', '“Comprehensive documentation and a user guide” nằm dưới section Deliverables; 42/123 đúng EXPECTED_RESULT, 2026 chọn ACTIVITY. Đây là bằng chứng ngữ nghĩa ủng hộ gold, nhưng vẫn cần người kiểm tra xác nhận. Không có cờ hạn chế context tự động ở item này.'),
        (805, 'Nhãn tham chiếu có khả năng mơ hồ', 'Cấu trúc “If ... should report ...” có thể là quy tắc báo cáo/điều kiện chứ không chỉ đầu ra. Gold EXPECTED_RESULT cần phân xử; 123 khớp gold không chứng minh 123 đúng về ngữ nghĩa.'),
        (1114, 'Model / vai trò context', 'Section ghi Contributions to Mathesar và danh sách issue có mã #2747, hỗ trợ cách đọc đây là đóng góp/lịch sử. 42/123 chọn OTHER; 2026 chọn REQUIREMENT. Cần nguồn xác nhận vai trò đoạn; không có cờ thiếu section hoặc cắt context ở item này.'),
        (732, 'Model / phân biệt kết quả có sẵn với kết quả dự kiến', 'Câu có dẫn nghiên cứu Husain et al., 2024, nằm cạnh nhận xét về nghiên cứu trước; 123 chọn OTHER, 42/2026 EXPECTED_RESULT. Cần guideline phân biệt related work với kết quả dự kiến. Chưa xác minh PDF.')]
    for index, category, note in examples:
        row = combined.loc[index]
        assert index in set(loss.validation_row)
        text += f'**{row.document_id}, row {index} — {category}**\n\nCâu: {row.text}\n\nSection: {row.section or "(trống)"}\n\nPrevious: {row.prev_text or "(trống)"}\n\nNext: {row.next_text or "(trống)"}\n\n'
        text += table(['Gold', '42', '123', '2026'], [[row.true_label, row.prediction_42, row.prediction_123, row.prediction_2026]])
        text += f'\n\nNhận định: {note} **Trạng thái: cần người kiểm tra; chưa đổi nhãn.**\n\n'
    text += '## 7. Đối chiếu 57 đề xuất sửa nhãn AI\n\n'
    text += f'Đối chiếu training_original_labels với training_revised_labels và overlay: **{len(proposals)} thay đổi có hiệu lực**, từ 68 thay đổi đề xuất trong 200 câu; features/thứ tự training giữ nguyên. Cả ba seed đang phân tích đều use_label_overlay=false, nên **không dùng 57 nhãn đề xuất**. Các item này thuộc train, không thuộc validation; giao document_id với validation bằng 0. Vì vậy không có dự đoán validation của ba seed cho chính 57 câu này; không suy diễn chúng gây ra lỗi seed 2026.\n\n'
    text += 'Toàn bộ câu, nhãn gốc, nhãn AI đề xuất, lý do AI và source_line: ' + link(DEST / 'ai_57_effective_proposals.csv') + '. Lý do AI được giữ nguyên kể cả ký tự lỗi; không coi đó là bằng chứng chuyên gia.\n\n'
    text += table(['Nhãn gốc → AI đề xuất', 'Số câu train', '42: gold→pred cùng hướng', '123', '2026'], [[r['original_label'] + ' → ' + r['proposed_label'], int(r['ai_proposals_training']), int(r[42]), int(r[123]), int(r[2026])] for _, r in overlap.iterrows()])
    text += '\n\nHai phía bảng trên là các tập câu khác nhau: chỉ đối chiếu hướng ranh giới nhãn, không phải item matching hay đo độ đúng của AI. Chưa có nhãn phân xử thủ công cho 57 câu nên không tính precision của AI, Kappa, hoặc số “sửa đúng”.\n\n'
    text += '## 8. Hành động ưu tiên\n\n1. Người kiểm tra phân xử REQUIREMENT–ACTIVITY, EXPECTED_RESULT và các trường hợp ảnh hưởng lớn, dùng cả đoạn nguồn và ghi lý do/quy tắc. Giữ nhãn validation hiện tại để tái lập; nếu phát hành nhãn mới phải có version và báo cáo tách riêng.\n2. Kiểm tra các hạn chế context và PDF trên đúng item regression, ghi trang/span; chỉ quy lỗi extraction khi có đối chiếu nguồn. Không mở loạt thí nghiệm mới.\n3. Hoàn thiện quyết định chọn checkpoint bằng validation sau rà soát. Seed 123 là ứng viên có điểm cao nhất hiện tại; báo cáo đồng thời mean/std ba seed, không chỉ điểm tốt nhất. Ghi rõ khác biệt metric Trainer và FP32 trước khi đóng băng giao thức inference.\n4. Lưu checkpoint, tokenizer, input_protocol/context version, dataset/validation hash và other_bias=0 cùng quyết định lựa chọn; không tự thay ensemble production.\n5. Final test chỉ đánh giá sau xác minh nguồn/nhãn và model được chốt; không dùng kết quả để chọn tiếp seed/tham số. Báo cáo này không sử dụng dữ liệu test/final test.\n\n'
    text += '## 9. Bằng chứng trực tiếp\n\n'
    for seed, (study, name, model_group) in SOURCES.items():
        folder = study / 'experiments' / name
        model = ROOT / 'models/controlled_study' / model_group / name
        text += f'- Seed {seed}: ' + ', '.join(link(p) for p in [folder / 'validation_predictions.csv', folder / 'validation_metrics.json', folder / 'validation_confusion.csv', folder / 'provenance.json', model / 'training_state.json', study / f'{name}.stdout.log', study / f'{name}.stderr.log']) + '.\n'
    text += '\nHash file đã đọc: ' + link(DEST / 'evidence_manifest.json') + '; kết quả kiểm tra: ' + link(DEST / 'integrity.json') + '.\n'
    text += '\nĐối chiếu stdout log và training_state: mỗi seed có đủ 5 lần validation, điểm làm tròn trong stdout khớp state trong sai số 0,0001; đọc stderr để ghi lại warning/traceback trong ' + link(DEST / 'log_checks.json') + '. Training rows thực tế khớp từng ô giữa ba seed; không chỉ dựa vào hash khai báo.\n'
    (DEST / 'PHAN_TICH_3_SEED.md').write_text(text, encoding='utf-8')
    write_json(DEST / 'integrity.json', integrity)
    write_json(DEST / 'log_checks.json', log_checks)
    write_json(DEST / 'evidence_manifest.json', dict(files=evidence, no_training=True, no_test_access=True, no_production_changes=True))
    write_json(DEST / 'summary.json', dict(seeds=summary, mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)),
                regressions=len(loss), both_other_seeds_correct=int((loss.correct_42 & loss.correct_123).sum()),
                effective_ai_proposals=len(proposals), ai_labels_ground_truth=False))
    print(DEST / 'PHAN_TICH_3_SEED.md')
    print(json.dumps(read_json(DEST / 'summary.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
