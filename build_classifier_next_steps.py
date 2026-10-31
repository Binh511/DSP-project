"""Summarize source audit and propose one inference protocol; no model selection."""
from pathlib import Path

import pandas as pd

from build_classifier_report import table
from study_data import ROOT, OUT, sha, read_json, write_json

STUDY = OUT / 'studies/classifier_stability_v1'
DEST = STUDY / 'next_steps'


def link(path):
    import os
    return f'[{path.name}]({Path(os.path.relpath(path, DEST)).as_posix()})'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    audit_folder = STUDY / 'section_audit_agent'
    audit = read_json(audit_folder / 'summary.json')
    runs = {42: (OUT / 'studies/user_labels_v1', 'baseline_original_labels', 'user_labels_v1'),
            123: (STUDY, 'seed_123', STUDY.name), 2026: (STUDY, 'seed_2026', STUDY.name)}
    sources, candidates, comparison = {}, [], []
    for seed, (study, name, group) in runs.items():
        folder = study / 'experiments' / name
        model = ROOT / 'models/controlled_study' / group / name
        metric = read_json(folder / 'validation_metrics.json')
        state = read_json(model / 'training_state.json')
        provenance = read_json(folder / 'provenance.json')
        candidates.append(dict(seed=seed, model_path=str(model.relative_to(ROOT)).replace('\\', '/'),
                               model_weight_sha256=metric['model_weight_sha256'],
                               tokenizer_sha256=sha(model / 'tokenizer.json'),
                               inference_config_sha256=sha(model / 'inference_config.json'),
                               validation_metrics_sha256=sha(folder / 'validation_metrics.json'),
                               validation_rows_file_sha256=sha(folder / 'validation_rows.csv'),
                               train_content_sha256=metric['train_sha256'], validation_content_sha256=metric['validation_sha256'],
                               library_versions=provenance['library_versions']))
        comparison.append([seed, state['best_global_step'], f"{state['best_metric']:.6f}",
                           f"{metric['macro_f1']:.6f}", f"{metric['macro_f1']-state['best_metric']:+.6f}"])
    for name in ['study_inputs.py', 'study_evaluation.py', 'study_experiments.py', 'classifier_training.py', 'train_human_cv.py', 'label_import.py']:
        sources[name] = sha(ROOT / name)
    write_json(DEST / 'inference_protocol_proposal.json', dict(
        name='psai-study-v1-export-fp32-bs8-v1', status='PROPOSED; no final model selected',
        input=dict(protocol='study-v1', context_mode='full', text_max_chars=600, section_last_chars=120,
                   previous_last_chars=150, next_first_chars=150, pos='int(pos * 10)',
                   extra_features='Bullet and Cue from frozen study_inputs.py',
                   max_length=256, truncation='longest_first', padding='dynamic',
                   fill_missing_context='', validation_order='saved validation_rows.csv order; no reshuffle'),
        inference=dict(batch_size=8, parameter_dtype='float32', autocast=False, mode='eval + torch.inference_mode',
                       tokenizer='saved local tokenizer; local_files_only=True', other_bias=0,
                       labels=['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER'],
                       decision='softmax then argmax; canonical label order',
                       confidence_filter=None, abstention_threshold=None),
        metric=dict(primary='macro-F1', labels='all five even if a class has no predictions', zero_division=0,
                    secondary=['per-class precision/recall/F1/support', 'confusion counts', 'accuracy']),
        numerical_limits=dict(recorded_historical_flags=False,
                              unrecorded_flags=['TF32', 'float32_matmul_precision', 'cuDNN flags', 'deterministic algorithms'],
                              bitwise_reproducibility_claim=False,
                              next_inference_must_record=['GPU/device', 'CUDA/cuDNN versions', 'TF32/matmul/determinism flags', 'package versions']),
        candidates=candidates, code_sha256=sources,
        selection_rule='Compare only same protocol on validation; no final choice until label/source review',
        original_trainer_selection='Existing export chosen by per-epoch Trainer BF16 validation score',
        warning='Not a rerun and not a model freeze; scores for unexported epoch checkpoints under FP32 remain unknown',
        final_test_accessed=False, production_replaced=False))
    text = '# Kiểm tra Section và công việc trước khi chốt classifier\n\n'
    text += '## Kết quả kiểm tra nguồn\n\n'
    text += f'Validation có **{audit["missing_section"]}/{audit["rows"]} câu Section trống ({100*audit["missing_section"]/audit["rows"]:.2f}%)**. Đối chiếu chỉ 29 tài liệu validation với Markdown hiện có: cả 29 đều có tiêu đề ở data_processed và data_cleaned. Không kiểm tra PDF trong đợt này.\n\n'
    status = audit['source_status_missing_section']
    text += table(['Trong 1.045 câu Section trống', 'Số câu', 'Mức kết luận'], [
        ['Khớp nội dung nguồn chuẩn hóa; một đường dẫn heading không rỗng', status['heading_present_in_cleaned_source'], 'Ứng viên có thể khôi phục từ Markdown; chưa sửa dữ liệu'],
        ['Khớp nhiều vị trí thuộc các heading khác nhau', status['multiple_source_matches_different_sections_needs_review'], 'Người kiểm tra phải xác định vị trí đúng'],
        ['Khớp trước heading đầu tiên / không có heading tại vị trí', status['matched_before_first_heading_or_no_heading'], 'Không tự gán heading từ nội dung xuất hiện sau']])
    text += '\n\n**Đã xác định một hạn chế trong cách dựng context:** ' + link(ROOT / 'train_human_cv.py') + ' dựng Section bằng các dòng bắt đầu bằng # còn trong CSV phân loại, không đọc heading của Markdown nguồn. ' + link(ROOT / 'label_import.py') + ' lấy các ô trong bảng gán nhãn vào CSV; heading nằm ngoài bảng không tự được mang sang. Do đó Section trống trong classifier không đồng nghĩa nguồn Markdown không có cấu trúc.\n\n'
    text += 'Đây là bằng chứng về thông tin cấu trúc có trong bản nguồn hiện có nhưng không được đưa vào input cũ. Chưa chứng minh việc bổ sung Section sẽ tăng F1, chưa xác minh cấu trúc Markdown trung thành với PDF, và chưa chứng minh đây là nguyên nhân riêng làm seed 2026 kém hơn: ba seed dùng cùng context. Nếu sửa cách tạo context, phải phát hành phiên bản mới và giữ nguyên số liệu cũ để đối chiếu.\n\n'
    text += '**Điểm cần chú ý khi truy nguồn:** source_line là dòng trong bảng Markdown gán nhãn, không phải số trang PDF hoặc dòng Markdown đã làm sạch. Cả 1.317 tham chiếu dòng khớp text; 81 nhãn trong file annotation hiện tại khác nhãn CSV đóng băng. Đây là sai khác giữa các bản dữ liệu, chưa cho biết bên nào đúng. Không âm thầm đồng bộ nhãn hoặc tính lại validation bằng nhãn khác.\n\n'
    text += 'Danh sách 81 trường hợp lệch nhãn: ' + link(audit_folder / 'annotation_label_discrepancies.csv') + '. Ngoài Section trống, còn **35 câu thuộc PSAI133** có Section khác giữa thuật toán stack hiện tại và stack theo cấp heading thực. Khi nhảy từ # sang #### rồi gặp #### tiếp theo, code hiện tại dùng độ sâu danh sách thay cho cấp heading, có thể giữ nhầm heading cùng cấp làm cha. Đây là lỗi phân cấp có thể tái hiện; chưa sửa formatter đã đóng băng hoặc đo ảnh hưởng F1. Bằng chứng: ' + link(audit_folder / 'heading_stack_discrepancies.csv') + '.\n\n'
    text += 'Chi tiết từng câu, heading nguồn và mức chắc chắn: ' + link(audit_folder / 'validation_section_trace.csv') + '. Tổng hợp theo tài liệu: ' + link(audit_folder / 'validation_section_documents.csv') + '. Phương pháp/giới hạn: ' + link(audit_folder / 'KIEM_TRA_SECTION.md') + '.\n\n'
    text += '## Phiếu phân xử nhãn đã chuẩn bị\n\n'
    text += 'Một phiếu gồm **91 câu**: 34 câu validation mang nhãn tham chiếu REQUIREMENT hoặc EXPECTED_RESULT trong nhóm regression của seed 2026, và toàn bộ 57 đề xuất AI có hiệu lực trong bản train sửa nhãn. Phần nội dung hiển thị không có nhãn gốc, đề xuất AI hoặc dự đoán model để người đọc đưa ra nhận định trước. Đây là chọn mẫu có chủ đích, không phải kiểm định ngẫu nhiên toàn corpus và không tạo được Kappa từ một người.\n\n'
    text += 'Mở ' + link(STUDY / 'human_review/PHAN_XU_91_CAU.html') + ', điền tên, nhãn, lý do và tình trạng đối chiếu nguồn; câu chưa đủ căn cứ được để trống. Có thể tải CSV đang làm và bản sao JSON. Nút tải CSV đã gán đủ yêu cầu có tên và đủ 91 nhãn/lý do. **Chuẩn bị phiếu chưa phải đã hoàn thành kiểm định**; không có nhãn người mới được tự tạo hoặc áp dụng.\n\n'
    text += '## Giao thức inference thống nhất được đề xuất\n\n'
    text += 'Ghi cụ thể giao thức `psai-study-v1-export-fp32-bs8-v1` trong ' + link(DEST / 'inference_protocol_proposal.json') + '. Dùng cùng encoder study-v1/full, text600, max_length256, longest_first, batch8, tokenizer local của checkpoint, eval/inference_mode, không autocast, OTHER bias0, không lọc confidence; tính macro-F1 cố định 5 nhãn và zero_division0. Không tự bổ sung Section vừa truy ra vào input của các điểm cũ.\n\n'
    text += table(['Seed', 'Best step Trainer', 'F1 Trainer', 'F1 export FP32', 'FP32 − Trainer'], comparison)
    text += '\n\nCode/provenance xác nhận Trainer bật BF16 còn predict_local nạp parameter FP32 và không dùng autocast. Chưa có logits Trainer lưu lại để xác định từng câu đổi dự đoán, nên không quy toàn bộ chênh lệch cho BF16. Các flag TF32/matmul/determinism chưa được lưu ở lượt inference cũ; không cam kết tái lập bitwise hoặc strict IEEE FP32. Lượt inference tiếp theo phải ghi môi trường và các flag này cùng hash đầu vào/model/tokenizer. Chuỗi PROTOCOL chung trong study_evaluation còn mô tả legacy; ba lượt này thực tế dispatch bằng input_protocol=study-v1, nên manifest nêu rõ encoder thực sự dùng. Không sửa code đã đóng băng.\n\n'
    text += 'Manifest hiện ghi hash của ba checkpoint export và tài liệu đánh giá có sẵn, không phải quyết định chốt model. Không dùng điểm Trainer trộn với FP32 để xếp hạng. Điểm FP32 của các epoch checkpoint chưa đánh giá vẫn chưa biết.\n\n'
    text += '## Thứ tự tiếp tục\n\n1. Người kiểm tra phân xử các câu gây tranh cãi và ghi nguồn/lý do; so sánh các phiên bản nhãn mà không tự xem AI là ground truth.\n2. Rà heading nguồn theo bảng truy vết; phân xử 25 vị trí mơ hồ trước khi đề xuất phiên bản context mới.\n3. Chỉ sau khi kiểm định, quyết định nhãn train nào được chấp nhận; mọi thay đổi phải có version. Không có thay đổi train/context được áp dụng trong đợt này.\n4. Dùng một giao thức validation thống nhất để cân nhắc checkpoint; seed 123 chỉ là ứng viên. Báo cáo đủ trung bình 0,6340 ± 0,0268 qua ba seed và giới hạn của validation phát triển.\n5. Final test chỉ đánh giá khi người phụ trách xác minh nguồn/nhãn và model đã chốt. Chưa chọn checkpoint cuối, chưa thay ensemble PSAI, không train thêm.\n'
    (DEST / 'KIEM_TRA_SECTION_VA_BUOC_TIEP.md').write_text(text, encoding='utf-8')
    print(DEST / 'KIEM_TRA_SECTION_VA_BUOC_TIEP.md')


if __name__ == '__main__':
    main()
