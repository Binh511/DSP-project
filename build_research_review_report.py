"""Export the response to the 12 review findings, embedding measured evidence."""
from pathlib import Path
import re

import pandas as pd

from build_classifier_report import export_docx, table
from study_data import ROOT, OUT, read_json


def main():
    registry = read_json(OUT / 'model_registry.json')
    investigation = read_json(OUT / 'investigation_08521.json')
    comparison = pd.read_csv(OUT / 'human_holdout_diagnostic/comparison.csv')
    errors = pd.read_csv(OUT / 'error_transitions.csv')
    plan = read_json(OUT / 'experiment_plan.json')
    annotation = read_json(OUT / 'annotation/manifest.json')
    final = pd.read_csv(OUT / 'final_test_candidate/labels_to_complete.csv')
    pipeline = read_json(OUT / 'pipeline_review/protocol.json')
    screening = read_json(OUT / 'final_test_candidate/screening.json')
    text = '''# KHẮC PHỤC CÁC VẤN ĐỀ TRONG BÁO CÁO TRAIN CLASSIFIER PSAI

Ngày tổng hợp: 26/10/2026.

## 1. Phạm vi và kết luận hiện tại

Đã bổ sung công cụ và thực hiện các kiểm tra có thể chạy từ bằng chứng hiện có. Không thể coi cả 12 vấn đề đã đóng: nhãn người độc lập, bằng chứng lịch sử đã mất và thực nghiệm dài chưa chạy vẫn được ghi là thiếu. Không tạo số Kappa, final-test F1, ablation hoặc mean/std giả để điền vào báo cáo.

Dataset gốc, checkpoint lịch sử và model đang dùng được giữ nguyên. Các lượt kiểm tra đường chạy được đặt tên smoke, tách khỏi model ứng viên để tránh xếp hạng nhầm. Chưa chọn hay triển khai model cuối mới.

## 2. Đối chiếu từng yêu cầu

'''
    statuses = [
        ('1', 'Đánh giá 19 model', 'Có registry và đo đồng nhất trên tập chẩn đoán; chưa có bảng independent final-test vì thiếu nhãn mới/lịch sử train.'),
        ('2', 'Điểm 0,8521', 'Đã tái tạo và khớp dự đoán cũ 100%; chưa xác minh được tập train lịch sử, không chứng nhận độc lập.'),
        ('3', 'Chất lượng nhãn', 'Có guideline và 200 câu phân tầng cho hai người; chờ nhãn độc lập, Kappa chưa đo.'),
        ('4', 'Final test mới', 'Có 8 tài liệu ứng viên, 1.060 câu và kiểm tra trùng; chờ người kiểm định nguồn và gán nhãn.'),
        ('5', 'Phân tích lỗi', 'Có chuyển nhãn, ví dụ, ngữ cảnh, confidence khi nguồn có; nguyên nhân cần người phân xử.'),
        ('6', 'Ablation', 'Có kế hoạch và runner giữ điều kiện cố định; chưa có lượt ablation hoàn chỉnh.'),
        ('7', 'Nhiều seed/fold', 'Có ba seed cố định và bộ tổng hợp chỉ nhận các lượt tương đương; chưa có mean/std thực đo.'),
        ('8', 'Loss và gradient', 'Đã sửa lỗi chuẩn hoá, kiểm tra batch lớn/microbatch và nhóm tích luỹ cuối; đạt kiểm tra hồi quy.'),
        ('9', 'Tìm tham số', 'Có kế hoạch giới hạn ngân sách và chọn bằng validation; chưa có kết quả tìm kiếm đầy đủ.'),
        ('10', 'Ensemble', 'Đã so sánh ba thành viên với ensemble trên cùng tập chẩn đoán; lưu mức trùng lỗi.'),
        ('11', 'Tác động pipeline', 'Có đầu ra đối chứng trên ba tài liệu và scorer exact-span; chờ reference. Chưa có đối chứng retrieval.'),
        ('12', 'Tốc độ/tài nguyên', 'Có benchmark từng model và benchmark ba lần của model đang dùng/ensemble; không suy diễn thành thời gian PDF 50 trang.'),
    ]
    text += table(['Mục', 'Vấn đề', 'Trạng thái có bằng chứng'], statuses)
    text += '\n\n## 3. Registry model và giới hạn lịch sử\n\n'
    text += table(['Model', 'Context', 'Chứng cứ train', 'Chọn checkpoint'],
                  [[r['model'], r['context_version'], r['training_provenance'], r['checkpoint_selection']] for r in registry])
    text += '''

Hash SHA256 trọng số hiện tại và cấu hình được lưu trong registry để các lần đánh giá sau có thể liên kết với đúng file model. Hash tạo hôm nay không chứng minh model đã dùng dữ liệu nào ở lần train cũ. Tên thư mục hoặc script chỉ là đầu mối điều tra, không thay thế log/manifest.

Model được extraction hiện tại sử dụng là ensemble final_s11, final_s22, final_s33.

## 4. Điều tra điểm lịch sử 0,8521

'''
    text += table(['Chỉ tiêu', 'Kết quả'], [[k, v] for k, v in investigation.items() if k != 'active_model_evidence'])
    text += '''

Kết quả tái tạo loại trừ giả thuyết đơn giản rằng checkpoint hiện tại chắc chắn cho dự đoán khác file dự đoán cũ. Nó không loại trừ vấn đề điều kiện train hoặc trùng lặp. Không có manifest train gốc và hash trọng số ghi tại thời điểm lịch sử nên không thể xác nhận độc lập chỉ bằng điểm số hoặc suy đoán.

## 5. So sánh đồng nhất trên tập chẩn đoán

Tất cả dòng sau dùng cùng 66 câu human holdout đã được xem trong quá trình phát triển. Dùng cấu hình context và OTHER bias đã lưu của từng model, cắt văn bản theo quy trình công bố và ánh xạ xác suất về cùng thứ tự nhãn. Legacy context không có phiên bản dùng chính sách Section cuối như bộ phân loại đơn hiện tại; chưa được chứng nhận khớp huấn luyện cũ.

Đây là bảng chẩn đoán, không phải independent final test, không dùng để chọn model cuối. Không xếp hạng model có lịch sử train chưa rõ như thể đã đảm bảo chưa thấy dữ liệu này.

'''
    text += table(['Model', 'N', 'Accuracy', 'Macro-F1'],
                  [[r.model, r.n, f'{r.accuracy:.4f}', f'{r.macro_f1:.4f}'] for r in comparison.itertuples()])
    ensemble = comparison[comparison.model.eq('ensemble_final_s11_s22_s33')].iloc[0]
    text += f'\n\nEnsemble có macro-F1 {ensemble.macro_f1:.4f} trên mẫu này; chưa có bằng chứng nó tốt hơn các thành viên hoặc tốt nhất trong 19 model. Chưa thay cấu hình production dựa trên mẫu chẩn đoán.\n\n'
    overlap = read_json(OUT / 'human_holdout_diagnostic/ensemble_error_overlap.json')
    text += table(['Thành viên A', 'Thành viên B', 'Cùng sai', 'Ít nhất một sai', 'Jaccard lỗi'],
                  [[r['a'], r['b'], r['both_wrong'], r['either_wrong'], f'{r["error_jaccard"]:.4f}' if r['error_jaccard'] is not None else 'Không xác định'] for r in overlap])
    text += '\n\n## 6. Phân tích lỗi và kiểm định nhãn\n\n'
    text += table(['Model', 'Tập', 'Nhãn dataset', 'Dự đoán', 'Số câu'],
                  [[r.model, r.dataset, r.true_label, r.predicted_label, r.n] for r in errors.head(12).itertuples()])
    text += '''

Đặc biệt có 50 câu REQUIREMENT được dự đoán ACTIVITY trong fair_fold0_human_once. Đây là thống kê bất đồng với nhãn dataset, không tự chứng minh cả 50 là lỗi của model. File error_analysis có cột để người đọc ghi nguyên nhân, nhãn sau phân xử và lý do. Các dấu hiệu như modal cue, văn bản dài hoặc ký tự lỗi chỉ là gợi ý.

Confidence của file lịch sử chỉ có hard label được để trống. Các lượt inference mới đã lưu đầy đủ xác suất và confidence; không điền confidence giả cho kết quả cũ.

Bộ kiểm định gồm 200 câu từ phần train, lấy 40 câu theo mỗi nhãn gốc. Hai bản A/B giữ cùng ID nhưng đảo thứ tự khác nhau và không chứa nhãn gốc. Người điều phối giữ khoá riêng. Công cụ chỉ tính agreement/Kappa khi hai người đã điền đủ nhãn hợp lệ, cùng ID và cùng nội dung; không thay ô trống bằng OTHER. Với mẫu phân tầng cân bằng, agreement chỉ mô tả mẫu kiểm định này.

## 7. Final test và bảo vệ quy trình chọn model

'''
    text += f'Tập ứng viên có {final.document_id.nunique()} tài liệu và {len(final)} câu. Không gán nhãn bằng model. Người phụ trách cần xác nhận nguồn và chất lượng nhãn trước khi được sử dụng.\n\n'
    text += table(['Tài liệu', 'Tài liệu phát triển gần nhất', 'Word-6gram Jaccard', 'Câu trùng chính xác'],
                  [[r['document_id'], r['closest_known_document'] or 'Không có', f'{r["word6gram_jaccard"]:.6f}', r['exact_sentence_overlaps']] for r in screening['documents']])
    text += '''

Kiểm tra này chỉ đối chiếu với dữ liệu phát triển đã biết. Overlap thấp không chứng minh tài liệu chưa xuất hiện trong những lịch sử không còn log. Review gate hiện chưa được phê duyệt.

Model được chọn phải có kết quả validation hoàn chỉnh, hash trọng số khớp và provenance train. Chọn xong sẽ đóng băng trước khi mở final test. Bộ đánh giá từ chối điểm diagnostic/smoke, nhãn thiếu, thay đổi nội dung mẫu, model thay đổi hoặc tài liệu trùng train/validation. Một attempt được ghi trước khi chạy; lỗi cũng giữ lại để xem xét thay vì tự xoá và đánh giá lặp.

## 8. Kế hoạch thực nghiệm có kiểm soát

'''
    reference_config = plan['experiments'][0]['config']
    text += table(['Baseline: tham số', 'Giá trị'], list(reference_config.items())) + '\n\n'
    text += table(['Tên', 'Nhóm', 'Thay đổi so với baseline', 'Kết quả đầy đủ'],
                  [[r['name'], r['group'], '; '.join(f'{k}={v}' for k, v in r['config'].items() if reference_config[k] != v) or 'Cấu hình tham chiếu', 'Chưa chạy'] for r in plan['experiments']])
    text += '''

Baseline và các lượt cùng điều kiện dùng cùng seed chia dữ liệu, cùng quy tắc lọc và cùng mã hoá đầu vào. Các seed train 42/123/2026 không thay seed chia tập. Thí nghiệm no_human cố ý bỏ phần bổ sung; các ablation khác giữ nguyên mẫu train. Mean/std chỉ được tính cho nhóm có cùng cấu hình và dấu vân tay dữ liệu, chỉ khác seed. Chưa có lượt đầy đủ trong kế hoạch nên chưa có kết luận Context/Class Weight/Human labels có đóng góp bao nhiêu, hoặc cấu hình nào tối ưu.

Đã chạy smoke bốn bước optimizer với tham số FP32, đánh giá validation, lưu và nạp lại model. Kiểm tra sau chạy xác nhận 202 tensor trọng số hữu hạn và 1.317 dòng xác suất hữu hạn, chuẩn hoá đúng. Kết quả smoke bị loại khỏi bảng chọn model và thống kê ổn định. Không dùng việc smoke chạy được để tuyên bố model đạt chất lượng tốt.

Các lỗi phát hiện trong chính quá trình kiểm tra runner được giữ lại: smoke_2 dừng trước train do API warmup_ratio không còn hỗ trợ, đã đổi thành warmup_steps; smoke_1 dùng FP16 ngầm định và AdamW thường tạo NaN, được đánh dấu invalid; smoke_3 có đầu ra hữu hạn nhưng tham số FP16 nên được đánh dấu superseded. Bản hiện tại buộc tham số FP32, giữ mixed precision ở mức tính toán và từ chối loss/logits/xác suất không hữu hạn. Các lỗi thử nghiệm này không sửa hoặc làm mất kết quả 19 model cũ.

## 9. Kiểm tra loss và gradient

Custom loss dùng microbatch mean nhưng Trainer đang cài suy ra model nhận loss kwargs từ chữ ký **kwargs. Điều này có thể làm bỏ bước chia theo số microbatch khi tích luỹ. Đã đặt model_accepts_loss_kwargs=False trong các Trainer tuỳ chỉnh và tách trainer dùng chung cho luồng mới.

Kiểm tra với mô hình tuyến tính, dữ liệu cố định, SGD, không dropout/clipping: batch lớn khớp microbatch sau sửa, cả nhóm cuối thiếu mẫu. Tái tạo cách cũ với tích luỹ tám bước cho cập nhật lớn gấp tám lần. Không suy diễn rằng mọi checkpoint cũ dùng cùng môi trường, hoặc lỗi này là nguyên nhân duy nhất của F1 thấp.

Bộ kiểm tra hồi quy hiện có 17 kiểm tra đạt, bao gồm loss/gradient, bảo toàn offset, input ablation, ánh xạ rater theo ID, chặn nhãn thiếu, chặn chọn bằng diagnostic, xác suất không hữu hạn và scorer pipeline. Các kiểm tra tính đúng phần mềm không thay thế kết quả thực nghiệm nhiều seed hoặc thẩm định nhãn người.

## 10. Đánh giá tích hợp pipeline

'''
    text += 'Ba tài liệu đối chứng: ' + ', '.join(pipeline['document_ids']) + '.\n\n'
    text += '''Đã tạo đầu ra bằng luật và bằng classifier ensemble hiện tại trên cùng văn bản, kèm danh sách công việc tương ứng. Reference để người kiểm tra điền tên các trường ở từng câu, hoặc NONE; không hiển thị dự đoán trong sheet gán nhãn. Scorer tính TP/FP/FN và P/R/F1 theo exact source sentence span, kiểm tra nội dung/offset không bị sửa. Reference còn trống nên chưa có điểm chất lượng đầu ra.

Đây chưa phải phép so sánh retrieval-only với classifier+retrieval: không có đầu ra truy xuất được đóng băng chung cho hai nhánh. Không tự gọi rules là retrieval. Muốn chứng minh tác động retrieval cần một tác vụ đầu ra thống nhất, cùng nguồn tham chiếu và quy trình truy xuất. Chất lượng planning cũng cần người chấm riêng; quote khớp không chứng minh công việc hợp lý.

## 11. Benchmark inference

'''
    bench_path = OUT / 'benchmark_deployed.json'
    if bench_path.exists():
        benchmark = read_json(bench_path)
        text += f'Đo trên {benchmark["n"]} câu, batch {benchmark["batch_size"]}, lặp {benchmark["repeats"]} lần; gồm tokenization, đồng bộ CUDA, bỏ thời gian load và warmup. Ensemble được nạp đồng thời đúng như code đang dùng.\n\n'
        text += table(['Model', 'Trung bình giây / 66 câu', 'Độ lệch chuẩn giây', 'Đỉnh VRAM cấp phát MiB'],
                      [[r['model'], f'{r["mean_seconds"]:.4f}', f'{r["std_seconds"]:.4f}', f'{(r["peak_allocated_vram_bytes"] or 0)/1024**2:.1f}'] for r in benchmark['results']])
        text += '\n\nBộ đo dùng một mẫu nhỏ và trạng thái GPU thực tế, nên chưa phải benchmark tổng quát mọi tải. Đỉnh cấp phát PyTorch khác tổng VRAM mà nvidia-smi báo. Không suy ra thời gian PDF 50 trang vì còn OCR, số câu và bước downstream.\n\n'
    else:
        text += 'Đã có benchmark một lượt cho từng model; benchmark lặp của model đang dùng chưa hoàn tất tại thời điểm xuất tài liệu.\n\n'
    text += '''## 12. Việc cần hoàn tất trước kết luận cuối

1. Hai người gán nhãn độc lập bộ 200 câu, đo agreement/Kappa và phân xử bất đồng; kiểm tra các lỗi có hệ thống.
2. Người phụ trách kiểm định nguồn và gán nhãn tập ứng viên mới; không tự đổi nhãn bằng kết quả model.
3. Chạy baseline hoàn chỉnh sau khi dữ liệu được chốt, rồi các ablation/seed/search trong ngân sách tài nguyên. Chỉ chọn bằng validation.
4. Điền reference pipeline, tính chất lượng trích xuất và bổ sung đánh giá planning/retrieval nếu muốn kết luận đầu cuối.
5. Đóng băng model được chọn và thực hiện final test theo gate. Báo cáo kết quả thật cùng giới hạn, không thay bằng điểm diagnostic cao nhất.

Tài liệu này tự chứa số liệu và trạng thái để người đọc không phải mở đường dẫn trên máy phát triển. Các file công cụ và dữ liệu rà soát nằm trong gói bàn giao riêng. Không có cơ sở tuyên bố “đã sửa xong mọi thiếu sót nghiên cứu” khi các bước nhãn người và thực nghiệm trên còn chưa hoàn thành.
'''
    (OUT / 'KET_QUA_KHAC_PHUC.md').write_text(text, encoding='utf-8')
    export_docx(text, OUT / 'KET_QUA_KHAC_PHUC.docx')
    print(OUT / 'KET_QUA_KHAC_PHUC.docx')


if __name__ == '__main__':
    main()
