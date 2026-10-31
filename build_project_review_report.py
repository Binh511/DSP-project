"""Build the current project assessment from actual saved evidence."""
from pathlib import Path
import json
import os

import pandas as pd

from build_classifier_report import export_docx, table
from study_data import ROOT, OUT, read_json, sha, write_json

DEST = ROOT / 'reports/project_review'


def link(path):
    return f'[{path.name}]({Path(os.path.relpath(path, DEST)).as_posix()})'


def main():
    checks = read_json(DEST / 'verification.json')
    study = OUT / 'studies/classifier_stability_v1'
    stats = read_json(study / 'stability_with_reference.json')
    section = read_json(study / 'section_audit_agent/summary.json')
    research = read_json(DEST / 'research_audit.json')
    assert checks['full_suite_passed'] and checks['unit_tests_passed'] >= 61
    text = '# Đánh giá và hoàn thiện toàn bộ project PSAI\n\n'
    text += '## 1. Nhận xét theo tiêu chuẩn giảng viên phản biện\n\n'
    text += '**Project có prototype hoạt động và bằng chứng thực nghiệm classifier kiểm tra lại được, nhưng chưa đủ điều kiện kết luận nghiên cứu hoặc nghiệm thu chất lượng toàn hệ thống.** Trở ngại chính hiện nay là nhãn tham chiếu chưa được phân xử, dữ liệu ứng viên final test từng xuất hiện trong lịch sử phát triển, và thiếu đánh giá độc lập PDF → nội dung → báo cáo. Những việc này không thể giải quyết bằng cách train thêm hoặc viết báo cáo khẳng định mạnh hơn.\n\n'
    text += 'Đợt này đã sửa lỗi code có thể kiểm chứng, bổ sung cách chạy thống nhất và kiểm tra hồi quy; giữ nguyên weights production, dataset frozen và dự đoán nghiên cứu đã lưu. Có chạy inference ensemble trên một ví dụ tự biên soạn để kiểm tra chức năng, không dùng ví dụ này làm tập đánh giá. Không train thêm thí nghiệm và không tính điểm test/final test.\n\n'
    text += table(['Tiêu chí', 'Đánh giá hiện tại', 'Điều còn thiếu để chấp nhận kết luận'], [
        ['Mục tiêu và phạm vi', 'Phân loại câu, trích thông tin, kế hoạch có nguồn và retrieval đã có code', 'Phải tách rõ đầu ra trích nguyên văn, quy tắc và mô tả AI; chưa có đánh giá hiểu sơ đồ/pipeline trong hình'],
        ['Dữ liệu và nhãn', 'Có version/hash, nhóm theo tài liệu và nhật ký đề xuất nhãn', 'Phân xử người cho nhãn mơ hồ; xử lý lệch nhãn giữa annotation và CSV'],
        ['Thí nghiệm classifier', 'Ba seed và đối chứng nhãn/context có số liệu thật', 'Chưa có so sánh độc lập hợp lệ với production ensemble; chưa đủ baseline/ablation chính thức để khẳng định ưu thế phương pháp'],
        ['Tính đúng đắn code', f'Đã sửa các lỗi nguồn, ghi đè, chunking, embedding và final gate; {checks["unit_tests_passed"]} kiểm tra đạt', 'Tests không thay thế đánh giá ngữ nghĩa bằng nhãn người'],
        ['Khả năng chạy lại', 'Có entry point, dependency groups, môi trường thực tế, manifests và ví dụ', 'Chưa có kiểm thử cài mới trên máy sạch; requirements chưa phải lock đa nền tảng'],
        ['Toàn bộ PDF/report/retrieval', 'Markdown → ensemble → báo cáo đã chạy thật; PDF adapter có kiểm tra giả lập', 'Chưa chạy PDF reader thật trong môi trường hiện tại và chưa benchmark retrieval với nhãn người'],
        ['Sẵn sàng nghiệm thu', 'Có thể trình diễn prototype và báo cáo trung thực những gì đã làm', 'Chưa được gọi là hệ thống hoàn thiện hoặc model đã chứng minh tốt hơn bản triển khai']])
    text += '\n\n## 2. Số liệu classifier có thể bảo vệ trước hội đồng\n\n'
    text += table(['Seed', 'Validation Macro-F1'], [[r['seed'], f"{r['macro_f1']:.6f}"] for r in stats['runs']])
    text += f'\n\nTrung bình **{stats["mean"]:.6f} ± {stats["sample_std"]:.6f}** (độ lệch chuẩn mẫu, n=3). Cùng 4.144 dòng train và 1.317 câu validation/29 tài liệu. Metric được kiểm lại từ predictions, per-class JSON và confusion CSV. Validation đã được sử dụng trong phát triển, không phải final test.\n\n'
    classes = pd.read_csv(study / 'analysis/per_class_comparison.csv')
    text += table(['Seed', 'Nhãn', 'Precision', 'Recall', 'F1', 'Support'], [[int(r['seed']), r['label'], f'{r["precision"]:.4f}', f'{r["recall"]:.4f}', f'{r["f1-score"]:.4f}', int(r['support'])] for _, r in classes.iterrows()])
    text += '\n\nCó 165 câu seed 2026 sai nhưng seed 42 hoặc 123 đúng theo nhãn hiện có; 73 câu cả hai seed kia đều đúng. Seed 2026 yếu nhất ở REQUIREMENT, nhưng nguyên nhân không chỉ là nhầm với ACTIVITY. EXPECTED_RESULT yếu ở cả ba seed; OTHER false positives cũng cần phân tích theo vai trò đoạn. Những quan sát này chưa xác nhận từng nhãn tham chiếu đều đúng.\n\n'
    text += 'Bằng chứng chi tiết: ' + link(study / 'analysis/PHAN_TICH_3_SEED.md') + ' và ' + link(study / 'analysis/CAC_CAU_SEED_2026_SAI.md') + '.\n\n'
    text += '## 3. Những vấn đề khoa học phải xử lý trước khi chốt model\n\n'
    text += '**Section và nguồn nhãn.** Có 1.045/1.317 câu thiếu Section. Trong số này 1.019 có heading ứng viên nhất quán khi truy Markdown, 25 cần xác định vị trí, 1 nằm trước heading đầu tiên. Có 35 câu PSAI133 chịu vấn đề stack phân cấp heading; 81 nhãn annotation hiện tại khác nhãn frozen. Không tự đồng bộ nhãn hoặc hồi tố thay context vào các số liệu cũ.\n\n'
    text += '**Nhãn AI.** 57 thay đổi có hiệu lực trong biến thể train sửa nhãn vẫn là đề xuất AI, không phải ground truth. Cả ba seed phân tích ở trên đều dùng nhãn gốc. Đã có phiếu 91 câu cho người phân xử, chưa có kết quả người mới được áp dụng. Một trường metadata label_origin ở seed42 gây hiểu nhầm đã được giải thích bằng sidecar, không sửa provenance lịch sử. Xem ' + link(DEST / 'provenance_corrections.json') + '.\n\n'
    text += '**Final test chưa độc lập.** Cả 8 tài liệu ứng viên có trong pseudo_candidates lịch sử: tổng 515 dòng, 386 dòng khớp text ứng viên hiện tại; 6 tài liệu còn có 9 câu khớp trong extra_labeled3. Đây là dấu vết từng được khai thác/gán nhãn trong quá trình phát triển, chưa chứng minh mọi model đã train trên chúng. Không đủ điều kiện gọi là tập hoàn toàn chưa sử dụng. CSV label_queue3 còn lỗi parser; phải sửa từ nguồn gốc có kiểm chứng hoặc có bản lưu thay thế, không bỏ qua dòng lỗi. Chi tiết trong ' + link(DEST / 'research_audit.md') + '.\n\n'
    text += '**So sánh với ensemble.** Code huấn luyện production hiện có dùng toàn corpus gốc; lịch sử chính xác từng weights chưa có đầy đủ. Do đó chấm ensemble trên validation mới có nguy cơ là dữ liệu nó đã thấy, không đủ tạo phép so sánh công bằng. Seed123 là ứng viên validation tốt nhất trong nhóm mới, chưa phải model cuối cùng hoặc bản tốt hơn ensemble.\n\n'
    text += '**Giao thức số học.** Trainer bật BF16 trong khi điểm sau export được tính qua đường FP32. Chưa lưu logits Trainer và một số flag TF32/determinism, không thể quy toàn bộ chênh lệch cho một yếu tố hoặc cam kết tái lập bitwise. Đã có manifest giao thức đề xuất; chưa chốt model và chưa thử điểm các epoch chưa được đánh giá.\n\n'
    text += 'Cơ chế chốt mới yêu cầu manifest gắn với chính lượt validation, gồm bias, thứ tự nhãn, tokenizer/config/code và cờ runtime. Các điểm lịch sử thiếu manifest này chưa đủ điều kiện đóng băng model. Công cụ ' + link(ROOT / 'replay_validation.py') + ' có thể đánh giá lại đúng validation frozen và tạo kết quả mới, giữ nguyên bằng chứng cũ; chưa chạy replay trên model thật trong đợt này. Kiểm tra nguồn final cũng buộc ID và từng span khớp đúng tài liệu, không chỉ kiểm tra hash của một file bất kỳ.\n\n'
    text += '## 4. Các sửa đổi kỹ thuật đã thực hiện\n\n'
    fixes = [
        ['Entry point và đầu ra', 'psai.py; README; ví dụ và dependency groups', 'Một lệnh kiểm tra môi trường, một lệnh phân tích tài liệu; đầu ra mới gồm report/source/extraction/plan/manifest, từ chối ghi đè'],
        ['PDF', 'pdf_preprocessing.py; ocr.py', 'Backend selectable text có metadata trang/hash, chặn PDF rỗng/scan hoặc mã hóa; Marker/VLM import khi dùng; VLM mặc định tắt và mô tả AI ở sidecar'],
        ['Trích xuất', 'extraction.py', 'Sửa stack heading/fence, offset và validation prediction; confidence được giữ cho deliverable suy từ model; metadata context có version'],
        ['Kế hoạch', 'planning.py', 'Kiểm tra hash/quote/value/span và ID; nội dung bước phải được bằng chứng hỗ trợ; loại công việc đã hoàn thành; phân biệt kiểm tra record với kiểm tra nguồn thật'],
        ['Batch extraction', 'run_extraction.py', 'Import không tự chạy; chỉ công bố output sau khi toàn batch thành công; output cũ còn nguyên khi có lỗi'],
        ['Cleaning/chunking', 'cleaning.py; chunking.py', 'Bỏ xóa toàn bộ thư mục mặc định; giữ câu ngắn/code fence/bảng; chia Unicode không mất ký tự; thêm hash và phạm vi dòng nguồn'],
        ['Embedding/retrieval', 'embedding.py; vector_db.py', 'Nạp dependency khi dùng, kiểm tra provenance/model/dimension/vector; upsert trước khi bỏ IDs cũ; retrieval theo tài liệu; từ chối dữ liệu không khớp'],
        ['Serving classifier', 'ensemble_classifier.py; sentence_classifier.py', 'Nạp file local, chặn logits/probabilities sai shape hoặc không hữu hạn; giữ nguyên ba checkpoint production'],
        ['Final gate', 'study_final_test.py; final_test_integrity.py', 'Ràng buộc validation/provenance/config, model/tokenizer/code/source hashes; sàng lọc cả pseudo/review/tuning artifacts; khóa attempt độc quyền và giữ identity khi hoàn tất/lỗi'],
        ['Tài liệu', 'README; báo cáo hiện tại; ghi chú báo cáo lịch sử', 'Các báo cáo cũ được đánh dấu lịch sử, giữ bản gốc; báo cáo mới tách rõ thực nghiệm, phần mềm và việc chưa xác minh']]
    text += table(['Nhánh', 'File', 'Kết quả thay đổi'], fixes)
    text += '\n\nSửa stack làm thay đổi context được dựng cho các lượt trích xuất mới. Phiên bản mới là source-context-v2-heading-stack-fences; không dùng metric cũ để tuyên bố chất lượng của context mới. Các ngưỡng lọc cũ được giữ và đánh dấu legacy, chưa tối ưu/chứng nhận lại.\n\n'
    text += '## 5. Kiểm chứng thực tế\n\n'
    demo = ROOT / 'outputs/project_review_demo'
    extraction = read_json(demo / 'extraction.json')
    plan = read_json(demo / 'plan.json')
    text += table(['Kiểm tra', 'Kết quả', 'Giới hạn'], [
        ['Unit/integration regression', f'{checks["unit_tests_passed"]}/{checks["unit_tests_passed"]} đạt', 'Bao gồm vector DB/embedding được mock và reader PDF giả lập; không phải chất lượng ngữ nghĩa toàn hệ thống'],
        ['Source spans', f'{checks["source_documents_checked"]} tài liệu; {checks["source_sentences_checked"]} câu; {checks["invalid_source_spans"]} offset sai', 'Đối chiếu Markdown cleaned, không phải PDF gốc'],
        ['Luồng ensemble thật', f'{sum(len(v) for v in extraction["fields"].values())} mục trích xuất theo field; {len(plan["steps"])} bước từ nguồn; grounding=source_hash_and_spans', 'Một ví dụ tự biên soạn; một câu có thể nằm nhiều field; không suy ra precision/recall hoặc F1'],
        ['Môi trường', 'Đã lưu Python và các package thực tế', 'Chưa cài lại trên máy sạch'],
        ['PDF thật', 'Chưa chạy trong môi trường này vì thiếu pypdf/Marker', 'Adapter chỉ kiểm tra bằng reader giả lập; không ghi “PDF end-to-end đã đạt”'],
        ['Retrieval với model/DB thật', 'Chưa chạy live BGE-M3/Chroma sau sửa', 'Tests kiểm tra logic bằng mock; DB nhiều batch chưa có transaction toàn phần']])
    text += '\n\nBằng chứng chức năng: ' + link(demo / 'report.md') + ', ' + link(demo / 'manifest.json') + ', ' + link(DEST / 'verification.json') + '. Không có điểm pipeline hay tỷ lệ hallucination được bịa ra. Một báo cáo trích nguyên văn vẫn có thể bỏ sót ý, phân loại nhầm hoặc chọn không đúng trọng tâm; cần nhãn người để đo.\n\n'
    text += '## 6. Cách chạy và bàn giao\n\n'
    text += 'Từ thư mục project:\n\n```powershell\npython psai.py doctor\npython -m unittest discover -p "test_*.py"\npython psai.py analyze --input examples/proposal.md --output outputs/demo_new --classifier rules\npython psai.py analyze --input examples/proposal.md --output outputs/demo_ensemble_new\n```\n\n'
    text += 'Dùng thư mục output mới mỗi lần. Đọc ' + link(ROOT / 'README.md') + ' để chọn dependency PDF/retrieval. Chạy rules không cần checkpoint, nhưng phải ghi rõ đây là quy tắc. Chạy ensemble cần ba bộ weights/tokenizer/config hiện có. Khi bàn giao model hoặc dữ liệu, dùng manifest và giữ split/version; không chỉ gửi file code rồi cho rằng người nhận tái lập được môi trường GPU.\n\n'
    text += '## 7. Các việc còn phải hoàn thành theo thứ tự\n\n'
    text += '1. Người kiểm định phân xử nhãn, ghi lý do và nguồn; giải quyết 81 lệch phiên bản và chấp nhận/từ chối từng đề xuất AI. Không sửa validation rồi gọi điểm mới là kết quả độc lập.\n2. Rà PDF gốc cho mẫu đại diện các lỗi, nhất là bảng/hình/thứ tự đọc; xác nhận bản trích text và heading. Tách riêng mô tả AI chưa được người kiểm tra.\n3. Chuẩn bị tập cuối thực sự chưa tham gia train/tuning/mining theo lịch sử có thể kiểm tra. 8 ứng viên hiện tại phải tiếp tục ở trạng thái chưa đủ điều kiện; có thể dùng cho phân tích phát triển nếu ghi rõ.\n4. Sau kiểm định mới quyết định version dữ liệu/context và giao thức inference; chọn checkpoint bằng validation, lưu đầy đủ weights/tokenizer/config/hash/flags. Chưa có thí nghiệm train mới được chạy trong đợt này.\n5. Đo extraction precision/recall, coverage so với PDF, mức đúng của nội dung báo cáo và retrieval recall theo câu hỏi người gán nhãn. Chạy live retrieval trên collection phiên bản mới trước phát hành.\n6. Chỉ mở final evaluation sau khi model và giao thức đã đóng băng; báo cáo cả kết quả không thuận lợi và giới hạn.\n\n'
    text += '## 8. Cách trình bày với giảng viên\n\n'
    text += f'“Em đã xây dựng và chạy được prototype phân tích đề xuất có trích dẫn nguồn. Em đã kiểm tra ba seed trên cùng validation, báo cáo đầy đủ trung bình/độ lệch chuẩn, và tìm ra vấn đề cấu trúc context cùng các nhãn cần phân xử. Sau rà soát, em sửa lỗi ghi đè dữ liệu, kiểm tra source span, tính nhất quán của kế hoạch, chunk/vector provenance và cơ chế khóa final test. Bộ kiểm tra hiện đạt {checks["unit_tests_passed"]} ca. Em chưa khẳng định model mới tốt hơn ensemble cũ; tập cuối hiện có dấu vết phát triển nên em chưa dùng nó để báo cáo đánh giá độc lập. Phần tiếp theo cần người kiểm định nhãn và PDF, rồi mới chốt model và đánh giá cuối.”\n\n'
    text += 'Đây là cách bảo vệ dự án bằng bằng chứng và giới hạn thực tế. Không nên nói “đã hoàn thiện toàn bộ”, “nhãn AI đã sửa đúng” hoặc “PDF/report đã đạt chất lượng” khi chưa có kiểm định tương ứng.\n'
    md = DEST / 'DANH_GIA_VA_HOAN_THIEN_PROJECT.md'
    md.write_text(text, encoding='utf-8')
    export_docx(text, DEST / 'DANH_GIA_VA_HOAN_THIEN_PROJECT.docx')
    notice = '> Tài liệu lịch sử: số liệu và hướng dẫn bên dưới phản ánh thời điểm biên soạn. Trạng thái hiện tại và giới hạn đã kiểm chứng nằm trong [báo cáo rà soát toàn project](project_review/DANH_GIA_VA_HOAN_THIEN_PROJECT.md) và [README](../README.md).\n\n'
    for backup in (DEST / 'historical_reports').glob('*.md'):
        target = ROOT / 'reports' / backup.name
        target.write_text(notice + backup.read_text(encoding='utf-8'), encoding='utf-8')
    files = ['psai.py', 'pdf_preprocessing.py', 'ocr.py', 'extraction.py', 'planning.py', 'run_extraction.py',
             'cleaning.py', 'chunking.py', 'embedding.py', 'vector_db.py', 'ensemble_classifier.py',
             'sentence_classifier.py', 'study_final_test.py', 'final_test_integrity.py', 'replay_validation.py',
             'test_project_runtime.py', 'test_pipeline_quality.py', 'test_retrieval_quality.py', 'README.md']
    write_json(DEST / 'changed_source_manifest.json', {name: sha(ROOT / name) for name in files})
    print(md)


if __name__ == '__main__':
    main()
