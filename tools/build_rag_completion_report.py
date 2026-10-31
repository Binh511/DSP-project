"""Build the RAG handoff report from existing measurements, without inference."""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'reports/rag_studies/retrieval_genq_lora_v1'
QUALITY = ROOT / 'psai_rag_quality/artifacts/retrieval_genq_lora_v1'
RELEASE = ROOT / 'reports/rag_release_v1'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]


def html_document(markdown):
    """Small escaped report renderer; measurements and source snippets are text."""
    output, table, bullets = [], False, False
    for line in markdown.splitlines():
        if not line.startswith('|') and table:
            output.append('</tbody></table>'); table = False
        if not line.startswith('- ') and bullets:
            output.append('</ul>'); bullets = False
        if line.startswith('|'):
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if all(set(cell) <= set('-: ') for cell in cells):
                continue
            if not table:
                output.append('<table><tbody>'); table = True
            output.append('<tr>' + ''.join('<td>' + escape(cell) + '</td>' for cell in cells) + '</tr>')
        elif line.startswith('#'):
            count = len(line) - len(line.lstrip('#'))
            output.append(f'<h{count}>' + escape(line[count:].strip()) + f'</h{count}>')
        elif line.startswith('- '):
            if not bullets: output.append('<ul>'); bullets = True
            output.append('<li>' + escape(line[2:]) + '</li>')
        elif line.startswith('> '):
            output.append('<blockquote>' + escape(line[2:]) + '</blockquote>')
        elif line.strip():
            output.append('<p>' + escape(line) + '</p>')
    if table: output.append('</tbody></table>')
    if bullets: output.append('</ul>')
    return ('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Báo cáo nhánh RAG PSAI</title><style>body{font:16px/1.6 Segoe UI,Arial;max-width:1120px;margin:32px auto;padding:0 24px;color:#193047}'
            'h1,h2{color:#105d7e}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td{border:1px solid #cad6df;padding:8px}'
            'tr:first-child{background:#e9f2f8;font-weight:bold}blockquote{background:#f4f7fa;border-left:4px solid #297fa1;margin:12px 0;padding:12px;white-space:pre-wrap}'
            'p,li{overflow-wrap:anywhere}@media print{body{max-width:none;margin:0}table{font-size:11px}}</style><body>'
            + ''.join(output) + '</body></html>')


def main():
    summary = load(QUALITY / 'paired_analysis.json')
    training = load(ROOT / 'models/rag/retrieval_genq_lora_v1/training_receipt.json')
    split = load(STUDY / 'split_manifest.json')
    generation = load(STUDY / 'generation_report.json')
    pair_map = {row['query_id']: row for row in rows(STUDY / 'validation_pairs.jsonl')}
    corpus = {row['chunk_id']: row for row in rows(STUDY / 'corpus.jsonl')}
    with (QUALITY / 'paired_queries.csv').open(encoding='utf-8-sig', newline='') as handle:
        changed = [row for row in csv.DictReader(handle)
                   if row['scope'] == 'within_doc' and row['method'] == 'dense' and row['outcome'] != 'tie']
    packet = load(QUALITY / 'priority_review_packet/packet_manifest.json')
    verification = load(RELEASE / 'verification.json') if (RELEASE / 'verification.json').exists() else {}
    primary = summary['summary']['within_doc/dense']
    ci = primary['mrr_delta_bootstrap']
    report = ['# Báo cáo hoàn thiện nhánh RAG — PSAI', '',
        'Bản runtime đã nối vào luồng phân tích tài liệu và xuất báo cáo có nguồn. Lượt train LoRA đã hoàn thành; '
        'chưa có bằng chứng đủ mạnh để khẳng định tốt hơn BGE-M3 gốc trên truy vấn thực tế.', '',
        '## 1. Phần đã thực hiện', '',
        '- Chuẩn bị dữ liệu theo nhóm tài liệu trùng/gần trùng, giữ hash nguồn và tách train/validation/test trước sinh query.',
        '- Sinh query bằng T5-small, lấy negative BM25 trong cùng tài liệu, lọc bằng cross-encoder; toàn bộ là nhãn tự động.',
        '- Train BGE-M3 dense + LoRA rank8, một epoch, seed42; CachedMNRL batch16/mini1, BF16 encoder và logits FP32.',
        '- Chạy baseline BM25, dense và RRF; đánh giá base/adapter trên cùng validation và phân tích sai số theo nhóm nguồn.',
        '- Hoàn thiện runtime index/search: nhiều truy vấn, giới hạn tài liệu, tài nguyên tham khảo riêng, hybrid/RRF và reranker tùy chọn.',
        '- Giữ quote nguyên văn và char spans; heading được thêm vào embedding riêng; kiểm tra hash model/tokenizer/adapter/index và nguồn.',
        '- Tích hợp Markdown/text/PDF có text → trích nội dung → truy xuất → báo cáo HTML/Markdown/JSON. HTML nhúng nguồn để mở độc lập.', '',
        'Luồng: PDF/text → văn bản + hash → chia đoạn theo cấu trúc → BM25/dense → RRF/rerank tùy chọn → hợp nhất → '
        'đối chiếu nguồn → nội dung và báo cáo có dẫn chứng. Luồng LLM lập kế hoạch không được tính là đã hoàn thành trong nhánh RAG này.', '',
        '## 2. Dữ liệu và lượt huấn luyện', '',
        f"{split['counts']['source_documents']} tài liệu ban đầu, {split['counts']['source_groups']} nhóm nguồn; "
        f"{split['counts']['corpus_chunks']} đoạn theo tokenizer BGE-M3.", '',
        '| Phần | Nhóm nguồn | Đoạn | Cặp/query sau lọc |', '|---|---:|---:|---:|']
    for name in ['train','validation','test']:
        part = split['splits'][name]
        count = generation['coverage'].get(name, {}).get('selected_pair_count', 0)
        report.append(f"| {name} | {part['groups']} | {part['chunks']} | {count} |")
    report += ['', f"Train hoàn thành {training['global_step']}/{training['planned_steps']} bước; "
                   f"{training['parameters']['trainable']:,} tham số trainable; {training['training_seconds']:.1f} giây. "
                   'Checkpoint có commit/hash và bằng chứng gradient khác 0, trọng số được cập nhật.', '',
        '## 3. Kết quả validation thực tế', '',
        '54 query tự sinh trên 22 nhóm nguồn. Positive là đoạn dùng để sinh query; không phải nhãn relevance do người kiểm định. '
        'Các số dưới đây thuộc dữ liệu study đã đóng băng, không phải điểm chất lượng end-to-end của runtime.', '',
        '| Phạm vi | Cách truy xuất | Phiên bản | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for key, result in sorted(summary['summary'].items()):
        for variant in ['base','adapter']:
            if result['method'] == 'bm25' and variant == 'adapter': continue
            values = result[variant]
            report.append('| ' + ' | '.join([result['scope'], result['method'], variant]
                + [f'{values[name]:.4f}' for name in ['Recall@1','Recall@5','Recall@10','MRR@10','nDCG@10']]) + ' |')
    report += ['', '## 4. Độ chắc chắn của mức cải thiện', '',
        'Bootstrap ghép cặp 10.000 lần, lấy mẫu lại toàn nhóm nguồn để giữ các query cùng tài liệu đi cùng nhau. '
        'Khoảng tin cậy chỉ phản ánh biến thiên lấy mẫu trên tập tự sinh này, chưa bao gồm lỗi nhãn hoặc biến thiên seed train.', '',
        '| Phạm vi/phương pháp | Delta MRR | CI95% thấp | CI95% cao | Thắng/thua/hòa |', '|---|---:|---:|---:|---|']
    for key, result in sorted(summary['summary'].items()):
        interval = result['mrr_delta_bootstrap']
        report.append(f"| {key} | {interval['estimate']:+.6f} | {interval['ci95_low']:+.6f} | "
                      f"{interval['ci95_high']:+.6f} | {result['wins']}/{result['losses']}/{result['ties']} |")
    report += ['', f"Tiêu chí chính dense/within_doc tăng {ci['estimate']:+.6f}, nhưng CI95% "
                   f"[{ci['ci95_low']:+.6f}; {ci['ci95_high']:+.6f}] chứa 0. Không đủ bằng chứng kết luận cải thiện ổn định; "
                   'cũng không chứng minh hai mô hình tương đương. Default runtime dense giữ base; adapter được bật có chủ đích để nghiên cứu.', '',
        '## 5. Các query đổi thứ hạng', '',
        'Đây là thay đổi theo nhãn tự động hiện có. Cần người kiểm tra xem query có hợp lý và positive có thật sự liên quan.']
    for row in changed:
        query = pair_map[row['query_id']]
        passage = corpus[query['positive_id']]
        report += ['', f"### {row['doc_id']} — {row['outcome']}", '', query['query'], '',
            f"Query ID: {row['query_id']}; thứ hạng positive: {row['base_positive_rank'] or '>10'} → "
            f"{row['adapter_positive_rank'] or '>10'}; delta MRR {float(row['delta']):+.6f}.", '',
            f"Đoạn input được gắn positive tự động: {query['positive_id']}; nguồn bao quanh dòng "
            f"{passage['source_start_line']}–{passage['source_end_line']}. Text dưới có prefix heading của study; không phải quote PDF có char spans.", '',
            '> ' + query['positive'][:600].replace('\n', '\n> ') + ('…' if len(query['positive']) > 600 else '')]
    report += ['', '## 6. Giới hạn dữ liệu và điều kiện chốt', '',
        '- Teacher margin không chứng minh positive liên quan: 499/1.536 cặp train và 19/54 query validation có positive-logit âm; '
        'logit này không phải xác suất và không tự là nhãn sai.',
        '- Có query méo nghĩa và văn bản nguồn lỗi; lỗi nguồn đã được ghi tại DATA_SUPERVISION_AUDIT.md. '
        'Không tự sửa nhãn hoặc gọi các suy đoán này là ground truth.',
        '- Bộ sinh query/teacher hướng tiếng Anh; chưa có đánh giá truy xuất tiếng Việt do người kiểm định.',
        '- Runtime giữ nguyên khoảng trắng/char spans còn study có chuẩn hóa block và cửa sổ riêng. '
        'Dùng cùng quy ước prefix heading chưa biến điểm study thành chứng nhận chất lượng runtime.',
        '- Chưa đánh giá cuối trên tập query thực độc lập. Không dùng final test để chọn tham số; không thay ensemble classifier/Chroma production.', '',
        '## 7. Kiểm tra chức năng và bàn giao', '',
        f"Unittest discovery đã kiểm tra: {verification.get('unit_test_count', 'xem verification.json')} test. "
        'Các test fixture kiểm tra code, không phải phép đánh giá chất lượng nghiên cứu.',
        '- Demo PDF tự tạo hai trang: kiểm tra đủ câu mẫu, hash/page spans, toàn bộ quote và link HTML; PDF trắng yêu cầu OCR.',
        '- Demo model thật: hybrid + LoRA + MiniLM reranker trả về bằng chứng từ tài liệu và tài nguyên riêng, có kiểm tra nguồn.',
        '- Báo cáo HTML chứa văn bản nguồn, mở không cần project gốc. Nguồn tài nguyên trùng text vẫn giữ riêng origin và snapshot.',
        '- Index mới độc lập với Chroma cũ; thay nguồn/model/tokenizer/adapter hoặc artifact sai hash thì bị chặn.', '',
        '## 8. Phần cần người hoàn tất', '',
        f"Packet ưu tiên có {packet['queries']} query và {packet['items']} cặp query–đoạn, được chọn để chẩn đoán; "
        'không đại diện toàn benchmark. Hai người chấm độc lập và phân xử bất đồng; phần chưa rõ giữ là chưa biết.',
        'Sau đó cần query thực tế và relevance trên tài liệu giữ lại để đánh giá mô hình đã đóng băng. '
        'Đây là công việc còn thiếu về bằng chứng nghiên cứu, không thể hoàn tất trung thực bằng cách tự sinh thêm nhãn AI.', '',
        '## 9. Mở tài liệu và chạy demo', '',
        'Chạy tại gốc project: python tools/demo_rag.py',
        'Model thật: python tools/demo_rag.py --mode hybrid --device cuda',
        'Adapter thử nghiệm: python tools/demo_rag.py --mode hybrid --use-adapter --rerank --device cuda',
        'Kiểm tra release: python tools/finalize_rag_release.py --verify-only', '',
        'Các bằng chứng máy đọc: [paired analysis](../../psai_rag_quality/artifacts/retrieval_genq_lora_v1/paired_analysis.json), '
        '[review ưu tiên](../../psai_rag_quality/artifacts/retrieval_genq_lora_v1/priority_review_packet/REVIEW.html), '
        '[kết quả train](../rag_studies/retrieval_genq_lora_v1/KET_QUA_RAG_PILOT.md), '
        '[manifest release](./RELEASE_MANIFEST.json).', '']
    RELEASE.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(report)
    (RELEASE / 'BAO_CAO_RAG.md').write_text(text, encoding='utf-8')
    (RELEASE / 'BAO_CAO_RAG.html').write_text(html_document(text), encoding='utf-8')
    (STUDY / 'QUALITY_AND_ERROR_ANALYSIS.md').write_text(
        '# Kiểm tra chất lượng RAG sau pilot\n\nPhân tích đã được tích hợp, với bảng số liệu và ví dụ trực tiếp, tại '
        '[báo cáo RAG đầy đủ](../../rag_release_v1/BAO_CAO_RAG.md).\n\n'
        f"Dense within-doc: delta MRR {ci['estimate']:+.6f}; CI95% [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}]; "
        f"{primary['wins']} thắng/{primary['losses']} thua/{primary['ties']} hòa. Chưa đủ bằng chứng cải thiện ổn định.\n",
        encoding='utf-8')
    print(RELEASE / 'BAO_CAO_RAG.html')


if __name__ == '__main__':
    main()
