"""Prepare a blinded review sample from a new, locally cached external source.

No labels are inferred, no classifier is loaded and no test score is computed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from extraction import sentences_with_context


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, output: Path, sample_size=80, seed=2027):
    if output.exists():
        raise FileExistsError('Never replace an assigned review packet')
    text = source.read_text(encoding='utf-8-sig')
    candidates = list(sentences_with_context(text))
    if sample_size < 1 or len(candidates) < sample_size:
        raise ValueError('Invalid sample size')
    indices = sorted(random.Random(seed).sample(range(len(candidates)), sample_size))
    rows = []
    for index in indices:
        item = candidates[index]
        if text[item['start']:item['end']] != item['text']:
            raise ValueError('Source span mismatch')
        rows.append(dict(item_id=hashlib.sha256(f'{sha(source)}:{index}:{item["start"]}'.encode()).hexdigest(),
            document_id='EXT_GSOC2025_FPM_EMMA', source_sentence_index=index, text=item['text'],
            section=item['section'], prev_text=item['prev'], next_text=item['next_text'], pos=item['pos'],
            char_start=item['start'], char_end=item['end'], label='', reviewer='', review_reason=''))
    output.mkdir(parents=True)
    with (output / 'reference_to_complete.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    immutable = [{key: value for key, value in row.items() if key not in ['label', 'reviewer', 'review_reason']} for row in rows]
    content_hash = hashlib.sha256(json.dumps(immutable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest = dict(source_path=str(source.resolve()), source_sha256=sha(source), sentence_candidates=len(candidates),
        sampled_sentences=len(rows), sampling='uniform without replacement; sorted back to source order', seed=seed,
        source_document_count=1, content_sha256=content_hash,
        source_review_complete=False, human_annotation_complete=False, approved_for_final_test=False,
        model_inference_performed=False, test_scored=False,
        limitation='One-source pilot review packet, not a complete independent final test set.')
    (output / 'review_gate.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    page = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Phiếu nguồn mới — classifier</title>
<style>body{font:16px system-ui;max-width:1000px;margin:25px auto;padding:0 20px;color:#183048}header{position:sticky;top:0;background:white;padding:10px;border-bottom:1px solid #aaa}article{padding:18px;background:#f2f6fa;margin:15px 0}p{white-space:pre-wrap}input,textarea,select,button{font:inherit;padding:8px;margin:5px}textarea{width:95%;min-height:55px}.sentence{font-weight:bold}</style>
<h1>Phiếu kiểm định classifier trên nguồn mới</h1>
<p>Đây là một tài liệu ứng viên mới, chưa được phê duyệt là final test. Mẫu được chọn ngẫu nhiên trước khi chạy classifier; không có dự đoán model trong phiếu. Không dùng kết quả phiếu để chọn tham số rồi gọi đó là test độc lập.</p>
<p>OBJECTIVE: mục tiêu tổng thể; REQUIREMENT: điều kiện/ràng buộc/tiêu chí bắt buộc; ACTIVITY: việc thực hiện; EXPECTED_RESULT: sản phẩm/trạng thái/kết quả dự kiến; OTHER: phần còn lại. Đọc context và nguồn đầy đủ khi câu mơ hồ; có thể để chưa quyết định và ghi lý do.</p>
<p>Nguồn gốc: <a href="https://gist.github.com/emmabastas/80b016dacbc2cafc5c5f3b8117eaeb17">đề xuất fpm của tác giả</a>. Nhận xét người gán chưa được phần mềm chứng nhận đúng hoặc độc lập; việc duyệt final test cần bước riêng.</p>
<header><label>Người gán <input id="reviewer"></label><strong id="count"></strong><br><button id="save">Tải CSV đang làm</button><button id="complete">Tải CSV đã gán đủ</button></header><main id="items"></main>
<script>
const rows=__ROWS__,labels=['OBJECTIVE','REQUIREMENT','ACTIVITY','EXPECTED_RESULT','OTHER'],key='psai_external___HASH__';let state={reviewer:'',answers:{}};
try{const saved=JSON.parse(localStorage.getItem(key));if(saved)state=saved;}catch(e){}
const reviewer=document.getElementById('reviewer');reviewer.value=state.reviewer||'';
function persist(){state.reviewer=reviewer.value;try{localStorage.setItem(key,JSON.stringify(state));}catch(e){}document.getElementById('count').textContent=rows.filter(r=>labels.includes((state.answers[r.item_id]||{}).label)).length+'/'+rows.length;}
reviewer.oninput=persist;
rows.forEach((r,i)=>{const card=document.createElement('article');const title=document.createElement('h2');title.textContent='Câu '+(i+1);card.append(title);for(const [name,field] of [['Câu','text'],['Section','section'],['Trước','prev_text'],['Sau','next_text']]){const p=document.createElement('p');p.textContent=name+': '+(r[field]||'(trống)');if(field==='text')p.className='sentence';card.append(p);}const select=document.createElement('select');for(const label of ['',...labels]){const op=document.createElement('option');op.value=label;op.textContent=label||'Chưa quyết định';select.append(op);}const reason=document.createElement('textarea');reason.placeholder='Lý do/nhãn thay thế nếu còn tranh cãi';const old=state.answers[r.item_id]||{};select.value=old.label||'';reason.value=old.review_reason||'';function update(){state.answers[r.item_id]={label:select.value,review_reason:reason.value};persist();}select.onchange=reason.oninput=update;card.append(select,reason);document.getElementById('items').append(card);});
function download(complete){persist();if(complete&&(!state.reviewer.trim()||rows.some(r=>{const a=state.answers[r.item_id]||{};return !labels.includes(a.label)||!a.review_reason?.trim();}))){alert('Cần tên và đủ nhãn/lý do. Có thể tải bản đang làm.');return;}const columns=Object.keys(rows[0]),quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const content=[columns.map(quote).join(','),...rows.map(r=>{const value={...r,...(state.answers[r.item_id]||{}),reviewer:state.reviewer};return columns.map(c=>quote(value[c])).join(',');})].join('\r\n');const url=URL.createObjectURL(new Blob(['\ufeff'+content],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='external_reference_'+(complete?'completed':'progress')+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.getElementById('save').onclick=()=>download(false);document.getElementById('complete').onclick=()=>download(true);persist();
</script></html>'''
    page = page.replace('__ROWS__', json.dumps(rows, ensure_ascii=False).replace('<', r'\u003c')).replace('__HASH__', content_hash)
    # In the raw template, use real JavaScript CSV escapes, not literal backslashes.
    page = page.replace(r".join('\\r\\n')", r".join('\r\n')").replace(r"['\\ufeff'+content]", r"['\ufeff'+content]")
    (output / 'GAN_NHAN_NGUON_MOI.html').write_text(page, encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.sample_size), indent=2))
