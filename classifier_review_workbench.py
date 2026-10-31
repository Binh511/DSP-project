"""Prepare a human review workbench without applying AI or human labels."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from study_data import OUT, read_json, write_json, sha

STUDY = OUT / 'studies/classifier_stability_v1'
DEST = STUDY / 'human_review'


def main():
    source = STUDY / 'analysis'
    regressions = pd.read_csv(source / 'seed_2026_regressions.csv', keep_default_na=False)
    candidates = regressions[regressions.true_label.isin(['REQUIREMENT', 'EXPECTED_RESULT'])].copy()
    candidates['review_scope'] = 'validation_disagreement'
    candidates['reference_label'] = candidates.true_label
    proposals = pd.read_csv(source / 'ai_57_effective_proposals.csv', keep_default_na=False)
    proposals['review_scope'] = 'training_ai_proposal'
    proposals['reference_label'] = proposals.original_label
    joined = pd.concat([candidates, proposals], ignore_index=True).fillna('')
    assert len(candidates) == 34 and len(proposals) == 57 and joined.item_id.nunique() == 91
    columns = ['item_id', 'review_scope', 'document_id', 'text', 'section', 'prev_text', 'next_text']
    public = joined[columns].copy()
    public['content_sha256'] = [hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest() for row in public.to_dict('records')]
    for field in ['reviewer', 'annotation', 'reason', 'source_checked', 'source_page']:
        public[field] = ''
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / 'PHAN_XU_91_CAU.csv'
    if target.exists():
        previous = pd.read_csv(target, keep_default_na=False)
        if not public[['item_id', 'content_sha256']].equals(previous[['item_id', 'content_sha256']]):
            raise ValueError('Review content changed; create a new version')
        public = previous
    else:
        public.to_csv(target, index=False, encoding='utf-8-sig')
    joined.to_csv(DEST / 'coordinator_key.csv', index=False, encoding='utf-8-sig')
    template = r'''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Phân xử nhãn classifier PSAI</title>
<style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 20px;color:#1c3145}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #abc}article{background:#f1f5fa;margin:18px 0;padding:18px;border-radius:10px}p{white-space:pre-wrap}input,textarea,select,button{font:inherit;padding:8px;margin:5px}textarea{width:95%;min-height:60px}button{cursor:pointer}.note{background:#fff3ce;padding:14px}.sentence{font-weight:600}summary{cursor:pointer}</style>
<h1>Phân xử nhãn classifier PSAI</h1>
<p>34 câu validation có nhãn tham chiếu REQUIREMENT hoặc EXPECTED_RESULT mà seed 2026 sai, ít nhất một seed khác đúng; và 57 đề xuất AI trong train. Đây là mẫu ưu tiên, không đại diện ngẫu nhiên toàn tập.</p>
<div class="note">Gán theo nội dung và ngữ cảnh trước khi xem nhãn gốc hoặc đề xuất AI. Phiếu này không hiển thị đáp án model. Nhãn được lưu vào phiếu, không tự sửa dataset hoặc điểm validation.<br>OBJECTIVE: mục tiêu tổng thể. REQUIREMENT: điều kiện/ràng buộc/tiêu chí bắt buộc. ACTIVITY: công việc sẽ thực hiện. EXPECTED_RESULT: sản phẩm/trạng thái/kết quả dự kiến. OTHER: nội dung còn lại.<br>Nếu câu có nhiều chức năng hoặc thiếu nguồn, để chưa quyết định và ghi lý do; không đoán nhãn. Chỉ xác nhận lỗi PDF sau khi mở nguồn và ghi trang.</div>
<header><label>Người kiểm tra <input id="reviewer"></label><strong id="counter"></strong><br><button id="download">Tải CSV đang làm</button><button id="complete">Tải CSV đã gán đủ</button><button id="backup">Tải bản sao JSON</button><label>Khôi phục JSON <input id="restore" type="file" accept=".json"></label><p>Phiếu tự lưu trong trình duyệt nếu được cho phép. Tải bản sao JSON trước khi chuyển máy hoặc trình duyệt.</p></header><main id="items"></main>
<script>
const rows=__ROWS__,labels=['OBJECTIVE','REQUIREMENT','ACTIVITY','EXPECTED_RESULT','OTHER'];
const storageKey='psai_human_review_91___HASH__';let state={reviewer:'',answers:{}};
try{const saved=JSON.parse(localStorage.getItem(storageKey));if(saved)state=saved;}catch(e){}
const reviewer=document.getElementById('reviewer');reviewer.value=state.reviewer||'';
function persist(){state.reviewer=reviewer.value;try{localStorage.setItem(storageKey,JSON.stringify(state));}catch(e){}document.getElementById('counter').textContent=rows.filter(r=>labels.includes((state.answers[r.item_id]||{}).annotation)).length+'/'+rows.length+' câu';}
reviewer.oninput=persist;
rows.forEach((r,i)=>{const card=document.createElement('article');const heading=document.createElement('h2');heading.textContent=(i+1)+'. '+r.document_id+' — '+(r.review_scope==='validation_disagreement'?'Validation':'Train');card.append(heading);const sentence=document.createElement('p');sentence.className='sentence';sentence.textContent=r.text;card.append(sentence);const details=document.createElement('details');details.open=true;const summary=document.createElement('summary');summary.textContent='Ngữ cảnh đang lưu trong dữ liệu';details.append(summary);for(const [label,field] of [['Section','section'],['Câu trước','prev_text'],['Câu sau','next_text']]){const p=document.createElement('p');p.textContent=label+': '+(r[field]||'(trống)');details.append(p);}card.append(details);const answer=state.answers[r.item_id]||{};const select=document.createElement('select');for(const label of ['',...labels]){const option=document.createElement('option');option.value=label;option.textContent=label||'Chưa quyết định';select.append(option);}select.value=answer.annotation||'';const reason=document.createElement('textarea');reason.placeholder='Lý do chọn nhãn, chức năng chính, nhãn thay thế nếu còn tranh cãi';reason.value=answer.reason||'';const source=document.createElement('select');for(const value of ['unchecked','yes','unavailable']){const option=document.createElement('option');option.value=value;option.textContent={unchecked:'Chưa đối chiếu nguồn',yes:'Đã đối chiếu nguồn',unavailable:'Chưa có nguồn để đối chiếu'}[value];source.append(option);}source.value=answer.source_checked||'unchecked';const page=document.createElement('input');page.placeholder='Tên nguồn, trang / dòng';page.value=answer.source_page||'';function update(){state.answers[r.item_id]={annotation:select.value,reason:reason.value,source_checked:source.value,source_page:page.value};persist();}select.onchange=source.onchange=reason.oninput=page.oninput=update;card.append(select,reason,source,page);document.getElementById('items').append(card);});
function download(text,name,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportCsv(complete){persist();if(complete&&(!state.reviewer.trim()||rows.some(r=>{const a=state.answers[r.item_id]||{};return !labels.includes(a.annotation)||!a.reason?.trim();}))){alert('Cần tên người kiểm tra và đủ nhãn/lý do. Có thể tải CSV đang làm nếu còn bất đồng.');return;}const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const columns=[...Object.keys(rows[0]),'reviewer','annotation','reason','source_checked','source_page'];const result=[columns.map(quote).join(','),...rows.map(r=>{const value={...r,...(state.answers[r.item_id]||{}),reviewer:state.reviewer};return columns.map(c=>quote(value[c])).join(',');})].join('\r\n');download('\ufeff'+result,'PHAN_XU_91_CAU_'+(complete?'completed':'progress')+'.csv','text/csv;charset=utf-8');}
document.getElementById('download').onclick=()=>exportCsv(false);document.getElementById('complete').onclick=()=>exportCsv(true);
document.getElementById('backup').onclick=()=>{persist();download(JSON.stringify({packet:storageKey,state},null,2),'PHAN_XU_91_CAU_backup.json','application/json');};
document.getElementById('restore').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;const saved=JSON.parse(await file.text());if(saved.packet!==storageKey||typeof saved.state?.reviewer!=='string'||!saved.state.answers)throw Error('Bản sao không thuộc phiếu này');const allowed=new Set(rows.map(r=>r.item_id));for(const [id,a] of Object.entries(saved.state.answers)){if(!allowed.has(id)||typeof a!=='object'||(a.annotation&&!labels.includes(a.annotation))||['annotation','reason','source_checked','source_page'].some(k=>a[k]!==undefined&&typeof a[k]!=='string'))throw Error('Bản sao có dữ liệu không hợp lệ');}if(!confirm('Khôi phục câu trả lời từ bản sao này?'))return;localStorage.setItem(storageKey,JSON.stringify(saved.state));location.reload();}catch(error){alert(error.message);}};persist();
</script></html>'''
    rows = public.drop(columns=['reviewer', 'annotation', 'reason', 'source_checked', 'source_page']).to_dict('records')
    payload = json.dumps(rows, ensure_ascii=False).replace('<', r'\u003c')
    packet_hash = hashlib.sha256(payload.encode()).hexdigest()
    html = template.replace('__ROWS__', payload).replace('__HASH__', packet_hash)
    (DEST / 'PHAN_XU_91_CAU.html').write_text(html, encoding='utf-8')
    write_json(DEST / 'manifest.json', dict(items=91, validation_items=34, training_items=57,
               source_sha256={name: sha(source / name) for name in ['seed_2026_regressions.csv', 'ai_57_effective_proposals.csv']},
               packet_sha256=packet_hash, human_completed=False, labels_applied=False,
               warning='One targeted adjudication packet; no inter-rater independence or Kappa claim'))
    print(DEST / 'PHAN_XU_91_CAU.html')


if __name__ == '__main__':
    main()
