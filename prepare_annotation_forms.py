"""Create offline, self-contained annotation forms without exposing gold labels."""
import csv
import json
from pathlib import Path
import zipfile

from study_data import OUT, sha
from build_classifier_report import export_docx


def main():
    guideline = (OUT / 'ANNOTATION_GUIDELINE.md').read_text(encoding='utf-8')
    export_docx(guideline, OUT / 'HUONG_DAN_KIEM_DINH_NHAN.docx')
    template = r'''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Kiểm định nhãn PSAI — __RATER__</title>
<style>body{font:16px system-ui;max-width:1040px;margin:30px auto;padding:0 18px;color:#183048}header{position:sticky;top:0;background:#fff;padding:12px 0;border-bottom:1px solid #abc;z-index:2}button,select,input,textarea{font:inherit;padding:8px;margin:4px 0}button{cursor:pointer}article{padding:16px;background:#f2f6fb;margin:12px 0;border-radius:10px}textarea{width:95%;min-height:50px}.sentence{white-space:pre-wrap;font-size:18px;font-weight:600}details{margin:10px 0}.ctx{white-space:pre-wrap;color:#42576b}#notice{color:#8d3200}label{display:block}summary{cursor:pointer}.help{background:#fff4d6;padding:14px}select{min-width:240px}</style>
<h1>Kiểm định nhãn — người __RATER__</h1>
<p>200 câu. Mỗi câu chọn một nhãn theo vai trò trong ngữ cảnh. Không có nhãn gốc hoặc dự đoán model trong phiếu này.</p>
<div class="help"><strong>Quy tắc nhãn:</strong><ul>
<li>OBJECTIVE: mục đích tổng thể dự án muốn đạt.</li>
<li>REQUIREMENT: điều kiện, ràng buộc hoặc tiêu chuẩn cần thoả mãn.</li>
<li>ACTIVITY: công việc hoặc bước dự kiến thực hiện.</li>
<li>EXPECTED_RESULT: sản phẩm, trạng thái hoặc lợi ích sau thực hiện.</li>
<li>OTHER: nội dung không thuộc bốn nhóm trên.</li></ul>
Đọc cả mục, câu trước và câu sau. Không quyết định chỉ bằng từ “will”, “must” hay “develop”. Với trường hợp khó, ghi lý do và nhãn có thể thay thế. Nếu chưa thể quyết định, để trống và lưu bản đang làm để người điều phối xem; chưa tính Kappa khi còn thiếu nhãn.
<p>Đọc hướng dẫn Word đi kèm trước khi bắt đầu. Hai người dùng cùng phiên bản hướng dẫn, gán độc lập, không xem khoá nhãn, dự đoán model hoặc kết quả của người kia. Người điều phối nên giải thích guideline bằng mẫu tập dượt riêng trước; không dùng 200 câu này để tập dượt chung.</p></div>
<header><label>Tên người gán <input id="reviewer" placeholder="Nhập họ tên"></label>
<label><input id="ack" type="checkbox"> Tôi đã đọc hướng dẫn và gán độc lập, không xem nhãn gốc hoặc nhãn người kia.</label>
<strong id="counter"></strong><br><button id="save">Tải CSV hoàn chỉnh</button> <button id="progress">Tải CSV đang làm</button> <label>Khôi phục từ CSV đã lưu <input id="restore" type="file" accept=".csv"></label><p id="notice">Nếu đóng trình duyệt hoặc đổi máy, hãy tải CSV đang làm. Lưu tự động trong trình duyệt có thể không hoạt động với file offline hoặc chế độ riêng tư.</p></header>
<main id="items"></main>
<script>
const rows=__ROWS__,labels=['OBJECTIVE','REQUIREMENT','ACTIVITY','EXPECTED_RESULT','OTHER'];
const key='psai_annotation___RATER_____HASH__';
const state={reviewer:'',ack:false,answers:{}};
try{const old=JSON.parse(localStorage.getItem(key));if(old)Object.assign(state,old);}catch(e){}
const reviewer=document.getElementById('reviewer'),ack=document.getElementById('ack');reviewer.value=state.reviewer;ack.checked=state.ack;
function save(){state.reviewer=reviewer.value;state.ack=ack.checked;try{localStorage.setItem(key,JSON.stringify(state));}catch(e){}document.getElementById('counter').textContent=rows.filter(r=>labels.includes((state.answers[r.item_id]||{}).annotation)).length+' / '+rows.length+' câu đã gán';}
reviewer.oninput=save;ack.onchange=save;
const host=document.getElementById('items');
rows.forEach((r,i)=>{const card=document.createElement('article');const title=document.createElement('p');title.textContent='Câu '+(i+1)+' — '+r.document_id+' — dòng '+r.source_line;card.append(title);const sentence=document.createElement('p');sentence.className='sentence';sentence.textContent=r.text;card.append(sentence);const details=document.createElement('details');details.open=true;const summary=document.createElement('summary');summary.textContent='Ngữ cảnh';details.append(summary);for(const [label,k] of [['Mục','section'],['Câu trước','prev_text'],['Câu sau','next_text']]){const p=document.createElement('p');p.className='ctx';p.textContent=label+': '+(r[k]||'(trống)');details.append(p);}card.append(details);const select=document.createElement('select');select.setAttribute('aria-label','Nhãn câu '+(i+1));for(const l of ['',...labels]){const op=document.createElement('option');op.value=l;op.textContent=l||'— Chưa chọn —';select.append(op);}const answer=state.answers[r.item_id]||{};select.value=answer.annotation||'';const reason=document.createElement('textarea');reason.placeholder='Lý do; nhãn có thể thay thế nếu còn phân vân';reason.setAttribute('aria-label','Lý do câu '+(i+1));reason.value=answer.reason||'';function update(){state.answers[r.item_id]={annotation:select.value,reason:reason.value};save();}select.onchange=update;reason.oninput=update;card.append(select,document.createElement('br'),reason);host.append(card);});
function quote(v){return '"'+String(v??'').replaceAll('"','""')+'"';}
function download(complete){save();const n=rows.filter(r=>labels.includes((state.answers[r.item_id]||{}).annotation)).length;if(complete&&(!state.reviewer.trim()||!state.ack||n!==rows.length)){alert('Cần họ tên, xác nhận độc lập và đủ '+rows.length+' nhãn hợp lệ. Có thể tải CSV đang làm nếu chưa xong.');return;}const cols=[...Object.keys(rows[0]),'reviewer','independence_ack'];const content=[cols.map(quote).join(','),...rows.map(r=>{const a=state.answers[r.item_id]||{};const record={...r,annotation:a.annotation||'',reason:a.reason||'',reviewer:state.reviewer,independence_ack:state.ack?'yes':'no'};return cols.map(c=>quote(record[c])).join(',');})].join('\r\n');const blob=new Blob(['\ufeff'+content],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='rater___RATER__'+(complete?'_completed':'_progress')+'.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.getElementById('save').onclick=()=>download(true);document.getElementById('progress').onclick=()=>download(false);
// A small RFC4180 parser handles commas, escaped quotes and multiline context.
function csvParse(s){const result=[];let row=[],value='',quoted=false;for(let i=0;i<s.length;i++){const c=s[i];if(c==='"'){if(quoted&&s[i+1]==='"'){value+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(value);value='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&s[i+1]==='\n')i++;row.push(value);if(row.some(v=>v!==''))result.push(row);row=[];value='';}else value+=c;}if(quoted)throw Error('CSV có dấu ngoặc kép chưa đóng');if(value||row.length){row.push(value);result.push(row);}return result;}
document.getElementById('restore').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const parsed=csvParse((await file.text()).replace(/^\ufeff/,''));const headers=parsed.shift(),records=parsed.map(a=>Object.fromEntries(headers.map((h,i)=>[h,a[i]??''])));const map=new Map(records.map(r=>[r.item_id,r]));if(records.length!==rows.length||map.size!==rows.length)throw Error('CSV phải đủ 200 ID duy nhất');for(const r of rows){const loaded=map.get(r.item_id);if(!loaded)throw Error('CSV không khớp bộ câu');for(const k of Object.keys(r).filter(k=>!['annotation','reason'].includes(k))){if(loaded[k]!==r[k])throw Error('Nội dung hoặc ngữ cảnh đã thay đổi');}if(loaded.annotation&&!labels.includes(loaded.annotation))throw Error('Nhãn không hợp lệ');}if(!confirm('Khôi phục nhãn và ghi chú từ CSV? Nội dung nguồn sẽ giữ nguyên.'))return;for(const r of rows){const a=map.get(r.item_id);state.answers[r.item_id]={annotation:a.annotation,reason:a.reason||''};}state.reviewer=records[0].reviewer||'';state.ack=records[0].independence_ack==='yes';try{localStorage.setItem(key,JSON.stringify(state));}catch(e){throw Error('Không lưu được trong trình duyệt. Hãy dùng một trình duyệt cho phép lưu localStorage.');}location.reload();}catch(err){alert(err.message);}};save();
</script></html>'''
    destination = OUT / 'annotation_forms'
    destination.mkdir(exist_ok=True)
    for rater in ('A', 'B'):
        csv_path = OUT / f'annotation/rater_{rater}.csv'
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 200 or any('label' in r or 'prediction' in r for r in rows):
            raise ValueError('Annotation form must contain 200 blinded rows')
        encoded = json.dumps(rows, ensure_ascii=False).replace('<', r'\u003c')
        page = template.replace('__ROWS__', encoded).replace('__RATER__', rater).replace('__HASH__', sha(csv_path))
        path = destination / f'GAN_NHAN_{rater}.html'
        path.write_text(page, encoding='utf-8')
        with zipfile.ZipFile(ROOT_REPORTS / f'KIEM_DINH_NHAN_{rater}_DE_SU_DUNG.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(path, path.name)
            z.write(OUT / 'HUONG_DAN_KIEM_DINH_NHAN.docx', 'HUONG_DAN_KIEM_DINH_NHAN.docx')
        print(path)


ROOT_REPORTS = OUT.parent

if __name__ == '__main__':
    main()
