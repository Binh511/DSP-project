"""Generate a local, escaped review interface; no network or remote scripts."""
import json
from pathlib import Path

from .analysis import read_jsonl
from .review import FORM_FIELDS


def write_review_html(packet_dir: Path) -> Path:
    packet_dir = Path(packet_dir)
    data = json.dumps({"items": read_jsonl(packet_dir / "items.jsonl"), "fields": FORM_FIELDS},
                      ensure_ascii=True).replace("<", "\\u003c")
    target = packet_dir / "REVIEW.html"
    target.write_text(HTML.replace("__PACKET_DATA__", data), encoding="utf-8")
    return target


HTML = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rà soát relevance — chỉ chẩn đoán validation</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 18px;color:#172536;background:#f5f7fa}button,input,select,textarea{font:inherit;padding:8px;margin:4px 0}button{cursor:pointer}header,article{background:white;padding:18px;border-radius:12px;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:46vh;overflow:auto;background:#f1f4f8;padding:14px}label{display:block;margin:8px 0}.row{display:flex;gap:12px;flex-wrap:wrap}small{color:#526171}textarea{width:95%}.notice{border-left:4px solid #b47c12;padding-left:12px}</style>
<header><h1>Rà soát relevance validation</h1><p class="notice">Bộ ưu tiên chẩn đoán; không đại diện toàn benchmark. Điền nhãn độc lập, chưa xem nhãn người khác. “Không rõ” được giữ là chưa biết, không chuyển thành không liên quan.</p>
<div class="row"><label>Người chấm <input id="reviewer" placeholder="Tên hoặc mã người chấm"></label><label>Tìm truy vấn / nguồn <input id="search" placeholder="Tìm từ, query_id, doc_id"></label><label>Lọc <select id="filter"><option value="all">Tất cả</option><option value="missing">Chưa đủ nhãn</option><option value="unknown">Không rõ / cần xem lại</option><option value="done">Đã đủ nhãn chắc chắn</option></select></label></div>
<button id="importButton">Nhập CSV đã lưu</button><input id="importFile" type="file" accept=".csv" hidden><button id="exportButton">Xuất CSV để lưu</button><p id="status" aria-live="polite"></p><small>Dữ liệu chỉ xử lý trong trình duyệt; không gửi mạng. Xuất CSV trước khi đóng trang. Không có tự lưu lên máy chủ.</small></header>
<article><div class="row"><button id="prev">← Trước</button><button id="next">Sau →</button><span id="position"></span></div><h2 id="query"></h2><small id="identity"></small><pre id="passage"></pre>
<label>Truy vấn có rõ và thực tế? <select id="query_realism"><option value="">Chưa chấm</option><option value="realistic">Rõ / có thể sử dụng</option><option value="malformed">Méo nghĩa / không thực tế</option><option value="uncertain">Không rõ</option></select></label>
<label>Nguồn có đọc hiểu được? <select id="source_quality"><option value="">Chưa chấm</option><option value="readable">Đọc hiểu được</option><option value="corrupt">Lỗi nguồn / không đọc hiểu</option><option value="uncertain">Không rõ</option></select></label>
<label>Mức liên quan của đoạn với truy vấn <select id="relevance"><option value="">Chưa chấm</option><option value="0">0 — Không trả lời</option><option value="1">1 — Liên quan một phần</option><option value="2">2 — Trả lời trực tiếp</option><option value="unknown">Không rõ — cần người phân xử</option></select></label>
<label>Ghi chú <textarea id="notes" rows="3"></textarea></label><p><small>Giữ cùng đánh giá về tính rõ ràng của một truy vấn ở các đoạn khác nhau. Không sửa query hoặc đoạn nguồn; yêu cầu viết lại phải tạo phiên bản truy vấn mới.</small></p></article>
<script type="application/json" id="packet">__PACKET_DATA__</script><script>
const packet=JSON.parse(document.getElementById('packet').textContent), rows=packet.items.map(x=>({...x,query_realism:'',source_quality:'',relevance:'',reviewer:'',notes:''}));let current=0,visible=[];
const el=id=>document.getElementById(id), labels=['query_realism','source_quality','relevance','notes'];
function known(r){return r.query_realism&&r.source_quality&&r.relevance&&r.query_realism!=='uncertain'&&r.source_quality!=='uncertain'&&r.relevance!=='unknown'&&r.reviewer.trim()}
function save(){const r=rows[current];if(!r)return;for(const k of labels)r[k]=el(k).value;if(el('reviewer').value.trim())r.reviewer=el('reviewer').value.trim()}
function render(){const term=el('search').value.toLowerCase(),f=el('filter').value;visible=rows.map((r,i)=>({r,i})).filter(({r})=>(r.query+' '+r.doc_id+' '+r.query_id+' '+r.text).toLowerCase().includes(term)&&(f==='all'||f==='done'&&known(r)||f==='missing'&&!known(r)||f==='unknown'&&(r.relevance==='unknown'||r.source_quality==='uncertain'||r.query_realism==='uncertain'))).map(x=>x.i);if(!visible.includes(current))current=visible[0]??-1;el('status').textContent=knownCount()+' / '+rows.length+' mục đủ nhãn chắc chắn; đang hiển thị '+visible.length;const r=rows[current];if(!r){el('query').textContent='Không có mục phù hợp';el('passage').textContent='';return}el('query').textContent=r.query;el('identity').textContent=r.query_id+' | '+r.passage_doc_id+' | '+r.chunk_id+' | dòng '+r.source_start_line+'–'+r.source_end_line;el('passage').textContent=r.text;el('position').textContent=(visible.indexOf(current)+1)+' / '+visible.length;for(const k of labels)el(k).value=r[k]||''}
function knownCount(){return rows.filter(known).length}for(const k of labels)el(k).addEventListener('change',()=>{save();el('status').textContent=knownCount()+' / '+rows.length+' mục đủ nhãn chắc chắn'});for(const k of ['search','filter'])el(k).addEventListener('input',()=>{save();render()});for(const [id,d]of [['prev',-1],['next',1]])el(id).onclick=()=>{save();current=visible[(visible.indexOf(current)+d+visible.length)%visible.length]??-1;render()};
function safe(x){x=String(x??'');return /^[=+@\\t\\r-]/.test(x)?"'"+x:x}function cell(x){return '"'+safe(x).replaceAll('"','""')+'"'}
el('exportButton').onclick=()=>{save();const csv='\\ufeff'+[packet.fields.map(cell).join(','),...rows.map(r=>packet.fields.map(k=>cell(r[k])).join(','))].join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='human_review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
function parseCSV(text){text=text.replace(/^\\uFEFF/,'');const out=[];let row=[],value='',quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){value+='"';i++}else quoted=!quoted}else if(c===','&&!quoted){row.push(value);value=''}else if((c==='\\n'||c==='\\r')&&!quoted){if(c==='\\r'&&text[i+1]==='\\n')i++;row.push(value);if(row.some(Boolean))out.push(row);row=[];value=''}else value+=c}if(quoted)throw Error('CSV có dấu nháy chưa đóng');if(value||row.length){row.push(value);out.push(row)}return out}
el('importButton').onclick=()=>el('importFile').click();el('importFile').onchange=async()=>{try{const table=parseCSV(await el('importFile').files[0].text()),head=table.shift();if(JSON.stringify(head)!==JSON.stringify(packet.fields)||table.length!==rows.length)throw Error('CSV không khớp cấu trúc / số mục');const byId=new Map(rows.map(r=>[r.item_id,r])),pending=[],seen=new Set;for(const values of table){const x=Object.fromEntries(head.map((k,i)=>[k,values[i]??''])),r=byId.get(x.item_id);if(!r||seen.has(x.item_id))throw Error('Mã mục không hợp lệ hoặc trùng');seen.add(x.item_id);for(const k of packet.fields.filter(k=>!['query_realism','source_quality','relevance','reviewer','notes'].includes(k)))if(x[k]!==safe(r[k]))throw Error('Query / nguồn đã thay đổi');pending.push([r,x])}for(const[r,x]of pending)for(const k of ['query_realism','source_quality','relevance','reviewer','notes'])r[k]=x[k];render();alert('Đã nhập CSV. Python validator sẽ kiểm tra nhãn khi bàn giao.')}catch(error){alert(error.message)}};render();
</script></html>'''
