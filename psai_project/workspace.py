"""Standalone, offline project tracking, kept separate from model evidence.

Task completion is a person's progress entry. It never changes classification,
source provenance, relevance labels or independent evaluation results.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import hashlib
from html import escape
import json
import re

STATUSES = ('proposed', 'in_progress', 'blocked', 'done')
DECISIONS = ('unsure', 'approve', 'reject')
_HASH = re.compile(r'^[0-9a-f]{64}$')
_ID = re.compile(r'^[A-Za-z0-9_-]{1,80}$')
_MARKUP = re.compile(r'<\s*/?\s*[a-zA-Z!]|javascript\s*:|\bon[a-z]+\s*=', re.I)


def _canonical(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Analysis and progress must contain finite JSON values') from error


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f'{name} has missing or unknown fields')


def _note(value, name='notes', maximum=10000):
    if (type(value) is not str or len(value) > maximum or _MARKUP.search(value)
            or any(ord(character) < 32 and character not in '\t\r\n' for character in value)):
        raise ValueError(f'{name} must be plain text without HTML, scripts or control characters')
    return value


def _catalog(analysis):
    if type(analysis) is not dict or type(analysis.get('plan')) is not dict:
        raise ValueError('Workspace requires a canonical analysis with a plan')
    document = analysis.get('document_id')
    if type(document) is not str or not document.strip():
        raise ValueError('Analysis document_id must be nonempty')
    source_hash = analysis.get('extraction', {}).get('source_sha256')
    if type(source_hash) is not str or not _HASH.fullmatch(source_hash):
        raise ValueError('Analysis must contain an exact SHA256 source identity')
    metadata_hash = analysis.get('metadata', {}).get('source_text_sha256', source_hash)
    if metadata_hash != source_hash or analysis['plan'].get('source_sha256', source_hash) != source_hash:
        raise ValueError('Analysis source hashes disagree')
    tasks, identities = [], set()
    steps = analysis['plan'].get('steps')
    if type(steps) is not list:
        raise ValueError('Analysis plan steps must be a list')
    for index, step in enumerate(steps, 1):
        if type(step) is not dict:
            raise ValueError('Each plan step must be an object')
        task_id = step.get('task_id', f'S{index:03d}')
        if type(task_id) is not str or not _ID.fullmatch(task_id) or task_id in identities:
            raise ValueError('Task IDs must be distinct safe identifiers')
        identities.add(task_id)
        checklist = step.get('checklist', [])
        if type(checklist) is not list or any(type(item) is not dict for item in checklist):
            raise ValueError('Task checklist must contain objects')
        tasks.append({'task_id': task_id, 'step': step,
                      'checklist_ids': [f'{task_id}:C{number:03d}' for number in range(1, len(checklist) + 1)]})
    for task in tasks:
        parent = task['step'].get('parent_task_id')
        if parent is not None and (parent not in identities or parent == task['task_id']):
            raise ValueError('Candidate task hierarchy references an unknown/self task')
    dependencies, dependency_ids = [], set()
    candidates = analysis['plan'].get('dependency_candidates', [])
    if type(candidates) is not list:
        raise ValueError('Dependency candidates must be a list')
    for candidate in candidates:
        if type(candidate) is not dict:
            raise ValueError('Dependency candidate must be an object')
        before, after = candidate.get('before_task_id'), candidate.get('after_task_id')
        if before not in identities or after not in identities or before == after:
            raise ValueError('Dependency candidate references an unknown/self task')
        dependency_id = f'{before}->{after}'
        if dependency_id in dependency_ids:
            raise ValueError('Repeated dependency candidate')
        dependency_ids.add(dependency_id)
        dependencies.append({'dependency_id': dependency_id, 'before_task_id': before, 'after_task_id': after})
    identity = {'schema_version': 1, 'document_id': document, 'source_sha256': source_hash,
                'analysis_sha256': hashlib.sha256(_canonical(analysis).encode('utf-8')).hexdigest(),
                'source_review_status': 'unreviewed'}
    return identity, tasks, dependencies


def initial_progress(analysis: dict) -> dict:
    identity, tasks, dependencies = _catalog(analysis)
    return {**identity,
            'tasks': [{'task_id': task['task_id'], 'status': 'proposed',
                       'checklist': [{'item_id': item_id, 'done': False} for item_id in task['checklist_ids']],
                       'notes': ''} for task in tasks],
            'dependencies': [{'dependency_id': item['dependency_id'], 'decision': 'unsure',
                              'reviewer': '', 'notes': ''} for item in dependencies], 'notes': ''}


def validate_progress(progress: dict, analysis: dict) -> dict:
    """Reject foreign/tampered progress; return a copy without changing analysis."""
    expected = initial_progress(analysis)
    _keys(progress, expected, 'Progress')
    _canonical(progress)
    for key in ('schema_version', 'document_id', 'source_sha256', 'analysis_sha256', 'source_review_status'):
        if type(progress[key]) is not type(expected[key]) or progress[key] != expected[key]:
            raise ValueError(f'Progress {key} differs from this analysis')
    if type(progress['tasks']) is not list or len(progress['tasks']) != len(expected['tasks']):
        raise ValueError('Progress task count differs from this plan')
    tasks = {task['task_id']: task for task in expected['tasks']}
    seen = set()
    for task in progress['tasks']:
        _keys(task, ('task_id', 'status', 'checklist', 'notes'), 'Task progress')
        identity = task['task_id']
        if type(identity) is not str or identity not in tasks or identity in seen:
            raise ValueError('Unknown or repeated progress task ID')
        seen.add(identity)
        if type(task['status']) is not str or task['status'] not in STATUSES:
            raise ValueError('Unknown task status')
        _note(task['notes'])
        entries = task['checklist']
        wanted = {entry['item_id'] for entry in tasks[identity]['checklist']}
        if type(entries) is not list or len(entries) != len(wanted):
            raise ValueError('Progress checklist count differs from this task')
        found = set()
        for entry in entries:
            _keys(entry, ('item_id', 'done'), 'Checklist progress')
            if type(entry['item_id']) is not str or entry['item_id'] not in wanted or entry['item_id'] in found:
                raise ValueError('Unknown or repeated checklist item ID')
            if type(entry['done']) is not bool:
                raise ValueError('Checklist done must be a JSON boolean')
            found.add(entry['item_id'])
    dependencies = {item['dependency_id']: item for item in expected['dependencies']}
    if type(progress['dependencies']) is not list or len(progress['dependencies']) != len(dependencies):
        raise ValueError('Progress dependency count differs from this plan')
    seen, graph = set(), {}
    for dependency in progress['dependencies']:
        _keys(dependency, ('dependency_id', 'decision', 'reviewer', 'notes'), 'Dependency review')
        identity = dependency['dependency_id']
        if type(identity) is not str or identity not in dependencies or identity in seen:
            raise ValueError('Unknown or repeated dependency ID')
        seen.add(identity)
        if type(dependency['decision']) is not str or dependency['decision'] not in DECISIONS:
            raise ValueError('Unknown dependency review decision')
        reviewer = _note(dependency['reviewer'], 'reviewer', 120)
        _note(dependency['notes'])
        if dependency['decision'] in ('approve', 'reject') and not reviewer.strip():
            raise ValueError('Approving or rejecting a dependency requires a human reviewer name')
        if dependency['decision'] == 'approve':
            before, after = identity.split('->')
            graph.setdefault(before, []).append(after)
    active, complete = set(), set()
    def visit(node):
        if node in active:
            raise ValueError('Approved dependencies contain a cycle')
        if node in complete:
            return
        active.add(node)
        for successor in graph.get(node, []):
            visit(successor)
        active.remove(node)
        complete.add(node)
    for node in graph:
        visit(node)
    _note(progress['notes'])
    return deepcopy(progress)


def progress_summary(progress: dict, analysis: dict) -> dict:
    progress = validate_progress(progress, analysis)
    counts = {status: sum(task['status'] == status for task in progress['tasks']) for status in STATUSES}
    total = len(progress['tasks'])
    return {'tasks_total': total, 'tasks_by_status': counts,
            'task_done_percent': round(100 * counts['done'] / total, 2) if total else 0.0,
            'checklist_total': sum(len(task['checklist']) for task in progress['tasks']),
            'checklist_done': sum(item['done'] for task in progress['tasks'] for item in task['checklist']),
            'dependencies_reviewed': sum(item['decision'] != 'unsure' for item in progress['dependencies']),
            'research_quality_measured': False}


def _script_json(value):
    return _canonical(value).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def render_workspace(analysis: dict) -> str:
    """Return a self-contained HTML tracker; all edits remain separate progress."""
    identity, tasks, dependencies = _catalog(analysis)
    progress = initial_progress(analysis)
    evidence = {entry['evidence_id']: entry for entries in analysis['extraction'].get('fields', {}).values()
                for entry in entries}
    starts = analysis.get('metadata', {}).get('source_line_starts', [0])
    if (type(starts) is not list or not starts or starts[0] != 0
            or any(type(number) is not int or number < 0 for number in starts)
            or starts != sorted(set(starts))):
        raise ValueError('Source line starts must be an ordered list of character offsets')
    def refs(ids):
        result = []
        for item_id in ids:
            if item_id not in evidence:
                raise ValueError('Workspace task references unknown source evidence')
            entry = evidence[item_id]
            line = bisect_right(starts, entry['char_start'])
            result.append({'id': item_id, 'url': f'report.html#src-{identity["source_sha256"]}-L{line}',
                           'label': f'{item_id} · [{entry["char_start"]}, {entry["char_end"]})'})
        return result
    view_tasks = []
    for task in tasks:
        step = task['step']
        view_tasks.append({'task_id': task['task_id'], 'description': step['description'],
                           'parent_task_id': step.get('parent_task_id'), 'refs': refs(step['evidence_ids']),
                           'checklist': [{'item_id': item_id, 'description': item.get('description', item.get('text', '')),
                                          'refs': refs(item.get('evidence_ids', [item.get('evidence_id')]))}
                                         for item_id, item in zip(task['checklist_ids'], step.get('checklist', []))],
                           'resources': [{'description': item.get('description', item.get('text', '')),
                                          'refs': refs(item.get('evidence_ids', [item.get('evidence_id')]))}
                                         for item in step.get('required_resources', [])]})
    warnings = list(analysis['plan'].get('warnings', []))
    warnings += analysis.get('resource_matching', {}).get('warnings', [])
    data = {'identity': identity, 'tasks': view_tasks, 'dependencies': dependencies,
            'warnings': warnings, 'initial_progress': progress}
    before, after = _HTML.rsplit('__DATA__', 1)
    return before.replace('__TITLE__', escape(identity['document_id'])) + _script_json(data) + after


_HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PSAI · Theo dõi __TITLE__</title><style>
:root{font:16px/1.5 system-ui,sans-serif;color:#183347;background:#f2f5f7;--muted:#597082;--line:#dce5eb;--accent:#145d81}
*{box-sizing:border-box}body{margin:0}header{background:#14364c;color:white;padding:30px max(24px,calc((100vw - 1120px)/2))}h1{font-size:26px;margin:0 0 8px}h2{font-size:20px}h3{font-size:17px;margin:0 0 12px}main{max-width:1120px;margin:24px auto;padding:0 20px}a{color:#0d658e}header a{color:#bfe7ff}.notice{background:#fff6d9;border-left:4px solid #c58d18;padding:14px 18px;border-radius:6px;margin:16px 0}.muted{color:var(--muted);font-size:14px}.toolbar,.stats{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.toolbar{margin:20px 0}button,.button{background:var(--accent);color:white;border:0;border-radius:7px;padding:10px 16px;font:inherit;cursor:pointer}button.secondary{background:#e4edf3;color:#234459}input[type=file]{max-width:100%}.stats>div{background:white;border:1px solid var(--line);border-radius:10px;padding:16px;flex:1;min-width:145px}.stats strong{display:block;font-size:25px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:22px;margin:16px 0}.task-top{display:flex;gap:16px;justify-content:space-between;align-items:start}.badge{font:12px ui-monospace,monospace;display:inline-block;background:#e7f1f7;padding:4px 8px;border-radius:5px}.refs{display:flex;gap:10px;flex-wrap:wrap;font-size:13px;margin:10px 0}.refs a{overflow-wrap:anywhere}label{display:block;font-size:14px;margin:12px 0 5px}select,input[type=text],textarea{font:inherit;border:1px solid #b7c8d3;border-radius:6px;padding:9px;background:#fff;color:#183347}textarea{width:100%;min-height:80px;resize:vertical}select{max-width:100%}input[type=checkbox]{width:18px;height:18px;vertical-align:middle;margin-right:8px}.check{display:flex;align-items:start;gap:4px}.check input{flex:none;margin-top:5px}.hash{font:12px ui-monospace,monospace;overflow-wrap:anywhere}.dependency-fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}.message{min-height:24px;color:#244d66}.error{color:#a32626}footer{margin:30px 0;color:var(--muted);font-size:13px}details summary{cursor:pointer;font-weight:600}ul{padding-left:22px}@media(max-width:650px){.task-top{display:block}.dependency-fields{grid-template-columns:1fr}header{padding:24px}.card{padding:16px}}
</style></head><body><header><h1>Không gian dự án · __TITLE__</h1><div>Theo dõi công việc và rà soát đề xuất từ tài liệu</div><p><a href="report.html">Mở báo cáo có dẫn chứng</a> · <a href="analysis.json">Dữ liệu phân tích gốc</a></p></header>
<main><div class="notice">Kế hoạch, nhãn và quan hệ phụ thuộc vẫn cần người kiểm tra. Đánh dấu công việc “Hoàn thành” là tiến độ do bạn nhập; không chứng minh mô hình đúng hoặc dự án đã đầy đủ 100%.</div>
<details><summary>Danh tính tài liệu và dữ liệu gốc</summary><p>Source SHA256:</p><p class="hash" id="source-hash"></p><p>Analysis SHA256:</p><p class="hash" id="analysis-hash"></p><p class="muted">Trạng thái kiểm định nguồn: chưa kiểm định. Tiến độ không sửa nội dung phân tích hay provenance của mô hình.</p></details>
<div class="toolbar"><button id="export">Xuất tiến độ JSON</button><label class="button" style="margin:0" for="import">Nhập tiến độ JSON</label><input id="import" type="file" accept="application/json,.json" hidden><button class="secondary" id="reset">Đặt lại tiến độ</button></div>
<p class="muted">Dữ liệu ở trên máy bạn. Trình duyệt có thể không lưu được khi mở file://; hãy xuất JSON để giữ và chuyển tiến độ. Không có dữ liệu nào được tải lên mạng.</p><p class="message" id="message" role="status" aria-live="polite"></p>
<section class="stats" id="stats" aria-label="Tiến độ do người dùng nhập"></section>
<h2>Công việc đề xuất</h2><section id="tasks"></section><h2>Rà soát quan hệ phụ thuộc</h2><section id="dependencies"></section>
<section class="card"><h2>Ghi chú chung</h2><label for="general-notes">Ghi chú cá nhân, tách khỏi nội dung nguồn và kết quả mô hình</label><textarea id="general-notes" maxlength="10000"></textarea></section>
<section class="card"><h2>Cảnh báo cần kiểm tra</h2><ul id="warnings"></ul></section><footer>Mọi dẫn chứng mở báo cáo gốc. Tiến độ chỉ là dữ liệu quản lý công việc; không phải nhãn chuẩn hoặc kết quả kiểm định nghiên cứu.</footer></main>
<script id="workspace-data" type="application/json">__DATA__</script><script>
'use strict';
const model=JSON.parse(document.getElementById('workspace-data').textContent);
const clone=x=>JSON.parse(JSON.stringify(x));
let progress=clone(model.initial_progress);
const storageKey='psai-progress-'+model.identity.analysis_sha256;
const statusNames={proposed:'Đề xuất',in_progress:'Đang làm',blocked:'Đang vướng',done:'Hoàn thành'};
const decisionNames={unsure:'Chưa chắc / chưa duyệt',approve:'Chấp nhận',reject:'Bác bỏ'};
const byId=id=>document.getElementById(id);
const element=(tag,text,cls)=>{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;};
function message(text,error=false){byId('message').textContent=text;byId('message').className=error?'message error':'message';}
function keys(x,wanted){if(x===null||typeof x!=='object'||Array.isArray(x)||Object.keys(x).sort().join('|')!==wanted.slice().sort().join('|'))throw Error('Trường dữ liệu không đúng schema.');}
function note(x,max=10000){if(typeof x!=='string'||x.length>max||/<\s*\/?\s*[a-zA-Z!]|javascript\s*:|\bon[a-z]+\s*=/i.test(x)||/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(x))throw Error('Ghi chú/tên cần là văn bản thuần, không có HTML hoặc script.');}
function validate(value){
 const initial=model.initial_progress;keys(value,Object.keys(initial));
 for(const key of ['schema_version','document_id','source_sha256','analysis_sha256','source_review_status'])if(typeof value[key]!==typeof initial[key]||value[key]!==initial[key])throw Error('File tiến độ thuộc tài liệu hoặc phiên bản phân tích khác.');
 if(!Array.isArray(value.tasks)||value.tasks.length!==initial.tasks.length)throw Error('Số công việc không khớp.');
 const seen=new Set();for(const task of value.tasks){keys(task,['task_id','status','checklist','notes']);const wanted=initial.tasks.find(x=>x.task_id===task.task_id);if(!wanted||seen.has(task.task_id))throw Error('ID công việc lạ hoặc lặp.');seen.add(task.task_id);if(typeof task.status!=='string'||!Object.hasOwn(statusNames,task.status))throw Error('Trạng thái công việc không hợp lệ.');note(task.notes);if(!Array.isArray(task.checklist)||task.checklist.length!==wanted.checklist.length)throw Error('Số checklist không khớp.');const checks=new Set();for(const item of task.checklist){keys(item,['item_id','done']);if(!wanted.checklist.some(x=>x.item_id===item.item_id)||checks.has(item.item_id)||typeof item.done!=='boolean')throw Error('Checklist sai ID hoặc kiểu dữ liệu.');checks.add(item.item_id);}}
 if(!Array.isArray(value.dependencies)||value.dependencies.length!==initial.dependencies.length)throw Error('Số quan hệ phụ thuộc không khớp.');
 const deps=new Set(),graph=new Map();for(const dependency of value.dependencies){keys(dependency,['dependency_id','decision','reviewer','notes']);if(!initial.dependencies.some(x=>x.dependency_id===dependency.dependency_id)||deps.has(dependency.dependency_id))throw Error('ID quan hệ lạ hoặc lặp.');deps.add(dependency.dependency_id);if(typeof dependency.decision!=='string'||!Object.hasOwn(decisionNames,dependency.decision))throw Error('Quyết định không hợp lệ.');note(dependency.reviewer,120);note(dependency.notes);if(dependency.decision!=='unsure'&&!dependency.reviewer.trim())throw Error('Cần nhập tên người kiểm tra trước khi duyệt hoặc bác bỏ.');if(dependency.decision==='approve'){const [before,after]=dependency.dependency_id.split('->');if(!graph.has(before))graph.set(before,[]);graph.get(before).push(after);}}
 const active=new Set(),done=new Set();function visit(node){if(active.has(node))throw Error('Các quan hệ đã chấp nhận tạo thành chu trình.');if(done.has(node))return;active.add(node);for(const next of graph.get(node)||[])visit(next);active.delete(node);done.add(node);}for(const node of graph.keys())visit(node);note(value.notes);return clone(value);
}
function save(){try{validate(progress);try{localStorage.setItem(storageKey,JSON.stringify(progress));message('Đã lưu thêm trong trình duyệt. Hãy xuất JSON để có bản giữ chắc chắn.');}catch(_){message('Trình duyệt không lưu được. Hãy bấm Xuất tiến độ JSON để giữ thay đổi.');}summary();}catch(error){message(error.message,true);}}
function change(apply){const previous=clone(progress);apply();try{validate(progress);save();}catch(error){progress=previous;message(error.message,true);render();}}
function links(parent,items){const box=element('div',undefined,'refs');for(const item of items){const a=element('a',item.label);a.href=item.url;a.target='_blank';a.rel='noopener';box.append(a);}parent.append(box);}
function select(values,current,onChange){const box=element('select');for(const [value,name] of Object.entries(values)){const option=element('option',name);option.value=value;box.append(option);}box.value=current;box.addEventListener('change',()=>onChange(box.value));return box;}
function summary(){const total=progress.tasks.length,done=progress.tasks.filter(x=>x.status==='done').length,blocked=progress.tasks.filter(x=>x.status==='blocked').length,checkTotal=progress.tasks.reduce((s,x)=>s+x.checklist.length,0),checked=progress.tasks.reduce((s,x)=>s+x.checklist.filter(c=>c.done).length,0);byId('stats').replaceChildren();for(const [value,label] of [[done+'/'+total,'Công việc do bạn đánh dấu xong'],[(total?Math.round(10000*done/total)/100:0)+'%','Tiến độ công việc, không phải chất lượng AI'],[blocked,'Công việc đang vướng'],[checked+'/'+checkTotal,'Checklist đã đánh dấu']]){const box=element('div');box.append(element('strong',String(value)),element('span',label,'muted'));byId('stats').append(box);}}
function render(){
 byId('source-hash').textContent=model.identity.source_sha256;byId('analysis-hash').textContent=model.identity.analysis_sha256;summary();byId('tasks').replaceChildren();
 for(const task of model.tasks){const state=progress.tasks.find(x=>x.task_id===task.task_id),card=element('article',undefined,'card'),top=element('div',undefined,'task-top'),title=element('div');title.append(element('span',task.task_id,'badge'),element('h3',task.description));top.append(title,select(statusNames,state.status,value=>change(()=>state.status=value)));card.append(top);if(task.parent_task_id)card.append(element('p','Đề xuất công việc cha: '+task.parent_task_id+' · cần người kiểm tra','muted'));links(card,task.refs);
 if(task.checklist.length){card.append(element('h3','Checklist trích từ nguồn'));for(const item of task.checklist){const row=element('label',undefined,'check'),input=element('input');input.type='checkbox';const mark=state.checklist.find(x=>x.item_id===item.item_id);input.checked=mark.done;input.addEventListener('change',()=>change(()=>mark.done=input.checked));row.append(input,element('span',item.description));card.append(row);links(card,item.refs);}}
 if(task.resources.length){card.append(element('h3','Tài nguyên được đề xuất gắn vào công việc'));for(const resource of task.resources){card.append(element('p',resource.description));links(card,resource.refs);}card.append(element('p','Cần người kiểm tra tính cần thiết; việc xuất hiện trong nguồn không chứng minh công việc thiếu tài nguyên.','muted'));}
 const label=element('label','Ghi chú công việc'),notes=element('textarea');notes.value=state.notes;notes.maxLength=10000;notes.addEventListener('change',()=>change(()=>state.notes=notes.value));card.append(label,notes);byId('tasks').append(card);}
 if(!model.tasks.length)byId('tasks').append(element('p','Không có công việc đủ điều kiện từ nguồn. Không tự tạo công việc thay thế.','card'));
 byId('dependencies').replaceChildren();for(const dep of model.dependencies){const state=progress.dependencies.find(x=>x.dependency_id===dep.dependency_id),card=element('article',undefined,'card');card.append(element('h3',dep.before_task_id+' → '+dep.after_task_id),element('p','Đề xuất của mô hình, cần người rà soát.','muted'));const fields=element('div',undefined,'dependency-fields'),nameBox=element('div'),name=element('input');name.type='text';name.maxLength=120;name.value=state.reviewer;name.addEventListener('change',()=>change(()=>state.reviewer=name.value));nameBox.append(element('label','Tên người kiểm tra'),name);const decisionBox=element('div');decisionBox.append(element('label','Kết luận của người kiểm tra'),select(decisionNames,state.decision,value=>change(()=>state.decision=value)));fields.append(nameBox,decisionBox);const notes=element('textarea');notes.value=state.notes;notes.maxLength=10000;notes.addEventListener('change',()=>change(()=>state.notes=notes.value));card.append(fields,element('label','Lý do / ghi chú của người kiểm tra'),notes);byId('dependencies').append(card);}
 if(!model.dependencies.length)byId('dependencies').append(element('p','Mô hình không đề xuất quan hệ phụ thuộc. Điều này không chứng minh dự án không có phụ thuộc.','card'));
 byId('general-notes').value=progress.notes;byId('warnings').replaceChildren();for(const warning of model.warnings)byId('warnings').append(element('li',String(warning)));if(!model.warnings.length)byId('warnings').append(element('li','Nguồn, nhãn, tính đầy đủ và đề xuất của mô hình vẫn cần người kiểm tra.'));
}
byId('general-notes').addEventListener('change',event=>change(()=>progress.notes=event.target.value));
byId('export').addEventListener('click',()=>{try{validate(progress);const url=URL.createObjectURL(new Blob([JSON.stringify(progress,null,2)],{type:'application/json;charset=utf-8'}));const a=element('a');a.href=url;a.download='PSAI_progress_'+model.identity.analysis_sha256.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message('Đã xuất JSON. Lưu file này để giữ và chuyển tiến độ.');}catch(error){message(error.message,true);}});
byId('import').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>2000000)throw Error('File tiến độ vượt giới hạn 2 MB.');const imported=validate(JSON.parse(await file.text()));progress=imported;render();save();}catch(error){message('Không nhập dữ liệu: '+error.message,true);}finally{event.target.value='';}});
byId('reset').addEventListener('click',()=>{if(confirm('Đặt lại tiến độ đang mở? Hãy xuất JSON trước nếu cần giữ bản hiện tại.')){progress=clone(model.initial_progress);render();save();}});
try{const cached=localStorage.getItem(storageKey);if(cached){progress=validate(JSON.parse(cached));message('Đã khôi phục tiến độ lưu trong trình duyệt.');}}catch(_){message('Chưa có bản lưu trình duyệt sử dụng được; hãy nhập JSON nếu có.');}
render();
</script></body></html>'''
