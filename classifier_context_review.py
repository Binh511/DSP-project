"""Expose exact source/neighbor evidence for unresolved classifier Section rows.

No class labels, dataset rows, checkpoints or source manifests are modified.
Proposals require both neighbors immediately adjacent in v2.1 normalized text.
Unmatched, conflicting and headingless cases stay unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
DEFAULT_VERSION = ROOT / 'reports/research_review/studies/classifier_context_v2_1'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_csv(raw: bytes) -> list[dict]:
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError('Missing or duplicate CSV headers')
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed CSV row width')
    return rows


def source_snippet(text: str, start: int, end: int, radius: int = 2) -> dict:
    lines = text.splitlines()
    if not 1 <= start <= end <= len(lines):
        raise ValueError('Source line range is invalid')
    lo, hi = max(1, start - radius), min(len(lines), end + radius)
    return dict(start_line=lo, end_line=hi,
                text='\n'.join(f'{line}: {lines[line - 1]}' for line in range(lo, hi + 1)))


def neighbor_supported_section(parsed: dict, row: dict, matches: list[dict]) -> dict:
    from classifier_context_v2 import normalize
    previous, following = normalize(row['prev_text']), normalize(row['next_text'])
    supported = []
    if previous and following:
        for match in matches:
            a, b = match['normalized_start'], match['normalized_end']
            source = parsed['normalized']
            if (source[max(0, a - len(previous) - 1):a] == previous + ' '
                    and source[b:b + len(following) + 1] == ' ' + following):
                supported.append(match)
    paths = {path for match in supported for path in match['section_paths']}
    safe = (bool(supported) and len(paths) == 1 and '' not in paths
            and all(len(match['section_paths']) == 1 for match in supported))
    return dict(status='exact_two_neighbors_support_one_section' if safe else 'unknown_needs_human_review',
                candidate_section=next(iter(paths)) if safe else '',
                supported_matches=supported, human_verified=False, pdf_verified=False,
                normalization='same classifier-context-v2.1 normalization; exact adjacency; no fuzzy matching')


def build_evidence(version_dir: Path) -> dict:
    from classifier_context_v2 import parse_source, find_matches
    version_dir = Path(version_dir).resolve()
    manifest_raw = (version_dir / 'preparation_manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    needed = ['section_unresolved_for_review.csv', 'section_changes_and_provenance.csv',
              'source_manifest.json', 'training_rows.csv', 'validation_rows.csv']
    input_bytes = {}
    for name in needed:
        raw = (version_dir / name).read_bytes()
        if manifest['evidence_files'].get(name) != sha(raw):
            raise ValueError(f'Prepared evidence hash changed: {name}')
        input_bytes[name] = raw
    originals, prepared, input_hashes = {}, {}, {}
    for split, filename in [('train', 'training_rows.csv'), ('validation', 'validation_rows.csv')]:
        original_path = Path(manifest['splits'][split]['original_input_file'])
        raw = original_path.read_bytes()
        if sha(raw) != manifest['splits'][split]['original_input_file_sha256']:
            raise ValueError(f'Frozen original {split} file changed')
        originals[split] = read_csv(raw)
        prepared[split] = read_csv(input_bytes[filename])
        if len(originals[split]) != len(prepared[split]):
            raise ValueError('Prepared/frozen row counts differ')
        input_hashes[str(original_path.resolve())] = sha(raw)
    sources = {}
    for doc in json.loads(input_bytes['source_manifest.json']):
        sources[doc['document_id']] = doc['sources']
    trace = read_csv(input_bytes['section_unresolved_for_review.csv'])
    all_trace = {(row['split'], row['row_index']): row
                 for row in read_csv(input_bytes['section_changes_and_provenance.csv'])}
    cache, entries, seen = {}, [], set()
    for item in trace:
        key = (item['split'], item['row_index'])
        if key in seen or key not in all_trace or item != all_trace[key]:
            raise ValueError('Unresolved rows differ from the complete frozen trace')
        seen.add(key)
        split, index = item['split'], int(item['row_index'])
        if split not in originals or not 0 <= index < len(originals[split]):
            raise ValueError('Invalid split/row index')
        row, revised = originals[split][index], prepared[split][index]
        if (item['resolved'].casefold() != 'false'
                or item['document_id'] != row['document_id'] or item['text'] != row['text']
                or item['reference_label_unchanged'] != row['label'] or item['old_section'] != row['section']
                or item['new_section'] != revised['section']
                or any(row[field] != revised[field] for field in row if field != 'section')):
            raise ValueError('Trace does not bind the unchanged frozen row')
        matches, source_views = [], []
        for source in sources.get(row['document_id'], []):
            path = Path(source['path']).resolve()
            if path not in cache:
                raw = path.read_bytes()
                if sha(raw) != source['sha256']:
                    raise ValueError(f'Markdown source changed: {path}')
                text = raw.decode('utf-8-sig')
                cache[path] = (text, parse_source(text))
            text, parsed = cache[path]
            current = [dict(source_path=str(path), **match) for match in find_matches(parsed, row['text'])]
            matches.extend(current)
            targets = [{**match, 'snippet': source_snippet(text, match['source_start_line'], match['source_end_line'])}
                       for match in current]
            anchors = {}
            for field in ('prev_text', 'next_text'):
                exact = find_matches(parsed, row[field]) if row[field].strip() else []
                anchors[field] = dict(total_matches=len(exact), shown_matches=[
                    {**match, 'snippet': source_snippet(text, match['source_start_line'], match['source_end_line'])}
                    for match in exact[:8]])
            source_views.append(dict(source_path=str(path), source_uri=path.as_uri(),
                                     sha256=source['sha256'], target_matches=targets, neighbors=anchors))
        if matches != json.loads(item['matches']):
            raise ValueError('Recomputed source matches differ from the prepared trace')
        proposal = dict(status='unknown_needs_human_review', candidate_section='', supported_matches=[],
                        human_verified=False, pdf_verified=False)
        if item['status'] == 'multiple_section_paths_needs_review' and len(source_views) == 1:
            parsed = cache[Path(source_views[0]['source_path'])][1]
            proposal = neighbor_supported_section(parsed, row, matches)
        public = dict(item_id=sha(f'{split}:{index}:{row["document_id"]}:{row["text"]}'.encode()),
                      split=split, row_index=index, document_id=row['document_id'], text=row['text'],
                      prev_text=row['prev_text'], next_text=row['next_text'],
                      original_section=row['section'], prepared_section=revised['section'],
                      original_status=item['status'], evidence_status=proposal['status'],
                      proposed_section=proposal['candidate_section'], reviewer='', review_decision='',
                      reviewed_section='', review_reason='')
        entries.append(dict(item=public, proposal=proposal, sources=source_views))
    supported = [entry for entry in entries if entry['item']['proposed_section']]
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_version=manifest['version'],
                   unresolved_input_rows=len(entries), original_status_counts=dict(Counter(e['item']['original_status'] for e in entries)),
                   exact_neighbor_supported=len(supported),
                   supported_by_split=dict(Counter(e['item']['split'] for e in supported)),
                   remaining_unknown=len(entries) - len(supported),
                   dataset_rows_modified=False, semantic_labels_adjudicated=False, human_verified=False,
                   pdf_verified=False, training_or_inference_performed=False, test_or_final_test_used=False,
                   preparation_manifest_sha256=sha(manifest_raw),
                   evidence_input_sha256={name: sha(raw) for name, raw in input_bytes.items()},
                   original_file_sha256=input_hashes,
                   markdown_source_sha256={str(path): sha(path.read_bytes()) for path in cache},
                   caveat='Proposals are exact matches in v2.1 normalized Markdown; not PDF/expert verification. Existing rows remain unchanged.')
    return dict(entries=entries, summary=summary, preparation_manifest_raw=manifest_raw)


HTML = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>PSAI — kiểm tra Section</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:12px}article{background:#f3f6fa;padding:18px;margin:18px 0;border-radius:8px}pre,p{white-space:pre-wrap;overflow-wrap:anywhere}pre{background:white;padding:12px}input,select,textarea,button{font:inherit;padding:8px;margin:5px}textarea{width:95%}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #ccc}.note{background:#fff0cb;padding:14px}</style>
<h1>Đối chiếu Section — 110 câu cần rà soát</h1><p class="note">Chỉ kiểm tra vị trí và heading trong Markdown nguồn, không gán nhãn chức năng. Đề xuất dùng cả câu trước và sau khớp liền kề theo normalization v2.1; chưa được người/PDF xác nhận. Những câu chưa có bằng chứng vẫn để trống. Dataset và model không bị sửa.</p>
<header><label>Người kiểm tra <input id="reviewer"></label><select id="filter"><option value="all">Tất cả</option><option value="supported">Có đề xuất hai ngữ cảnh</option><option value="unknown">Còn chưa xác định</option></select><button id="download">Tải CSV nhận xét</button><span id="counts"></span></header><main id="cards"></main>
<script>const entries=__ENTRIES__,answers={};document.getElementById('counts').textContent=entries.filter(e=>e.item.proposed_section).length+' đề xuất / '+entries.length+' câu';
function add(parent,tag,text){const node=document.createElement(tag);node.textContent=text;parent.append(node);return node;}
for(const e of entries){const r=e.item,card=add(document.getElementById('cards'),'article','');card.dataset.supported=String(Boolean(r.proposed_section));add(card,'h2',r.split+' #'+r.row_index+' — '+r.document_id);add(card,'p',r.text);add(card,'p','Trước: '+r.prev_text+'\nSau: '+r.next_text+'\nSection cũ: '+r.original_section+'\nTrạng thái gốc: '+r.original_status+'\nĐề xuất có bằng chứng: '+(r.proposed_section||'(chưa xác định)'));
for(const s of e.sources){const link=add(card,'a','Mở Markdown nguồn');link.href=s.source_uri;link.target='_blank';link.rel='noopener';add(card,'p',s.source_path+'\nSHA256: '+s.sha256);for(const match of s.target_matches){add(card,'h3','Câu đích — dòng '+match.source_start_line+'–'+match.source_end_line);add(card,'p','Heading: '+match.section_paths.join(' || '));add(card,'pre',match.snippet.text);}if(!s.target_matches.length)add(card,'p','Không có khớp normalized cho câu đích. Không suy đoán vị trí.');for(const [name,anchor] of Object.entries(s.neighbors)){const details=add(card,'details','');add(details,'summary',(name==='prev_text'?'Câu trước':'Câu sau')+': '+anchor.total_matches+' vị trí khớp; hiển thị tối đa 8');for(const hit of anchor.shown_matches)add(details,'pre',hit.snippet.text);}}
const choice=add(card,'select','');for(const value of ['', 'CONFIRM_SOURCE_SECTION','SOURCE_MISMATCH','UNRESOLVED']){const option=add(choice,'option',value||'Chưa quyết định');option.value=value;}const section=add(card,'input','');section.placeholder='Đường dẫn heading sau đối chiếu';const reason=add(card,'textarea','');reason.placeholder='Lý do và dòng nguồn đã đối chiếu';const update=()=>answers[r.item_id]={review_decision:choice.value,reviewed_section:section.value,review_reason:reason.value};choice.onchange=section.oninput=reason.oninput=update;}
document.getElementById('filter').onchange=event=>{for(const card of document.querySelectorAll('article'))card.hidden=event.target.value==='supported'&&card.dataset.supported!=='true'||event.target.value==='unknown'&&card.dataset.supported==='true';};
document.getElementById('download').onclick=()=>{const cols=Object.keys(entries[0].item),quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=entries.map(e=>({...e.item,...(answers[e.item.item_id]||{}),reviewer:document.getElementById('reviewer').value}));const csv='\ufeff'+[cols.map(quote).join(','),...rows.map(r=>cols.map(c=>quote(r[c])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='PSAI_SECTION_REVIEW.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};</script></html>'''


def prepare(version_dir: Path, output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Choose a new review directory; old packets are preserved')
    evidence = build_evidence(version_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.context-review-', dir=output_dir.parent))
    try:
        for filename, value in [('source_evidence.json', evidence['entries']), ('summary.json', evidence['summary'])]:
            (stage / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        rows = [entry['item'] for entry in evidence['entries']]
        if not rows:
            raise ValueError('No unresolved items to review')
        with (stage / 'review_items.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        payload = json.dumps(evidence['entries'], ensure_ascii=False).replace('<', r'\u003c')
        (stage / 'KIEM_TRA_SECTION.html').write_text(HTML.replace('__ENTRIES__', payload), encoding='utf-8')
        (stage / 'input_preparation_manifest.json').write_bytes(evidence['preparation_manifest_raw'])
        stage.rename(output_dir)
    finally:
        if stage.exists():
            if stage.parent.resolve() != output_dir.parent or not stage.name.startswith('.context-review-'):
                raise RuntimeError('Refusing cleanup outside staging directory')
            shutil.rmtree(stage)
    return evidence['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version-dir', type=Path, default=DEFAULT_VERSION)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.version_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
