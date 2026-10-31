# -*- coding: utf-8 -*-
"""PSAI: find potentially unseen Markdown proposals by DOCUMENT, not by file extension.
Run: python psai_tim_md_test_doc_moi.py
Safe: reads CSV/MD, writes candidate lists only; no training, no model changes.
This is screening, not proof of independence. Manual review of content similarity needed.
"""
from pathlib import Path
from collections import defaultdict
import csv
import re
import hashlib

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'data_segment_cls' / 'segment_classification.csv'
HUMAN = ROOT / 'extra_human.csv'
OUT = ROOT / 'runs' / 'independent_test'
EXCLUDE = {'.git', '.venv', 'venv', '__pycache__', 'models', 'chroma_db', 'node_modules', 'runs', '.cache'}
PAT = re.compile(r'^PSAI\d+\.md$', re.I)


def load_counts(path):
    if not path.is_file():
        raise SystemExit(f'Cannot find {path}. Run this from the PSAI project folder with both labeled CSV files present.')
    result = defaultdict(int)
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            k = str(row.get('document_id', '')).strip().upper()
            if k:
                result[k] += 1
    return result


def normal(s):
    return ' '.join(re.findall(r'\w+', s.casefold(), flags=re.UNICODE))


def fingerprint(s):
    return hashlib.sha256(normal(s).encode('utf-8')).hexdigest()


def shingles(s, k=6):
    words = normal(s).split()
    if len(words) < k:
        return set()
    return set(hash(tuple(words[i:i+k])) for i in range(len(words)-k+1))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    old, human = load_counts(ORIGINAL), load_counts(HUMAN)
    files = []
    for path in ROOT.rglob('*.md'):
        if not PAT.fullmatch(path.name):
            continue
        if any(part.lower() in EXCLUDE for part in path.relative_to(ROOT).parts[:-1]):
            continue
        files.append(path)
    # Prefer the longest non-empty file if there are multiple copies of the same ID.
    docs = {}
    for path in files:
        name = path.stem.upper()
        s = path.read_text(encoding='utf-8-sig', errors='replace')
        if name not in docs or len(s) > len(docs[name][1]):
            docs[name] = (path, s)
    print(f'Markdown proposal files found: {len(files)} | distinct IDs: {len(docs)}')
    print(f'Already labeled document IDs (original CSV): {len(old)}')
    print(f'Document IDs appearing in extra_human.csv: {len(human)}')
    for did in [f'PSAI{x}' for x in range(151, 156)]:
        print(f'{did}: markdown={did in docs} original_label_rows={old[did]} extra_human_rows={human[did]}')

    # Compare every candidate to available USED Markdown documents, and to earlier candidates.
    # ID-based filtering is the primary safeguard; similarity is a secondary screen.
    used = [(did, shingles(body)) for did, (_path, body) in docs.items() if old[did] or human[did]]
    candidates = [(did, path, body, shingles(body)) for did, (path, body) in sorted(docs.items()) if not old[did] and not human[did]]
    rows = []
    for did, path, body, sh in candidates:
        best_id, best_similarity = '', 0.0
        for other_id, other_sh in used:
            sim = jaccard(sh, other_sh)
            if sim > best_similarity:
                best_id, best_similarity = other_id, sim
        state = 'MANUAL_REVIEW_REQUIRED'
        if len(body.strip()) < 800:
            state = 'TOO_SHORT'
        elif best_similarity >= 0.65:
            state = 'NEAR_DUPLICATE_WARNING'
        rows.append({'document_id':did, 'file_path':str(path.relative_to(ROOT)),
                     'characters':len(body),'exact_text_sha256':fingerprint(body),
                     'original_label_rows':0,'extra_human_rows':0,
                     'most_similar_used_md':best_id,
                     'max_used_md_6gram_jaccard':f'{best_similarity:.4f}',
                     'screening_status':state})
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT/'fresh_md_candidates.csv'
    with out.open('w',encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f,fieldnames=['document_id','file_path','characters','exact_text_sha256',
                                        'original_label_rows','extra_human_rows','most_similar_used_md',
                                        'max_used_md_6gram_jaccard','screening_status'])
        wr.writeheader(); wr.writerows(rows)
    print(f'Candidate IDs absent from BOTH labeled CSVs: {len(rows)}')
    print(f'Near duplicates among available used MD docs: {sum(r["screening_status"] == "NEAR_DUPLICATE_WARNING" for r in rows)}')
    print('First potential candidates:', ', '.join(r['document_id'] for r in rows[:15]) or '(none)')
    print('Saved:',out)
    print('IMPORTANT: This does NOT prove a completely independent holdout. '
          'Verify semantic document/version duplicates and annotate BLINDLY before scoring.')

if __name__ == '__main__':
    main()
