# -*- coding: utf-8 -*-
"""Make a NEW PSAI classification CSV from AI-reviewed proposals, NEVER modify original.

Put this file AND PSAI_ra_soat_321_cau_AI.csv in D:\DSP_PROJECT\DSP_PROJECT.
Preview: python tao_dataset_v2_psai.py --mode preview
Export:  python tao_dataset_v2_psai.py --mode export
Default: apply only HIGH confidence AI proposals (still not expert ground truth).
Optional for ANALYSIS ONLY: --include-medium (also apply MEDIUM confidence proposals).
Do not use a previously inspected / edited test set as independent final evaluation.
"""
import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIG = ROOT / 'data_segment_cls' / 'segment_classification.csv'
REVIEW = ROOT / 'PSAI_ra_soat_321_cau_AI.csv'
OUT_DIR = ROOT / 'runs' / 'label_audit'
EXPECTED_SHA = '45cb04aa78072c5a0607ff44af4cf71c1e699032bce8adbf19fdfcdadcc41c99'
LABELS = {'OBJECTIVE','REQUIREMENT','ACTIVITY','EXPECTED_RESULT','OTHER'}

def norm(x):
    return re.sub(r'\s+', ' ', str(x)).casefold().strip()

def label(x):
    return re.sub(r'^[BI]-', '', str(x).replace('\\','').replace('`','').strip().upper())

def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames:raise ValueError('File CSV không có header: '+str(path))
        return list(rd), rd.fieldnames

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', choices=['preview','export'],default='preview')
    p.add_argument('--include-medium',action='store_true',help='Optional only; not confirmed by annotators')
    args=p.parse_args()
    if not ORIG.is_file() or not REVIEW.is_file():
        raise SystemExit('Thiếu dữ liệu gốc hoặc PSAI_ra_soat_321_cau_AI.csv tại '+str(ROOT))
    sha=hashlib.sha256(ORIG.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise SystemExit('DỪNG: SHA256 dataset gốc khác báo cáo audit. Không áp dụng nhãn tự động.\n'
                         f'  expected: {EXPECTED_SHA}\n  actual:   {sha}')
    src,headers=read_csv(ORIG)
    audit,cols=read_csv(REVIEW)
    required={'document_id','source_line','text','nhan_hien_tai','nhan_AI_de_xuat','quyet_dinh_AI','do_tin_cay_AI'}
    if not required<=set(cols):raise SystemExit('File rà soát thiếu cột: '+', '.join(sorted(required-set(cols))))
    if not {'document_id','source_line','text','label'}<=set(headers):raise SystemExit('Dataset gốc thiếu cột bắt buộc')
    allowed={'Cao','Vừa'} if args.include_medium else {'Cao'}
    patches={}
    for r in audit:
        if r['quyet_dinh_AI']!='ĐỔI' or r['do_tin_cay_AI'] not in allowed:continue
        old,new=r['nhan_hien_tai'],r['nhan_AI_de_xuat']
        if old not in LABELS or new not in LABELS:raise SystemExit('Nhãn không hợp lệ trong review')
        key=(r['document_id'].strip(),r['source_line'].strip(),norm(r['text']))
        if key in patches:raise SystemExit('Trùng key trong review: '+str(key))
        patches[key]=(old,new)
    found=Counter()
    out=[]
    for r in src:
        row=dict(r)
        key=(str(r['document_id']).strip(),str(r['source_line']).strip(),norm(r['text']))
        if key in patches:
            old,new=patches[key]
            if label(r['label'])!=old:raise SystemExit(f'Nhãn gốc khác review tại {key[:2]}: {r["label"]} vs {old}')
            found[key]+=1
            if found[key]>1:raise SystemExit('DỪNG: source row key không duy nhất: '+str(key))
            prefix='B-' if str(r['label']).strip().startswith('B-') else 'I-' if str(r['label']).strip().startswith('I-') else ''
            row['label']=prefix+new
        out.append(row)
    missing=set(patches)-set(found)
    if missing:raise SystemExit(f'DỪNG: {len(missing)} đề xuất không khớp với dataset gốc. Ví dụ: {list(missing)[:3]}')
    print('Bản gốc:',len(src),'dòng; SHA256:',sha)
    print('Chế độ:',args.mode,'| Chỉ AI confidence cao' if not args.include_medium else '| Gồm confidence vừa')
    print('Số thay đổi đề xuất khớp đúng vị trí:',sum(found.values()))
    print('Dữ liệu test đã được phân tích lỗi: KHÔNG coi v2 là test độc lập.')
    if args.mode=='preview':
        print('Không có file nào được thay đổi. Muốn xuất bản sao: --mode export')
        return
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    name='segment_classification_v2_AI_high_plus_medium.csv' if args.include_medium else 'segment_classification_v2_AI_high.csv'
    dest=OUT_DIR/name
    if dest.exists():raise SystemExit('File kết quả đã tồn tại, tránh ghi đè: '+str(dest))
    with dest.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=headers)
        wr.writeheader();wr.writerows(out)
    print('ĐÃ XUẤT BẢN SAO (không ghi đè dữ liệu gốc):',dest)
    print('Chưa cập nhật script train hay các checkpoint.')

if __name__=='__main__':main()
