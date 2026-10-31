import csv, shutil, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
LABELS = {"OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER", ""}
P = Path("label_relabel.csv")
BACKUP = P.with_name("label_relabel.before_fix.csv")

with P.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))
header, body = rows[0], rows[1:]
print("Tiêu đề:", header, "| số dòng dữ liệu:", len(body))

new_body, fixed, dropped, unfixable = [], [], [], []
for i, r in enumerate(body, start=2):
    is_last = (i == len(body) + 1)
    if len(r) == 4:
        new_body.append(r)
    elif len(r) == 5 and r[3] == "" and r[4].strip().upper() in LABELS:
        new_body.append(r[:3] + [r[4]])
        fixed.append((i, r[0], r[4]))
    elif is_last and len(r) < 4:
        dropped.append((i, r))
    else:
        unfixable.append((i, len(r), [c[:50] for c in r]))

print("Đã sửa:", fixed)
print("Bỏ dòng cuối bị cắt dở:", dropped)
if unfixable:
    print("\nKHÔNG tự sửa được, cần xem tay:")
    for u in unfixable:
        print("  ", u)
    raise SystemExit("Dừng, chưa ghi gì.")

if not BACKUP.exists():
    shutil.copy2(P, BACKUP)
    print("Đã sao lưu:", BACKUP)
with P.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(new_body)
print("Đã ghi lại", P, "-", len(new_body), "dòng, tất cả đủ 4 cột")