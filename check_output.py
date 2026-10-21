import json, random, statistics as st
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

P = "data_extracted/extraction_model.jsonl"
rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]
FIELDS = ["objectives", "requirements", "activities",
          "deliverables", "expected_results", "technologies", "resources"]

allitems = [it for r in rows for f in FIELDS for it in r["fields"][f]]
ok = sum(1 for it in allitems if it.get("verified"))
print(f"verified: {ok}/{len(allitems)} ({ok/len(allitems):.1%})")

print("\nField              median   p90    max   docs_rỗng")
for f in FIELDS:
    c = sorted(len(r["fields"][f]) for r in rows)
    print(f"{f:18s} {st.median(c):6.0f} {c[int(len(c)*0.9)]:5d} {c[-1]:6d} {sum(1 for x in c if x == 0):8d}")

bins = Counter()
for r in rows:
    for it in r["fields"]["activities"]:
        s = it.get("label_score", 0)
        bins[round(min(s, 0.99) // 0.05 * 0.05, 2)] += 1
print("\nđiểm activities:")
for k in sorted(bins):
    print(f"  {k:.2f}-{k+0.05:.2f}: {bins[k]}")

random.seed(0)
low = [(r["document_id"], it) for r in rows for it in r["fields"]["activities"]
       if it.get("label_score", 1) < 0.8]
print(f"\n20 câu activities điểm 0.70-0.80 (tổng {len(low)}):")
for d, it in random.sample(low, min(20, len(low))):
    print(f"  [{it['label_score']:.2f}] {d}: {it['value'][:100]}")