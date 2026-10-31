import json, sys
import planning

sys.stdout.reconfigure(encoding="utf-8")
DOC_ID = sys.argv[1] if len(sys.argv) > 1 else "PSAI001"

record = None
with open("data_extracted/extraction_model.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r["document_id"] == DOC_ID:
                record = r
                break
if record is None:
    raise SystemExit(f"Không thấy {DOC_ID}")

plan = planning.generate_plan(record)
print(f"=== {DOC_ID}: {len(plan['steps'])} bước kế hoạch ===")
for s in plan["steps"][:30]:
    print(f"{s['step']:3d}. {s['description'][:120]}")
