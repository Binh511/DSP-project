import json, random
rows = [json.loads(l) for l in open("data_extracted/extraction_model.jsonl", encoding="utf-8")]
random.seed(1)
pool = [(r["document_id"], it) for r in rows for it in r["fields"]["activities"]]
for d, it in random.sample(pool, 25):
    print(f"[{it['label_score']:.2f}] {d} | {it['section'][:40]} | {it['value'][:90]}")