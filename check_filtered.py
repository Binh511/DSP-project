import sys
from pathlib import Path

import extraction
import sentence_classifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DOC_ID = "PSAI001"
path = next(Path("data_cleaned").rglob(f"{DOC_ID}/*.md"))
text = path.read_text(encoding="utf-8-sig")

items = list(extraction.sentences_with_context(text))
preds = sentence_classifier.classify([
    {"text": i["text"], "section": i["section"], "prev_text": i["prev"],
     "pos": i["pos"], "next_text": i["next_text"]}
    for i in items
])

thr = extraction.MIN_SCORE_BY_LABEL["ACTIVITY"]
print("Câu mô hình đoán ACTIVITY nhưng bị bộ lọc loại:")
count = 0
for it, p in zip(items, preds):
    sentence = it["text"]
    value = extraction._clean_value(sentence)
    if p["label"] == "ACTIVITY" and p["score"] >= thr and extraction._is_noise(value):
        count += 1
        reason = "bảng" if sentence.count("|") >= 2 else "ngắn"
        print(f"  [{reason}] {p['score']:.2f}  {sentence[:100]}")
print("Tổng:", count)
