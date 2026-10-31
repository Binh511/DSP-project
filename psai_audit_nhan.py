"""PSAI: audit original sentence-classification labels without training or modifying data.

Place this file in D:\\DSP_PROJECT\\DSP_PROJECT and run:
    python psai_audit_nhan.py

Inputs: data_segment_cls/segment_classification.csv
Optional: PSAI_7_cau_de_xuat_doi_nhan.csv (previous AI-reviewed examples)
Outputs: runs/label_audit/* (new reports only)

IMPORTANT: This script detects *candidates for human review*, NOT automatic ground truth.
It does not touch the frozen split, dataset, labels, predictions, or any model.
"""
from __future__ import annotations

import csv
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data_segment_cls" / "segment_classification.csv"
REVIEW7 = ROOT / "PSAI_7_cau_de_xuat_doi_nhan.csv"
OUT = ROOT / "runs" / "label_audit"
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]

REQUIREMENT_CUE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?(?:the (?:system|application|module|platform)\s+)?"
    r"(?:must|shall|should|needs? to|is required to|support(?:s)?\b|enable(?:s)?\b|allow(?:s)?\b)", re.I
)
DELIVERABLE_CUE = re.compile(r"\b(?:this deliverable will|deliverables? include|expected deliverables?|will be delivered)\b", re.I)
HEADING_CUE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[:.)\s-]+)?(?:phase|milestone|week|month|sprint)\b", re.I)
ACTION_CUE = re.compile(r"\b(?:implement|build|develop|design|test|deploy|integrate|write|create|collect|review|fix|optimize|update|document|configure|refactor)\b", re.I)
STRIP_PREFIX = re.compile(r"^[BI]-")


def label_of(value: str) -> str:
    return STRIP_PREFIX.sub("", str(value).replace("\\", "").replace("`", "").strip().upper())


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).casefold().strip()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"CSV không có header: {path}")
        return list(reader), set(reader.fieldnames)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not SOURCE.is_file():
        raise SystemExit("Không thấy file gốc: " + str(SOURCE))
    data, cols = read_csv(SOURCE)
    required = {"document_id", "source_line", "text", "label"}
    if not required <= cols:
        raise SystemExit("Thiếu cột trong dataset: " + ", ".join(sorted(required - cols)))

    before = len(data)
    cleaned, skipped = [], []
    for src_index, row in enumerate(data, 2):
        label = label_of(row.get("label", ""))
        txt = str(row.get("text") or "").strip()
        doc = str(row.get("document_id") or "").strip()
        if label not in LABELS or not txt or not doc:
            skipped.append({"csv_line": src_index, "document_id": doc, "text": txt, "label": row.get("label", "")})
            continue
        cleaned.append({
            "csv_line": src_index,
            "document_id": doc, "source_line": str(row.get("source_line") or "").strip(),
            "text": txt, "label": label, "norm": norm(txt),
        })
    if not cleaned:
        raise SystemExit("Không tìm thấy câu hợp lệ. Dừng kiểm tra tên cột/dữ liệu.")

    # Original row order is preserved when line numbers are non-numeric.
    def sort_key(r):
        try:
            return (r["document_id"], 0, int(r["source_line"]), r["csv_line"])
        except ValueError:
            return (r["document_id"], 1, r["source_line"], r["csv_line"])

    ordered = sorted(cleaned, key=sort_key)
    docs = defaultdict(list)
    for row in ordered:
        docs[row["document_id"]].append(row)
    for docrows in docs.values():
        stack = []
        for idx, row in enumerate(docrows):
            match = re.match(r"^(#+)\s*(.*)$", row["text"])
            if match:
                stack = stack[:len(match.group(1)) - 1] + [match.group(2).strip()]
            row["section"] = " > ".join(stack)
            row["prev_text"] = docrows[idx-1]["text"] if idx else ""
            row["next_text"] = docrows[idx+1]["text"] if idx+1 < len(docrows) else ""

    by_text = defaultdict(list)
    for row in cleaned:
        by_text[row["norm"]].append(row)

    collisions = {k: rows for k, rows in by_text.items() if len({r["label"] for r in rows}) > 1}
    conflicts = []
    for key, rows in collisions.items():
        labs = ", ".join(sorted({r["label"] for r in rows}))
        ids = ", ".join(sorted({r["document_id"] for r in rows}))
        for r in rows:
            conflicts.append({
                "document_id": r["document_id"], "source_line": r["source_line"],
                "text": r["text"], "label": r["label"], "labels_for_same_text": labs,
                "document_ids": ids, "repeat_count": len(rows),
                "section": r["section"], "prev_text": r["prev_text"], "next_text": r["next_text"],
            })

    proposals = {}
    if REVIEW7.is_file():
        previous, fields = read_csv(REVIEW7)
        if {"document_id", "text", "nhan_moi"} <= fields:
            for p in previous:
                key = (str(p["document_id"]).strip(), norm(p["text"]))
                proposals[key] = (str(p["nhan_moi"]).strip(), str(p.get("ly_do", "")).strip())

    queue = []
    for row in cleaned:
        lab, txt = row["label"], row["text"]
        reasons = []
        score = 0
        if row["norm"] in collisions:
            score += 8
            reasons.append("Cùng nội dung nhưng có nhãn khác nhau trong dataset")
        if lab == "ACTIVITY" and REQUIREMENT_CUE.search(txt):
            score += 3
            reasons.append("ACTIVITY có dạng mô tả chức năng/yêu cầu: cân nhắc REQUIREMENT")
        if lab == "ACTIVITY" and DELIVERABLE_CUE.search(txt):
            score += 2
            reasons.append("ACTIVITY nói về deliverable: xem lại ACTIVITY/EXPECTED_RESULT")
        if lab == "ACTIVITY" and HEADING_CUE.search(txt):
            score += 2
            reasons.append("ACTIVITY có dạng tiêu đề milestone/phase; cần xem ngữ cảnh")
        if lab == "ACTIVITY" and len(txt.split()) <= 7 and not ACTION_CUE.search(txt):
            score += 2
            reasons.append("ACTIVITY quá ngắn/không có động từ công việc; cần xem heading")
        if lab == "OTHER" and re.search(r"\b(?:must|shall|should|is required to)\b", txt, re.I):
            score += 1
            reasons.append("OTHER có dấu hiệu câu yêu cầu")
        prior = proposals.get((row["document_id"], row["norm"]))
        suggestion = ""
        if prior:
            suggestion = prior[0]
            if suggestion != lab:
                score += 5
                reasons.append("Có đề xuất rà soát trước đây (AI, chưa phải ground truth)")
        if score == 0:
            continue
        queue.append({
            "do_uu_tien": score,
            "document_id": row["document_id"], "source_line": row["source_line"],
            "text": txt, "label_hien_tai": lab,
            "nhan_tu_de_xuat_cu": suggestion,
            "ly_do_can_xem": " | ".join(reasons),
            "section": row["section"], "prev_text": row["prev_text"], "next_text": row["next_text"],
            "nhan_sau_khi_kiem_duyet": "", "ghi_chu_kiem_duyet": "",
        })
    queue.sort(key=lambda x: (-x["do_uu_tien"], x["document_id"], x["source_line"]))

    OUT.mkdir(parents=True, exist_ok=True)
    fields_q = ["do_uu_tien", "document_id", "source_line", "text", "label_hien_tai",
                "nhan_tu_de_xuat_cu", "ly_do_can_xem", "section", "prev_text", "next_text",
                "nhan_sau_khi_kiem_duyet", "ghi_chu_kiem_duyet"]
    write_csv(OUT / "hang_cho_ra_soat.csv", queue, fields_q)
    write_csv(OUT / "nhan_mau_thuan.csv", conflicts,
              ["document_id", "source_line", "text", "label", "labels_for_same_text",
               "document_ids", "repeat_count", "section", "prev_text", "next_text"])
    write_csv(OUT / "hang_khong_hop_le.csv", skipped, ["csv_line", "document_id", "text", "label"])

    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    freq = Counter(r["label"] for r in cleaned)
    counts = "\n".join(f"  {lab}: {freq[lab]}" for lab in LABELS)
    report = (f"PSAI LABEL AUDIT - KHÔNG TỰ ĐỘNG SỬA NHÃN\n"
              f"Nguồn: {SOURCE}\n"
              f"SHA256: {digest}\n"
              f"Số hàng CSV: {before}\n"
              f"Số câu hợp lệ: {len(cleaned)}\n"
              f"Số tài liệu: {len(docs)}\n"
              f"Số hàng không hợp lệ/bị bỏ qua: {len(skipped)}\n"
              f"Phân bố nhãn:\n{counts}\n"
              f"Số nội dung xuất hiện nhiều lần: {sum(len(v)>1 for v in by_text.values())}\n"
              f"Số nội dung có nhãn MÂU THUẪN: {len(collisions)}\n"
              f"Số hàng thuộc nhóm nhãn mâu thuẫn: {len(conflicts)}\n"
              f"Số hàng được đề xuất RÀ SOÁT: {len(queue)}\n"
              f"Số câu khớp đề xuất AI cũ: {sum(bool(proposals.get((r['document_id'],r['norm']))) for r in cleaned)}\n\n"
              "QUAN TRỌNG:\n"
              "- Đây chỉ là danh sách câu đáng kiểm tra, không phải nhãn sửa tự động.\n"
              "- Cùng câu nhưng khác nhãn có thể hợp lý khi ngữ cảnh khác nhau.\n"
              "- Không sửa tập test đã xem rồi để dùng làm điểm test độc lập.\n"
              "- KHÔNG ghi đè segment_classification.csv, model hoặc frozen_split.json.\n")
    (OUT / "bao_cao.txt").write_text(report, encoding="utf-8-sig")
    print(report)
    print("Đã xuất:")
    for p in ["bao_cao.txt", "nhan_mau_thuan.csv", "hang_cho_ra_soat.csv", "hang_khong_hop_le.csv"]:
        print("  " + str(OUT / p))
    print("\nBước tiếp theo: xem nhan_mau_thuan.csv, sau đó hang_cho_ra_soat.csv.")


if __name__ == "__main__":
    main()
