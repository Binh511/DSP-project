"""Train-only quality flags; these heuristics never manufacture human labels."""
from collections import Counter
import re
import unicodedata


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def quality_flags(query: str, passage: str) -> list[str]:
    flags = []
    words, source_words = re.findall(r"\w+", normalize(query)), re.findall(r"\w+", normalize(passage))
    if not 4 <= len(words) <= 40:
        flags.append("query_outside_4_40_words")
    if normalize(query) == normalize(passage) or (words and len(words) >= 0.8 * len(source_words)
            and (" " + " ".join(words) + " ") in (" " + " ".join(source_words) + " ")):
        flags.append("copied_source_span")
    if "\ufffd" in passage:
        flags.append("unicode_replacement_character")
    unassigned = sum(unicodedata.category(char) == "Cn" for char in passage)
    if unassigned:
        flags.append("unicode_unassigned_character")
    nonspace = [char for char in passage if not char.isspace()]
    latin1 = sum("\u00c0" <= char <= "\u00ff" for char in nonspace)
    if latin1 >= 6 and latin1 / max(1, len(nonspace)) >= 0.10:
        flags.append("high_latin1_density_possible_garbled_source")
    return flags


def inspect_training(rows: list[dict]) -> dict:
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Training-quality design must not inspect validation or test labels")
    counts, flagged = Counter(), []
    for row in rows:
        flags = quality_flags(row["query"], row["positive"])
        for flag in flags:
            counts[flag] += 1
        if flags:
            flagged.append({"query_id": row["query_id"], "doc_id": row["doc_id"], "flags": flags})
    return {"split": "train", "pairs": len(rows), "flagged_pairs": len(flagged), "flag_counts": dict(counts),
            "flagged": flagged, "positive_teacher_logit_below_zero": sum(row["teacher_positive_score"] < 0 for row in rows),
            "filtered_v2_created": False, "training_run_started": False,
            "interpretation": "Flags request source/human review; negative teacher logits are not proven wrong labels. No threshold selected from validation."}
