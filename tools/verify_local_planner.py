"""Verify one real local-LLM plan against an existing extraction/report artifact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "reports/project_completion/local_planner_model.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/project_completion/planner_smoke")
    args = parser.parse_args(argv)
    source_dir = args.input_dir.resolve()
    model_info = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    record = json.loads((source_dir / "extraction.json").read_text(encoding="utf-8"))
    source_text = (source_dir / "source.txt").read_text(encoding="utf-8")
    retrieval_path = source_dir / "retrieval.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8")) if retrieval_path.exists() else None
    if retrieval is not None:
        snapshots = {}
        for key, source in retrieval.get("sources", {}).items():
            artifact = (source_dir / source["artifact"]).resolve()
            if not artifact.is_relative_to(source_dir) or not artifact.is_file():
                raise ValueError("Retrieval snapshot is outside the input report or missing")
            snapshots[key] = artifact.read_text(encoding="utf-8")
        retrieval = {**retrieval, "source_texts": snapshots}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = args.output_root.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    inputs = {name: digest(source_dir / name) for name in ("source.txt", "extraction.json")}
    if retrieval_path.exists():
        inputs["retrieval.json"] = digest(retrieval_path)
    receipt = {"run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(),
               "input_dir": str(source_dir), "input_hashes": inputs,
               "model_manifest_sha256": digest(args.model_manifest),
               "model_repo_id": model_info["repo_id"], "model_revision": model_info["revision"],
               "training_performed": False, "status": "running", "production_model_changed": False}
    write_json(output / "receipt.json", receipt)
    begin = time.perf_counter()
    print(json.dumps({"status": "running", "output": str(output)}, ensure_ascii=False), flush=True)
    try:
        import torch
        from psai_planner import plan_project
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this verification explicitly requires the local GPU")
        torch.cuda.reset_peak_memory_stats()
        plan = plan_project(record, source_text, retrieval, model_info["local_path"],
                            {"device": "cuda", "model_revision": model_info["revision"]})
        if plan["status"] != "validated_candidate" or not plan["steps"] or not plan["provenance"]["model_loaded"]:
            raise RuntimeError("Real-model smoke requires nonempty validated tasks and model_loaded=true")
        write_json(output / "plan.json", plan)
        (output / "raw_response.txt").write_text(plan["raw_response"], encoding="utf-8", newline="")
        write_json(output / "prompt.json", {"messages": plan["messages"]})
        write_json(output / "catalog.json", plan["catalog"])
        receipt.update(status="passed", task_count=len(plan["steps"]),
                       dependency_candidates=len(plan["dependency_candidates"]),
                       human_review_complete=False, quality_score=None,
                       input_tokens=plan["provenance"]["input_tokens"],
                       output_tokens=plan["provenance"]["output_tokens"],
                       gpu_name=torch.cuda.get_device_name(),
                       gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       plan_sha256=digest(output / "plan.json"))
        return_code = 0
    except Exception as error:
        failure = {"error_type": type(error).__name__, "message": str(error),
                   "raw_response": getattr(error, "raw_response", None),
                   "messages": getattr(error, "messages", None),
                   "provenance": getattr(error, "provenance", None)}
        write_json(output / "failure.json", failure)
        if failure["raw_response"] is not None:
            (output / "raw_response.txt").write_text(failure["raw_response"], encoding="utf-8", newline="")
        receipt.update(status="failed", error_type=type(error).__name__, message=str(error))
        return_code = 2
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - begin)
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
