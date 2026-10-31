"""One explicit CUDA optimizer step on disposable, in-memory retrieval pairs.

Run ``python -m psai_rag.verify_training_stack``. The only persisted output is
the smoke receipt; production adapters and all historical models are untouched.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

from psai_rag.train import _float32_cosine, _sha256, _write_json


PINNED_BASE = Path(
    "C:/Users/Admin/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)


def run_smoke(root: Path, base_model_dir: Path = PINNED_BASE, output: Path | None = None) -> dict:
    root, base = Path(root).resolve(), Path(base_model_dir).resolve()
    output = (Path(output) if output else root / "reports/rag_studies/retrieval_genq_lora_v1/gpu_smoke.json").resolve()
    if not output.is_relative_to(root / "reports") or output.suffix != ".json":
        raise ValueError("The smoke receipt must be a JSON file under the project reports directory")
    # Existing cache files suffice; any unexpected network access must fail.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.monotonic()
    receipt = {"schema_version": 1, "status": "starting", "base_snapshot": str(base),
               "base_revision": base.name, "optimizer_steps_requested": 1,
               "production_adapter_saved": False, "pairs_are_disposable_synthetic_fixtures": True,
               "source_sha256": {"verify_training_stack.py": _sha256(Path(__file__)),
                                 "train.py": _sha256(Path(__file__).with_name("train.py"))}}

    def stage(name: str, **values) -> None:
        receipt.update({"stage": name, "elapsed_seconds": time.monotonic() - started, **values})
        _write_json(output, receipt)
        print(json.dumps({"event": "gpu_smoke", "stage": name, **values}, ensure_ascii=False), flush=True)

    torch = model = loss_fn = optimizer = None
    try:
        stage("importing_training_stack")
        import torch
        from peft import LoraConfig, TaskType
        from sentence_transformers import SentenceTransformer, losses

        receipt["versions"] = {name: importlib.metadata.version(name) for name in
                               ("torch", "sentence-transformers", "transformers", "peft")}
        receipt["loss_forward_signature"] = str(inspect.signature(losses.CachedMultipleNegativesRankingLoss.forward))
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The smoke requires a BF16-capable CUDA device")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        receipt["cuda"] = {"device": torch.cuda.get_device_name(0),
                           "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        native_modules = json.loads((base / "modules.json").read_text(encoding="utf-8"))
        pooling = json.loads((base / "1_Pooling/config.json").read_text(encoding="utf-8"))
        if not pooling.get("pooling_mode_cls_token") or pooling.get("pooling_mode_mean_tokens"):
            raise ValueError("Expected native CLS pooling")
        if not any(module["type"].endswith(".Normalize") for module in native_modules):
            raise ValueError("Expected native normalized embeddings")
        receipt["pooling"] = pooling
        receipt["normalized_embeddings"] = True
        torch.cuda.reset_peak_memory_stats()
        stage("loading_local_base")
        model = SentenceTransformer(str(base), device="cuda", local_files_only=True,
                                    model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
        model.max_seq_length = 256
        encoder = model.transformers_model
        if getattr(encoder, "peft_config", None):
            raise RuntimeError("Pinned base unexpectedly contains adapters")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False,
                                    r=8, lora_alpha=16, lora_dropout=0.05,
                                    target_modules=["query", "value"], bias="none"))
        trainable = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"Unexpected trainable base parameter: {name}")
                parameter.data = parameter.data.float()
                trainable.append((name, parameter))
        receipt["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                  "trainable": sum(p.numel() for _, p in trainable),
                                  "trainable_tensors": len(trainable),
                                  "frozen_dtypes": sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad}),
                                  "trainable_dtypes": sorted({str(p.dtype) for _, p in trainable})}
        if receipt["parameters"]["trainable"] != 786432:
            raise RuntimeError("Unexpected rank-8 query/value LoRA parameter count")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        encoder.config.use_cache = False
        model.train()
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=2e-5, weight_decay=0.01)
        loss_fn = losses.CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=1, scale=20, similarity_fct=_float32_cosine,
            gather_across_devices=False, show_progress_bar=False)
        # Related distractors keep this tiny fixture a meaningful ranking task.
        rows = [
            {"query": "How long can a student borrow a library book?",
             "positive": "Students can borrow ordinary library books for fourteen days and renew a loan once.",
             "negative": "Teachers can borrow library books for thirty days; students can use reference books only inside the library."},
            {"query": "Sinh viên đăng ký học phần ở đâu?",
             "positive": "Sinh viên đăng ký học phần trên cổng đào tạo trực tuyến của trường trong thời gian quy định.",
             "negative": "Giảng viên cập nhật điểm học phần trên hệ thống quản lý đào tạo sau khi kết thúc kỳ thi."},
        ]
        features = [{name: value.to("cuda") if hasattr(value, "to") else value
                     for name, value in model.tokenize([row[column] for row in rows]).items()}
                    for column in ("query", "positive", "negative")]
        stage("contrastive_forward", logical_batch_size=2, mini_batch_size=1,
              max_length=256, explicit_negatives_per_query=1,
              gradient_checkpointing_use_reentrant=False, parameters=receipt["parameters"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # ST 6.1 requires the labels argument even for this unlabeled loss.
            loss = loss_fn(features, labels=None)
            loss_value = float(loss.detach().float().cpu())
            if not math.isfinite(loss_value) or loss_value <= 0:
                raise RuntimeError(f"Loss has no finite positive training signal: {loss_value}")
            stage("contrastive_backward", loss=loss_value, loss_dtype=str(loss.dtype))
            loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for _, p in trainable], 1.0, error_if_nonfinite=True).cpu())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("Adapter gradients are zero or nonfinite")
        probe = next(((name, p, p.detach().clone()) for name, p in trainable
                      if p.grad is not None and bool(torch.any(p.grad != 0))), None)
        if probe is None:
            raise RuntimeError("No adapter received a nonzero gradient")
        optimizer.step()
        torch.cuda.synchronize()
        delta = float((probe[1].detach() - probe[2]).abs().max().float().cpu())
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError("The optimizer did not change an adapter parameter")
        receipt.update({"status": "passed", "optimizer_steps_completed": 1,
                        "loss": loss_value, "gradient_norm_before_clip": gradient_norm,
                        "updated_parameter": probe[0], "parameter_max_absolute_change": delta,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "memory_scope": "includes local base loading and the optimizer step"})
        stage("completed", status="passed", optimizer_steps_completed=1,
              peak_allocated_bytes=receipt["peak_allocated_bytes"], peak_reserved_bytes=receipt["peak_reserved_bytes"],
              gradient_norm_before_clip=gradient_norm, parameter_max_absolute_change=delta)
        return receipt
    except BaseException as error:
        receipt.update({"status": "failed", "error_type": type(error).__name__,
                        "error": str(error), "traceback": traceback.format_exc()})
        if torch is not None and torch.cuda.is_available():
            receipt.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        stage("failed", status="failed", error=str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-model-dir", type=Path, default=PINNED_BASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_smoke(args.root, args.base_model_dir, args.output)


if __name__ == "__main__":
    main()
