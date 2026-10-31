"""Dense retrieval adapter training, isolated from historical classifier adapters.

Only ``train_adapter`` imports the training stack. Validation and batching can be
tested on a CPU without importing Torch or constructing a model.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time
import traceback
import unicodedata
import uuid


DEFAULTS = {
    "seed": 42, "epochs": 1, "learning_rate": 2e-5,
    "logical_batch_size": 16, "mini_batch_size": 1, "max_length": 256,
    "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
    "warmup_ratio": 0.1, "weight_decay": 0.01, "max_grad_norm": 1.0,
    "save_steps": 10, "gradient_accumulation_steps": 1, "loss_scale": 20.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """Replace a receipt atomically so interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{number}: expected an object")
            rows.append(value)
    return rows


def _publish_checkpoint(staging: Path, committed: Path, latest_path: Path, step: int) -> None:
    """A failed publication cannot modify files in the previous checkpoint."""
    os.replace(staging, committed)
    _write_json(latest_path, {"adapter_dir": str(committed), "global_step": step,
                              "commit_sha256": _sha256(committed / "commit.json")})


def _text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _float32_cosine(left, right):
    """Keep contrastive logits precise while encoder activations remain BF16."""
    import torch
    import torch.nn.functional as functional

    with torch.autocast(device_type=left.device.type, enabled=False):
        return functional.normalize(left.float(), dim=1) @ functional.normalize(right.float(), dim=1).T


def _group(row: dict) -> str:
    return str(row.get("source_group_id") or row.get("family_id") or "")


def validate_training_data(rows: list[dict], manifest: dict,
                           corpus: list[dict] | None = None) -> dict:
    """Reject split leakage and mislabeled/colliding candidates before loading a GPU."""
    if not rows:
        raise ValueError("train_pairs.jsonl is empty")
    split_groups = manifest.get("split_groups")
    if split_groups is None:
        split_groups = {
            key: value.get("source_group_ids", [])
            for key, value in manifest.get("splits", {}).items()
        }
    if not isinstance(split_groups, dict) or not split_groups.get("train"):
        raise ValueError("split_manifest.json needs nonempty split_groups.train")
    groups = {name: {str(item) for item in values}
              for name, values in split_groups.items()}
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            if groups[name] & groups[other]:
                raise ValueError(f"Source-family leakage between {name} and {other}")
    explicit_negatives = [bool(row.get("negative")) for row in rows]
    if any(explicit_negatives) and not all(explicit_negatives):
        raise ValueError("Use a consistent pair or triplet format in train_pairs.jsonl")
    corpus_by_id = None
    if corpus is not None:
        corpus_by_id = {}
        for chunk in corpus:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in corpus_by_id:
                raise ValueError(f"Duplicate corpus chunk ID: {chunk_id}")
            corpus_by_id[chunk_id] = chunk
    query_ids = set()
    all_positives: dict[str, set[str]] = {}
    for row in rows:
        for column in ("query", "positive"):
            if not isinstance(row.get(column), str) or not row[column].strip():
                raise ValueError(f"Missing nonempty {column} in a training row")
        if row.get("negative") and not isinstance(row["negative"], str):
            raise ValueError("A negative passage must be text")
        if row.get("split") != "train":
            raise ValueError("Every training row must explicitly have split='train'")
        if _group(row) not in groups["train"]:
            raise ValueError("Training row belongs to an absent or held-out source family")
        if not row.get("doc_id"):
            raise ValueError("Training rows require doc_id for source-aware batching")
        if row.get("query_id"):
            if row["query_id"] in query_ids:
                raise ValueError("Duplicate query_id in training rows")
            query_ids.add(row["query_id"])
        keys = [_text_key(row["query"]), _text_key(row["positive"])]
        if row.get("negative"):
            keys.append(_text_key(row["negative"]))
        if len(set(keys)) != len(keys):
            raise ValueError("A query, positive, and negative must be distinct texts")
        all_positives.setdefault(keys[0], set()).add(keys[1])
        if corpus_by_id is not None:
            for column, id_column in (("positive", "positive_id"), ("negative", "negative_id")):
                if not row.get(column):
                    continue
                chunk = corpus_by_id.get(str(row.get(id_column, "")))
                if chunk is None:
                    raise ValueError(f"Unknown corpus {id_column}")
                if (chunk.get("split") != "train" or _group(chunk) != _group(row)
                        or str(chunk.get("doc_id")) != str(row["doc_id"])):
                    raise ValueError(f"{column} must come from the query's training document")
                if chunk.get("text") != row[column]:
                    raise ValueError(f"{column} text differs from its corpus chunk")
    for row in rows:
        if row.get("negative") and _text_key(row["negative"]) in all_positives[_text_key(row["query"])]:
            raise ValueError("An explicit negative is a known positive for the same query")
    return {
        "rows": len(rows), "source_groups": len({_group(row) for row in rows}),
        "documents": len({str(row["doc_id"]) for row in rows}),
        "explicit_negatives": all(explicit_negatives),
        "corpus_references_verified": corpus_by_id is not None,
        "supervision": sorted({str(row.get("supervision", "unspecified")) for row in rows}),
        "labels_are_human_ground_truth": False,
    }


class SourceAwareBatchSampler:
    """Deterministic batches with one row per source family and document.

    All query/positive/negative texts are unique within a batch. Rows deferred by
    a collision are retried in the next batch; no training row is discarded.
    """
    def __init__(self, rows: list[dict], batch_size: int, seed: int = 42):
        if batch_size < 2:
            raise ValueError("logical_batch_size must be at least 2")
        self.rows, self.batch_size, self.seed = rows, batch_size, seed

    def batches(self, epoch: int = 0) -> list[list[int]]:
        remaining = list(range(len(self.rows)))
        random.Random(self.seed + epoch).shuffle(remaining)
        result = []
        while remaining:
            batch, deferred = [], []
            families, documents, texts = set(), set(), set()
            for index in remaining:
                row = self.rows[index]
                row_texts = {_text_key(row[key]) for key in ("query", "positive", "negative") if row.get(key)}
                if (len(batch) >= self.batch_size or _group(row) in families
                        or str(row["doc_id"]) in documents or row_texts & texts):
                    deferred.append(index)
                    continue
                batch.append(index)
                families.add(_group(row))
                documents.add(str(row["doc_id"]))
                texts.update(row_texts)
            if len(batch) == 1 and not self.rows[batch[0]].get("negative"):
                raise ValueError("A pair-only singleton batch has no contrastive negatives; supply negatives or more sources")
            result.append(batch)
            remaining = deferred
        return result


def _settings(root: Path, config: dict) -> dict:
    settings = {**DEFAULTS, **config}
    output = Path(settings.get("output_dir", root / "models/rag/retrieval_genq_lora_v1"))
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if not output.is_relative_to(root.resolve()) or output == root.resolve():
        raise ValueError("Adapter output_dir must be a child of the project root")
    settings["output_dir"] = str(output)
    for key in ("epochs", "logical_batch_size", "mini_batch_size", "max_length", "lora_rank", "save_steps"):
        if isinstance(settings[key], bool) or int(settings[key]) != settings[key] or int(settings[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
        settings[key] = int(settings[key])
    if settings["gradient_accumulation_steps"] != 1:
        raise ValueError("This contrastive loop uses gradient_accumulation_steps=1; GradCache supplies the negative pool")
    if not 0 <= float(settings["warmup_ratio"]) < 1:
        raise ValueError("warmup_ratio must be in [0,1)")
    if float(settings["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if settings.get("max_steps") is not None and int(settings["max_steps"]) < 1:
        raise ValueError("max_steps must be positive when supplied")
    return settings


def _base_provenance(base: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", base.name):
        raise ValueError("base_model_dir must be a pinned 40-character Hub snapshot revision")
    if (base / "adapter_config.json").exists():
        raise ValueError("A classifier/PEFT adapter cannot be used as the base encoder")
    required = ["config.json", "modules.json", "tokenizer_config.json", "1_Pooling/config.json"]
    for name in required:
        if not (base / name).is_file():
            raise ValueError(f"Pinned base snapshot is incomplete: {name}")
    architecture = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if (architecture.get("model_type") != "xlm-roberta" or architecture.get("hidden_size") != 1024
            or architecture.get("architectures") != ["XLMRobertaModel"]):
        raise ValueError("Expected a BGE-M3 XLM-R encoder with 1024-dimensional embeddings")
    pooling = json.loads((base / "1_Pooling/config.json").read_text(encoding="utf-8"))
    if not pooling.get("pooling_mode_cls_token") or any(
        value for name, value in pooling.items() if name.startswith("pooling_mode_") and name != "pooling_mode_cls_token"
    ):
        raise ValueError("Pinned base must use CLS-only pooling")
    modules = json.loads((base / "modules.json").read_text(encoding="utf-8"))
    if not any(module["type"].endswith(".Normalize") for module in modules):
        raise ValueError("Pinned base must include its embedding normalization module")
    weights = sorted(base.glob("*.safetensors")) + sorted(base.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("Pinned base snapshot has no encoder weights")
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")
    return {
        "model_id": "BAAI/bge-m3", "revision": base.name, "snapshot": str(base),
        "weights_sha256": {path.name: _sha256(path) for path in weights},
        "tokenizer_sha256": {name: _sha256(base / name) for name in tokenizer_names if (base / name).is_file()},
        "configuration_sha256": {name: _sha256(base / name) for name in required},
        "pooling": pooling, "normalized": True,
    }


def train_adapter(root: Path, study_dir: Path, base_model_dir: Path, config: dict) -> dict:
    """Train a fresh dense-only LoRA adapter, returning a durable run receipt.

    CUDA, BF16, and the installed training libraries are required only when this
    function is invoked. It never downloads weights or saves full base weights.
    Checkpoints contain adapter weights plus small optimizer/RNG state; setting
    ``resume=True`` continues the last committed checkpoint after interruption.
    """
    root, study_dir, base = Path(root).resolve(), Path(study_dir).resolve(), Path(base_model_dir).resolve()
    settings = _settings(root, config)
    output = Path(settings["output_dir"])
    receipt_path = output / "training_receipt.json"
    checkpoints_dir = output / "checkpoints"
    latest_path = output / "latest_checkpoint.json"
    last_dir = None
    resume = bool(settings.get("resume", False))
    if receipt_path.exists() and not resume:
        raise FileExistsError(f"Training output already has a receipt: {receipt_path}; choose a new output or resume=True")
    if (latest_path.exists() or (checkpoints_dir.exists() and any(checkpoints_dir.iterdir()))) and not resume:
        raise FileExistsError("Refusing to overwrite existing checkpoints; choose a fresh output_dir or resume=True")
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1, "status": "preparing", "started_at_unix": time.time(),
        "config": settings, "adapter_dir": None, "global_step": 0,
        "supervision_note": "Synthetic GenQ and teacher-filtered pairs are pseudo-labels, not human ground truth.",
        "best_selection": "Not selected in training; driver evaluates base and last on held-out queries.",
    }
    model = optimizer = loss_fn = None
    torch = None

    def progress(event: str, **values) -> None:
        record = {"event": event, "time_unix": time.time(), **values}
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        _write_json(receipt_path, receipt)
        pairs_path, manifest_path = study_dir / "train_pairs.jsonl", study_dir / "split_manifest.json"
        rows = _read_jsonl(pairs_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        corpus_path = study_dir / "corpus.jsonl"
        corpus = _read_jsonl(corpus_path) if corpus_path.exists() else None
        receipt["data_validation"] = validate_training_data(rows, manifest, corpus)
        if not receipt["data_validation"]["corpus_references_verified"]:
            raise ValueError("corpus.jsonl is required to verify positive/negative provenance")
        receipt["input_sha256"] = {path.name: _sha256(path) for path in (pairs_path, manifest_path, corpus_path)}
        sampler = SourceAwareBatchSampler(rows, settings["logical_batch_size"], settings["seed"])
        plan = [(epoch, batch_number, indices) for epoch in range(settings["epochs"])
                for batch_number, indices in enumerate(sampler.batches(epoch))]
        if settings.get("max_steps") is not None:
            plan = plan[:int(settings["max_steps"])]
        if not plan:
            raise ValueError("No valid contrastive training batches")
        receipt["planned_steps"] = len(plan)
        receipt["actual_batch_sizes"] = {"minimum": min(len(item[2]) for item in plan),
                                          "maximum": max(len(item[2]) for item in plan)}
        progress("hashing_pinned_base", revision=base.name)
        receipt["base"] = _base_provenance(base)
        receipt["source_sha256"] = {str(path.relative_to(root)): _sha256(path)
                                      for path in sorted((root / "psai_rag").glob("*.py"))}
        receipt["versions"] = {name: importlib.metadata.version(name) for name in
                               ("torch", "sentence-transformers", "transformers", "peft", "safetensors")}
        identity = {"config": {key: value for key, value in settings.items() if key != "resume"},
                    "inputs": receipt["input_sha256"], "base": receipt["base"],
                    "source": receipt["source_sha256"], "versions": receipt["versions"]}
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        receipt["run_identity_sha256"] = identity_hash
        checkpoint = None
        if resume:
            pointer = json.loads(latest_path.read_text(encoding="utf-8"))
            last_dir = Path(pointer["adapter_dir"]).resolve()
            if last_dir.parent != checkpoints_dir.resolve():
                raise ValueError("Latest checkpoint points outside the run checkpoint directory")
            commit_path = last_dir / "commit.json"
            if _sha256(commit_path) != pointer["commit_sha256"]:
                raise ValueError("Latest checkpoint commit hash does not match the atomic pointer")
            checkpoint = json.loads(commit_path.read_text(encoding="utf-8"))
            if checkpoint["run_identity_sha256"] != identity_hash:
                raise ValueError("Resume configuration, base, code, versions, or input hashes differ from checkpoint")
            for name, expected in checkpoint["files_sha256"].items():
                if Path(name).name != name:
                    raise ValueError("Checkpoint filenames must be direct child filenames")
                if _sha256(last_dir / name) != expected:
                    raise ValueError(f"Incomplete checkpoint commit: {name}")
            receipt["adapter_dir"] = str(last_dir)
        _write_json(receipt_path, receipt)

        # Keep expensive imports and all CUDA/model activity inside this API.
        import torch
        from peft import LoraConfig, TaskType
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer import losses

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This training configuration requires CUDA with BF16 support")
        random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        progress("loading_fresh_base", device="cuda", dtype="bfloat16", logical_batches=len(plan))
        model = SentenceTransformer(str(base), device="cuda", local_files_only=True,
                                    model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
        model.max_seq_length = settings["max_length"]
        encoder = model.transformers_model
        if getattr(encoder, "peft_config", None):
            raise RuntimeError("Fresh base unexpectedly contains adapters")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if checkpoint:
            model.load_adapter(str(last_dir), adapter_name="default", is_trainable=True,
                               local_files_only=True)
        else:
            model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False,
                                        r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
                                        lora_dropout=settings["lora_dropout"], target_modules=["query", "value"],
                                        bias="none"))
        # PEFT's native Transformers integration can preserve the base dtype;
        # explicitly retain FP32 trainable adapter parameters for stable updates.
        trainable = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"Unexpected trainable base parameter: {name}")
                parameter.data = parameter.data.float()
                trainable.append((name, parameter))
        total = sum(parameter.numel() for parameter in model.parameters())
        count = sum(parameter.numel() for _, parameter in trainable)
        if count <= 0 or count / total > 0.05:
            raise RuntimeError(f"Unexpected LoRA parameter count: {count}/{total}")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        encoder.config.use_cache = False
        model.train()
        receipt["parameters"] = {"total": total, "trainable": count, "fraction": count / total,
                                  "trainable_names": [name for name, _ in trainable],
                                  "frozen_base_dtype": "bfloat16", "adapter_dtype": "float32"}
        receipt["cuda"] = {"device": torch.cuda.get_device_name(0),
                           "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        warmup = math.ceil(len(plan) * settings["warmup_ratio"])

        def lr_factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return max(0.0, (len(plan) - step) / max(1, len(plan) - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        loss_fn = losses.CachedMultipleNegativesRankingLoss(
            model, scale=settings["loss_scale"], mini_batch_size=settings["mini_batch_size"],
            similarity_fct=_float32_cosine,
            gather_across_devices=False, show_progress_bar=False)
        start_step = 0
        if checkpoint:
            state = torch.load(last_dir / "optimizer_state.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            random.setstate(state["python_rng"])
            start_step = state["global_step"]
            if start_step != checkpoint["global_step"] or start_step != pointer["global_step"] or start_step > len(plan):
                raise ValueError("Resume cursor differs between checkpoint state, commit, pointer, or current plan")
            receipt["first_step_proof"] = checkpoint.get("first_step_proof")
            receipt["resumed_from_step"] = start_step
        receipt["status"] = "training"
        receipt["global_step"] = start_step
        _write_json(receipt_path, receipt)
        progress("training_started", trainable_parameters=count, total_parameters=total,
                 planned_steps=len(plan), starting_step=start_step,
                 actual_batch_sizes=receipt["actual_batch_sizes"])

        def save_last(step: int) -> None:
            nonlocal last_dir
            if not getattr(encoder, "_hf_peft_config_loaded", False):
                raise RuntimeError("Native PEFT saving is not active; refusing to save full base weights")
            checkpoints_dir.mkdir(exist_ok=True)
            token = uuid.uuid4().hex
            staging = checkpoints_dir / (".pending_" + token)
            committed = checkpoints_dir / (f"step_{step:06d}_" + token)
            if staging.resolve().parent != checkpoints_dir.resolve() or committed.resolve().parent != checkpoints_dir.resolve():
                raise ValueError("Checkpoint targets must stay inside the run checkpoint directory")
            staging.mkdir()
            # Native Transformers PEFT saving writes adapter weights only.
            encoder.save_pretrained(str(staging), safe_serialization=True)
            if not (staging / "adapter_model.safetensors").exists() or not (staging / "adapter_config.json").exists():
                raise RuntimeError("Expected an adapter-only safetensors checkpoint")
            if (staging / "model.safetensors").exists() or (staging / "pytorch_model.bin").exists():
                raise RuntimeError("Refusing a checkpoint containing full base weights")
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "global_step": step, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()},
                       staging / "optimizer_state.pt")
            files = {path.name: _sha256(path) for path in staging.iterdir() if path.is_file()}
            _write_json(staging / "commit.json", {
                "global_step": step, "run_identity_sha256": identity_hash,
                "files_sha256": files, "first_step_proof": receipt.get("first_step_proof")})
            # Publish a complete immutable directory first, then atomically move
            # the pointer. An interrupted save leaves the prior commit usable.
            _publish_checkpoint(staging, committed, latest_path, step)
            last_dir = committed
            receipt["adapter_dir"] = str(committed)
            receipt["last_committed_step"] = step
            receipt["adapter_sha256"] = files["adapter_model.safetensors"]

        training_start = time.monotonic()
        for step_index in range(start_step, len(plan)):
            epoch, batch_number, indices = plan[step_index]
            batch = [rows[index] for index in indices]
            columns = ["query", "positive"] + (["negative"] if batch[0].get("negative") else [])
            optimizer.zero_grad(set_to_none=True)
            if step_index == start_step:
                torch.cuda.reset_peak_memory_stats()
            features = [{key: value.to("cuda") if hasattr(value, "to") else value
                         for key, value in model.tokenize([row[column] for row in batch]).items()}
                        for column in columns]
            # GradCache replays encoder forwards from a backward hook, so retain
            # the same autocast context around both the loss and its backward.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = loss_fn(features, labels=None)
                loss_value = float(loss.detach().float().cpu())
                if not math.isfinite(loss_value) or loss_value <= 0:
                    raise RuntimeError(f"Contrastive loss is nonfinite or has no signal: {loss_value}")
                loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable], settings["max_grad_norm"], error_if_nonfinite=True).cpu())
            if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError("Adapter gradient norm is zero or nonfinite")
            probe = None
            if step_index == 0:
                for name, parameter in trainable:
                    if parameter.grad is not None and bool(torch.any(parameter.grad != 0)):
                        probe = (name, parameter, parameter.detach().clone())
                        break
            used_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            if step_index == 0:
                if probe is None:
                    raise RuntimeError("No adapter parameter received a nonzero gradient")
                delta = float((probe[1].detach() - probe[2]).abs().max().float().cpu())
                if not math.isfinite(delta) or delta <= 0:
                    raise RuntimeError("First optimizer step did not change adapter weights")
                receipt["first_step_proof"] = {
                    "loss": loss_value, "gradient_norm_before_clip": gradient_norm,
                    "updated_parameter": probe[0], "parameter_max_absolute_change": delta,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "actual_logical_batch_size": len(batch), "physical_mini_batch_size": settings["mini_batch_size"],
                    "explicit_negatives_per_query": int("negative" in columns),
                }
            step = step_index + 1
            record = {"global_step": step, "planned_steps": len(plan), "epoch": epoch + 1,
                      "batch_in_epoch": batch_number + 1, "logical_batch_size": len(batch),
                      "loss": loss_value, "gradient_norm": gradient_norm, "learning_rate": used_lr,
                      "elapsed_seconds": time.monotonic() - training_start,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            with (output / "training_steps.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            receipt.update({"global_step": step, "last_step": record})
            del features, loss, probe
            if step == 1 or step % settings["save_steps"] == 0 or step == len(plan):
                save_last(step)
            _write_json(receipt_path, receipt)
            progress("optimizer_step", **record)
        receipt["status"] = "completed"
        receipt["completed_at_unix"] = time.time()
        receipt["training_seconds"] = time.monotonic() - training_start
        _write_json(receipt_path, receipt)
        progress("training_completed", global_step=receipt["global_step"], adapter_dir=str(last_dir))
        return receipt
    except BaseException as error:
        receipt.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        "failed_at_unix": time.time(), "error_type": type(error).__name__,
                        "error": str(error), "traceback": traceback.format_exc()})
        _write_json(receipt_path, receipt)
        progress("training_stopped", status=receipt["status"], global_step=receipt["global_step"], error=str(error))
        raise
    finally:
        loss_fn = optimizer = model = None
        if torch is not None:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
