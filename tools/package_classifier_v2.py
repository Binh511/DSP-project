"""Package frozen classifier v2 code/evidence without importing ML libraries.

Run only after finalizing the v2 documents::

    python tools/package_classifier_v2.py --dry-run
    python tools/package_classifier_v2.py

This is an evidence/code package. Model weights, raw source documents, runtime
environments and machine-specific historical provenance paths are not made
portable by this helper. It never trains a model or runs inference.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile


RELEASE = Path("reports/classifier/release_source_clean_v2")
STUDY = Path("reports/research_review/studies/classifier_source_clean_context_study_v1")
PREPARED = Path("reports/research_review/studies/classifier_source_clean_v1")
CONTEXT = Path("reports/research_review/studies/classifier_context_v2_1")
ORIGINAL_PREP = Path("reports/research_review/studies/user_labels_v1/experiments/baseline_original_labels")
SELECTED_MODEL = Path("models/controlled_study/classifier_source_clean_context_study_v1/baseline_seed_2026")
ARCHIVE_NAME = "PSAI_CLASSIFIER_V2_CODE_VA_BANG_CHUNG.zip"
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
RUNS = tuple(f"{variant}_seed_{seed}" for variant in ("baseline", "context") for seed in (42, 123, 2026))
RUN_EVIDENCE = ("validation_metrics.json", "validation_predictions.csv", "validation_confusion.csv",
                "validation_errors.csv", "provenance.json", "run_status.json")
SOURCE_SEEDS = (
    "classifier_predict.py", "classifier_demo.py", "train_human_cv.py", "fair_fold0.py",
    "classifier_training.py", "study_inputs.py", "study_data.py", "study_evaluation.py",
    "study_experiments.py", "study_final_test.py", "final_test_integrity.py",
    "classifier_source_split.py", "classifier_context_v2.py", "classifier_context_experiment.py",
    "classifier_context_review.py", "classifier_adjudication.py", "classifier_analysis.py",
    "classifier_review_workbench.py", "classifier_followup.py", "classifier_external_packet.py",
    "classifier_validation_replay.py", "replay_validation.py", "reviewed_study.py",
    "analyze_classifier_seeds.py", "build_classifier_report.py",
    "tools/package_classifier_v2.py",
)
OPTIONAL_TOOLS = ("tools/demo_classifier_final.py", "tools/finalize_classifier_release.py",
                  "tools/build_classifier_release_report.py", "tools/test_classifier_release.py")
HISTORICAL_SOURCES = (
    "apply_review.py", "compare30.py", "compare30b.py", "fix_relabel.py", "kfold_step5.py",
    "kfold_step6.py", "kfold_step7.py", "make_check30.py", "make_check30b.py",
    "make_label_queue.py", "make_label_queue2.py", "make_label_queue3.py", "make_relabel.py",
    "merge_extra.py", "merge_extra2.py",
)
LINK_DOCUMENTS = (
    "README_CLASSIFIER.md", "reports/classifier/HUONG_DAN_DEMO_CHO_THAY.md",
    "reports/classifier/release_source_clean_v2/model_card.md",
    "reports/classifier/release_source_clean_v2/DEMO.md", "archive/classifier_legacy/README.md",
)
BLOCKED_MODULES = {
    "rag", "serve", "server", "app", "ocr", "embedding", "vector_db", "chunking", "cleaning",
    "planning", "psai", "pdf_preprocessing", "run_extraction", "study_pipeline", "research_review",
    "build_classifier_handoff", "classifier_training_keepawake",
}
BLOCKED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "cache", "caches", "outputs", "logs", "checkpoint", "checkpoints", "weights", "optimizer", "node_modules",
}
BLOCKED_SUFFIXES = {".zip", ".pdf", ".xlsx", ".xls", ".safetensors", ".bin", ".pt", ".pth",
                    ".ckpt", ".onnx", ".pkl", ".pickle", ".pyc", ".log"}
EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt", ".docx"}
EXCLUDED_RELEASE_FILES = {"package_verification.json", "package_smoke_verification.json"}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_arcname(name):
    """Validate archive names independently of the host operating system."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive member: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"Unsafe archive member: {name!r}")
    if path.as_posix() != name:
        raise ValueError(f"Noncanonical archive member: {name!r}")
    if any(part.casefold() in BLOCKED_PARTS for part in path.parts):
        raise ValueError(f"Excluded directory in archive member: {name}")
    if path.suffix.casefold() in BLOCKED_SUFFIXES:
        raise ValueError(f"Excluded file type in archive member: {name}")
    if any(part.casefold().startswith("checkpoint-") for part in path.parts):
        raise ValueError(f"Checkpoint in archive member: {name}")
    return name


def _within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _local_imports(path, root, local_modules):
    """Inspect the source AST; importing classifier modules could load torch."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path.relative_to(root)))
    candidates = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            candidates.add(module)
            candidates.update(f"{module}.{alias.name}" for alias in node.names)
    return sorted({local_modules[mod] for name in candidates
                   for mod in (name, name.split(".")[0]) if mod in local_modules})


def source_closure(root, seeds):
    """Return only selected sources and their transitive local import closure."""
    modules = {path.stem: path for path in root.glob("*.py")}
    modules.update({f"tools.{path.stem}": path for path in (root / "tools").glob("*.py")})
    pending = list(seeds)
    found = set()
    graph = {}
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if path.stem in BLOCKED_MODULES:
            raise ValueError(f"Classifier source closure reached excluded module: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(f"Required classifier source is missing: {path}")
        dependencies = _local_imports(path, root, modules)
        found.add(path)
        graph[path.relative_to(root).as_posix()] = [p.relative_to(root).as_posix() for p in dependencies]
        pending.extend(dependencies)
    return sorted(found), dict(sorted(graph.items()))


def _selection_matches(selection):
    expected = SELECTED_MODEL.as_posix().casefold()
    refs = [selection.get(key) for key in ("model_dir", "model_path", "selected_model_dir", "candidate_model")]
    for ref in refs:
        if isinstance(ref, str):
            portable = ref.replace("\\", "/").rstrip("/").casefold()
            if portable == expected or portable.endswith("/" + expected):
                return True
    return False


def collect_files(root, release, require_complete=True):
    """Collect by contract; never recurse through the repository or model tree."""
    files = {}
    missing = []

    def add(path, role, required=False):
        path = Path(path)
        if not path.is_file():
            if required:
                missing.append(path.relative_to(root).as_posix())
            return
        resolved = path.resolve()
        if not _within(resolved, root) or path.is_symlink():
            raise ValueError(f"Source escapes the workspace or is a symlink: {path}")
        name = safe_arcname(path.relative_to(root).as_posix())
        previous = files.get(name)
        if previous is None:
            files[name] = (path, role)

    def top(folder, role, suffixes=EVIDENCE_SUFFIXES):
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if (path.is_file() and path.suffix.casefold() in suffixes
                        and path.name.casefold() not in EXCLUDED_RELEASE_FILES):
                    add(path, role)

    add(root / "README_CLASSIFIER.md", "entry_documentation", True)
    for name in ("HUONG_DAN_DEMO_CHO_THAY.md", "HUONG_DAN_DEMO_CHO_THAY.txt", "CURRENT_CLASSIFIER_RELEASE.json"):
        add(root / "reports/classifier" / name, "current_v2_entry_documentation", True)
    add(root / "reports/classifier/handoff_source/history_inventory.json",
        "historical_inventory_not_current_selection", True)
    for name in ("reports/classifier/CLASSIFIER_CLEANUP_20261027.json",
                 "reports/classifier/FINAL_MODEL_MANIFEST.json", "reports/research_review/final_selection.json"):
        add(root / name, "historical_reference_not_current_selection", True)
    historical = root / "archive/classifier_legacy"
    add(historical / "README.md", "historical_source_documentation", True)
    # These historical bytes are references only: do not import them, inspect
    # their import closure, or execute their old data-writing workflows.
    for name in HISTORICAL_SOURCES:
        add(historical / name, "historical_source_not_active", True)
    for name in ("requirements.txt", "requirements-research.txt"):
        add(root / name, "runtime_dependencies", True)
    for path in sorted((root / "examples").glob("classifier*.json")):
        add(path, "demo_input")
    add(root / "examples/classifier_input.json", "demo_input", True)

    source_seeds = [root / name for name in SOURCE_SEEDS]
    source_seeds.extend(root / name for name in OPTIONAL_TOOLS if (root / name).is_file())
    source_seeds.extend(root.glob("test_classifier*.py"))
    sources, graph = source_closure(root, source_seeds)
    for path in sources:
        add(path, "classifier_test" if path.name.startswith("test_") else "classifier_source", True)

    add(release / "FINAL_MODEL_MANIFEST.json", "frozen_v2_model_manifest", True)
    add(release / "final_selection.json", "frozen_v2_selection", True)
    top(release, "v2_release_documentation_and_evidence")
    for subfolder in ("evidence", "source_snapshot"):
        folder = release / subfolder
        if folder.is_dir():
            allowed = EVIDENCE_SUFFIXES | ({".py"} if subfolder == "source_snapshot" else set())
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.casefold() in allowed:
                    if path.name.casefold() not in EXCLUDED_RELEASE_FILES:
                        add(path, "v2_release_evidence")

    study = root / STUDY
    for name in ("experiment_plan.json", "summary.json", "selection_recommendation.json",
                 "completed_experiments.csv", "per_class_comparison.csv", "KET_QUA_DOI_CHUNG.md", "queue_status.json"):
        add(study / name, "six_run_source_clean_study", True)
    for run in RUNS:
        experiment = study / "experiments" / run
        for name in RUN_EVIDENCE:
            add(experiment / name, f"completed_run:{run}", True)
        if (experiment / "run_status.json").is_file():
            status = json.loads((experiment / "run_status.json").read_text(encoding="utf-8-sig"))
            if status.get("state") != "complete":
                raise ValueError(f"Cannot package unfinished experiment: {run}")
        for path in sorted((experiment / "source_snapshot").glob("*.py")):
            add(path, f"run_source_snapshot:{run}")
    for path in sorted((study / "source_snapshot").glob("*.py")):
        add(path, "frozen_study_source_snapshot")

    prepared = root / PREPARED
    for name in ("baseline_training_rows.csv", "baseline_validation_rows.csv", "context_training_rows.csv",
                 "context_validation_rows.csv", "preparation_manifest.json", "source_groups.json", "removed_training_rows.csv", "README.md"):
        add(prepared / name, "source_disjoint_prepared_data", True)
    for path in sorted((prepared / "source_snapshot").glob("*.py")):
        add(path, "preparation_source_snapshot")

    # These are preparation ancestors, not competing current/v1 score reports.
    for name in ("training_rows.csv", "validation_rows.csv"):
        add(root / ORIGINAL_PREP / name, "original_context_preparation_input", True)
    for name in ("preparation_manifest.json", "source_manifest.json"):
        add(root / CONTEXT / name, "revised_context_preparation_input", True)
    top(root / CONTEXT, "revised_context_preparation_input")
    for path in sorted((root / CONTEXT / "source_snapshot").glob("*.py")):
        add(path, "context_preparation_source_snapshot")
    add(root / "runs/fair_fold0/frozen_split.json", "historical_split_preparation_metadata")

    # Selected-export metadata remains under its model path; tokenizer/weights
    # must travel as the complete separate model directory for live inference.
    for name in ("config.json", "inference_config.json", "tokenizer.json", "tokenizer_config.json",
                 "training_provenance.json", "training_state.json"):
        add(root / SELECTED_MODEL / name, "selected_model_metadata", True)

    selection_path = release / "final_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
        if not isinstance(selection, dict) or not _selection_matches(selection):
            raise ValueError("v2 final_selection.json must identify baseline_seed_2026 in the source-clean study")
    if require_complete and missing:
        raise FileNotFoundError("Release package inputs are missing:\n  " + "\n  ".join(sorted(set(missing))))
    return dict(sorted(files.items())), graph, sorted(set(missing))


def start_here(release_name):
    return (
        "PSAI CLASSIFIER RELEASE V2 - CODE AND EVIDENCE\n\n"
        f"Start with README_CLASSIFIER.md and {release_name}/FINAL_MODEL_MANIFEST.json.\n"
        f"The frozen selection is {release_name}/final_selection.json.\n"
        "Selected run: source-clean baseline_seed_2026, seed 2026; validation-based selection.\n"
        "This ZIP contains all six completed comparison runs' saved evidence and the prepared train/validation CSVs.\n"
        "It does not certify independent final-test performance. No training or inference was run to build it.\n\n"
        "LIVE DEMO ON ANOTHER COMPUTER\n"
        "1. Extract the ZIP, preserving the relative directory layout.\n"
        "2. Install a compatible Python environment with the packages in requirements.txt; install the CPU build\n"
        "   of PyTorch for a CPU-only demo. The local reference CPU environment used torch 2.14.1+cpu and\n"
        "   torchvision 0.29.1+cpu; follow the release setup guide for the target machine. The original machine's\n"
        "   .venv and native DLLs are not included. Its system-site-packages setting is specific to that machine.\n"
        "3. Transfer the COMPLETE selected model export separately (weights, tokenizer, config and inference config).\n"
        f"   Original relative model location: {SELECTED_MODEL.as_posix()}\n"
        "4. Follow the v2 demo guide and pass --model-dir with that export's path on the new computer.\n"
        "   tools/demo_classifier_final.py is the frozen-release wrapper when present.\n"
        "   classifier_demo.py also accepts an explicit --model-dir for live CPU inference.\n\n"
        "PORTABILITY AND PROVENANCE\n"
        "PACKAGE_MANIFEST.json uses relative POSIX member paths and SHA256 hashes. Historical provenance files\n"
        "retain original absolute paths as evidence; those paths are not portable configuration. Prepared CSVs\n"
        "are bundled, but rebuilding preparation from raw source Markdown needs the original source corpus,\n"
        "and reproducing training additionally needs the pretrained base model and a compatible ML environment.\n"
        "The code/evidence ZIP is not a self-contained portable training environment.\n"
        "Model weights, optimizer state, checkpoints, PDFs, private human-label spreadsheets, logs and caches\n"
        "are excluded. The v1 release has not been overwritten. No production model is replaced.\n"
    ).encode("utf-8")


def _zip_write(stream, name, data):
    safe_arcname(name)
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    stream.writestr(info, data)


def verify_archive(archive):
    """Verify every member, including the manifest, without extracting files."""
    with zipfile.ZipFile(archive, "r") as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise ValueError("Duplicate archive member names")
        for name in names:
            safe_arcname(name)
        if stream.testzip() is not None:
            raise ValueError("ZIP CRC integrity verification failed")
        manifest_data = stream.read(MANIFEST_NAME)
        manifest = json.loads(manifest_data)
        entries = manifest["entries"]
        if set(names) != set(entries) | {MANIFEST_NAME}:
            raise ValueError("Archive members differ from the contents manifest")
        verified = {}
        for name in sorted(names):
            data = stream.read(name)
            actual = {"sha256": digest(data), "bytes": len(data)}
            if name == MANIFEST_NAME:
                actual["role"] = "portable_contents_manifest"
            else:
                recorded = entries[name]
                if actual["sha256"] != recorded["sha256"] or actual["bytes"] != recorded["bytes"]:
                    raise ValueError(f"Archive SHA256 or byte count mismatch: {name}")
                actual["role"] = recorded["role"]
            verified[name] = actual
        return manifest, verified


def frozen_artifact_coverage(root, members):
    """Require every frozen artifact except the separately transferred weights."""
    frozen = json.loads((root / RELEASE / "FINAL_MODEL_MANIFEST.json").read_text(encoding="utf-8-sig"))
    bindings = frozen["artifact_sha256"]
    excluded_weight = (SELECTED_MODEL / "model.safetensors").as_posix()
    expected = set(bindings)
    covered = expected.intersection(members)
    intentionally_excluded = expected.intersection({excluded_weight})
    missing = expected - covered - intentionally_excluded
    coverage = {
        "frozen_manifest": (RELEASE / "FINAL_MODEL_MANIFEST.json").as_posix(),
        "bound_artifacts": len(bindings), "bound_artifacts_included": len(covered),
        "bound_artifacts_intentionally_excluded": {
            name: {"sha256": bindings[name], "reason": "selected model weights transferred separately"}
            for name in sorted(intentionally_excluded)
        },
        "missing_required_bound_artifacts": sorted(missing),
        "all_required_bound_artifacts_included": not missing,
    }
    if missing:
        raise ValueError("Frozen artifacts need explicit package allowlist entries:\n  " + "\n  ".join(sorted(missing)))
    for name in covered:
        safe_arcname(name)
        if isinstance(members[name], dict) and members[name].get("sha256") != bindings[name]:
            raise ValueError(f"Packaged bytes differ from the frozen artifact hash: {name}")
    return coverage


def verify_documentation_links(archive):
    """Resolve entry-guide links against ZIP members, stripping :line anchors."""
    checked = []
    failures = []
    with zipfile.ZipFile(archive) as stream:
        names = set(stream.namelist())
        for document in LINK_DOCUMENTS:
            text = stream.read(document).decode("utf-8-sig")
            links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
            local_count = 0
            for target in links:
                target = target.strip()
                target = target[1:target.index(">")] if target.startswith("<") and ">" in target else target.split()[0]
                target = re.sub(r":\d+(?::\d+)?(?=$|[#?])", "", target)
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                local_count += 1
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(document), unquote(parsed.path)))
                directory = resolved.rstrip("/") + "/"
                if (resolved.startswith(("../", "/")) or
                        not (resolved in names or any(name.startswith(directory) for name in names))):
                    failures.append({"document": document, "target": target, "resolved": resolved})
            checked.append({"document": document, "local_links_checked": local_count})
    if failures:
        raise ValueError("Entry-guide links are missing from the ZIP: " + json.dumps(failures, ensure_ascii=False))
    return {"passed": True, "documents": checked,
            "local_links_checked": sum(item["local_links_checked"] for item in checked), "broken_links": []}


def build_package(root, release, archive):
    files, graph, _ = collect_files(root, release)
    frozen_artifact_coverage(root, files)
    generated = {"START_HERE.txt": start_here(release.relative_to(root).as_posix())}
    if any(name in files for name in (MANIFEST_NAME, *generated)):
        raise ValueError("A source file collides with a generated package member")
    entries = {}
    release.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".classifier-v2-package-", suffix=".zip", dir=release, delete=False) as tmp:
        temporary_archive = Path(tmp.name)
    try:
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
            for name, (path, role) in files.items():
                data = path.read_bytes()
                entries[name] = {"sha256": digest(data), "bytes": len(data), "role": role}
                _zip_write(stream, name, data)
            for name, data in generated.items():
                entries[name] = {"sha256": digest(data), "bytes": len(data), "role": "package_entry_guide"}
                _zip_write(stream, name, data)
            coverage = frozen_artifact_coverage(root, entries)
            manifest = {
                "schema_version": "psai-classifier-v2-code-evidence-package-v1",
                "release": "source-clean-v2", "selected_model_relative_path": SELECTED_MODEL.as_posix(),
                "member_path_format": "relative POSIX paths; no absolute paths or traversal",
                "hash_algorithm": "SHA256", "manifest_self_hash": "recorded externally in package_verification.json",
                "weights_included": False, "frozen_artifact_coverage": coverage,
                "code_dependency_graph": graph, "entries": dict(sorted(entries.items())),
            }
            _zip_write(stream, MANIFEST_NAME, json_bytes(manifest))
        _, verified = verify_archive(temporary_archive)
        if verified[MANIFEST_NAME]["sha256"] != digest(json_bytes(manifest)):
            raise ValueError("The written contents manifest differs from the generated manifest")
        link_verification = verify_documentation_links(temporary_archive)
        archive_hash = file_digest(temporary_archive)
        archive_size = temporary_archive.stat().st_size
        temporary_archive.replace(archive)
    finally:
        temporary_archive.unlink(missing_ok=True)
    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release": "source-clean-v2", "archive": archive.relative_to(root).as_posix(),
        "sha256": archive_hash, "bytes": archive_size, "files": len(verified),
        "zip_integrity_passed": True, "all_members_sha256_verified": True,
        "portable_archive_member_paths": True, "weights_included": False, "gpu_used": False,
        "training_performed": False, "inference_performed": False,
        "original_absolute_provenance_paths_preserved": True,
        "runtime_and_model_weights_require_separate_setup": True, "entries": verified,
        "frozen_artifact_coverage": coverage,
        "documentation_link_verification": link_verification,
    }
    destination = release / "package_verification.json"
    with tempfile.NamedTemporaryFile(prefix=".classifier-v2-verification-", suffix=".tmp", dir=release, delete=False) as tmp:
        verification_tmp = Path(tmp.name)
        tmp.write(json_bytes(verification))
    try:
        verification_tmp.replace(destination)
    finally:
        verification_tmp.unlink(missing_ok=True)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Workspace root")
    parser.add_argument("--output", type=Path, help="ZIP path inside the v2 release folder")
    parser.add_argument("--dry-run", action="store_true", help="List inputs and missing final documents; write nothing")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    release = (root / RELEASE).resolve()
    if not _within(release, root):
        parser.error("The v2 release folder must remain inside the workspace")
    archive = args.output or release / ARCHIVE_NAME
    if not archive.is_absolute():
        archive = root / archive
    archive = archive.resolve()
    if archive.parent != release or archive.suffix.casefold() != ".zip":
        parser.error("--output must be a ZIP directly inside reports/classifier/release_source_clean_v2")
    if args.dry_run:
        files, graph, missing = collect_files(root, release, require_complete=False)
        print(json.dumps({"dry_run": True, "files": len(files), "missing_required": missing,
                          "archive": archive.relative_to(root).as_posix(), "code_dependency_graph": graph,
                          "members": list(files)}, ensure_ascii=False, indent=2))
        return 0
    verification = build_package(root, release, archive)
    print(json.dumps({key: verification[key] for key in ("archive", "files", "bytes", "sha256", "zip_integrity_passed")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Classifier v2 package failed: {error}", file=sys.stderr)
        raise SystemExit(1)
