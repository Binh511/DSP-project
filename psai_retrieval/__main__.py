"""Command line entry point for isolated local evidence indexes."""
import argparse
import json
from pathlib import Path
import sys

from .runtime import build_index, search_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("index", help="Build a fresh source-verified Markdown index")
    build.add_argument("--input", "--input-dir", dest="input_path", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--resource-dir", type=Path)
    search = commands.add_parser("search", help="Search only a verified local index")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", action="append", required=True)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--document-id", action="append")
    search.add_argument("--include-resources", action="store_true")
    search.add_argument("--rerank-model-dir", type=Path)
    for command in (build, search):
        command.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
        command.add_argument("--base-model-dir", type=Path)
        command.add_argument("--adapter-dir", type=Path)
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    try:
        common = {key: getattr(args, key) for key in ("mode", "base_model_dir", "adapter_dir", "device")}
        if args.command == "index":
            result = build_index(args.input_path, args.output, resource_dir=args.resource_dir, **common)
        else:
            result = search_index(args.index, args.query, top_k=args.top_k,
                                  document_ids=args.document_id, include_resources=args.include_resources,
                                  rerank_model_dir=args.rerank_model_dir, **common)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(f"Retrieval failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
