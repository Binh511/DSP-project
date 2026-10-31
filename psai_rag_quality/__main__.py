"""Explicit CPU diagnostics and CSV review commands; no model dependencies."""
import argparse
import json
from pathlib import Path

from .analysis import analyze, read_jsonl
from .gates import inspect_training
from .review import import_reviews, make_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["analyze", "make-review", "train-quality"]:
        command = commands.add_parser(name)
        command.add_argument("--study", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    commands.choices["analyze"].add_argument("--samples", type=int, default=10000)
    commands.choices["make-review"].add_argument("--depth", type=int, default=5)
    commands.choices["make-review"].add_argument("--priority", action="store_true")
    command = commands.add_parser("import-review")
    command.add_argument("--packet", type=Path, required=True)
    command.add_argument("--rater-a", type=Path, required=True)
    command.add_argument("--rater-b", type=Path, required=True)
    command.add_argument("--adjudication", type=Path)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        result = analyze(args.study, args.output, samples=args.samples)
    elif args.command == "make-review":
        result = make_review(args.study, args.output, depth=args.depth, priority=args.priority)
    elif args.command == "import-review":
        result = import_reviews(args.packet, args.rater_a, args.rater_b, args.output, adjudication_path=args.adjudication)
    else:
        result = inspect_training(read_jsonl(args.study / "train_pairs.jsonl"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
