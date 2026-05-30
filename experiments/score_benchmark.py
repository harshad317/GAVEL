"""Score benchmark predictions against normalized benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact_el.benchmarks.scoring import (
    load_normalized_examples,
    load_predictions,
    score_predictions,
    summarize_scores,
)
from pact_el.ux import install_rich_tracebacks, print_score_summary, print_title


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Normalized benchmark JSONL.")
    parser.add_argument("--predictions", required=True, help="JSON/JSONL predictions keyed by example_id.")
    parser.add_argument("--out", help="Optional JSONL score output path.")
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Execute generated code for MBPP pass@1 scoring.",
    )
    parser.add_argument(
        "--workers",
        type=_parse_workers,
        default=1,
        help="Process workers for local scoring.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of rich output.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    install_rich_tracebacks()
    args = build_parser().parse_args()
    if not args.json:
        print_title("GAVEL Benchmark Scoring", Path(args.dataset).name)
    examples = load_normalized_examples(Path(args.dataset))
    predictions = load_predictions(Path(args.predictions))
    results = score_predictions(
        examples,
        predictions,
        allow_code_execution=args.allow_code_execution,
        show_progress=not args.no_progress,
        description=f"Scoring {Path(args.dataset).stem}",
        workers=args.workers,
    )
    if args.out:
        Path(args.out).write_text(
            "\n".join(result.model_dump_json() for result in results)
            + ("\n" if results else "")
        )
    summary = summarize_scores(results)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_score_summary("Score Summary", summary)


def _parse_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--workers must be an integer") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("--workers must be at least 1")
    return workers


if __name__ == "__main__":
    main()
