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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = load_normalized_examples(Path(args.dataset))
    predictions = load_predictions(Path(args.predictions))
    results = score_predictions(
        examples,
        predictions,
        allow_code_execution=args.allow_code_execution,
    )
    if args.out:
        Path(args.out).write_text(
            "\n".join(result.model_dump_json() for result in results)
            + ("\n" if results else "")
        )
    print(json.dumps(summarize_scores(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
