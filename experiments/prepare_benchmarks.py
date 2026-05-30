"""Prepare normalized benchmark JSONL files from official sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact_el.benchmarks.prepare import prepare_all, prepare_benchmark
from pact_el.benchmarks.registry import list_benchmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark",
        nargs="?",
        default="list",
        help="Benchmark id, `all`, or `list`.",
    )
    parser.add_argument("--split", help="Split to prepare. Defaults to each benchmark's default split.")
    parser.add_argument("--out", default="data/benchmarks")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.benchmark == "list":
        rows = [
            {
                "benchmark_id": spec.benchmark_id,
                "display_name": spec.display_name,
                "default_split": spec.default_split,
                "task_type": spec.task_type.value,
                "metrics": [metric.value for metric in spec.metrics],
                "official_url": spec.official_url,
                "source_url": spec.source_url,
            }
            for spec in list_benchmarks()
        ]
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    out_dir = Path(args.out)
    if args.benchmark == "all":
        paths = prepare_all(
            out_dir=out_dir,
            limit=args.limit,
            force=args.force,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
        return

    path = prepare_benchmark(
        args.benchmark,
        out_dir=out_dir,
        split=args.split,
        limit=args.limit,
        force=args.force,
    )
    print(path)


if __name__ == "__main__":
    main()
