"""Prepare normalized benchmark JSONL files from official sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact_el.benchmarks.prepare import prepare_all, prepare_benchmark
from pact_el.benchmarks.registry import list_benchmarks
from pact_el.ux import (
    console,
    install_rich_tracebacks,
    print_benchmark_table,
    print_paths_table,
    print_title,
)


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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of rich output.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    install_rich_tracebacks()
    args = build_parser().parse_args()
    show_progress = not args.no_progress
    if args.benchmark == "list":
        specs = list_benchmarks()
        if args.json:
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
                for spec in specs
            ]
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            print_title("GAVEL Benchmark Sources", "official websites, GitHub repositories, and dataset hosts")
            print_benchmark_table(specs)
        return

    out_dir = Path(args.out)
    if args.benchmark == "all":
        if not args.json:
            print_title("Preparing Benchmarks", f"out={out_dir} limit={args.limit or 'all'}")
        paths = prepare_all(
            out_dir=out_dir,
            limit=args.limit,
            force=args.force,
            show_progress=show_progress,
        )
        if args.json:
            print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
        else:
            print_paths_table("Prepared Datasets", paths)
        return

    if not args.json:
        console.rule(f"[bold cyan]Preparing {args.benchmark}[/bold cyan]")
    path = prepare_benchmark(
        args.benchmark,
        out_dir=out_dir,
        split=args.split,
        limit=args.limit,
        force=args.force,
        show_progress=show_progress,
    )
    if args.json:
        print(json.dumps({"path": str(path)}, indent=2, sort_keys=True))
    else:
        print_paths_table("Prepared Dataset", {args.benchmark: path})


if __name__ == "__main__":
    main()
