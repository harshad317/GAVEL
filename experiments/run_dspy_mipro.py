"""Run DSPy direct, MIPROv2, and GEPA baselines over normalized benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact_el.baselines.dspy_mipro import (
    DSPyMIPROConfig,
    MissingDSPyError,
    load_examples,
    run_dspy_baseline,
)
from pact_el.benchmarks.prepare import prepare_benchmark
from pact_el.benchmarks.registry import get_benchmark_spec
from pact_el.benchmarks.splitting import select_three_way
from pact_el.ux import (
    install_rich_tracebacks,
    print_config_table,
    print_method_results_table,
    print_selection_summary,
    print_run_summary,
    print_title,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        help="Official benchmark id to prepare and split inside this command, e.g. ifbench.",
    )
    parser.add_argument("--benchmark-out", default="data/benchmarks")
    parser.add_argument("--train-split", help="Official split to use as the train pool with --benchmark.")
    parser.add_argument("--val-split", help="Official split to use as the validation pool with --benchmark.")
    parser.add_argument("--test-split", help="Official split to use as the test pool with --benchmark.")
    parser.add_argument("--force-prepare", action="store_true", help="Redownload/rebuild benchmark JSONL files.")
    parser.add_argument("--eval-dataset", help="Normalized benchmark JSONL to evaluate. Alias: --test-dataset.")
    parser.add_argument("--test-dataset", help="Normalized benchmark JSONL used as the held-out test set.")
    parser.add_argument("--train-dataset", help="Normalized benchmark JSONL used by MIPROv2/GEPA.")
    parser.add_argument("--val-dataset", help="Optional normalized validation JSONL for MIPROv2/GEPA.")
    parser.add_argument("--out", default="output/baselines/dspy_mipro")
    parser.add_argument("--optimizer", choices=["dspy", "mipro", "gepa"], default="mipro")
    parser.add_argument("--program", choices=["predict", "cot", "chain_of_thought"], default="cot")
    parser.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--model-type", default="chat")
    parser.add_argument(
        "--temperature",
        type=float,
        help=(
            "Sampling temperature for the task LM, between 0 and 2. "
            "If omitted, DSPy/provider default is used."
        ),
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--workers",
        type=_parse_workers,
        default=1,
        help="Concurrent workers for final evaluation; also used as optimizer num_threads unless overridden.",
    )
    parser.add_argument(
        "--cache",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=None,
        help="Enable or disable DSPy LM caching. Use --cache True or --cache False.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Legacy alias for --cache False.")
    parser.add_argument("--lm-kwargs", default="{}", help="Additional JSON kwargs passed to dspy.LM.")
    parser.add_argument("--test-n", type=int, help="Number of leakage-free test examples to evaluate.")
    parser.add_argument("--train-n", type=int, help="Number of leakage-free training examples.")
    parser.add_argument("--val-n", type=int, help="Number of leakage-free validation examples.")
    parser.add_argument("--limit", type=int, help="Deprecated alias for --test-n.")
    parser.add_argument("--train-limit", type=int, help="Deprecated alias for --train-n.")
    parser.add_argument("--val-limit", type=int, help="Deprecated alias for --val-n.")
    parser.add_argument("--derived-val-size", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, help="Seed for deterministic split selection.")
    parser.add_argument("--auto", choices=["light", "medium", "heavy", "none"], default="light")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--num-threads", type=int, help="Override optimizer compile/evaluation threads.")
    parser.add_argument("--max-metric-calls", type=int, help="GEPA metric-call budget when --auto none.")
    parser.add_argument("--max-full-evals", type=int, help="GEPA full-evaluation budget when --auto none.")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=4)
    parser.add_argument("--metric-threshold", type=float)
    parser.add_argument("--no-minibatch", action="store_true")
    parser.add_argument("--minibatch-size", type=int, default=35)
    parser.add_argument("--minibatch-full-eval-steps", type=int, default=5)
    parser.add_argument("--reflection-model", help="GEPA reflection LM. Defaults to --model.")
    parser.add_argument(
        "--reflection-temperature",
        type=float,
        help="GEPA reflection LM temperature. Defaults to --temperature, then provider default.",
    )
    parser.add_argument("--reflection-max-tokens", type=int)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Allow MBPP generated code execution during scoring.",
    )
    parser.add_argument("--no-save-program", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of rich output.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    install_rich_tracebacks()
    args = build_parser().parse_args()
    test_dataset = args.test_dataset or args.eval_dataset
    if test_dataset is None and args.benchmark is None:
        raise SystemExit("Provide --test-dataset/--eval-dataset or --benchmark.")
    train_n = _coalesce_count(args.train_n, args.train_limit, "train")
    val_n = _coalesce_count(args.val_n, args.val_limit, "val")
    test_n = _coalesce_count(args.test_n, args.limit, "test")
    selection_seed = args.selection_seed if args.selection_seed is not None else args.seed
    if (
        args.optimizer in {"mipro", "gepa"}
        and (args.train_dataset or args.benchmark)
        and val_n is None
        and args.val_dataset is None
        and args.val_split is None
    ):
        val_n = args.derived_val_size
    cache = _resolve_cache(args.cache, args.no_cache)
    lm_kwargs = _parse_lm_kwargs(args.lm_kwargs)
    config = DSPyMIPROConfig(
        model=args.model,
        optimizer=args.optimizer,
        program=args.program,
        output_dir=Path(args.out),
        api_key=args.api_key,
        api_base=args.api_base,
        model_type=args.model_type,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache=cache,
        lm_kwargs=lm_kwargs,
        auto=None if args.auto == "none" else args.auto,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        num_threads=args.num_threads,
        max_metric_calls=args.max_metric_calls,
        max_full_evals=args.max_full_evals,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        metric_threshold=args.metric_threshold,
        minibatch=not args.no_minibatch,
        minibatch_size=args.minibatch_size,
        minibatch_full_eval_steps=args.minibatch_full_eval_steps,
        reflection_model=args.reflection_model,
        reflection_temperature=args.reflection_temperature,
        reflection_max_tokens=args.reflection_max_tokens,
        reflection_minibatch_size=args.reflection_minibatch_size,
        seed=args.seed,
        allow_code_execution=args.allow_code_execution,
        save_program=not args.no_save_program,
        show_progress=not args.no_progress,
        workers=args.workers,
    )

    train_pool, val_pool, test_pool, dataset_info = _load_or_prepare_pools(
        args,
        test_dataset=test_dataset,
        show_progress=not args.no_progress,
    )
    needs_optimization_data = args.optimizer in {"mipro", "gepa"}
    optimizer_label = "MIPROv2" if args.optimizer == "mipro" else "GEPA"
    if needs_optimization_data and train_pool is None:
        raise SystemExit(f"{optimizer_label} requires --train-dataset or --benchmark.")
    if (
        needs_optimization_data
        and dataset_info.get("single_official_pool")
        and (train_n is None or val_n is None or test_n is None)
    ):
        raise SystemExit(
            f"{dataset_info['benchmark']} has one configured official split in this registry; "
            "pass --train-n, --val-n, and --test-n so the command can make disjoint subsets."
        )
    if (
        needs_optimization_data
        and train_pool is not None
        and val_pool is None
        and val_n is not None
        and train_n is None
        and not dataset_info.get("single_official_pool")
    ):
        train_n = max(0, len(train_pool) - val_n)

    if not args.json:
        print_title("GAVEL DSPy Baseline", f"{args.optimizer} / {args.program}")
        print_config_table(
            "Run Configuration",
            {
                "optimizer": args.optimizer,
                "program": args.program,
                "model": args.model,
                "benchmark": args.benchmark,
                "test_dataset": dataset_info.get("test_dataset") or test_dataset,
                "train_dataset": dataset_info.get("train_dataset") or args.train_dataset,
                "val_dataset": dataset_info.get("val_dataset") or args.val_dataset,
                "benchmark_out": args.benchmark_out if args.benchmark else None,
                "auto": args.auto,
                "temperature": args.temperature if args.temperature is not None else "provider_default",
                "workers": args.workers,
                "cache": cache,
                "num_threads": (
                    args.num_threads
                    if args.num_threads is not None
                    else args.workers if needs_optimization_data else None
                ),
                "reflection_model": (
                    (args.reflection_model or args.model) if args.optimizer == "gepa" else None
                ),
                "reflection_temperature": (
                    _effective_reflection_temperature(args) if args.optimizer == "gepa" else None
                ),
                "max_metric_calls": args.max_metric_calls,
                "max_full_evals": args.max_full_evals,
                "train_n": train_n,
                "val_n": val_n,
                "test_n": test_n,
                "selection_seed": selection_seed,
                "out": args.out,
            },
        )

    selection = select_three_way(
        train_pool=train_pool,
        validation_pool=val_pool,
        test_pool=test_pool,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        seed=selection_seed,
    )
    if dataset_info:
        selection.manifest["dataset"] = dataset_info
    if needs_optimization_data and not selection.validation:
        raise SystemExit(
            f"{optimizer_label} requires a non-empty validation set; pass --val-n or --val-dataset."
        )
    if not args.json:
        print_selection_summary(selection.manifest)

    try:
        result = run_dspy_baseline(
            selection.test,
            config=config,
            train_examples=selection.train,
            val_examples=selection.validation,
            selection_summary=selection.manifest,
        )
    except MissingDSPyError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(result.summary, indent=2, sort_keys=True))
    else:
        print_method_results_table(result.summary)
        print_run_summary(result.summary)


def _parse_lm_kwargs(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--lm-kwargs must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--lm-kwargs must decode to a JSON object")
    return parsed


def _load_or_prepare_pools(
    args: argparse.Namespace,
    *,
    test_dataset: Optional[str],
    show_progress: bool,
) -> tuple[Any, Any, Any, Dict[str, Any]]:
    if args.benchmark is None:
        train_pool = load_examples(Path(args.train_dataset)) if args.train_dataset else None
        val_pool = load_examples(Path(args.val_dataset)) if args.val_dataset else None
        if test_dataset is None:
            raise SystemExit("Provide --test-dataset or --eval-dataset.")
        test_pool = load_examples(Path(test_dataset))
        return train_pool, val_pool, test_pool, {}

    spec = get_benchmark_spec(args.benchmark)
    out_dir = Path(args.benchmark_out)
    prepared_paths: Dict[str, str] = {}

    def prepare_split(split: str) -> Path:
        if split not in prepared_paths:
            path = prepare_benchmark(
                spec.benchmark_id,
                out_dir=out_dir,
                split=split,
                force=args.force_prepare,
                show_progress=show_progress,
            )
            prepared_paths[split] = str(path)
        return Path(prepared_paths[split])

    test_split = args.test_split or spec.default_split
    test_path = Path(test_dataset) if test_dataset else prepare_split(test_split)
    test_pool = load_examples(test_path)

    train_path = Path(args.train_dataset) if args.train_dataset else None
    train_split = args.train_split
    single_official_pool = False
    if args.optimizer in {"mipro", "gepa"}:
        if train_path is not None:
            train_pool = load_examples(train_path)
        elif train_split is not None:
            train_path = prepare_split(train_split)
            train_pool = load_examples(train_path)
        else:
            train_pool = test_pool
            train_split = test_split
            train_path = test_path
            single_official_pool = True
    else:
        train_pool = None

    val_path = Path(args.val_dataset) if args.val_dataset else None
    val_split = args.val_split
    if val_path is not None:
        val_pool = load_examples(val_path)
    elif val_split is not None:
        val_path = prepare_split(val_split)
        val_pool = load_examples(val_path)
    else:
        val_pool = None

    dataset_info = {
        "benchmark": spec.benchmark_id,
        "official_url": spec.official_url,
        "source_url": spec.source_url,
        "prepared_paths": prepared_paths,
        "train_split": train_split,
        "validation_split": val_split,
        "test_split": test_split,
        "train_dataset": str(train_path) if train_path else None,
        "val_dataset": str(val_path) if val_path else None,
        "test_dataset": str(test_path),
        "single_official_pool": single_official_pool,
    }
    return train_pool, val_pool, test_pool, dataset_info


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def _parse_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--workers must be an integer") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("--workers must be at least 1")
    return workers


def _resolve_cache(cache: Optional[bool], no_cache: bool) -> bool:
    if cache is not None and no_cache and cache:
        raise SystemExit("Use either --cache True or --no-cache, not both.")
    if no_cache:
        return False
    return True if cache is None else cache


def _effective_reflection_temperature(args: argparse.Namespace) -> str | float:
    if args.reflection_temperature is not None:
        return args.reflection_temperature
    if args.temperature is not None:
        return args.temperature
    return "provider_default"


def _coalesce_count(primary: Optional[int], alias: Optional[int], name: str) -> Optional[int]:
    if primary is not None and alias is not None and primary != alias:
        raise SystemExit(f"Use either --{name}-n or its legacy alias, not conflicting values.")
    return primary if primary is not None else alias


if __name__ == "__main__":
    main()
