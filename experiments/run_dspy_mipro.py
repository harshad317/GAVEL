"""Run DSPy direct and MIPROv2 baselines over normalized benchmark JSONL files."""

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
from pact_el.benchmarks.splitting import select_three_way
from pact_el.ux import (
    install_rich_tracebacks,
    print_config_table,
    print_selection_summary,
    print_run_summary,
    print_title,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dataset", help="Normalized benchmark JSONL to evaluate. Alias: --test-dataset.")
    parser.add_argument("--test-dataset", help="Normalized benchmark JSONL used as the held-out test set.")
    parser.add_argument("--train-dataset", help="Normalized benchmark JSONL used by MIPROv2.")
    parser.add_argument("--val-dataset", help="Optional normalized validation JSONL for MIPROv2.")
    parser.add_argument("--out", default="output/baselines/dspy_mipro")
    parser.add_argument("--optimizer", choices=["dspy", "mipro"], default="mipro")
    parser.add_argument("--program", choices=["predict", "cot", "chain_of_thought"], default="cot")
    parser.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--model-type", default="chat")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--workers",
        type=_parse_workers,
        default=1,
        help="Concurrent workers for final evaluation; also used as MIPRO num_threads unless overridden.",
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
    parser.add_argument("--train-n", type=int, help="Number of leakage-free training examples for MIPROv2.")
    parser.add_argument("--val-n", type=int, help="Number of leakage-free validation examples for MIPROv2.")
    parser.add_argument("--limit", type=int, help="Deprecated alias for --test-n.")
    parser.add_argument("--train-limit", type=int, help="Deprecated alias for --train-n.")
    parser.add_argument("--val-limit", type=int, help="Deprecated alias for --val-n.")
    parser.add_argument("--derived-val-size", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, help="Seed for deterministic split selection.")
    parser.add_argument("--auto", choices=["light", "medium", "heavy", "none"], default="light")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--num-threads", type=int, help="Override MIPROv2 compile threads.")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=4)
    parser.add_argument("--metric-threshold", type=float)
    parser.add_argument("--no-minibatch", action="store_true")
    parser.add_argument("--minibatch-size", type=int, default=35)
    parser.add_argument("--minibatch-full-eval-steps", type=int, default=5)
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
    if test_dataset is None:
        raise SystemExit("Provide --test-dataset or --eval-dataset.")
    train_n = _coalesce_count(args.train_n, args.train_limit, "train")
    val_n = _coalesce_count(args.val_n, args.val_limit, "val")
    test_n = _coalesce_count(args.test_n, args.limit, "test")
    selection_seed = args.selection_seed if args.selection_seed is not None else args.seed
    if args.optimizer == "mipro" and args.train_dataset and val_n is None and args.val_dataset is None:
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
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        metric_threshold=args.metric_threshold,
        minibatch=not args.no_minibatch,
        minibatch_size=args.minibatch_size,
        minibatch_full_eval_steps=args.minibatch_full_eval_steps,
        seed=args.seed,
        allow_code_execution=args.allow_code_execution,
        save_program=not args.no_save_program,
        show_progress=not args.no_progress,
        workers=args.workers,
    )

    train_pool = load_examples(Path(args.train_dataset)) if args.train_dataset else None
    val_pool = load_examples(Path(args.val_dataset)) if args.val_dataset else None
    test_pool = load_examples(Path(test_dataset))
    if args.optimizer == "mipro" and train_pool is None:
        raise SystemExit("MIPROv2 requires --train-dataset.")
    if (
        args.optimizer == "mipro"
        and train_pool is not None
        and val_pool is None
        and val_n is not None
        and train_n is None
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
                "test_dataset": test_dataset,
                "train_dataset": args.train_dataset,
                "val_dataset": args.val_dataset,
                "auto": args.auto,
                "workers": args.workers,
                "cache": cache,
                "num_threads": (
                    args.num_threads
                    if args.num_threads is not None
                    else args.workers if args.optimizer == "mipro" else None
                ),
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
    if args.optimizer == "mipro" and not selection.validation:
        raise SystemExit("MIPROv2 requires a non-empty validation set; pass --val-n or --val-dataset.")
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
        print_run_summary(result.summary)


def _parse_lm_kwargs(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--lm-kwargs must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--lm-kwargs must decode to a JSON object")
    return parsed


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


def _coalesce_count(primary: Optional[int], alias: Optional[int], name: str) -> Optional[int]:
    if primary is not None and alias is not None and primary != alias:
        raise SystemExit(f"Use either --{name}-n or its legacy alias, not conflicting values.")
    return primary if primary is not None else alias


if __name__ == "__main__":
    main()
