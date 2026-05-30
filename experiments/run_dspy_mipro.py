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
from pact_el.ux import (
    install_rich_tracebacks,
    print_config_table,
    print_run_summary,
    print_title,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dataset", required=True, help="Normalized benchmark JSONL to evaluate.")
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
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--lm-kwargs", default="{}", help="Additional JSON kwargs passed to dspy.LM.")
    parser.add_argument("--limit", type=int, help="Limit evaluation rows.")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--derived-val-size", type=int, default=50)
    parser.add_argument("--auto", choices=["light", "medium", "heavy", "none"], default="light")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--num-threads", type=int)
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
        cache=not args.no_cache,
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
    )

    if not args.json:
        print_title("GAVEL DSPy Baseline", f"{args.optimizer} / {args.program}")
        print_config_table(
            "Run Configuration",
            {
                "optimizer": args.optimizer,
                "program": args.program,
                "model": args.model,
                "eval_dataset": args.eval_dataset,
                "train_dataset": args.train_dataset,
                "val_dataset": args.val_dataset,
                "auto": args.auto,
                "limit": args.limit,
                "train_limit": args.train_limit,
                "out": args.out,
            },
        )

    eval_examples = load_examples(Path(args.eval_dataset), limit=args.limit)
    train_examples = (
        load_examples(Path(args.train_dataset), limit=args.train_limit)
        if args.train_dataset
        else None
    )
    val_examples = (
        load_examples(Path(args.val_dataset), limit=args.val_limit)
        if args.val_dataset
        else None
    )

    if args.optimizer == "mipro" and train_examples and val_examples is None:
        train_examples, val_examples = _derive_val_split(
            train_examples,
            val_size=args.derived_val_size,
            seed=args.seed,
        )

    try:
        result = run_dspy_baseline(
            eval_examples,
            config=config,
            train_examples=train_examples,
            val_examples=val_examples,
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


def _derive_val_split(
    examples: list,
    val_size: int,
    seed: int,
) -> tuple[list, Optional[list]]:
    from pact_el.baselines.dspy_mipro import split_train_val

    train_split, val_split = split_train_val(
        examples,
        val_size=val_size,
        seed=seed,
    )
    return train_split, val_split


if __name__ == "__main__":
    main()
