"""Cached matched-budget experiment runner for PACT-EL.

This runner is deliberately API-agnostic. For reproducibility it consumes cached
optimizer and target responses, then writes complete OptimizationReport JSON.
Provider-backed experiments can supply LiteLLM clients in a small wrapper while
reusing this dataset and reporting shape.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact_el.clients import ReplayOptimizerClient, ReplayTargetClient
from pact_el.optimize import pact_optimize


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "cases" in payload:
        return list(payload["cases"])
    raise ValueError("dataset must be a JSON list, a JSON object with cases, or JSONL")


def load_cache(path: Path) -> List[Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "outputs" in payload:
        return list(payload["outputs"])
    raise ValueError("cache file must be a JSON list or an object with outputs")


def apply_ablation(case: Dict[str, Any], ablations: Iterable[str]) -> Dict[str, Any]:
    copied = dict(case)
    ablation_set = set(ablations)
    metadata = dict(copied.get("metadata") or {})
    metadata["ablations"] = sorted(ablation_set)
    copied["metadata"] = metadata
    if "no_graph" in ablation_set:
        copied["task_spec"] = (
            copied.get("task_spec", "")
            + "\nAblation: do not use explicit graph structure in analysis."
        )
    if "no_canary_gate" in ablation_set:
        copied["rubric"] = (
            copied.get("rubric", "")
            + "\nAblation: canary gate disabled in downstream analysis."
        )
    if "broad_rewrite" in ablation_set:
        copied["task_spec"] = (
            copied.get("task_spec", "")
            + "\nAblation: allow broad prompt rewrite instead of minimal patch."
        )
    if "no_logic" in ablation_set:
        copied["task_spec"] = (
            copied.get("task_spec", "")
            + "\nAblation: keep guarantees as prose rather than GuaranteeScript."
        )
    return copied


async def run(args: argparse.Namespace) -> None:
    dataset = load_json_or_jsonl(Path(args.dataset))
    optimizer_outputs = load_cache(Path(args.optimizer_cache))
    target_outputs = load_cache(Path(args.target_cache))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer_client = ReplayOptimizerClient(optimizer_outputs)
    target_client = ReplayTargetClient(target_outputs)

    reports = []
    for index, raw_case in enumerate(dataset):
        case = apply_ablation(raw_case, args.ablation)
        report = await pact_optimize(
            prompt=case["prompt"],
            task_spec=case.get("task_spec", ""),
            rubric=case.get("rubric", ""),
            logs=case.get("logs"),
            examples=case.get("examples"),
            optimizer_client=optimizer_client,
            target_client=target_client,
            budget=args.budget,
            allow_one_repair=not args.disable_repair,
        )
        report_path = out_dir / f"report_{index:04d}.json"
        report_path.write_text(report.model_dump_json(indent=2))
        reports.append(report)

    summary = {
        "cases": len(reports),
        "accepted": sum(1 for report in reports if report.accepted),
        "rejected": sum(1 for report in reports if not report.accepted),
        "optimizer_calls": sum(report.call_ledger.optimizer_calls for report in reports),
        "target_calls": sum(report.call_ledger.target_calls for report in reports),
        "judge_calls": sum(report.call_ledger.judge_calls for report in reports),
        "cost_usd": sum(report.call_ledger.total_cost_usd for report in reports),
        "ablations": args.ablation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--optimizer-cache", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--out", default="output/experiments")
    parser.add_argument("--budget", type=int, default=9)
    parser.add_argument("--disable-repair", action="store_true")
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        choices=["no_graph", "no_canary_gate", "broad_rewrite", "no_logic"],
    )
    return parser


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
