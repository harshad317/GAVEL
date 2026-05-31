"""Terminal UX helpers for benchmark commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_traceback


console = Console()


def install_rich_tracebacks() -> None:
    install_traceback(show_locals=False)


def print_title(title: str, subtitle: str = "") -> None:
    body = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        body += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel.fit(body, border_style="cyan", box=box.ROUNDED))


def print_benchmark_table(specs: Sequence[Any]) -> None:
    table = Table(
        title="Official Benchmark Registry",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Split", style="green", no_wrap=True)
    table.add_column("Task", style="yellow")
    table.add_column("Metrics", style="blue")
    table.add_column("Official Source", style="dim")
    for spec in specs:
        table.add_row(
            spec.benchmark_id,
            spec.display_name,
            spec.default_split,
            spec.task_type.value,
            ", ".join(metric.value for metric in spec.metrics),
            spec.official_url,
        )
    console.print(table)


def print_paths_table(title: str, paths: Mapping[str, Path | str]) -> None:
    table = Table(title=title, box=box.SIMPLE, header_style="bold green")
    table.add_column("Benchmark", style="bold cyan")
    table.add_column("Output", style="white")
    for name, path in paths.items():
        table.add_row(str(name), str(path))
    console.print(table)


def print_score_summary(title: str, summary: Mapping[str, Any]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total", str(summary.get("total", 0)), style="white")
    table.add_row("Scored", str(summary.get("scored", 0)), style="green")
    table.add_row("Unscored", str(summary.get("unscored", 0)), style=_count_style(summary.get("unscored", 0)))
    table.add_row("Passed", str(summary.get("passed", 0)), style="green")
    accuracy = summary.get("accuracy")
    table.add_row(
        "Accuracy",
        "n/a" if accuracy is None else f"{float(accuracy):.2%}",
        style=_score_style(accuracy),
    )
    mean_score = summary.get("mean_score")
    table.add_row(
        "Mean Score",
        "n/a" if mean_score is None else f"{float(mean_score):.4f}",
        style=_score_style(mean_score),
    )
    console.print(table)


def print_run_summary(summary: Mapping[str, Any]) -> None:
    table = Table(title="Baseline Run Summary", box=box.SIMPLE_HEAVY, header_style="bold magenta")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    for key in (
        "optimizer",
        "program",
        "model",
        "optimizer_model",
        "temperature",
        "effective_temperature",
        "optimizer_temperature",
        "reflection_model",
        "reflection_temperature",
        "effective_reflection_temperature",
        "benchmark",
        "eval_examples",
        "train_examples",
        "val_examples",
        "workers",
        "max_in_flight",
        "cache",
        "budget",
        "prompt_portfolio",
        "self_refine_rounds",
        "execution_modes",
        "validation_confidence_z",
        "rejected_candidate_margin",
        "prompt_complexity_margin",
        "num_threads",
        "accepted",
        "decision",
        "selected_prompt",
        "selected_execution_mode",
        "accuracy",
        "mean_score",
        "passed",
        "scored",
        "unscored",
        "predictions_path",
        "scores_path",
        "program_path",
        "report_path",
        "prompt_path",
    ):
        value = summary.get(key)
        style = _score_style(value) if key in {"accuracy", "mean_score"} else "white"
        if key == "accuracy" and value is not None:
            value = f"{float(value):.2%}"
        if key == "mean_score" and value is not None:
            value = f"{float(value):.4f}"
        table.add_row(key, "n/a" if value is None else str(value), style=style)
    console.print(table)


def print_method_results_table(summary: Mapping[str, Any]) -> None:
    method = str(summary.get("method") or summary.get("optimizer") or "method")
    split_results = summary.get("split_results") or {}
    table = Table(box=box.ROUNDED, header_style="bold", show_lines=False)
    table.add_column("Split", style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("StdDev", justify="right")
    table.add_column("Scored", justify="right")
    table.add_column("Unscored", justify="right")
    table.add_column("API calls", justify="right")
    for split_name in ("train", "val", "test", "optimization"):
        row = split_results.get(split_name) or {}
        table.add_row(
            split_name,
            _format_float(row.get("score")),
            _format_percent(row.get("accuracy")),
            _format_float(row.get("stddev")),
            _format_count(row.get("scored")),
            _format_count(row.get("unscored")),
            _format_count(row.get("api_calls")),
            style=_score_style(row.get("score")) if split_name != "optimization" else "dim",
        )
    console.print(Panel(table, title=f" {method} ", border_style="magenta", box=box.ROUNDED))
    _print_scoring_issues(split_results)


def print_selection_summary(manifest: Mapping[str, Any]) -> None:
    table = Table(title="Leakage-Safe Split Selection", box=box.SIMPLE_HEAVY, header_style="bold green")
    table.add_column("Split", style="bold cyan")
    table.add_column("Requested", justify="right")
    table.add_column("Selected", justify="right", style="green")
    table.add_column("Pool", justify="right")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Derived", style="dim")
    for split_name in ("train", "validation", "test"):
        split = (manifest.get("splits") or {}).get(split_name, {})
        table.add_row(
            split_name,
            "all" if split.get("requested") is None else str(split.get("requested")),
            str(split.get("selected", 0)),
            str(split.get("candidate_pool", 0)),
            str(split.get("skipped_for_leakage", 0)),
            str(split.get("derived_from", "")),
        )
    check = manifest.get("leakage_check", {})
    caption = "Leakage check: passed" if check.get("passed") else "Leakage check: not run"
    table.caption = caption
    console.print(table)


def print_config_table(title: str, rows: Mapping[str, Any]) -> None:
    table = Table(title=title, box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Option", style="bold")
    table.add_column("Value", style="white")
    for key, value in rows.items():
        table.add_row(str(key), "n/a" if value is None else str(value))
    console.print(table)


def _score_style(value: Any) -> str:
    if value is None:
        return "dim"
    score = float(value)
    if score >= 0.8:
        return "bold green"
    if score >= 0.5:
        return "bold yellow"
    return "bold red"


def _count_style(value: Any) -> str:
    try:
        return "green" if int(value) == 0 else "yellow"
    except (TypeError, ValueError):
        return "dim"


def _format_float(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2%}"


def _format_count(value: Any) -> str:
    if value is None:
        return "—"
    return str(int(value))


def _print_scoring_issues(split_results: Mapping[str, Any]) -> None:
    issue_rows = []
    for split_name in ("train", "val", "test"):
        row = split_results.get(split_name) or {}
        unscored = int(row.get("unscored") or 0)
        if unscored <= 0:
            continue
        reasons = row.get("unscored_reasons") or []
        if not reasons:
            issue_rows.append((split_name, str(unscored), "score unavailable", "open scores JSONL"))
            continue
        for reason in reasons[:3]:
            fix = reason.get("install") or (
                f"missing module: {reason['missing_module']}"
                if reason.get("missing_module")
                else "open scores JSONL"
            )
            issue_rows.append(
                (
                    split_name,
                    str(reason.get("count", unscored)),
                    escape(str(reason.get("reason", "score unavailable"))),
                    escape(str(fix)),
                )
            )
    if not issue_rows:
        return
    table = Table(box=box.ROUNDED, header_style="bold yellow")
    table.add_column("Split", style="bold cyan")
    table.add_column("Unscored", justify="right", style="yellow")
    table.add_column("Reason", style="white")
    table.add_column("Fix", style="green")
    for row in issue_rows:
        table.add_row(*row)
    console.print(Panel(table, title=" Scoring issue ", border_style="yellow", box=box.ROUNDED))
