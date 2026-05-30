"""Terminal UX helpers for benchmark commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rich import box
from rich.console import Console
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
        "eval_examples",
        "train_examples",
        "val_examples",
        "mean_score",
        "passed",
        "scored",
        "unscored",
        "predictions_path",
        "scores_path",
        "program_path",
    ):
        value = summary.get(key)
        style = _score_style(value) if key == "mean_score" else "white"
        if key == "mean_score" and value is not None:
            value = f"{float(value):.4f}"
        table.add_row(key, "n/a" if value is None else str(value), style=style)
    console.print(table)


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
