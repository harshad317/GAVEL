"""Optional external baseline integrations."""

from pact_el.baselines.dspy_mipro import (
    DSPyMIPROConfig,
    DSPyRunResult,
    build_dspy_metric,
    build_gepa_metric,
    run_dspy_baseline,
)

__all__ = [
    "DSPyMIPROConfig",
    "DSPyRunResult",
    "build_dspy_metric",
    "build_gepa_metric",
    "run_dspy_baseline",
]
