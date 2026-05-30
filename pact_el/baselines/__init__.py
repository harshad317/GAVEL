"""Optional external baseline integrations."""

from pact_el.baselines.dspy_mipro import (
    DSPyMIPROConfig,
    DSPyRunResult,
    build_dspy_metric,
    build_gepa_metric,
    run_dspy_baseline,
)
from pact_el.baselines.gavel import GavelConfig, GavelRunResult, run_gavel_baseline

__all__ = [
    "DSPyMIPROConfig",
    "DSPyRunResult",
    "GavelConfig",
    "GavelRunResult",
    "build_dspy_metric",
    "build_gepa_metric",
    "run_dspy_baseline",
    "run_gavel_baseline",
]
