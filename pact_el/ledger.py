"""Evidence Ledger construction and ranking utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pact_el.schemas import (
    ContractArea,
    EvidenceOutcome,
    EvidenceRow,
    EvidenceSource,
    Severity,
    StrictModel,
    SEVERITY_WEIGHT,
)


class EvidenceLedger(StrictModel):
    """Structured behavior evidence used by the contract compiler."""

    rows: List[EvidenceRow]

    def add(self, row: EvidenceRow) -> None:
        self.rows.append(row)

    def extend(self, rows: Iterable[EvidenceRow]) -> None:
        self.rows.extend(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def has_behavior_evidence(self) -> bool:
        return any(
            row.success_or_failure in {EvidenceOutcome.SUCCESS, EvidenceOutcome.FAILURE}
            and bool(row.expected_behavior)
            and bool(row.observed_behavior)
            for row in self.rows
        )

    @property
    def failures(self) -> List[EvidenceRow]:
        return [
            row
            for row in self.rows
            if row.success_or_failure == EvidenceOutcome.FAILURE
        ]

    @property
    def successes(self) -> List[EvidenceRow]:
        return [
            row
            for row in self.rows
            if row.success_or_failure == EvidenceOutcome.SUCCESS
        ]

    def prompt_fixable_failures(self, threshold: float = 0.5) -> List[EvidenceRow]:
        return [
            row
            for row in self.failures
            if row.prompt_fixability >= threshold
        ]

    def suspected_area_counts(self) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for row in self.failures:
            if row.suspected_contract_area is not None:
                counter[row.suspected_contract_area.value] += 1
        return dict(counter)

    def weighted_failure_score_by_area(self) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for row in self.failures:
            area = (
                row.suspected_contract_area.value
                if row.suspected_contract_area is not None
                else "unknown"
            )
            scores[area] += (
                SEVERITY_WEIGHT[row.severity]
                * row.confidence
                * row.prompt_fixability
            )
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def to_optimizer_table(self) -> List[Dict[str, Any]]:
        """Return a JSON-serializable table for the optimizer prompt."""

        return [
            {
                "evidence_id": row.evidence_id,
                "source": row.source.value,
                "input": row.input,
                "expected_behavior": row.expected_behavior,
                "observed_behavior": row.observed_behavior,
                "success_or_failure": row.success_or_failure.value,
                "severity": row.severity.value,
                "suspected_contract_area": (
                    row.suspected_contract_area.value
                    if row.suspected_contract_area is not None
                    else None
                ),
                "confidence": row.confidence,
                "prompt_fixability": row.prompt_fixability,
            }
            for row in self.rows
        ]

    def compact_summary(self) -> Dict[str, Any]:
        return {
            "total_rows": len(self.rows),
            "successes": len(self.successes),
            "failures": len(self.failures),
            "suspected_area_counts": self.suspected_area_counts(),
            "weighted_failure_score_by_area": self.weighted_failure_score_by_area(),
        }


def _guess_area(value: Optional[str]) -> Optional[ContractArea]:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    for area in ContractArea:
        if area.value == normalized:
            return area
    return None


def _guess_severity(value: Any, default: Severity = Severity.MEDIUM) -> Severity:
    if value is None:
        return default
    if isinstance(value, Severity):
        return value
    normalized = str(value).strip().lower()
    for severity in Severity:
        if severity.value == normalized:
            return severity
    return default


def build_evidence_ledger(
    logs: Optional[Sequence[Mapping[str, Any]]] = None,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    rows: Optional[Sequence[EvidenceRow]] = None,
) -> EvidenceLedger:
    """Normalize logs and examples into an Evidence Ledger.

    Expected log keys are intentionally permissive because benchmark harnesses
    rarely agree on exact names. The normalized output remains strict.
    """

    ledger = EvidenceLedger(rows=list(rows or []))
    for index, item in enumerate(logs or []):
        passed = item.get("passed")
        if passed is None:
            passed = item.get("success")
        outcome = (
            EvidenceOutcome.SUCCESS
            if passed is True
            else EvidenceOutcome.FAILURE
            if passed is False
            else EvidenceOutcome.UNKNOWN
        )
        ledger.add(
            EvidenceRow(
                evidence_id=str(item.get("evidence_id") or f"log_{index}"),
                source=EvidenceSource.LOG,
                input=item.get("input"),
                expected_behavior=str(
                    item.get("expected_behavior")
                    or item.get("expected")
                    or item.get("label")
                    or "Match the task rubric."
                ),
                observed_behavior=str(
                    item.get("observed_behavior")
                    or item.get("observed")
                    or item.get("failure")
                    or item.get("output")
                    or "No observed behavior supplied."
                ),
                output=item.get("output"),
                success_or_failure=outcome,
                severity=_guess_severity(item.get("severity")),
                suspected_contract_area=_guess_area(
                    item.get("suspected_contract_area")
                    or item.get("contract_area")
                    or item.get("area")
                ),
                confidence=float(item.get("confidence", 0.5)),
                prompt_fixability=float(item.get("prompt_fixability", 0.5)),
                metadata=dict(item.get("metadata") or {}),
            )
        )

    for index, item in enumerate(examples or []):
        expected = (
            item.get("expected_behavior")
            or item.get("expected")
            or item.get("expected_output")
            or "Produce the expected task output."
        )
        observed = (
            item.get("observed_behavior")
            or item.get("observed")
            or item.get("output")
            or "Example supplied without a target output."
        )
        passed = item.get("passed")
        if passed is None:
            passed = item.get("output") == item.get("expected_output")
        ledger.add(
            EvidenceRow(
                evidence_id=str(item.get("evidence_id") or f"example_{index}"),
                source=EvidenceSource.EXAMPLE,
                input=item.get("input"),
                expected_behavior=str(expected),
                observed_behavior=str(observed),
                output=item.get("output"),
                success_or_failure=(
                    EvidenceOutcome.SUCCESS
                    if passed is True
                    else EvidenceOutcome.FAILURE
                    if passed is False
                    else EvidenceOutcome.UNKNOWN
                ),
                severity=_guess_severity(item.get("severity"), Severity.LOW),
                suspected_contract_area=_guess_area(
                    item.get("suspected_contract_area")
                    or item.get("contract_area")
                    or item.get("area")
                ),
                confidence=float(item.get("confidence", 0.5)),
                prompt_fixability=float(item.get("prompt_fixability", 0.5)),
                metadata=dict(item.get("metadata") or {}),
            )
        )

    return ledger


def choose_four_probe_inputs(task_spec: str, rubric: str) -> List[Dict[str, str]]:
    """Create four cold-start probes when no behavior logs exist."""

    return [
        {
            "probe_id": "cold_start_nominal",
            "purpose": "nominal task success",
            "input": (
                "Nominal case for the task. Follow the task specification exactly: "
                f"{task_spec}"
            ),
        },
        {
            "probe_id": "cold_start_boundary",
            "purpose": "boundary and refusal behavior",
            "input": (
                "Boundary case: include ambiguous or underspecified information and "
                f"apply the rubric without inventing unsupported facts. Rubric: {rubric}"
            ),
        },
        {
            "probe_id": "cold_start_format",
            "purpose": "format/schema preservation",
            "input": (
                "Format stress case: produce the required output shape exactly and "
                "avoid extra prose, headers, or commentary."
            ),
        },
        {
            "probe_id": "cold_start_regression",
            "purpose": "easy case regression guard",
            "input": (
                "Easy regression case: solve a simple representative instance while "
                "preserving every explicit instruction."
            ),
        },
    ]


def evidence_from_probe_outputs(
    probes: Sequence[Mapping[str, Any]],
    outputs: Sequence[Any],
    expected_behavior: str = "Follow the task specification and rubric.",
) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for probe, output in zip(probes, outputs):
        rows.append(
            EvidenceRow(
                evidence_id=str(probe.get("probe_id") or f"probe_{len(rows)}"),
                source=EvidenceSource.PROBE,
                input=probe.get("input"),
                expected_behavior=expected_behavior,
                observed_behavior=str(output),
                output=output,
                success_or_failure=EvidenceOutcome.UNKNOWN,
                severity=Severity.MEDIUM,
                suspected_contract_area=None,
                confidence=0.5,
                prompt_fixability=0.5,
                metadata={"probe_purpose": probe.get("purpose")},
            )
        )
    return rows

