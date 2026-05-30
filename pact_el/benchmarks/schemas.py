"""Schemas for benchmark source metadata and normalized examples."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from pact_el.schemas import StrictModel


class SourceKind(str, Enum):
    FILE = "file"
    ARCHIVE = "archive"
    HF_ROWS = "hf_rows"


class BenchmarkTaskType(str, Enum):
    MATH = "math"
    INSTRUCTION_FOLLOWING = "instruction_following"
    QUESTION_ANSWERING = "question_answering"
    CODE_GENERATION = "code_generation"
    TRUTHFULNESS = "truthfulness"
    MULTIPLE_CHOICE = "multiple_choice"


class MetricKind(str, Enum):
    NUMERIC_EXACT = "numeric_exact"
    EXACT_MATCH = "exact_match"
    TOKEN_F1 = "token_f1"
    MULTIPLE_CHOICE_ACCURACY = "multiple_choice_accuracy"
    PASS_AT_1 = "pass_at_1"
    OFFICIAL_EVALUATOR = "official_evaluator"


class BenchmarkSource(StrictModel):
    source_id: str
    kind: SourceKind
    url: Optional[str] = None
    local_name: Optional[str] = None
    split: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    hf_config: str = "default"
    hf_split: str = "test"
    archive_members: List[str] = Field(default_factory=list)
    required: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def _validate_source(self) -> "BenchmarkSource":
        if self.kind in {SourceKind.FILE, SourceKind.ARCHIVE} and not self.url:
            raise ValueError("file and archive sources require a url")
        if self.kind == SourceKind.HF_ROWS and not self.hf_dataset_id:
            raise ValueError("hf_rows sources require an hf_dataset_id")
        return self


class BenchmarkSpec(StrictModel):
    benchmark_id: str
    display_name: str
    task_type: BenchmarkTaskType
    adapter: str
    default_split: str
    metrics: List[MetricKind]
    official_url: str
    source_url: str
    paper_url: Optional[str] = None
    license: Optional[str] = None
    evaluator: str
    sources: List[BenchmarkSource]
    description: str
    notes: str = ""


class BenchmarkExample(StrictModel):
    benchmark_id: str
    example_id: str
    split: str
    prompt: str
    expected_answer: Optional[Any] = None
    choices: List[str] = Field(default_factory=list)
    metric: MetricKind
    source_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoreResult(StrictModel):
    example_id: str
    metric: MetricKind
    score: Optional[float]
    passed: Optional[bool]
    prediction: Any
    expected: Any = None
    details: Dict[str, Any] = Field(default_factory=dict)

