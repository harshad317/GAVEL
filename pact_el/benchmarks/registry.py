"""Registry of official benchmark source locations."""

from __future__ import annotations

from typing import Dict, List

from pact_el.benchmarks.schemas import (
    BenchmarkSource,
    BenchmarkSpec,
    BenchmarkTaskType,
    MetricKind,
    SourceKind,
)


OFFICIAL_BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        benchmark_id="gsm8k",
        display_name="GSM8K",
        task_type=BenchmarkTaskType.MATH,
        adapter="gsm8k",
        default_split="test",
        metrics=[MetricKind.NUMERIC_EXACT],
        official_url="https://github.com/openai/grade-school-math",
        source_url="https://github.com/openai/grade-school-math",
        paper_url="https://arxiv.org/abs/2110.14168",
        license="MIT",
        evaluator="numeric final-answer exact match after extracting the answer following ####",
        sources=[
            BenchmarkSource(
                source_id="gsm8k_train",
                kind=SourceKind.FILE,
                split="train",
                url="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
                local_name="train.jsonl",
            ),
            BenchmarkSource(
                source_id="gsm8k_test",
                kind=SourceKind.FILE,
                split="test",
                url="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
                local_name="test.jsonl",
            ),
        ],
        description="Grade-school math word problems with final answers marked by ####.",
    ),
    "ifbench": BenchmarkSpec(
        benchmark_id="ifbench",
        display_name="IFBench",
        task_type=BenchmarkTaskType.INSTRUCTION_FOLLOWING,
        adapter="ifbench",
        default_split="test",
        metrics=[MetricKind.OFFICIAL_EVALUATOR],
        official_url="https://github.com/allenai/IFBench",
        source_url="https://github.com/allenai/IFBench",
        paper_url="https://arxiv.org/abs/2507.02833",
        license="Apache-2.0 code; ODC-BY-1.0 data",
        evaluator="official IFBench verifier functions via run_eval.py",
        sources=[
            BenchmarkSource(
                source_id="ifbench_test",
                kind=SourceKind.FILE,
                split="test",
                url="https://raw.githubusercontent.com/allenai/IFBench/main/data/IFBench_test.jsonl",
                local_name="IFBench_test.jsonl",
            )
        ],
        description="Precise instruction-following prompts with verifiable output constraints.",
        notes="Use the official IFBench evaluator for strict and loose prompt-level accuracy.",
    ),
    "hotpotqa": BenchmarkSpec(
        benchmark_id="hotpotqa",
        display_name="HotpotQA",
        task_type=BenchmarkTaskType.QUESTION_ANSWERING,
        adapter="hotpotqa",
        default_split="dev_distractor",
        metrics=[MetricKind.EXACT_MATCH, MetricKind.TOKEN_F1],
        official_url="https://hotpotqa.github.io/",
        source_url="https://github.com/hotpotqa/hotpot",
        paper_url="https://arxiv.org/abs/1809.09600",
        license="CC BY-SA 4.0",
        evaluator="official hotpot_evaluate_v1.py exact match and F1",
        sources=[
            BenchmarkSource(
                source_id="hotpot_dev_distractor",
                kind=SourceKind.HF_ROWS,
                split="dev_distractor",
                hf_dataset_id="hotpotqa/hotpot_qa",
                hf_config="distractor",
                hf_split="validation",
                local_name="hotpot_dev_distractor.rows.jsonl",
            ),
            BenchmarkSource(
                source_id="hotpot_dev_fullwiki",
                kind=SourceKind.HF_ROWS,
                split="dev_fullwiki",
                hf_dataset_id="hotpotqa/hotpot_qa",
                hf_config="fullwiki",
                hf_split="validation",
                local_name="hotpot_dev_fullwiki.rows.jsonl",
            ),
            BenchmarkSource(
                source_id="hotpot_train",
                kind=SourceKind.HF_ROWS,
                split="train",
                hf_dataset_id="hotpotqa/hotpot_qa",
                hf_config="distractor",
                hf_split="train",
                local_name="hotpot_train.rows.jsonl",
            ),
            BenchmarkSource(
                source_id="hotpot_test_fullwiki",
                kind=SourceKind.HF_ROWS,
                split="test_fullwiki",
                hf_dataset_id="hotpotqa/hotpot_qa",
                hf_config="fullwiki",
                hf_split="test",
                local_name="hotpot_test_fullwiki.rows.jsonl",
            ),
        ],
        description="Multi-hop Wikipedia question answering with supporting facts.",
        notes=(
            "The official GitHub download script points to curtis.ml.cmu.edu; "
            "the default preparer uses the official hotpotqa Hugging Face dataset mirror "
            "for reliable row paging."
        ),
    ),
    "drop": BenchmarkSpec(
        benchmark_id="drop",
        display_name="DROP",
        task_type=BenchmarkTaskType.QUESTION_ANSWERING,
        adapter="drop",
        default_split="dev",
        metrics=[MetricKind.EXACT_MATCH, MetricKind.TOKEN_F1],
        official_url="https://allenai.org/data/drop",
        source_url="https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip",
        paper_url="https://arxiv.org/abs/1903.00161",
        license="CC BY-SA 4.0",
        evaluator="DROP exact match and token F1 over answer bags",
        sources=[
            BenchmarkSource(
                source_id="drop_dataset",
                kind=SourceKind.ARCHIVE,
                split="all",
                url="https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip",
                local_name="drop_dataset.zip",
                archive_members=[
                    "drop_dataset/drop_dataset_train.json",
                    "drop_dataset/drop_dataset_dev.json",
                ],
            )
        ],
        description="Reading comprehension requiring discrete reasoning over paragraphs.",
    ),
    "mbpp": BenchmarkSpec(
        benchmark_id="mbpp",
        display_name="MBPP",
        task_type=BenchmarkTaskType.CODE_GENERATION,
        adapter="mbpp",
        default_split="sanitized",
        metrics=[MetricKind.PASS_AT_1],
        official_url="https://github.com/google-research/google-research/tree/master/mbpp",
        source_url="https://github.com/google-research/google-research/tree/master/mbpp",
        paper_url="https://arxiv.org/abs/2108.07732",
        license="Apache-2.0",
        evaluator="execute generated Python against official tests; code execution must be explicitly enabled",
        sources=[
            BenchmarkSource(
                source_id="mbpp_sanitized",
                kind=SourceKind.FILE,
                split="sanitized",
                url="https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
                local_name="sanitized-mbpp.json",
            ),
            BenchmarkSource(
                source_id="mbpp_full",
                kind=SourceKind.FILE,
                split="full",
                url="https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
                local_name="mbpp.jsonl",
            ),
        ],
        description="Mostly Basic Programming Problems for short Python program synthesis.",
    ),
    "truthfulqa": BenchmarkSpec(
        benchmark_id="truthfulqa",
        display_name="TruthfulQA",
        task_type=BenchmarkTaskType.TRUTHFULNESS,
        adapter="truthfulqa",
        default_split="generation",
        metrics=[MetricKind.OFFICIAL_EVALUATOR, MetricKind.EXACT_MATCH],
        official_url="https://github.com/sylinrl/TruthfulQA",
        source_url="https://github.com/sylinrl/TruthfulQA",
        paper_url="https://arxiv.org/abs/2109.07958",
        license="Apache-2.0",
        evaluator="official TruthfulQA generation/MC evaluators; local exact-alias scoring is diagnostic only",
        sources=[
            BenchmarkSource(
                source_id="truthfulqa_generation",
                kind=SourceKind.FILE,
                split="generation",
                url="https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
                local_name="TruthfulQA.csv",
            ),
            BenchmarkSource(
                source_id="truthfulqa_mc",
                kind=SourceKind.FILE,
                split="mc",
                url="https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/v1/mc_task.json",
                local_name="mc_task.json",
            ),
        ],
        description="Truthfulness questions designed to expose imitative falsehoods.",
    ),
    "livebench_math": BenchmarkSpec(
        benchmark_id="livebench_math",
        display_name="LiveBench Math",
        task_type=BenchmarkTaskType.MATH,
        adapter="livebench_math",
        default_split="test",
        metrics=[MetricKind.EXACT_MATCH],
        official_url="https://github.com/livebench/livebench",
        source_url="https://huggingface.co/datasets/livebench/math",
        paper_url="https://arxiv.org/abs/2406.19314",
        license="Apache-2.0",
        evaluator="LiveBench objective task scorers; exact ground-truth matching is a local first pass",
        sources=[
            BenchmarkSource(
                source_id="livebench_math_test",
                kind=SourceKind.HF_ROWS,
                split="test",
                hf_dataset_id="livebench/math",
                hf_split="test",
                local_name="livebench_math_test.rows.jsonl",
            )
        ],
        description="LiveBench mathematics category loaded from the official LiveBench Hugging Face dataset.",
    ),
    "mmlu": BenchmarkSpec(
        benchmark_id="mmlu",
        display_name="MMLU",
        task_type=BenchmarkTaskType.MULTIPLE_CHOICE,
        adapter="mmlu",
        default_split="test",
        metrics=[MetricKind.MULTIPLE_CHOICE_ACCURACY],
        official_url="https://github.com/hendrycks/test",
        source_url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        paper_url="https://arxiv.org/abs/2009.03300",
        license="MIT",
        evaluator="multiple-choice accuracy over 57 subjects",
        sources=[
            BenchmarkSource(
                source_id="mmlu_hf_all_test",
                kind=SourceKind.HF_ROWS,
                split="test",
                hf_dataset_id="cais/mmlu",
                hf_config="all",
                hf_split="test",
                local_name="mmlu_all_test.rows.jsonl",
            ),
            BenchmarkSource(
                source_id="mmlu_hf_all_validation",
                kind=SourceKind.HF_ROWS,
                split="validation",
                hf_dataset_id="cais/mmlu",
                hf_config="all",
                hf_split="validation",
                local_name="mmlu_all_validation.rows.jsonl",
            ),
            BenchmarkSource(
                source_id="mmlu_hf_all_dev",
                kind=SourceKind.HF_ROWS,
                split="dev",
                hf_dataset_id="cais/mmlu",
                hf_config="all",
                hf_split="dev",
                local_name="mmlu_all_dev.rows.jsonl",
            )
        ],
        description="Massive Multitask Language Understanding across 57 subjects.",
        notes=(
            "The original Hendrycks GitHub points to "
            "https://people.eecs.berkeley.edu/~hendrycks/data.tar; the preparer uses "
            "the CAIS Hugging Face mirror for reliable row paging."
        ),
    ),
    "mmlu_pro": BenchmarkSpec(
        benchmark_id="mmlu_pro",
        display_name="MMLU-Pro",
        task_type=BenchmarkTaskType.MULTIPLE_CHOICE,
        adapter="mmlu_pro",
        default_split="test",
        metrics=[MetricKind.MULTIPLE_CHOICE_ACCURACY],
        official_url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        source_url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        paper_url="https://arxiv.org/abs/2406.01574",
        license="Apache-2.0",
        evaluator="official MMLU-Pro answer extraction and accuracy scripts",
        sources=[
            BenchmarkSource(
                source_id="mmlu_pro_test",
                kind=SourceKind.HF_ROWS,
                split="test",
                hf_dataset_id="TIGER-Lab/MMLU-Pro",
                hf_split="test",
                local_name="mmlu_pro_test.rows.jsonl",
            ),
            BenchmarkSource(
                source_id="mmlu_pro_validation",
                kind=SourceKind.HF_ROWS,
                split="validation",
                hf_dataset_id="TIGER-Lab/MMLU-Pro",
                hf_split="validation",
                local_name="mmlu_pro_validation.rows.jsonl",
            ),
        ],
        description="Harder MMLU variant with reasoning-focused questions and up to ten choices.",
    ),
}


def list_benchmarks() -> List[BenchmarkSpec]:
    return [OFFICIAL_BENCHMARKS[key] for key in sorted(OFFICIAL_BENCHMARKS)]


def get_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    normalized = benchmark_id.strip().lower().replace("-", "_")
    aliases = {
        "livebench math": "livebench_math",
        "livebench-math": "livebench_math",
        "mmlu-pro": "mmlu_pro",
        "mmlu pro": "mmlu_pro",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in OFFICIAL_BENCHMARKS:
        known = ", ".join(sorted(OFFICIAL_BENCHMARKS))
        raise KeyError(f"unknown benchmark {benchmark_id!r}; known benchmarks: {known}")
    return OFFICIAL_BENCHMARKS[normalized]
