"""Normalize official benchmark files into BenchmarkExample JSONL rows."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from pact_el.benchmarks.schemas import BenchmarkExample, BenchmarkSpec, MetricKind


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize_examples(
    spec: BenchmarkSpec,
    split: str,
    source_paths: Sequence[Path],
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[BenchmarkExample]:
    if spec.adapter == "gsm8k":
        return list(_normalize_gsm8k(spec, split, source_paths[0]))
    if spec.adapter == "ifbench":
        return list(_normalize_ifbench(spec, split, source_paths[0]))
    if spec.adapter == "hotpotqa":
        if rows is not None:
            return list(_normalize_hotpotqa_rows(spec, split, rows))
        return list(_normalize_hotpotqa(spec, split, source_paths[0]))
    if spec.adapter == "drop":
        return list(_normalize_drop(spec, split, source_paths))
    if spec.adapter == "mbpp":
        return list(_normalize_mbpp(spec, split, source_paths[0]))
    if spec.adapter == "truthfulqa":
        return list(_normalize_truthfulqa(spec, split, source_paths[0]))
    if spec.adapter == "livebench_math":
        return list(_normalize_livebench_math(spec, split, rows or _read_jsonl(source_paths[0])))
    if spec.adapter == "mmlu":
        if rows is not None:
            return list(_normalize_mmlu_rows(spec, split, rows))
        return list(_normalize_mmlu(spec, split, source_paths))
    if spec.adapter == "mmlu_pro":
        return list(_normalize_mmlu_pro(spec, split, rows or _read_jsonl(source_paths[0])))
    raise ValueError(f"unsupported benchmark adapter: {spec.adapter}")


def _normalize_gsm8k(
    spec: BenchmarkSpec,
    split: str,
    path: Path,
) -> Iterator[BenchmarkExample]:
    for index, row in enumerate(_read_jsonl(path)):
        full_answer = str(row["answer"])
        final_answer = extract_gsm8k_answer(full_answer)
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"gsm8k:{split}:{index}",
            split=split,
            prompt=str(row["question"]),
            expected_answer=final_answer,
            metric=MetricKind.NUMERIC_EXACT,
            source_url=spec.source_url,
            metadata={"answer_full": full_answer},
        )


def _normalize_ifbench(
    spec: BenchmarkSpec,
    split: str,
    path: Path,
) -> Iterator[BenchmarkExample]:
    for row in _read_jsonl(path):
        example_id = str(row.get("key", row.get("prompt_hash", "")))
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"ifbench:{split}:{example_id}",
            split=split,
            prompt=str(row["prompt"]),
            expected_answer=None,
            metric=MetricKind.OFFICIAL_EVALUATOR,
            source_url=spec.source_url,
            metadata={
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
                "official_input_row": row,
            },
        )


def _normalize_hotpotqa(
    spec: BenchmarkSpec,
    split: str,
    path: Path,
) -> Iterator[BenchmarkExample]:
    data = json.loads(path.read_text())
    for index, row in enumerate(data):
        context_text = _format_hotpot_context(row.get("context", []))
        question = str(row["question"])
        prompt = (
            "Answer the question using the provided HotpotQA context. "
            "Give only the final answer.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}"
        )
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"hotpotqa:{split}:{row.get('_id', index)}",
            split=split,
            prompt=prompt,
            expected_answer=str(row["answer"]),
            metric=MetricKind.TOKEN_F1,
            source_url=spec.source_url,
            metadata={
                "question": question,
                "answer": row.get("answer"),
                "type": row.get("type"),
                "level": row.get("level"),
                "supporting_facts": row.get("supporting_facts", []),
                "context": row.get("context", []),
            },
        )


def _normalize_hotpotqa_rows(
    spec: BenchmarkSpec,
    split: str,
    rows: Sequence[Mapping[str, Any]],
) -> Iterator[BenchmarkExample]:
    for index, row in enumerate(rows):
        context_text = _format_hotpot_context(row.get("context", []))
        question = str(row["question"])
        prompt = (
            "Answer the question using the provided HotpotQA context. "
            "Give only the final answer.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}"
        )
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"hotpotqa:{split}:{row.get('id', index)}",
            split=split,
            prompt=prompt,
            expected_answer=str(row["answer"]),
            metric=MetricKind.TOKEN_F1,
            source_url=spec.source_url,
            metadata={
                "question": question,
                "answer": row.get("answer"),
                "type": row.get("type"),
                "level": row.get("level"),
                "supporting_facts": row.get("supporting_facts", []),
                "context": row.get("context", []),
            },
        )


def _normalize_drop(
    spec: BenchmarkSpec,
    split: str,
    paths: Sequence[Path],
) -> Iterator[BenchmarkExample]:
    split_name = "dev" if split in {"dev", "validation"} else split
    matching = [
        path
        for path in paths
        if f"_{split_name}.json" in path.name or path.name.endswith(f"{split_name}.json")
    ]
    if not matching and split_name == "dev":
        matching = [path for path in paths if "dev" in path.name]
    if not matching:
        raise FileNotFoundError(f"no DROP source file found for split {split}")
    data = json.loads(matching[0].read_text())
    for passage_id, passage_info in data.items():
        passage = passage_info.get("passage", "")
        for qa_index, qa in enumerate(passage_info.get("qa_pairs", [])):
            answers = drop_answer_to_strings(qa.get("answer", {}))
            validated = [
                answer
                for validation in qa.get("validated_answers", [])
                for answer in drop_answer_to_strings(validation)
            ]
            aliases = list(dict.fromkeys([*answers, *validated]))
            question = qa.get("question", "")
            prompt = (
                "Answer the DROP question using the passage. Give only the final answer.\n\n"
                f"Passage: {passage}\n\nQuestion: {question}"
            )
            yield BenchmarkExample(
                benchmark_id=spec.benchmark_id,
                example_id=f"drop:{split}:{qa.get('query_id', passage_id + ':' + str(qa_index))}",
                split=split,
                prompt=prompt,
                expected_answer=aliases[0] if aliases else "",
                metric=MetricKind.TOKEN_F1,
                source_url=spec.source_url,
                metadata={
                    "passage_id": passage_id,
                    "question": question,
                    "answer_aliases": aliases,
                    "answer": qa.get("answer", {}),
                    "validated_answers": qa.get("validated_answers", []),
                },
            )


def _normalize_mbpp(
    spec: BenchmarkSpec,
    split: str,
    path: Path,
) -> Iterator[BenchmarkExample]:
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    else:
        rows = json.loads(path.read_text())
    for row in rows:
        task_id = str(row.get("task_id"))
        tests = list(row.get("test_list") or row.get("test") or [])
        test_imports = list(row.get("test_imports") or [])
        prompt = (
            "Write a Python function that satisfies the specification. "
            "Return only code, with no markdown fences.\n\n"
            f"Specification: {row['prompt']}"
        )
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"mbpp:{split}:{task_id}",
            split=split,
            prompt=prompt,
            expected_answer=row.get("code"),
            metric=MetricKind.PASS_AT_1,
            source_url=spec.source_url,
            metadata={
                "task_id": row.get("task_id"),
                "canonical_solution": row.get("code"),
                "test_imports": test_imports,
                "test_list": tests,
                "source_file": row.get("source_file"),
            },
        )


def _normalize_truthfulqa(
    spec: BenchmarkSpec,
    split: str,
    path: Path,
) -> Iterator[BenchmarkExample]:
    if path.suffix == ".json":
        yield from _normalize_truthfulqa_mc(spec, split, path)
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            correct_answers = _split_truthfulqa_answers(row.get("Correct Answers", ""))
            incorrect_answers = _split_truthfulqa_answers(row.get("Incorrect Answers", ""))
            best_answer = row.get("Best Answer", "")
            aliases = list(dict.fromkeys([best_answer, *correct_answers]))
            yield BenchmarkExample(
                benchmark_id=spec.benchmark_id,
                example_id=f"truthfulqa:{split}:{index}",
                split=split,
                prompt=str(row["Question"]),
                expected_answer=best_answer,
                metric=MetricKind.EXACT_MATCH,
                source_url=spec.source_url,
                metadata={
                    "type": row.get("Type"),
                    "category": row.get("Category"),
                    "correct_answers": aliases,
                    "incorrect_answers": incorrect_answers,
                    "source": row.get("Source"),
                    "official_metric_note": (
                        "TruthfulQA generation scoring should use the official evaluator; "
                        "this exact-alias score is diagnostic."
                    ),
                },
            )


def _normalize_truthfulqa_mc(
    spec: BenchmarkSpec,
    split: str,
    path: Path,
) -> Iterator[BenchmarkExample]:
    data = json.loads(path.read_text())
    for key, row in data.get("examples", data).items():
        choices = row.get("mc1_targets", {}).get("choices") or row.get("choices", [])
        labels = row.get("mc1_targets", {}).get("labels") or row.get("labels", [])
        correct_indices = [idx for idx, label in enumerate(labels) if label == 1]
        answer_label = CHOICE_LABELS[correct_indices[0]] if correct_indices else None
        prompt = _format_multiple_choice_prompt(str(row.get("question", key)), choices)
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"truthfulqa:{split}:{key}",
            split=split,
            prompt=prompt,
            expected_answer=answer_label,
            choices=list(choices),
            metric=MetricKind.MULTIPLE_CHOICE_ACCURACY,
            source_url=spec.source_url,
            metadata={"labels": labels, "raw": row},
        )


def _normalize_livebench_math(
    spec: BenchmarkSpec,
    split: str,
    rows: Sequence[Mapping[str, Any]],
) -> Iterator[BenchmarkExample]:
    for row in rows:
        turns = row.get("turns") or []
        prompt = "\n\n".join(str(turn) for turn in turns)
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"livebench_math:{split}:{row.get('question_id')}",
            split=split,
            prompt=prompt,
            expected_answer=row.get("ground_truth"),
            metric=MetricKind.EXACT_MATCH,
            source_url=spec.source_url,
            metadata={
                "category": row.get("category"),
                "task": row.get("task"),
                "subtask": row.get("subtask"),
                "year": row.get("year"),
                "hardness": row.get("hardness"),
                "release_date": row.get("livebench_release_date") or row.get("release_date"),
                "raw": row,
            },
        )


def _normalize_mmlu(
    spec: BenchmarkSpec,
    split: str,
    paths: Sequence[Path],
) -> Iterator[BenchmarkExample]:
    split_key = "val" if split == "validation" else split
    csv_paths = sorted(
        path
        for path in paths
        if path.suffix == ".csv"
        and f"_{split_key}.csv" in path.name
        and f"/{split_key}/" in path.as_posix()
    )
    if not csv_paths:
        csv_paths = sorted(
            path
            for path in paths
            if path.suffix == ".csv" and f"_{split_key}.csv" in path.name
        )
    if not csv_paths:
        raise FileNotFoundError(f"no MMLU CSV files found for split {split}")
    for path in csv_paths:
        subject = path.name.removesuffix(f"_{split_key}.csv")
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            for row_index, row in enumerate(reader):
                if len(row) < 6:
                    continue
                question, choices, answer = row[0], row[1:5], row[5].strip()
                yield BenchmarkExample(
                    benchmark_id=spec.benchmark_id,
                    example_id=f"mmlu:{split}:{subject}:{row_index}",
                    split=split,
                    prompt=_format_multiple_choice_prompt(question, choices),
                    expected_answer=answer,
                    choices=choices,
                    metric=MetricKind.MULTIPLE_CHOICE_ACCURACY,
                    source_url=spec.source_url,
                    metadata={"subject": subject, "row": row},
                )


def _normalize_mmlu_rows(
    spec: BenchmarkSpec,
    split: str,
    rows: Sequence[Mapping[str, Any]],
) -> Iterator[BenchmarkExample]:
    for row_index, row in enumerate(rows):
        choices = list(row.get("choices") or [])
        answer = row.get("answer")
        if isinstance(answer, int):
            answer_label = CHOICE_LABELS[answer]
        else:
            answer_label = str(answer).strip().upper()
        subject = str(row.get("subject", "all"))
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"mmlu:{split}:{subject}:{row_index}",
            split=split,
            prompt=_format_multiple_choice_prompt(str(row.get("question")), choices),
            expected_answer=answer_label,
            choices=choices,
            metric=MetricKind.MULTIPLE_CHOICE_ACCURACY,
            source_url=spec.source_url,
            metadata={"subject": subject, "row": dict(row)},
        )


def _normalize_mmlu_pro(
    spec: BenchmarkSpec,
    split: str,
    rows: Sequence[Mapping[str, Any]],
) -> Iterator[BenchmarkExample]:
    for row in rows:
        choices = list(row.get("options") or [])
        yield BenchmarkExample(
            benchmark_id=spec.benchmark_id,
            example_id=f"mmlu_pro:{split}:{row.get('question_id')}",
            split=split,
            prompt=_format_multiple_choice_prompt(str(row.get("question")), choices),
            expected_answer=row.get("answer"),
            choices=choices,
            metric=MetricKind.MULTIPLE_CHOICE_ACCURACY,
            source_url=spec.source_url,
            metadata={
                "answer_index": row.get("answer_index"),
                "category": row.get("category"),
                "src": row.get("src"),
                "cot_content": row.get("cot_content"),
                "raw": row,
            },
        )


def extract_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.rsplit("####", 1)[1].strip()
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer_text)
    return numbers[-1].replace(",", "") if numbers else answer_text.strip()


def drop_answer_to_strings(answer: Mapping[str, Any]) -> List[str]:
    spans = [str(span) for span in answer.get("spans", []) if str(span).strip()]
    number = str(answer.get("number", "")).strip()
    date = answer.get("date", {}) or {}
    date_parts = [
        str(date.get(part, "")).strip()
        for part in ("day", "month", "year")
        if str(date.get(part, "")).strip()
    ]
    outputs: List[str] = []
    if spans:
        outputs.extend(spans)
        outputs.append(", ".join(spans))
    if number:
        outputs.append(number)
    if date_parts:
        outputs.append(" ".join(date_parts))
    return list(dict.fromkeys(outputs))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _format_hotpot_context(context: Sequence[Any]) -> str:
    parts: List[str] = []
    if isinstance(context, Mapping):
        titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        for title, sentences in zip(titles, sentence_groups):
            parts.append(f"[{title}] " + " ".join(str(sentence) for sentence in sentences))
        return "\n".join(parts)
    for item in context:
        if not item:
            continue
        title = str(item[0])
        sentences = item[1] if len(item) > 1 else []
        parts.append(f"[{title}] " + " ".join(str(sentence) for sentence in sentences))
    return "\n".join(parts)


def _format_multiple_choice_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [question.strip(), "", "Choices:"]
    for index, choice in enumerate(choices):
        label = CHOICE_LABELS[index]
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Answer with only the letter of the best choice.")
    return "\n".join(lines)


def _split_truthfulqa_answers(value: str) -> List[str]:
    return [part.strip() for part in value.split(";") if part.strip()]
