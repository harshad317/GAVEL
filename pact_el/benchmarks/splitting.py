"""Leakage-aware benchmark set selection."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableSet, Optional, Sequence

from pact_el.benchmarks.schemas import BenchmarkExample


@dataclass
class SelectedSplit:
    name: str
    requested: Optional[int]
    selected: List[BenchmarkExample]
    skipped_for_leakage: int = 0


@dataclass
class ThreeWaySelection:
    train: List[BenchmarkExample] = field(default_factory=list)
    validation: List[BenchmarkExample] = field(default_factory=list)
    test: List[BenchmarkExample] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)


def select_disjoint_examples(
    examples: Sequence[BenchmarkExample],
    *,
    name: str,
    n: Optional[int],
    used_keys: MutableSet[str],
    seed: int,
) -> SelectedSplit:
    """Select up to `n` examples without overlapping previous IDs or prompt fingerprints."""

    if n is not None and n < 0:
        raise ValueError(f"{name} size must be non-negative")

    candidates = list(examples)
    random.Random(seed).shuffle(candidates)
    selected: List[BenchmarkExample] = []
    skipped = 0

    for example in candidates:
        keys = leakage_keys(example)
        if any(key in used_keys for key in keys):
            skipped += 1
            continue
        selected.append(example)
        used_keys.update(keys)
        if n is not None and len(selected) >= n:
            break

    if n is not None and len(selected) < n:
        raise ValueError(
            f"could only select {len(selected)} leakage-free {name} examples "
            f"from {len(candidates)} candidates; requested {n}"
        )

    return SelectedSplit(
        name=name,
        requested=n,
        selected=selected,
        skipped_for_leakage=skipped,
    )


def select_three_way(
    *,
    train_pool: Optional[Sequence[BenchmarkExample]],
    validation_pool: Optional[Sequence[BenchmarkExample]],
    test_pool: Sequence[BenchmarkExample],
    train_n: Optional[int],
    val_n: Optional[int],
    test_n: Optional[int],
    seed: int,
) -> ThreeWaySelection:
    """Build deterministic disjoint train, validation, and test selections."""

    used_keys: set[str] = set()
    manifest: Dict[str, Any] = {
        "seed": seed,
        "leakage_policy": "disjoint example_id and normalized prompt fingerprint",
        "splits": {},
    }

    train_selection = SelectedSplit("train", train_n, [])
    validation_selection = SelectedSplit("validation", val_n, [])

    if train_pool is not None:
        train_selection = select_disjoint_examples(
            train_pool,
            name="train",
            n=train_n,
            used_keys=used_keys,
            seed=seed,
        )
        manifest["splits"]["train"] = _split_manifest(train_selection, train_pool)

    if validation_pool is not None:
        validation_selection = select_disjoint_examples(
            validation_pool,
            name="validation",
            n=val_n,
            used_keys=used_keys,
            seed=seed + 1,
        )
        manifest["splits"]["validation"] = _split_manifest(
            validation_selection,
            validation_pool,
        )
    elif train_pool is not None and val_n is not None:
        validation_selection = select_disjoint_examples(
            train_pool,
            name="validation",
            n=val_n,
            used_keys=used_keys,
            seed=seed + 1,
        )
        manifest["splits"]["validation"] = _split_manifest(
            validation_selection,
            train_pool,
            derived_from="train_pool",
        )

    test_selection = select_disjoint_examples(
        test_pool,
        name="test",
        n=test_n,
        used_keys=used_keys,
        seed=seed + 2,
    )
    manifest["splits"]["test"] = _split_manifest(test_selection, test_pool)
    manifest["selected_counts"] = {
        "train": len(train_selection.selected),
        "validation": len(validation_selection.selected),
        "test": len(test_selection.selected),
    }
    manifest["leakage_check"] = validate_no_leakage(
        {
            "train": train_selection.selected,
            "validation": validation_selection.selected,
            "test": test_selection.selected,
        }
    )
    return ThreeWaySelection(
        train=train_selection.selected,
        validation=validation_selection.selected,
        test=test_selection.selected,
        manifest=manifest,
    )


def validate_no_leakage(
    splits: Mapping[str, Sequence[BenchmarkExample]],
) -> Dict[str, Any]:
    seen: Dict[str, str] = {}
    overlaps: List[Dict[str, str]] = []
    for split_name, examples in splits.items():
        for example in examples:
            for key in leakage_keys(example):
                previous = seen.get(key)
                if previous is not None and previous != split_name:
                    overlaps.append(
                        {
                            "key": key,
                            "first_split": previous,
                            "second_split": split_name,
                            "example_id": example.example_id,
                        }
                    )
                else:
                    seen[key] = split_name
    if overlaps:
        raise ValueError(f"leakage detected across splits: {overlaps[:5]}")
    return {
        "passed": True,
        "checked_splits": sorted(splits),
        "checked_examples": sum(len(examples) for examples in splits.values()),
        "unique_leakage_keys": len(seen),
    }


def leakage_keys(example: BenchmarkExample) -> List[str]:
    return [
        f"id:{example.example_id}",
        f"prompt:{prompt_fingerprint(example)}",
    ]


def prompt_fingerprint(example: BenchmarkExample) -> str:
    normalized_prompt = re.sub(r"\s+", " ", example.prompt).strip().lower()
    payload = f"{example.benchmark_id}\n{normalized_prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_manifest(
    selection: SelectedSplit,
    pool: Sequence[BenchmarkExample],
    derived_from: Optional[str] = None,
) -> Dict[str, Any]:
    data = {
        "requested": selection.requested,
        "selected": len(selection.selected),
        "candidate_pool": len(pool),
        "skipped_for_leakage": selection.skipped_for_leakage,
        "example_ids": [example.example_id for example in selection.selected],
    }
    if derived_from:
        data["derived_from"] = derived_from
    return data
