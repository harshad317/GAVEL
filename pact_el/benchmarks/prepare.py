"""Download official benchmark sources and write normalized JSONL datasets."""

from __future__ import annotations

import json
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pact_el.benchmarks.adapters import normalize_examples
from pact_el.benchmarks.registry import get_benchmark_spec, list_benchmarks
from pact_el.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkSource,
    SourceKind,
)


def prepare_benchmark(
    benchmark_id: str,
    out_dir: Path,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> Path:
    """Download official data and write normalized examples to JSONL."""

    spec = get_benchmark_spec(benchmark_id)
    selected_split = split or spec.default_split
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "_sources" / spec.benchmark_id
    source_dir.mkdir(parents=True, exist_ok=True)

    source_paths: List[Path] = []
    hf_rows: Optional[List[Mapping[str, Any]]] = None
    for source in _sources_for_split(spec.sources, selected_split):
        if source.kind == SourceKind.HF_ROWS:
            hf_rows = _load_hf_rows(
                source,
                source_dir,
                force=force,
                max_rows=limit,
            )
            source_paths.append(source_dir / (source.local_name or f"{source.source_id}.jsonl"))
            continue
        source_path = _ensure_source(source, source_dir, force=force)
        if source.kind == SourceKind.ARCHIVE:
            source_paths.extend(_extract_archive(source_path, source_dir, force=force))
        else:
            source_paths.append(source_path)

    examples = normalize_examples(
        spec,
        selected_split,
        source_paths,
        rows=hf_rows,
    )
    if limit is not None:
        examples = examples[:limit]

    output_path = out_dir / f"{spec.benchmark_id}-{selected_split}.jsonl"
    _write_examples(output_path, examples)
    manifest_path = out_dir / f"{spec.benchmark_id}-{selected_split}.manifest.json"
    manifest = {
        "benchmark": spec.model_dump(mode="json"),
        "split": selected_split,
        "num_examples": len(examples),
        "output_path": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output_path


def prepare_all(
    out_dir: Path,
    split_overrides: Optional[Mapping[str, str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    overrides = dict(split_overrides or {})
    for spec in list_benchmarks():
        split = overrides.get(spec.benchmark_id, spec.default_split)
        paths[spec.benchmark_id] = prepare_benchmark(
            spec.benchmark_id,
            out_dir=out_dir,
            split=split,
            limit=limit,
            force=force,
        )
    return paths


def _sources_for_split(
    sources: Sequence[BenchmarkSource],
    split: str,
) -> List[BenchmarkSource]:
    selected = [
        source
        for source in sources
        if source.split in {split, "all", None}
    ]
    if not selected:
        available = ", ".join(sorted({str(source.split) for source in sources}))
        raise KeyError(f"no source for split {split}; available splits: {available}")
    return selected


def _ensure_source(source: BenchmarkSource, source_dir: Path, force: bool) -> Path:
    path = source_dir / (source.local_name or Path(urllib.parse.urlparse(source.url or "").path).name)
    if path.exists() and not force:
        return path
    if source.url is None:
        raise ValueError(f"source {source.source_id} has no URL")
    _download(source.url, path)
    return path


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pact-el-benchmark-preparer/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def _extract_archive(path: Path, source_dir: Path, force: bool) -> List[Path]:
    extract_dir = source_dir / path.stem
    if extract_dir.exists() and not force:
        return sorted(p for p in extract_dir.rglob("*") if p.is_file())
    extract_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = extract_dir / member.filename
                _assert_safe_extract_path(extract_dir, target)
            archive.extractall(extract_dir)
    elif path.suffix in {".tar", ".gz", ".tgz"} or path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                target = extract_dir / member.name
                _assert_safe_extract_path(extract_dir, target)
            archive.extractall(extract_dir)
    else:
        raise ValueError(f"unsupported archive type: {path}")
    return sorted(p for p in extract_dir.rglob("*") if p.is_file())


def _assert_safe_extract_path(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved != target_resolved and root_resolved not in target_resolved.parents:
        raise ValueError(f"archive member escapes extraction directory: {target}")


def _load_hf_rows(
    source: BenchmarkSource,
    source_dir: Path,
    force: bool,
    page_size: int = 100,
    max_rows: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    local_path = source_dir / (source.local_name or f"{source.source_id}.jsonl")
    if max_rows is not None:
        local_path = local_path.with_name(
            f"{local_path.stem}.limit{max_rows}{local_path.suffix}"
        )
    if local_path.exists() and not force:
        return _read_jsonl(local_path)

    rows: List[Mapping[str, Any]] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "dataset": source.hf_dataset_id,
                "config": source.hf_config,
                "split": source.hf_split,
                "offset": offset,
                "length": page_size,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "pact-el-benchmark-preparer/0.1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page_rows = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page_rows)
        if max_rows is not None and len(rows) >= max_rows:
            rows = rows[:max_rows]
            break
        if len(page_rows) < page_size:
            break
        offset += page_size

    local_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else "")
    )
    return rows


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_examples(path: Path, examples: Sequence[BenchmarkExample]) -> None:
    path.write_text(
        "\n".join(example.model_dump_json() for example in examples)
        + ("\n" if examples else "")
    )
