# PACT-EL / GAVEL

PACT-EL implements the GAVEL idea from `GAVEL.pdf`: optimize prompts by repairing a typed behavioral contract, not by searching across many freeform prompt candidates.

The package turns observed failures into an Evidence Ledger, compiles a Prompt Axiom Graph, applies one minimal AxiomPatch, renders a final prompt from graph nodes and executable guarantee clauses, then tries to falsify the patch with four canaries:

- fix canary
- boundary canary
- regression canary
- format/schema canary

If the patch fails, PACT-EL allows one targeted repair of only the falsified axiom or edge. If that also fails, it returns the original prompt with a no-patch diagnosis.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

For LiteLLM-backed optimizer or target clients:

```bash
python -m pip install -e ".[litellm,dev]"
```

For DSPy, MIPROv2, and GEPA benchmark baselines, use Python 3.10+ and install:

```bash
python -m pip install -e ".[baselines,dev]"
```

For IFBench's official verifier, install the IFBench extra too:

```bash
python -m pip install -e ".[baselines,ifbench,dev]"
```

For live GAVEL benchmark runs, install LiteLLM as well:

```bash
python -m pip install -e ".[litellm,baselines,ifbench,dev]"
```

## Minimal Example

Run the deterministic replay example:

```bash
python3 examples/basic.py
```

## Benchmarks

List the supported official-source benchmarks:

```bash
python3 experiments/prepare_benchmarks.py list
```

Benchmark CLIs use `rich` tables and `tqdm` progress bars by default. Every
evaluation method reports live `accuracy`, `passed`, and `scored` counts from
the same scorer. Parallel evaluators also display `in_flight` and
`max_in_flight` so `--workers` behavior is visible. Add `--json` for
machine-readable output or `--no-progress` when writing logs.

Prepare a normalized JSONL file:

```bash
python3 experiments/prepare_benchmarks.py gsm8k --out data/benchmarks
python3 experiments/prepare_benchmarks.py mmlu_pro --limit 100 --out data/benchmarks
python3 experiments/prepare_benchmarks.py all --limit 100 --out data/benchmarks
```

Score predictions against a normalized file:

```bash
python3 experiments/score_benchmark.py \
  --dataset data/benchmarks/gsm8k-test.jsonl \
  --predictions output/predictions/gsm8k.jsonl \
  --workers 16
```

Prediction rows should include `example_id` and `prediction`:

```json
{"example_id": "gsm8k:test:0", "prediction": "#### 72"}
```

Supported benchmark ids:

- `gsm8k`
- `ifbench`
- `hotpotqa`
- `drop`
- `mbpp`
- `truthfulqa`
- `livebench_math`
- `mmlu`
- `mmlu_pro`

Source policy:

- Every registry entry points at an official benchmark website, GitHub repository, or official dataset host.
- IFBench and TruthfulQA retain official-evaluator metadata because their headline metrics are not plain exact match.
- MBPP pass@1 requires executing generated code. Local scoring refuses to execute code unless `--allow-code-execution` is passed.
- HotpotQA, MMLU, LiveBench Math, and MMLU-Pro use official Hugging Face dataset mirrors for row paging where the canonical repo points to large archives or where the official codebase itself loads from Hugging Face.

### DSPy, MIPROv2, and GEPA Baselines

Run a direct DSPy program over a normalized benchmark:

```bash
python3 experiments/run_dspy_mipro.py \
  --optimizer dspy \
  --program cot \
  --model openai/gpt-4o-mini \
  --eval-dataset data/benchmarks/gsm8k-test.jsonl \
  --temperature 0.0 \
  --workers 16 \
  --cache True \
  --out output/baselines/gsm8k_dspy
```

Run DSPy's MIPROv2 optimizer, using an official training split and then evaluating
on the held-out split:

```bash
python3 experiments/prepare_benchmarks.py gsm8k --split train --out data/benchmarks
python3 experiments/prepare_benchmarks.py gsm8k --split test --out data/benchmarks

python3 experiments/run_dspy_mipro.py \
  --optimizer mipro \
  --program cot \
  --model openai/gpt-4o-mini \
  --train-dataset data/benchmarks/gsm8k-train.jsonl \
  --test-dataset data/benchmarks/gsm8k-test.jsonl \
  --train-n 50 \
  --val-n 50 \
  --test-n 500 \
  --temperature 0.0 \
  --workers 16 \
  --cache True \
  --auto light \
  --out output/baselines/gsm8k_mipro
```

Run DSPy's GEPA optimizer with the same benchmark/split flow:

```bash
python3 experiments/run_dspy_mipro.py \
  --benchmark ifbench \
  --optimizer gepa \
  --program cot \
  --model openai/gpt-4.1-mini \
  --train-n 50 \
  --val-n 50 \
  --test-n 200 \
  --temperature 0.0 \
  --workers 16 \
  --cache True \
  --auto heavy \
  --out output/baselines/ifbench_gepa
```

GEPA uses `--model` as the reflection LM by default. To use a stronger
reflection model without changing the task model, pass `--reflection-model`.
Use `--temperature` to set the task LM temperature. GEPA also accepts
`--reflection-temperature`; if it is omitted, GEPA uses `--temperature` for the
reflection LM, then the provider default if neither flag is set.

`--train-n`, `--val-n`, and `--test-n` select deterministic, leakage-checked
subsets. The runner rejects overlaps by `example_id` and by normalized prompt
fingerprint before calling DSPy/MIPROv2/GEPA.

`--workers` controls concurrent final evaluation. For MIPROv2 and GEPA it is
also used as `num_threads` unless `--num-threads` is passed. `--cache True`
and `--cache False` explicitly control DSPy's LM cache.

The runner follows the official DSPy GitHub API: `dspy.LM`,
`dspy.configure`, `dspy.Predict` / `dspy.ChainOfThought`, and
`dspy.MIPROv2.compile(...)` / `dspy.GEPA.compile(...)`. MIPROv2 requires
DSPy's `optuna` extra, included by this package's `baselines` extra. GEPA is
included in DSPy's runtime dependency set. The runner writes prediction JSONL, score
JSONL, a summary JSON, and the saved DSPy program when the compiled program
supports `save(...)`.

For benchmarks like IFBench, you can let the runner prepare the official data
and then build leakage-safe subsets in the same command:

```bash
python3 experiments/run_dspy_mipro.py \
  --benchmark ifbench \
  --optimizer mipro \
  --program cot \
  --model openai/gpt-4.1-mini \
  --train-n 50 \
  --val-n 50 \
  --test-n 200 \
  --temperature 0.0 \
  --workers 16 \
  --cache True \
  --auto light \
  --out output/baselines/ifbench_mipro
```

The current IFBench registry entry has one public official test split, so this
command derives disjoint train, validation, and test subsets from that pool and
records the split manifest in the summary JSON.

After each run, the CLI prints a method results table with train, validation,
test, and optimization rows. The split scores are final post-optimization
evaluations; the optimization row reports optimizer API calls when DSPy's LM
history exposes that count.

### GAVEL Benchmark Runner

Run GAVEL with the same official benchmark and leakage-safe split flow:

```bash
python3 experiments/run_gavel.py \
  --benchmark ifbench \
  --model openai/gpt-4.1-mini \
  --optimizer-model openai/gpt-4.1-mini \
  --train-n 50 \
  --val-n 50 \
  --test-n 200 \
  --temperature 0.0 \
  --optimizer-temperature 0.0 \
  --workers 16 \
  --cache True \
  --budget 9 \
  --pareto-candidates 6 \
  --pareto-frontier-size 6 \
  --validation-margin 0.0 \
  --out output/baselines/ifbench_gavel
```

The GAVEL runner first evaluates the base prompt on the selected train examples
to build an Evidence Ledger, compiles one typed PACT-EL/GAVEL prompt, then runs
GAVEL-Pareto: an optimizer-proposed population of complete prompt mutations
ranked for expected gain, evidence support, compactness, and low regression
risk. The held-out validation gate scores the Pareto frontier alongside the
benchmark-general task strategy prompt, stricter constraint-solver prompt,
aggregate evidence strategy prompt, compiled GAVEL candidate, and target
execution modes. By default it compares direct answering, label-free
self-refinement, label-free plan-and-answer, and a contract-aware
plan-answer-refine mode that first compiles the current user prompt into a task
contract, drafts the answer from that contract, then runs a label-free repair
pass against the original prompt plus the contract. A non-base prompt or
non-direct execution mode is used only when it clears the base direct validation
score by a confidence-aware effective margin. Canary-rejected and much longer
prompts need extra validation lift before they can be selected, which reduces
narrow prompt overfit. Use
`--disable-pareto-search` for the older single-compile portfolio,
`--disable-rejected-candidate-validation` for stricter canary-only ablations,
or `--disable-prompt-portfolio` to score only the compiled candidate. The runner
then evaluates the selected prompt and execution mode on train, validation, and
test. Use `--execution-modes direct,plan,plan_refine,self_refine` to control the
validation portfolio and `--self-refine-rounds` to set the depth of the
self-refinement mode. It writes prediction JSONL, score JSONL, the full GAVEL optimization
report, the selected prompt, and a summary JSON.

Base prompts and graph-rendered GAVEL prompts use the same compact structured
prompt frame: `Goal`, `Context`, `Role`, `Input`, `Task`, `Constraints`,
`Output Format`, and `Quality Bar`.

## Architecture

- `pact_el.schemas`: strict Pydantic models for ledgers, graph nodes, patches, canaries, calls, and reports.
- `pact_el.ledger`: Evidence Ledger builders from logs, examples, probes, and canary results.
- `pact_el.graph`: graph validation, indexing, and precedence-aware node ordering.
- `pact_el.compiler`: optimizer prompt construction and strict schema parsing without silent JSON repair.
- `pact_el.renderer`: deterministic prompt rendering from graph nodes, edges, and GuaranteeScript clauses.
- `pact_el.validators`: local deterministic validators, including JSON Schema, regex, exact match, enum, numeric predicates, and JSONLogic-style clauses.
- `pact_el.canaries`: construction and validation of the four required canary families.
- `pact_el.clients`: async target and optimizer clients, including replay clients and optional LiteLLM clients.
- `pact_el.falsification`: acceptance, rollback, token-delta, regression, and no-patch logic.
- `pact_el.optimize`: end-to-end PACT-EL orchestration.
- `pact_el.benchmarks`: official-source benchmark registry, preparation adapters, and local scoring utilities.
- `pact_el.baselines`: optional DSPy, MIPROv2, GEPA, and GAVEL runners for normalized benchmarks.
- `experiments/run.py`: cached matched-budget experiment runner with ablation toggles.
- `experiments/prepare_benchmarks.py`: download and normalize official benchmarks.
- `experiments/score_benchmark.py`: score JSON/JSONL predictions against normalized examples.
- `experiments/run_dspy_mipro.py`: run direct DSPy, DSPy MIPROv2, and DSPy GEPA baselines.
- `experiments/run_gavel.py`: run live GAVEL optimization and benchmark evaluation.

## Design Invariants

- The rendered prompt is derived from the Prompt Axiom Graph and GuaranteeScript, not copied from a freeform optimizer rewrite.
- Runtime prompts use a compact eight-section structure: Goal, Context, Role, Input, Task, Constraints, Output Format, and Quality Bar.
- Live GAVEL selection compares a small portfolio of structured prompts and label-free execution modes against the base direct run on held-out validation before test evaluation.
- Validation selection uses statistical, canary-rejection, and prompt-complexity margins so small validation wins do not automatically select brittle prompts.
- Runtime compiled graph clauses are conditional on the current user input; evidence-specific constraints must not become unconditional requirements for unrelated examples.
- Optimizer output must satisfy a strict JSON schema on the first parse. Markdown-fenced or repaired JSON is rejected.
- Deterministic validators are preferred. LLM judges are represented as counted fallback validators and must be explicitly configured.
- PACT-EL does not claim to fix missing knowledge, broken tools, bad retrieval, invalid upstream data, or impossible constraints.
- Patch acceptance requires a supported fix claim and no critical regression.
- Live benchmark runs add a held-out validation gate after compilation, so a prompt that regresses validation is rolled back before test evaluation, and a canary-rejected candidate can be rescued only by real validation improvement.
- Optimizer evidence includes compact scorer diagnostics, failed instruction ids, answer aliases, choices, or public tests when available; raw benchmark rows are not copied wholesale into the optimizer prompt.
