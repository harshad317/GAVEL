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

## Minimal Example

Run the deterministic replay example:

```bash
python3 examples/basic.py
```

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
- `experiments/run.py`: cached matched-budget experiment runner with ablation toggles.

## Design Invariants

- The rendered prompt is derived from the Prompt Axiom Graph and GuaranteeScript, not copied from a freeform optimizer rewrite.
- Optimizer output must satisfy a strict JSON schema on the first parse. Markdown-fenced or repaired JSON is rejected.
- Deterministic validators are preferred. LLM judges are represented as counted fallback validators and must be explicitly configured.
- PACT-EL does not claim to fix missing knowledge, broken tools, bad retrieval, invalid upstream data, or impossible constraints.
- Patch acceptance requires a supported fix claim and no critical regression.
