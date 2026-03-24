# Contributing to PromptCue

Thank you for your interest in contributing.

---

## Development setup

```bash
git clone https://github.com/informity/promptcue.git
cd promptcue

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,semantic,linguistic,keywords,detection]"
python -m spacy download en_core_web_sm
```

---

## Running tests

```bash
pytest
```

Tests that require optional dependencies (`sentence-transformers`, `spacy`, `keybert`,
`langdetect`) are skipped automatically when the relevant package is not installed.

Run the full suite with all extras:

```bash
pip install -e ".[dev,semantic,linguistic,keywords,detection]"
pytest -v
```

---

## Linting and type checking

```bash
ruff check src/ tests/ examples/
mypy src/
```

All linter errors must be clean before opening a pull request.

---

## Building the package

```bash
python -m build
```

This produces `dist/promptcue-*.whl` and `dist/promptcue-*.tar.gz`.

---

## Branch and PR conventions

- Branch from `master` using the pattern `feat/short-description` or `fix/short-description`.
- Keep pull requests focused — one logical change per PR.
- All tests must pass and `ruff check` must be clean.
- Update `CHANGELOG.md` under `[Unreleased]` for any user-facing change.

---

## Public API contract

The public API surface is exactly three symbols:

```python
from promptcue import PromptCueAnalyzer, PromptCueConfig, PromptCueQueryObject
```

- Do not remove or rename fields on `PromptCueQueryObject` without a major version bump.
- Do not change the signature of `PromptCueAnalyzer.analyze()` without a minor version bump.
- Do not add a new required dependency to `[project.dependencies]` — all ML/NLP packages
  must remain optional extras.

---

## Adding a new query type

Edit `src/promptcue/data/query_types.yaml`. Each entry requires:

- `label` — unique short string
- `description` — one sentence
- `triggers` — list of short phrases for deterministic matching
- `examples` — list of full sentences for semantic embedding anchors (at least 4)
- `routing_hints` — boolean flags: `needs_retrieval`, `needs_reasoning`, `needs_current_info`
- `scope` — one of `broad`, `focused`, `comparative`, `exploratory`
- `action_hints` — boolean response-generation directives

Run `pytest` after any registry change to confirm nothing regresses.
