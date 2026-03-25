# PromptCue — Prompt Intent Classifier for LLM Pipelines

[![PyPI version](https://img.shields.io/pypi/v/promptcue.svg)](https://pypi.org/project/promptcue/)
[![Python versions](https://img.shields.io/pypi/pyversions/promptcue.svg)](https://pypi.org/project/promptcue/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/informity/promptcue/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/informity/promptcue/actions/workflows/ci.yml)

PromptCue classifies the intent behind a natural-language prompt and returns structured
routing cues — telling your LLM pipeline, RAG system, or query router not just *what*
the user asked, but *how* it should be answered: retrieve, reason, compare, enumerate,
check recency, or ask for clarification.

---

## How it works

PromptCue uses a **cascade classifier**:

1. **Deterministic pass** — scores the query against a YAML registry of query types
   using trigger-phrase matching and vocabulary overlap. Fast, zero ML dependencies,
   returns immediately when confidence is high.
2. **Semantic fallback** — when the deterministic result is ambiguous or below threshold,
   sentence-level embeddings (`all-MiniLM-L6-v2`) re-score the query against example
   sentences per type. Activates automatically when `sentence-transformers` is installed.

The result is a Pydantic model (`PromptCueQueryObject`) carrying the classification, confidence,
scope, routing hints, action directives, and any enrichment you have enabled.

---

## Requirements

- Python **3.13+**
- Core dependencies: `pydantic`, `PyYAML`, `numpy` (always installed)
- All ML/NLP components are **optional** — the package installs and runs without them

---

## Install

Core install — deterministic classifier only, no ML dependencies:

```bash
pip install promptcue
```

With semantic scoring (`sentence-transformers`):

```bash
pip install "promptcue[semantic]"
```

With language detection (`langdetect`):

```bash
pip install "promptcue[detection]"
```

With linguistic enrichment (`spaCy`):

```bash
pip install "promptcue[linguistic]"
python -m spacy download en_core_web_sm
```

With keyword extraction (`KeyBERT`):

```bash
pip install "promptcue[keywords]"
```

With everything:

```bash
pip install "promptcue[all]"
python -m spacy download en_core_web_sm
```

Development install (editable, with test and lint tools):

```bash
pip install -e ".[dev]"
```

---

## Production deployment

PromptCue requires `sentence-transformers` to produce production-quality results.
The deterministic-only path (`pip install promptcue`, no `[semantic]`) achieves
approximately 40–50% accuracy on naturalistic queries and is **not a supported
production configuration** — it is suitable for evaluation or development only.

Every production deployment must:

1. Install `pip install "promptcue[semantic]"`.
2. Pre-download the model **before** the service starts — not on first query.
3. Call `warm_up()` (or `warm_up_async()`) at startup and gate readiness on it succeeding.

If the model cannot be loaded, PromptCue raises `PromptCueModelLoadError` immediately.
It never silently falls back to deterministic-only mode — a misconfigured deployment
fails loudly at startup rather than producing quietly wrong results at query time.

### Model cache location

By default the model is stored in HuggingFace's standard cache (`~/.cache/huggingface/`).
For deployments that cannot rely on the default cache, set the path explicitly:

```python
from pathlib import Path
from promptcue import PromptCueAnalyzer, PromptCueConfig

analyzer = PromptCueAnalyzer(PromptCueConfig(
    model_cache_dir=Path('/opt/models')
))
analyzer.warm_up()   # raises PromptCueModelLoadError if the model is not at that path
```

Or via environment variable — no code change required:

```bash
export PROMPTCUE_MODEL_CACHE=/opt/models
```

### Deployment patterns

| Environment | Model management approach |
|---|---|
| Local dev | Leave `model_cache_dir` unset — HuggingFace downloads on first `warm_up()` |
| EC2 / EBS | Pre-download to EBS volume; set `HF_HOME=/opt/models` or `model_cache_dir` |
| Lambda (container image) | Bake model into Docker image at build time — **required**, Lambda `/tmp` is ephemeral |
| Lambda (EFS mount) | Pre-populate EFS volume; set `model_cache_dir=Path('/mnt/models')` |
| Docker / CI | Download during image build; volume-mount for local dev |

For Lambda container images, bake the model in at build time:

```dockerfile
FROM python:3.11-slim
RUN pip install "promptcue[semantic]"
ENV HF_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"
```

## Quick start

### Basic — no ML dependencies required

```python
from promptcue import PromptCueAnalyzer

analyzer = PromptCueAnalyzer()
result   = analyzer.analyze('Compare Aurora and OpenSearch for RAG on AWS')

print(result.primary_query_type)   # comparison
print(result.scope)                # comparative
print(result.confidence)           # 0.9
print(result.routing_hints)        # {'needs_retrieval': True, 'needs_reasoning': True, ...}
print(result.action_hints)         # {'should_compare': True, ...}
```

### With semantic scoring — requires `pip install "promptcue[semantic]"`

Semantic scoring is **enabled automatically** when `sentence-transformers` is installed.
Call `warm_up()` at startup to pre-load the model and avoid first-query latency.

```python
from promptcue import PromptCueAnalyzer

analyzer = PromptCueAnalyzer()
analyzer.warm_up()  # loads ~90 MB model once; cached after first download

result = analyzer.analyze('Should we use DynamoDB or RDS for a high-read catalog?')
print(result.primary_query_type)   # recommendation
print(result.classification_basis) # semantic_similarity
print(result.confidence)           # 0.25
```

### With full enrichment

```python
from promptcue import PromptCueAnalyzer, PromptCueConfig

analyzer = PromptCueAnalyzer(PromptCueConfig(
    enable_language_detection    = True,   # requires promptcue[detection]
    enable_linguistic_extraction = True,   # requires promptcue[linguistic]
    enable_keyword_extraction    = True,   # requires promptcue[keywords]
))
analyzer.warm_up()

result = analyzer.analyze(
    'How do I set up a VPC with private subnets and NAT gateway step by step?'
)
print(result.language)       # en
print(result.main_verbs)     # ['set']
print(result.noun_phrases)   # ['a VPC', 'private subnets', 'NAT gateway']
print(result.keywords)       # [PromptCueKeyword(text='vpc private subnets', weight=0.72, ...), ...]
print(result.entities)       # []  (no named entities in this query)
```

### In an async application

Both `.warm_up_async()` and `.analyze_async()` delegate to `asyncio.to_thread()`,
so they are safe to await in FastAPI handlers or any other async framework without
blocking the event loop.

```python
import asyncio
from promptcue import PromptCueAnalyzer

async def main() -> None:
    analyzer = PromptCueAnalyzer()
    await analyzer.warm_up_async()

    result = await analyzer.analyze_async('Compare option A and option B')
    print(result.primary_query_type)   # comparison

asyncio.run(main())
```

### Full JSON output

```python
print(result.model_dump_json(indent=2))
```

---

## Query types

PromptCue ships with a default registry of 12 query types:

| Label | Scope | Description |
|---|---|---|
| `analysis` | exploratory | Deep evaluation of a system, architecture, or decision |
| `chitchat` | broad | Social or conversational, not a knowledge query |
| `comparison` | comparative | Asks to compare two or more options |
| `coverage` | broad | Broad overview or "tell me everything" request |
| `generation` | focused | Produce entirely new content from scratch with no existing source to condense |
| `lookup` | focused | Factual question with a single direct answer |
| `procedure` | focused | Step-by-step instructions for a task |
| `recommendation` | focused | Asks for a decision or suggestion given constraints |
| `summarization` | focused | Condense existing content — provided, referenced, or in-context — into a shorter form |
| `troubleshooting` | focused | Diagnosing or fixing a problem |
| `update` | focused | Latest news, releases, or changes |
| `validation` | focused | Verify or fact-check a specific stated claim, assumption, or belief |

You can replace or extend the registry by pointing `PromptCueConfig.registry_path` at your
own YAML file — the schema is documented in `src/promptcue/data/query_types_en.yaml`.

---

## Public API

### `PromptCueAnalyzer`

```python
PromptCueAnalyzer(config: PromptCueConfig | None = None)
```

| Method | Description |
|---|---|
| `.analyze(text: str) -> PromptCueQueryObject` | Analyze a query and return a structured result |
| `.warm_up() -> None` | Pre-load all enabled models at startup to avoid first-query latency |
| `.analyze_async(text: str) -> PromptCueQueryObject` | Async variant of `.analyze()`; delegates to `asyncio.to_thread()` |
| `.warm_up_async() -> None` | Async variant of `.warm_up()`; delegates to `asyncio.to_thread()` |

---

### `PromptCueConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `registry_path` | `Path \| None` | `None` | Custom YAML registry path; uses bundled default when `None` |
| `model_cache_dir` | `Path \| None` | env / `None` | Directory where the sentence-transformers model is cached. Falls back to `PROMPTCUE_MODEL_CACHE` env var, then HuggingFace default (`~/.cache/huggingface/`) |
| `similarity_threshold` | `float` | `0.55` | Minimum score for a deterministic match to be accepted |
| `semantic_similarity_threshold` | `float` | `0.20` | Minimum score for a semantic match to be accepted |
| `ambiguity_margin` | `float` | `0.08` | Min gap between top-2 scores before clarification is flagged |
| `semantic_fallback_threshold` | `float` | `0.75` | Deterministic score above which the semantic pass is skipped |
| `trigger_fallback_threshold` | `float` | `0.60` | When a trigger phrase matched and the score meets this value and the margin is clear, the deterministic result is trusted directly and semantic is skipped |
| `enable_semantic_scoring` | `bool` | auto | `True` when `sentence-transformers` is installed, else `False` |
| `embedding_model` | `str` | `all-MiniLM-L6-v2` | HuggingFace model name for semantic scoring |
| `enable_language_detection` | `bool` | `False` | Detect BCP-47 language code; requires `promptcue[detection]` |
| `enable_linguistic_extraction` | `bool` | `False` | Extract verbs, noun phrases, named entities; requires `promptcue[linguistic]` |
| `enable_keyword_extraction` | `bool` | `False` | Extract keyphrases via KeyBERT; requires `promptcue[keywords]` |
| `max_keywords` | `int` | `8` | Maximum number of keyphrases to extract |
| `spacy_model` | `str` | `en_core_web_sm` | spaCy model name for linguistic extraction |

---

### `PromptCueQueryObject` fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Output schema version (`"1.0"`) |
| `input_text` | `str` | Original query as provided by the caller |
| `normalized_text` | `str` | Unicode-normalised, whitespace-collapsed query |
| `language` | `str` | BCP-47 language code (`"en"`) or `"unknown"` when detection is off |
| `is_continuation` | `bool` | `True` when the query continues an ongoing conversation (e.g. "what about X?", "and for Y?") |
| `primary_query_type` | `str` | Top classified query type label, or `"unknown"` |
| `classification_basis` | `str` | How the result was reached: `trigger_match`, `word_overlap`, `semantic_similarity`, `below_threshold` |
| `candidate_query_types` | `list[PromptCueCandidate]` | All types ranked by score |
| `runner_up` | `PromptCueCandidate \| None` | Second-ranked candidate; `None` when fewer than two candidates exist |
| `confidence` | `float` | Score of the top candidate (0.0–1.0) |
| `confidence_band` | `str` | Coarse confidence tier: `high`, `medium`, or `low` |
| `ambiguity_score` | `float` | How close the top-2 candidates are (0.0 = clear, 1.0 = identical) |
| `scope` | `str` | Query scope: `broad`, `focused`, `comparative`, `exploratory`, or `unknown` |
| `main_verbs` | `list[str]` | Root verbs extracted by spaCy (empty when enrichment is off) |
| `noun_phrases` | `list[str]` | Noun chunks extracted by spaCy (empty when enrichment is off) |
| `named_entities` | `list[str]` | Named entity surface texts, plain strings (backward compat) |
| `entities` | `list[PromptCueEntity]` | Named entities with `text` and `entity_type` (spaCy label) |
| `keywords` | `list[PromptCueKeyword]` | Keyphrases with `text`, `weight`, and `kind` from KeyBERT |
| `routing_hints` | `dict[str, bool]` | `needs_retrieval`, `needs_reasoning`, `needs_current_info`, `needs_clarification` |
| `action_hints` | `dict[str, bool]` | Response-generation directives: `should_survey`, `should_enumerate`, `should_compare`, `should_direct_answer`, `should_check_recency`, `should_clarify`, `should_respond_conversationally` |
| `constraints` | `list[str]` | Reserved for future use |

---

## Development

```bash
git clone https://github.com/informity/promptcue.git
cd promptcue

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,semantic,linguistic,keywords,detection]"
python -m spacy download en_core_web_sm

pytest
ruff check src/ tests/ examples/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
