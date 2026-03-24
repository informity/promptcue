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

### Full JSON output

```python
print(result.model_dump_json(indent=2))
```

---

## Query types

PromptCue ships with a default registry of 9 query types:

| Label | Scope | Description |
|---|---|---|
| `lookup` | focused | Factual question with a single direct answer |
| `comparison` | comparative | Asks to compare two or more options |
| `recommendation` | focused | Asks for a decision or suggestion given constraints |
| `troubleshooting` | focused | Diagnosing or fixing a problem |
| `procedure` | focused | Step-by-step instructions for a task |
| `analysis` | exploratory | Deep evaluation of a system, architecture, or decision |
| `coverage` | broad | Broad overview or "tell me everything" request |
| `update` | focused | Latest news, releases, or changes |
| `chitchat` | focused | Social or conversational, not a knowledge query |

You can replace or extend the registry by pointing `PromptCueConfig.registry_path` at your
own YAML file — the schema is documented in `src/promptcue/data/query_types.yaml`.

---

## Public API

### `PromptCueAnalyzer`

```python
PromptCueAnalyzer(config: PromptCueConfig | None = None)
```

| Method | Description |
|---|---|
| `.analyze(text: str) -> PromptCueQueryObject` | Analyze a query and return structured result |
| `.warm_up() -> None` | Pre-load all enabled models at startup |

---

### `PromptCueConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `registry_path` | `Path \| None` | `None` | Custom YAML registry path; uses bundled default when `None` |
| `similarity_threshold` | `float` | `0.55` | Minimum score for a deterministic match to be accepted |
| `semantic_similarity_threshold` | `float` | `0.20` | Minimum score for a semantic match to be accepted |
| `ambiguity_margin` | `float` | `0.08` | Min gap between top-2 scores before clarification is flagged |
| `semantic_fallback_threshold` | `float` | `0.75` | Deterministic score above which the semantic pass is skipped |
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
| `primary_query_type` | `str` | Top classified query type label, or `"unknown"` |
| `classification_basis` | `str` | How the result was reached: `label_match`, `trigger_match`, `word_overlap`, `semantic_similarity`, `below_threshold` |
| `candidate_query_types` | `list[PromptCueCandidate]` | All types ranked by score |
| `confidence` | `float` | Score of the top candidate (0.0–1.0) |
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
