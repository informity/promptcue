# PromptCue — Improvement Backlog v1.0

**Status:** 🔄 ACTIVE
**Last updated:** 2026-03-23
**Scope:** Post-review improvement proposals — ordered by priority
**Maintainer:** Informity

---

Work top-to-bottom. When an item is done, move it to the "Resolved history" section at
the bottom with a pass date and a brief note of what was changed.

---

## Backlog — ordered by priority

### 8. Unit test coverage gaps ★ Test health · Small–Medium

Four untested modules and one under-tested area. Ship as a single `tests/test_core.py`
+ `tests/test_language.py`:

**8a — `normalization.py`**
Cover: NFKC normalization, multi-whitespace collapse, leading/trailing strip, empty string,
Unicode ligatures.

**8b — `language.py`**
Cover: disabled path returns `unknown`, below-min-length returns `unknown`, successful
detection, missing dep raises `ImportError`, exception degrades to `unknown`, `is_loaded`
before and after.

**8c — `decision.py`**
Cover: no-candidates path, below-threshold, ambiguous-above-threshold (margin < 0.08),
clean resolution, `threshold_override`.

**8d — `classifier.py` tiers**
Cover: label match → 0.90, trigger score proportional to phrase length, word overlap in
[0.10, 0.50], cascade skips semantic when trigger_confident, cascade falls through to
semantic when ambiguous.

**8e — `embedding.py` edge cases**
Cover: zero-norm vector → 0.0, negative cosine clamped to 0.0, `is_loaded` before/after
`warm_up`, `encode` with empty list.

**8f — New types in schema tests**
`summarization`, `generation`, and `validation` have no dedicated scope/action-hint/basis
assertions in `test_schema.py` or `test_semantic.py`.

---

### 13. Centroid embeddings (semantic path) ★ Accuracy + Performance · Medium

**File** — `src/promptcue/core/classifier.py`

**Problem**
`_classify_semantic` takes the **max** cosine similarity across all examples per type.
One outlier example can dominate and produce an unstable score.

**Fix**
Pre-compute the mean (centroid) embedding per type once in `_build_example_cache()`.
Store one vector per type instead of N. Replace the per-pair loop with a single
`cosine_similarity(query_vec, centroid_vec)`.

**Trade-off**
Max-similarity is better for thin/sparse registries where you want edge-case coverage.
Centroid is better as the registry grows denser. Validate accuracy against
`.internal/tests/queries/queries-1.json` before shipping.

---

### 14. Word-overlap formula: Jaccard similarity ★ Accuracy · Trivial

**File** — `src/promptcue/core/classifier.py` `_classify_deterministic`

**Problem**
Current overlap is `|Q ∩ T| / |Q|` (query-normalised). A long query with a small
intersection scores lower than a short query with the same intersection, even when the
type vocabulary is far larger.

**Fix** — switch to Jaccard `|Q ∩ T| / |Q ∪ T|`:

```python
union   = query_words | type_words
overlap = len(query_words & type_words) / len(union) if union else 0.0
```

**Trade-off**
Jaccard produces lower absolute scores (larger denominator), pushing more borderline
queries to the semantic fallback. Run against `queries-1.json` first.

---

## Resolved history

### Fifth pass — 2026-03-23

| Fix | File | Detail |
|-----|------|--------|
| Routing dict full shape on failure paths | `core/decision.py` | No-candidates and below-threshold paths returned only `{needs_clarification: True}`; consumers accessing other keys got `KeyError` |
| `should_clarify: true` removed from YAML | `data/query_types.yaml` | `recommendation` and `generation` fired should_clarify on every query; only `troubleshooting` keeps it intentionally |
| Empty/null YAML guard test added | `tests/test_registry.py` | 3 parametrized cases (`''`, `'null'`, `'- item'`) now cover the `isinstance(raw, dict)` guard |
| Language detector warm-up | `extraction/language.py`, `analyzer.py` | `warm_up()` added to `PromptCueLanguageDetector`; called from `PromptCueAnalyzer.warm_up()` |
| `PromptCueBasis` StrEnum added | `models/enums.py`, `models/__init__.py`, `__init__.py` | Typed equivalent for all six basis string values; exported from top-level package |
| `PromptCueBasis` import test added | `tests/test_import.py` | `TRIGGER_MATCH` and `BELOW_THRESHOLD` assertions added to `test_enum_imports` |
| `analysis` action_hints signal added | `data/query_types.yaml` | `should_direct_answer: true` — analysis produces a single coherent evaluation |
| `PromptCueLinguistics` entities sync | `models/schema.py` | `model_validator` keeps `named_entities` in sync when only `entities` is provided |
| Multi-trigger confidence boost | `core/classifier.py` | Each extra matched trigger adds +0.03 (max +0.06), capped at ≤ 0.91 |
| Weak chitchat triggers removed | `data/query_types.yaml` | Removed `thanks`, `awesome`, `ok thanks`, `got it`; kept `great thanks`, `cheers`, etc. |
| Negation guard for trigger matches | `core/classifier.py` | `_is_negated()` demotes triggers preceded by not/never/don't/avoid/etc. to word-overlap tier |
| Batch cosine similarity | `core/embedding.py`, `core/classifier.py` | `cosine_similarity_batch()` replaces Python loop; single NumPy matmul per query |
| `scope` typed as `PromptCueScope` | `models/schema.py` | Field changed from `str` to `PromptCueScope`; Pydantic coerces string inputs automatically |
| `PCUE_UNKNOWN` vs `PCUE_SCOPE_UNKNOWN` documented | `constants.py` | Two-comment block explains the semantic difference and future-proofing intent |
| `_top_margin` helper extracted | `core/classifier.py`, `core/decision.py` | Deduplicates top/second/margin derivation that was copy-pasted across both modules |
| `EmbeddingBackend.warm_up()` role documented | `core/embedding.py` | Docstring clarifies it is for direct `EmbeddingBackend` consumers, not the normal analyzer path |

### Fourth pass — 2026-03-23

| Fix | File | Detail |
|-----|------|--------|
| `py.typed` PEP 561 marker created | `src/promptcue/py.typed` | Declared in `package-data` and `"Typing :: Typed"` classifier but file was missing |
| `demo_run.py` updated to 12 query types | `examples/demo_run.py` | Header, print, and QUERIES list said "9 types"; added summarization/generation/validation |
| `classification_basis` docs corrected | `README.md` | `fallback` basis value was missing from the field description table |
| `asyncio_mode = "auto"` added to pytest config | `pyproject.toml` | Mode was implicitly strict; now explicit |

### Third pass — 2026-03-23

| Fix | File | Detail |
|-----|------|--------|
| `_ALL_ACTION_KEYS` unused dead code removed | `tests/test_schema.py` | Variable defined but never referenced |
| `PCUE_ACTION_DIRECT` unused import removed | `tests/test_schema.py` | Only referenced in now-removed set |
| Enum imports added to public import test | `tests/test_import.py` | `PromptCueActionHint`, `PromptCueRoutingHint`, `PromptCueScope` untested |
| Registry type count floor raised | `tests/test_registry.py` | `>= 9` → `>= 12` (stale after adding 3 types) |
| YAML header comment corrected | `data/query_types.yaml` | "substring matching" → "word-boundary regex matching" |
| `routing_hints`/`action_hints` typed `dict[str, bool]` | `models/schema.py` | Was `dict[str, Any]` |
| `import-untyped` mypy errors fixed | `extraction/language.py`, `extraction/keywords.py` | Inline `# type: ignore[import-untyped]` for langdetect and keybert |
