# PromptCue — Improvement Backlog v1.0
# Maintained after each comprehensive review pass
# Maintainer: Informity

Items are ordered by priority. Work top-to-bottom. Mark items done by moving them
to the "Resolved" section at the bottom with a brief note of what was done.

---

## Backlog — ordered by priority

### 1. Consistent routing dict shape on all paths ★ Correctness · Trivial

**File** — `src/promptcue/core/decision.py`

**Problem**
Successful classification always returns all four routing keys:
`needs_clarification`, `needs_retrieval`, `needs_reasoning`, `needs_current_info`.

The two failure paths (no candidates, below-threshold) return **only** `needs_clarification`.
A consumer accessing `result.routing_hints['needs_retrieval']` raises `KeyError` on any
query classified as `unknown`.

**Fix**
In both failure-path returns inside `resolve()`, replace the sparse dict with the full shape:

```python
routing_hints = {
    PCUE_HINT_CLARIFICATION: True,
    PCUE_HINT_RETRIEVAL:     False,
    PCUE_HINT_REASONING:     False,
    PCUE_HINT_CURRENT_INFO:  False,
}
```

---

### 2. Remove always-true `should_clarify` from YAML ★ Correctness · Trivial

**File** — `src/promptcue/data/query_types.yaml`

**Problem**
`recommendation` and `generation` both have `should_clarify: true` baked into their
`action_hints`. The decision engine copies YAML hints verbatim:

```python
action_hints = {k: bool(v) for k, v in yaml_actions.items()}
if is_ambiguous:
    action_hints[PCUE_ACTION_CLARIFY] = True
```

Because `should_clarify` is already `true` in YAML, it fires on **every** recommendation
or generation query — including clear ones like "Should I use tabs or spaces?" or
"Write me a hello-world script". A caller using this hint to gate a follow-up question
will always ask, which is wrong.

**Fix**
Remove `should_clarify: true` from both types. The `is_ambiguous` guard in the decision
engine will still set it when scores are genuinely close. The YAML should only encode
structural hints that are invariant for a type (survey, enumerate, compare, direct_answer)
— not context-sensitive flags like clarify.

---

### 3. Test: empty/null YAML guard ★ Correctness · Trivial

**File** — `tests/test_registry.py`

**Problem**
The `isinstance(raw, dict)` guard added in `registry.py` (raises `PromptCueRegistryError`
on empty or null YAML) has no tests.

**Fix**
Add three parametrised cases using `tmp_path`:

```python
@pytest.mark.parametrize('content', ['', 'null', '- item'])
def test_registry_rejects_invalid_yaml(tmp_path, content):
    p = tmp_path / 'bad.yaml'
    p.write_text(content)
    with pytest.raises(PromptCueRegistryError):
        PromptCueRegistry.from_yaml(p)
```

---

### 4. Language detector warm-up ★ Correctness · Trivial

**File** — `src/promptcue/extraction/language.py`, `src/promptcue/analyzer.py`

**Problem**
`PromptCueAnalyzer.warm_up()` pre-loads the embedding model, spaCy, and KeyBERT but
**not** `langdetect`. A caller who runs `warm_up()` at startup still pays the import
cost on the first `detect()` call.

**Fix**
Add `warm_up()` to `PromptCueLanguageDetector` (mirroring `LinguisticExtractor` and
`KeywordExtractor`) and call it from `PromptCueAnalyzer.warm_up()`:

```python
# language.py
def warm_up(self) -> None:
    if self.enabled:
        self._ensure_lib()

# analyzer.py  warm_up()
self.language_detector.warm_up()
```

---

### 5. `PromptCueBasis` StrEnum ★ API consistency · Trivial

**File** — `src/promptcue/models/enums.py`, `models/__init__.py`, `__init__.py`

**Problem**
`classification_basis` in `PromptCueQueryObject` is a raw string.
`PromptCueScope`, `PromptCueRoutingHint`, and `PromptCueActionHint` all have typed
StrEnums; `basis` is the only consumer-facing string field without one.

**Fix**

```python
class PromptCueBasis(StrEnum):
    LABEL_MATCH     = 'label_match'
    TRIGGER_MATCH   = 'trigger_match'
    WORD_OVERLAP    = 'word_overlap'
    FALLBACK        = 'fallback'
    SEMANTIC        = 'semantic_similarity'
    BELOW_THRESHOLD = 'below_threshold'
```

Export from `models/__init__.py` and top-level `__init__.py`. Update constants.py comment
to note that `PromptCueBasis` is the typed equivalent.

---

### 6. `analysis` action_hints — document intent or add signal ★ API clarity · Trivial

**File** — `src/promptcue/data/query_types.yaml`

**Problem**
`analysis` is the only type with all seven `action_hints` set to `false`. A consumer
iterating action hints gets zero signal and cannot drive LLM response formatting.

**Options (pick one)**
- **A (document):** Add a YAML comment explaining this is intentional — analysis responses
  are freeform evaluations and the LLM should choose its own structure.
- **B (add signal):** Set `should_direct_answer: true` — an evaluation is a single coherent
  response, not a survey or enumeration.
- **C (new key):** Add `should_evaluate: true` — requires updating `constants.py`,
  `enums.py`, all 12 YAML entries, and README. High effort, cleanest long-term.

**Recommendation** — ship Option B now (1 YAML line), revisit Option C in a future version.

---

### 7. `PromptCueLinguistics` entities/named_entities consistency ★ Data integrity · Small

**File** — `src/promptcue/models/schema.py`

**Problem**
`PromptCueLinguistics` has two parallel fields: `entities: list[PromptCueEntity]` and
`named_entities: list[str]` (plain-text backward-compat alias).
A consumer who constructs `PromptCueLinguistics(entities=[...])` directly gets `entities`
populated but `named_entities` empty — silent inconsistency.

**Fix**

```python
from pydantic import model_validator

@model_validator(mode='before')
@classmethod
def _sync_named_entities(cls, data: dict) -> dict:
    if not data.get('named_entities') and data.get('entities'):
        data['named_entities'] = [
            e['text'] if isinstance(e, dict) else e.text
            for e in data['entities']
        ]
    return data
```

---

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

### 9. Multi-trigger confidence boost ★ Accuracy · Trivial

**File** — `src/promptcue/core/classifier.py` `_classify_deterministic`

**Problem**
When multiple trigger phrases match, only the longest contributes to the score. Matching
"compare" and "pros and cons" is no stronger than matching "compare" alone.

**Fix**

```python
bonus = min((len(matched) - 1) * 0.03, 0.06)  # +0.03 per extra, max +0.06
score = round(0.60 + 0.25 * specificity + bonus, 4)
```

Keeps score ceiling at ≤ 0.91. Reduces unnecessary semantic fallback on richly-phrased
queries.

---

### 10. Chitchat trigger false-positive risk ★ Accuracy · Trivial

**File** — `src/promptcue/data/query_types.yaml`

**Problem**
Chitchat triggers include very short common words: `thanks`, `hey`, `awesome`, `ok thanks`.
A technical query ending in "thanks!" or beginning with "hey, how do I..." can edge toward
`chitchat` when scores are close.

**Options**
- **A (simpler):** Remove the four weakest triggers (`thanks`, `awesome`, `ok thanks`,
  `got it`) and rely on semantic scoring for those social phrases.
- **B (surgical):** Require chitchat triggers to appear at the **start or end** of the
  query (position-aware matching, ~5 lines in classifier).

**Recommendation** — ship Option A now; revisit B only if false positives persist.

---

### 11. Negation guard for trigger matches ★ Accuracy · Small

**File** — `src/promptcue/core/classifier.py`

**Problem**
"When NOT to use caching" fires on the `procedure` trigger "to use".
"I don't recommend comparing these" fires on "compare".

**Fix**

```python
_NEGATION_WORDS = frozenset({'not', 'never', "don't", 'no', 'avoid', 'without'})

def _is_negated(phrase: str, lowered: str) -> bool:
    idx = lowered.find(phrase.lower())
    if idx < 0:
        return False
    prefix = lowered[:idx].split()
    return bool(prefix) and prefix[-1] in _NEGATION_WORDS
```

When `_is_negated` returns True, demote score to word-overlap tier and change basis to
`word_overlap`.

---

### 12. Batch cosine similarity (NumPy vectorised) ★ Performance · Small

**File** — `src/promptcue/core/embedding.py`, `classifier.py`

**Problem**
`_classify_semantic` calls `cosine_similarity(query_vec, ex_vec)` in a Python loop,
allocating two `np.array` objects per example per query.

**Fix**

```python
def _cosine_batch(query: list[float], matrix: list[list[float]]) -> list[float]:
    q   = np.array(query,  dtype=np.float32)
    m   = np.array(matrix, dtype=np.float32)
    q_n = np.linalg.norm(q)
    m_n = np.linalg.norm(m, axis=1)
    denom = q_n * m_n
    denom[denom == 0] = 1.0
    return np.clip((m @ q) / denom, 0.0, 1.0).tolist()
```

Replace the per-example loop with `max(_cosine_batch(query_vec, example_vecs))`.
Combines naturally with the centroid proposal (item 13) or can stand alone.

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

### 15. `scope` field typed as `PromptCueScope` ★ Type safety · Small

**File** — `src/promptcue/models/schema.py`

**Problem**
`PromptCueQueryObject.scope: str` accepts any string. `PromptCueScope` exists but
is not enforced at the schema boundary.

**Fix**

```python
scope: PromptCueScope = PromptCueScope.UNKNOWN
```

`PromptCueScope` is a `StrEnum`, so Pydantic serialises it as a plain string and JSON
output is unchanged. Pydantic will reject invalid scope strings on construction —
stricter but safer.

---

### 16. `PCUE_UNKNOWN` / `PCUE_SCOPE_UNKNOWN` — document distinction ★ Documentation · Trivial

**File** — `src/promptcue/constants.py`

**Problem**
Both constants have the identical value `'unknown'` but different semantic roles:
`PCUE_UNKNOWN` is the sentinel for `primary_query_type`; `PCUE_SCOPE_UNKNOWN` is for
`scope`. A consumer comparing `result.scope == PCUE_UNKNOWN` gets the correct answer
by accident — but the two could diverge.

**Fix**
Add a two-line comment making the distinction explicit. Prefer `PromptCueScope.UNKNOWN`
over `PCUE_SCOPE_UNKNOWN` in internal code going forward.

---

### 17. Extract `_top_margin` helper ★ Code cleanliness · Trivial

**File** — `src/promptcue/core/classifier.py`, `core/decision.py`

**Problem**
The pattern of computing `top`, `second_score`, `margin` from a candidates list is
duplicated verbatim in `PromptCueClassifier.classify` and `PromptCueDecisionEngine.resolve`.

**Fix**

```python
def _top_margin(
    candidates: list[PromptCueCandidate],
) -> tuple[PromptCueCandidate | None, float]:
    """Return (top_candidate, margin_between_top_two)."""
    if not candidates:
        return None, 0.0
    second = candidates[1].score if len(candidates) > 1 else 0.0
    return candidates[0], candidates[0].score - second
```

---

### 18. `EmbeddingBackend.warm_up()` — decide its role ★ API surface · Trivial

**File** — `src/promptcue/core/embedding.py`

**Problem**
`PromptCueEmbeddingBackend.warm_up()` is never called from the normal path.
`PromptCueClassifier.warm_up()` reaches the model via `_build_example_cache()` →
`encode()` → `_ensure_model()`. The method is reachable only by consumers who
construct `PromptCueEmbeddingBackend` directly.

**Decision needed**
- Remove it (not part of the internal API, safe to cut).
- Keep it with a docstring note clarifying it is for direct `EmbeddingBackend` consumers
  only, not the normal `PromptCueAnalyzer.warm_up()` path.

---

## Resolved history

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
