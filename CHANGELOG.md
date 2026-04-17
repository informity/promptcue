# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.3.7] — 2026-04-17

### Added

- New generic intent label: `chat_summary` for prompts that ask to recap the current conversation
  (for example: "summarize our chat", "what did we discuss", "discussion topics in this chat").
- New routing hint: `routing_hints['needs_chat_history']` to signal that consumers should route
  to conversation-history summarization paths instead of retrieval-heavy corpus paths.
- New registry examples/triggers/negatives for `chat_summary` to improve semantic and deterministic
  detection without app-specific rules.

### Changed

- PromptCue default taxonomy expanded from 12 to 13 query types.
- README updated with the new query type and routing hint guidance.
- Test suite expanded with deterministic, analyzer, semantic, and registry coverage for
  `chat_summary` and `needs_chat_history`.

## [0.3.6] — 2026-04-08

### Fixed

- Broad “across all sources” entity/date prompts no longer fall back to `unknown`;
  analyzer now promotes these multi-item synthesis/listing shapes to `coverage`.
- Coverage promotion now also applies when the initial classifier label is `unknown`,
  as long as broad multi-item semantic signals are present.

### Changed

- Expanded generic enumeration detection for prompts like “what are the names/dates/...”
  to improve stable routing for corpus-wide listing requests.

## [0.3.5] — 2026-04-07

### Changed

- Intent routing now promotes focused-family labels to `coverage` for clearly broad, multi-item structured prompts (model-agnostic heuristic based on semantic hints).
- Coverage query examples/triggers were expanded for broad synthesis phrasing while keeping registry vocabulary generic (no app-specific routing dependency).
- Added analyzer regression tests for broad multi-item prompts that previously drifted to focused routing.

## [0.3.4] — 2026-04-06

### Changed

- Documentation alignment for explicit recency signals:
  `semantic_hints.explicit_recency` is now consistently documented across README and changelog
- Release notes cleanup: `0.3.3` and `0.3.2` sections normalized for accurate historical record

---

## [0.3.3] — 2026-04-06

### Added

- `semantic_hints.explicit_recency` on `PromptCueQueryObject` for explicit freshness wording
  (for example: `today`, `tomorrow`, `now`, `current`, `latest`, `this week/month/year`)
- Explicit-recency promotion in analyzer: when explicit freshness wording is detected,
  PromptCue now sets `routing_hints['needs_current_info']=True` and
  `action_hints['should_check_recency']=True` even when primary type is not `update`

### Changed

- README field guidance updated for freshness routing:
  prefer `routing_hints['needs_current_info']` and `semantic_hints.explicit_recency`
  for web-search/recency-aware pipelines

---

## [0.3.2] — 2026-03-28

### Added

- `PromptCueConfig.embed_fn: Callable[[str], list[float]] | None` — injectable embed
  function for **hosted mode**: when set, PromptCue delegates all vector computation to the
  caller's function and never loads a `sentence-transformers` model of its own. Intended for
  applications that already have an embedding model loaded (e.g. for RAG) and want to reuse
  it for query classification without a second model load
- `PromptCueEmbedFn` type alias — `Callable[[str], list[float]]`; exported from the package
  root for use in type annotations
- `PromptCueConfig` model validator — when `embed_fn` is set, `enable_semantic_scoring` is
  forced to `True` automatically; semantic classification runs even when `sentence-transformers`
  is not installed in the environment
- `PromptCueEmbeddingBackend` hosted-mode behaviour — `is_loaded` returns `True` immediately
  when `embed_fn` is provided; `warm_up()` is a no-op; `encode()` calls `embed_fn(text)` per
  text instead of running through the model
- 7 new tests in `tests/test_core.py::TestInjectableEmbedFn` covering the full hosted-mode
  path end-to-end without `sentence-transformers` installed
- README: new "Hosted mode" subsection under Production deployment, new Quick-start example,
  `embed_fn` row in `PromptCueConfig` fields table
- `semantic_hints` on `PromptCueQueryObject` with agnostic keys:
  `mentions_multiple_items`, `requests_comparison`, `requests_enumeration`,
  `requests_structure`, `mentions_time`, `requires_multi_period_analysis`
- `confidence_meta` on `PromptCueQueryObject`:
  `type_confidence_margin`, `scope_confidence`, `scope_confidence_margin`
- `explanations` on `PromptCueQueryObject`:
  `decision_notes`, `evidence_tokens`
- Temporal semantics now live only in `semantic_hints`:
  `mentions_time` + `requires_multi_period_analysis` (not duplicated in `routing_hints`)
- 25 new tests in `tests/test_core.py::TestTemporalScope` covering True/False detector cases
  and end-to-end routing_hints key presence
- `PromptCueConfig.show_progress_bar: bool` (default `False`) — standalone-mode control
  for `SentenceTransformer.encode(show_progress_bar=...)`; keeps batch tqdm output disabled
  by default for clean server logs, with explicit opt-in for local debugging
- 2 new tests in `tests/test_core.py::TestEmbeddingBackend` verifying `show_progress_bar`
  forwarding (`False` default and `True` opt-in) on standalone embedding calls

---

## [0.2.1] — 2026-03-25

### Added

- `PromptCueConfig.show_progress_bar: bool` (default `False`) to control standalone
  sentence-transformers batch progress output
- README documentation for `show_progress_bar` in production deployment guidance and
  the `PromptCueConfig` fields table
- standalone tests verifying `show_progress_bar` forwarding in `PromptCueEmbeddingBackend`

### Changed

- standalone semantic embedding calls now pass `show_progress_bar` explicitly to
  `SentenceTransformer.encode(...)` (silent by default)

---

## [0.2.0] — 2026-03-25

### Added

- `is_continuation` field on `PromptCueQueryObject` — `bool`, populated by pure-regex
  detection of leading continuation openers (`also`, `furthermore`, `following up`,
  `building on that`, etc.) before classification runs; no model dependency, no new type
- `confidence_band` field on `PromptCueQueryObject` — `PromptCueConfidenceBand` enum
  (`high` / `medium` / `low`); trigger-match results always map to `high`; semantic and
  word-overlap results mapped by `confidence_high_threshold` (0.65) and
  `confidence_medium_threshold` (0.35) in `PromptCueConfig`
- `runner_up` property on `PromptCueQueryObject` — returns the second-ranked
  `PromptCueCandidate` directly; previously only accessible by indexing
  `candidate_query_types[1]`
- `to_routing_dict()` method on `PromptCueQueryObject` — returns a flat `dict[str, bool]`
  merging `routing_hints` and `action_hints` for callers who only need the downstream hints
- `needs_structure` key in `routing_hints` — `True` when the query contains Markdown heading
  patterns, `format as table`, `output in bullet points`, `with sections:`, etc.; signals
  that the caller should handle format extraction before routing
- `PromptCueConfidenceBand` enum — exported from package root alongside existing enums
- `PromptCueConfig.strict()`, `.balanced()`, `.recall_heavy()` — named classmethods
  returning pre-calibrated `PromptCueConfig` instances with documented threshold rationale
- `PromptCueConfig.confidence_high_threshold` (default `0.65`) and
  `confidence_medium_threshold` (default `0.35`) — thresholds controlling `confidence_band`
  assignment
- `PromptCueConfig.model_cache_dir: Path | None` — explicit model cache directory passed as
  `cache_folder` to `SentenceTransformer`; falls back to `PROMPTCUE_MODEL_CACHE` env var,
  then HuggingFace default (`~/.cache/huggingface/`)
- `ambiguity_margin_override` per-type field in `query_types_en.yaml` — allows individual
  types to override the global `ambiguity_margin`; `analysis` ships with `override=0.05`
- `--matrix` flag in `.internal/tests/run.py` — prints per-type confusion matrix with
  precision, recall, and F1 after each evaluation run
- `--check` flag in `.internal/tests/run.py` — scans `query_types_en.yaml` and reports any
  trigger phrase shared across two or more types; exits 0 when clean
- 49 new unit tests in `tests/test_core.py` and `tests/test_language.py` — covering
  normalization, cosine similarity, `EmbeddingBackend`, `DecisionEngine`, all classifier
  tiers, all new Phase 1 schema fields, and the offline model-load failure path; total test
  count raised from 67 to 116
- Production deployment section in README — documents fail-fast contract,
  `model_cache_dir` / `PROMPTCUE_MODEL_CACHE` usage, and deployment patterns for
  local dev, EC2/EBS, Lambda container image, Lambda EFS, and Docker

### Changed

- `query_types_en.yaml` — 176 lines of new triggers, negatives, and examples validated
  against 174 queries across 4 independent test sets (40 self-generated, 53 blind
  self-generated, 41 Claude-generated, 40 Codex-generated); overall accuracy 100% with
  no trigger overlaps (`--check` clean). Additions cover informal vocabulary
  (`give me the gist`, `knock together`, `what have I missed`, `what's actually changed`),
  non-tech domains (compliance, legal, HR, business operations), and 12 trigger-greed
  fixes (e.g. `what changed in` removed from `update` to stop stealing troubleshooting
  queries about user-behaviour changes; `what does` given coverage/procedure negatives)
- `PromptCueBasis` — removed `label_match` and `fallback` values; `label_match` tier was
  removed in v0.1.4 and `fallback` was never emitted by the decision engine; both are
  now absent from the enum, the README, and `classification_basis` documentation
- Word-overlap tier now uses Jaccard similarity (`|Q∩T| / |Q∪T|`) instead of
  query-normalised overlap (`|Q∩T| / |Q|`); produces lower absolute scores, pushing more
  ambiguous queries to the semantic path rather than returning a low-confidence
  deterministic result
- `query_types_en.yaml` renamed from `query_types.yaml` — file name now includes language
  suffix (`_en`) for future multi-language registry support
- README `PromptCueQueryObject` fields table updated: `is_continuation`, `confidence_band`,
  and `runner_up` added; `classification_basis` values corrected
- README `PromptCueConfig` fields table updated: `model_cache_dir` row added
- README query types table sorted alphabetically

---

## [0.1.4] — 2026-03-23

### Added
- `PromptCueBasis` (`StrEnum`) — typed equivalent for all six `classification_basis` string values (`label_match`, `trigger_match`, `word_overlap`, `fallback`, `semantic_similarity`, `below_threshold`); exported from top-level package
- `PromptCueLanguageDetector.warm_up()` — pre-loads `langdetect` at startup; called automatically by `PromptCueAnalyzer.warm_up()`
- Negation guard (`_is_negated`) in `PromptCueClassifier` — trigger phrases preceded by `not`, `never`, `don't`, `avoid`, etc. are demoted to word-overlap tier, preventing false positives like "When NOT to use caching" firing on the `procedure` trigger "to use"
- Multi-trigger confidence boost — each additional matched trigger phrase adds +0.03 to the score (capped at +0.06), so richly-phrased queries like "compare and contrast — pros and cons" score higher than single-trigger matches
- `cosine_similarity_batch()` in `core/embedding.py` — vectorised NumPy batch replace the per-example Python loop in the semantic path; single matrix multiply per query instead of N dot products
- Three parametrised tests for the empty/null YAML guard in `tests/test_registry.py`
- `PromptCueBasis` assertions added to `tests/test_import.py`

### Changed
- `PromptCueQueryObject.scope` field type changed from `str` to `PromptCueScope` — Pydantic coerces string inputs automatically; JSON output is unchanged
- Both failure paths in `PromptCueDecisionEngine.resolve()` (no candidates, below threshold) now return the full four-key `routing_hints` dict — prevents `KeyError` on `unknown` queries
- `PromptCueLinguistics.named_entities` is now kept in sync automatically when only `entities` is provided, via a `model_validator`
- `analysis` query type now sets `should_direct_answer: true` — an evaluation is a single coherent response; previously all seven action hints were `false`
- `recommendation` and `generation` query types no longer hard-code `should_clarify: true` — the decision engine still sets it dynamically when scores are genuinely ambiguous
- Removed weak chitchat triggers (`thanks`, `awesome`, `ok thanks`, `got it`) that caused false positives on technical queries ending in casual acknowledgements
- `_top_margin()` helper extracted and shared between `PromptCueClassifier.classify()` and `PromptCueDecisionEngine.resolve()` — removes duplicated top/second/margin derivation
- `PCUE_UNKNOWN` and `PCUE_SCOPE_UNKNOWN` now have explicit comments documenting their distinct semantic roles
- `PromptCueEmbeddingBackend.warm_up()` docstring clarifies it is for direct `EmbeddingBackend` consumers; the normal `PromptCueAnalyzer.warm_up()` path reaches the model via the classifier's example cache build
- Added recommendation triggers: `what is the best way to`, `best way to handle`, `best approach to`

---

## [0.1.3] — 2026-03-23

### Added
- Three new query types expanding the default taxonomy from 9 to 12:
  - `summarization` — condense existing content (provided, referenced, or in-context) into a shorter form
  - `generation` — produce entirely new content from scratch with no existing source to condense
  - `validation` — verify or fact-check a specific stated claim, assumption, or belief
- `PromptCueAnalyzer.warm_up_async()` — async variant of `warm_up()`; delegates to `asyncio.to_thread()`
- `PromptCueAnalyzer.analyze_async()` — async variant of `analyze()`; delegates to `asyncio.to_thread()`
- `PromptCueConfig.trigger_fallback_threshold` (default `0.60`) — when a trigger phrase matched and the score meets this value with a clear margin, the deterministic result is trusted directly and the semantic pass is skipped
- `pytest-asyncio` added to `dev` optional dependency group to make the async test requirement explicit

### Changed
- Deterministic classifier now uses word-boundary regex matching (`\b…\b`) for all label and trigger comparisons — prevents false positives caused by substring containment (e.g. `"invalidation"` matching `validation`, `"devs"` matching `vs`)
- Cascade logic introduces a `trigger_confident` guard — strong trigger matches that meet `trigger_fallback_threshold` and `ambiguity_margin` no longer get overridden by the semantic pass
- All lazy model loading paths (`EmbeddingBackend`, `Classifier`, `LinguisticExtractor`, `KeywordExtractor`) are now protected by `threading.Lock` — safe for concurrent use
- Expanded triggers and semantic examples across `coverage`, `comparison`, `procedure`, `summarization`, `validation`, and `troubleshooting` to improve recall on natural/conversational phrasing

---

## [0.1.2] — 2026-03-23

### Changed
- Renamed repository from `informity-promptcue` to `promptcue`
- Replaced static PyPI version badge with live shields.io badge

---

## [0.1.1] — 2026-03-23

### Fixed
- Corrected package author metadata (Informity)

---

## [0.1.0] — 2026-03-22

### Added
- Initial release — installable Python package with stable public API
- `PromptCueAnalyzer` — main entry point; call `.analyze(text)` to get a `PromptCueQueryObject`
- `PromptCueConfig` — Pydantic config model with runtime toggles and thresholds
- `PromptCueQueryObject` — structured result with `schema_version`, `primary_query_type`, `classification_basis`, `confidence`, `ambiguity_score`, `candidate_query_types`, `routing_hints`, and optional enrichment fields
- `PromptCueRegistry` — loads query type definitions from YAML; supports custom registry via `PromptCueConfig.registry_path`
- 9-type default query taxonomy: `coverage`, `lookup`, `comparison`, `recommendation`, `troubleshooting`, `procedure`, `analysis`, `update`, `chitchat`
- Dual-path classifier: deterministic (trigger-phrase matching) and semantic (`sentence-transformers` cosine similarity)
- Lazy model loading — `sentence-transformers` model loads on first use, not at construction time
- `PromptCueAnalyzer.warm_up()` — pre-loads the embedding model and example embeddings at application startup
- Optional linguistic enrichment via `PromptCueLinguisticExtractor` (spaCy)
- Optional keyword enrichment via `PromptCueKeywordExtractor` (KeyBERT)
- Registry validation — raises `PromptCueRegistryError` on malformed entries
- `constants.py` — single source of truth for all package paths, sentinel values, and routing hint keys
- MIT license
- CI workflow (GitHub Actions) for tests and wheel build on every push

---

[0.1.4]: https://github.com/informity/promptcue/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/informity/promptcue/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/informity/promptcue/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/informity/promptcue/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/informity/promptcue/releases/tag/v0.1.0
