# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.3]: https://github.com/informity/promptcue/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/informity/promptcue/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/informity/promptcue/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/informity/promptcue/releases/tag/v0.1.0
