# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] — 2026-03-23

### Fixed
- Corrected package author metadata (Informity)

## [0.1.0] — 2026-03-22

### Added
- Initial Phase 1 release — installable Python package with stable public API
- `PromptCueAnalyzer` — main entry point; call `.analyze(text)` to get a `PromptCueQueryObject`
- `PromptCueConfig` — Pydantic config model with runtime toggles and thresholds
- `PromptCueQueryObject` — structured result with `schema_version`, `primary_query_type`, `classification_basis`, `confidence`, `ambiguity_score`, `candidate_query_types`, `routing_hints`, and optional enrichment fields
- `PromptCueRegistry` — loads query type definitions from YAML; supports custom registry via `PromptCueConfig.registry_path`
- 9-type default query taxonomy: `coverage`, `lookup`, `comparison`, `recommendation`, `troubleshooting`, `procedure`, `analysis`, `update`, `chitchat`
- Dual-path classifier: deterministic (substring trigger matching) and semantic (`sentence-transformers` cosine similarity)
- Lazy model loading — `sentence-transformers` model loads on first use, not at construction time
- `PromptCueAnalyzer.warm_up()` — pre-loads the embedding model and example embeddings at application startup
- Optional linguistic enrichment via `PromptCueLinguisticExtractor` (spaCy placeholder)
- Optional keyword enrichment via `PromptCueKeywordExtractor` (KeyBERT placeholder)
- Registry validation — raises `PromptCueRegistryError` on malformed entries
- `constants.py` — single source of truth for all package paths, sentinel values, and routing hint keys
- MIT license
- CI workflow (GitHub Actions) for tests and wheel build on every push

[0.1.0]: https://github.com/informity/promptcue/releases/tag/v0.1.0
