# Column analysis: plan and tracker

`EXPLAIN` / `outputs=['explanation']` also returns column analysis: per-column cohort
statistics computed by the client from the request's context rows, mirroring kumo-ml's RFM
column analysis (`kumoml/common/explain/column_analysis.py`). No NIM or engine change.

**Status:** `TO DO` · `WAITING (on X)` · `DONE (commit)`

| # | Issue | End state | Status |
|---|---|---|---|
| 1 | `result.explanations` is the per-row gradients dict; column analysis is per-request and has no home | `result.explanations` is a container: `.gradients` (today's `GradientExplanations`, unchanged) and `.column_analysis` (new `ColumnAnalysis`), each with its own `to_pandas()`; existing `result.explanations[...]` callers in tests and docs move to `.gradients[...]` | TO DO |
| 2 | `explanations/explanations.py` holds only the gradients class but is named for all of it | Renamed `explanations/gradient.py` | TO DO |
| 3 | No column analysis exists in the client | `ColumnAnalysis` in `explanations/column.py`, kumo-ml semantics: 10 quantile bins for numerics and `COUNT(*)`, per-category cohorts with 500-cap tail merged into other, time-before-anchor buckets; per cohort the population share and mean outcome; column score = population-weighted std of cohort means, normalized across columns | TO DO |
| 4 | The tabular payload never names the class order, so a single "positive share" per cohort is ambiguous | Classification reports a mean per class, binary included (two classes, named by the labels); class set and order from the context labels by frequency, 30-class cap; regression reports the plain mean | TO DO |
| 5 | The context is only reliably in the client's hands during a `predict` call | Computed on `predict` only, whenever explanations are asked, from the context the request in hand carries (verified: tabular requests always hold labeled context, session scoring requests must carry the pinned context, relational `materialize()` always gathers it) | TO DO |
| 6 | Relational payloads carry no explicit hops, per-entity links, or encoder decisions | Per (table, hop, column): hops by the BFS the decoder already has (`_hops_from_entity`, pointed at the context half); per-entity aggregation (mean / category shares / relative time); anchor from `task.anchor_time_column`, non-temporal graphs get no time cohorts; the encoder heuristics (all-NA and too-wide drop, ID→categorical) and multicategorical separator inference ported from kumo-ml | TO DO |
| 7 | Tabular has no graph or anchor time | Hop 0 only, `COUNT(*)` trivial, no time cohorts | TO DO |
| 8 | The computation needs one hook both backends pass through | Hooked in `adapter.answered(...)`, shared by wire and in-process; forecast steps and ranking parts need nothing special (each payload carries its own context; rankings refuse explanations anyway) | TO DO |
| 9 | No tests | Unit tests per the `unittesting` skill, both models, cohort values checked by hand | TO DO |
| 10 | Docs describe explanations as the gradients dict | Explanation sections gain `.gradients` / `.column_analysis`; CHANGELOG `## Unreleased` | TO DO |
| 11 | `tests/golden/public_api.json` pins the old shape | Snapshot regenerated (the container is a new public shape; #34 is unreleased) | TO DO |
| 12 | Local gate | ruff, format, mypy, hygiene, suites green; counts recorded here | TO DO |
