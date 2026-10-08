# Phase 0 tracker

Status values: not started, in progress, done, blocked.
Each row links its evidence when done.

| Task | Status | Evidence | Notes |
|---|---|---|---|
| T0.1 CI on the migration branches | done | CI run 37710872493 green on `rust-core-phase-0` | Also formatted the diagram script for ruff |
| T0.2 Archive the first attempt | done | [archive.md](archive.md); branch `archive/rust-core-attempt-1` at 3887762 | Catalog and workspace names redacted (45 files); commit message reworded |
| T0.3 Reference branch and environment | done | [determinism-fixes.md](determinism-fixes.md), [reference-environment.md](reference-environment.md); branch `rust-core-reference` | 3 fixes ported, guard works, repeatable under both hash seeds with and without torch; owner decided D1 (not adopted) and D2 (accepted); baseline tag `reference-baseline-fa0637d` pushed and protected by a tag ruleset |
| T0.4 Support matrix | done | [support-matrix.md](support-matrix.md) | Gaps G1 to G14 decided by the owner |
| T0.5 Scope manifests and case inventory format | done | [scope-format.md](scope-format.md), [scope-python.yaml](scope-python.yaml), [scope-java.yaml](scope-java.yaml), [databricks-jobs.md](databricks-jobs.md) | Written and validated; J1 to J3 decided and applied |
| T0.6 Random streams and time library | done | [random-streams.md](random-streams.md), [date-rules.md](date-rules.md), [chrono-check.md](chrono-check.md) | Five generators plus the C++ neighbor draw modelled exactly; chrono agrees on all needed operations; five owner decisions proposed as register entries |
| T0.7 Phase 1b case inventory | done | [phase-1b-inventory.yaml](phase-1b-inventory.yaml) | 40 DuckDB cases (24 succeed, 16 refuse today), all run against the reference in about 1.4 s; 13 flagged for register entries if Phase 1b changes them; runner kept outside the repo |
| T0.8 Register and decision record | in progress | | |
| T0.9 PQL asks | not started | | Sent by the owner; not a phase exit |
| Phase review | not started | | Fresh-agent review before the owner merges |
