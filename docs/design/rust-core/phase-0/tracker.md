# Phase 0 tracker

Status values: not started, in progress, done, blocked.
Each row links its evidence when done.

| Task | Status | Evidence | Notes |
|---|---|---|---|
| T0.1 CI on the migration branches | done | CI run 37710872493 green on `rust-core-phase-0` | Also formatted the diagram script for ruff |
| T0.2 Archive the first attempt | in progress | [archive.md](archive.md) | Built and scanned locally; waiting on the owner's call on the catalog name and commit message before pushing |
| T0.3 Reference branch and environment | in progress | [determinism-fixes.md](determinism-fixes.md), [reference-environment.md](reference-environment.md); branch `rust-core-reference` | 3 fixes ported, guard works, repeatable under both hash seeds with and without torch; 2 owner decisions and the baseline tag ruleset left |
| T0.4 Support matrix | in progress | [support-matrix.md](support-matrix.md) | Written; gaps G1 to G14 wait on owner decisions |
| T0.5 Scope manifests and case inventory format | in progress | [scope-format.md](scope-format.md), [scope-python.yaml](scope-python.yaml), [scope-java.yaml](scope-java.yaml), [databricks-jobs.md](databricks-jobs.md) | Written and validated; J1 to J3 wait on owner decisions |
| T0.6 Random streams and time library | not started | | |
| T0.7 Phase 1b case inventory | not started | | Depends on T0.5 |
| T0.8 Register and decision record | not started | | |
| T0.9 PQL asks | not started | | Sent by the owner; not a phase exit |
| Phase review | not started | | Fresh-agent review before the owner merges |
