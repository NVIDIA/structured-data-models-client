# Phase 2 tracker: T2.16

The slice gate (brief `T2.16.md`; design, the T2.16 row and "The evidence pack"; handbook, "Ending a phase").
Lane q, on the phase head f133c4b8 (`Land T2.14c`); the reports first ran on be55838a, T2.14c's candidate, whose tree f133c4b8's landing holds.

## What T2.16 holds

- Owner rows: KO2-90 (the two kept rules' real-data counts, check 2) and KO2-91 (the lean plan of 2026-10-17, never recorded until now); KO2-92, the orchestrator's decision under KO2-88 on the third kept rule, `aggregation`.
- The tracker's true package statuses and the stop-rule counters from the log; six per-package statuses the owner's answers had settled.
- [as-built.md](as-built.md); the "Carried into Phase 3" rows in migration-plan.md section 8.
- The evidence pack, [evidence/README.md](evidence/README.md).
- An answer's text encoding (the whole-phase review part 2's finding, a regression since T2.14b): the core reads UTF-8 with a mark, UTF-16 and UTF-32 as Python's reader decodes bytes; register P1B-24 widened to raw lone surrogates (the orchestrator's decision under KO2-88).
- DuckDB capped below 2.0 in the connectors' `duckdb` extra (2.0 broke CI's `engine` and `installed-wheel` jobs on aa44c1d9; the orchestrator's decision under KO2-88; P2-T2.16-12 carries 2.0's support); CI's `installed-wheel` job installs the candidate's own wheels first, since `--find-links` alone let pip take the published connectors of the same version.
- `scripts/rust_core/cells.sh` installs the connectors' SQLite and Postgres drivers, as CI's cells do, with one test (`tests/test_cells_drivers.py`).

## What the phase-end run needs (found with `phase_end.sh` dry)

- `SDM_INVENTORY=docs/design/rust-core/phase-0/phase-1b-inventory.yaml`: Phase 2's locked set lists `cases`, not an inventory file.
- `SDM_PYTHON=<the reference cell's python>`: the gate's `l1` step runs `sdm-testkit`'s SQLite tests, whose case loader runs under `SDM_PYTHON` (`python3` by default, which lacks `referencing` on the phase-end machine).
- Cells holding the SQLite and Postgres drivers (`cells.sh` from this package on), and `SDM_POSTGRES_TEST_DSN`.

## The held-out run

Run once by the orchestrator on a2178e5b: 47 cases, 140 legs, 0 panics, 0 new refusals, python-core 0 differences, rust-host 4 cases (12 legs) where the endpoint refuses a call, 2 of them approved before the run under P2G-1; within the threshold, GO ([evidence/heldout/](evidence/heldout/)).

## Review rounds

| Round | SHA | Claude | Codex | Findings |
|---|---|---|---|---|
| 2 | aa44c1d9 | APPROVE but for CI (DuckDB 2.0) | – | Three P3s, fixed in fix 2 |
| 1 | 4a835391 | CHANGES REQUIRED | – | P2-1 P1B-24's wording; P2-2 the final tree; P2-3 as-built's stale lines; P2-4 carried rows for every gap; five P3s; all in fix 1 |
