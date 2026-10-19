# Phase 2 tracker: T2.1f

The design branch owns `tracker.md`; this file holds T2.1f's section until it folds in.
Lane d, on the phase head after T2.3's landing (68f9b8e4); a judging package, landing alone before T2.18 (the orchestrator's decision).

## T2.1f

T2.1f teaches the comparator one owner-approved difference, KO2-36's: a column only the core keeps, which the reference leaves out.

### What changed

- **`core_only_columns`** (`register.schema.json`): an entry may list columns as `<table>.<column>`, split at the first dot (a table's name holds none; a column's may).
- **`compare.py`** reads them as it reads T2.1e's `statements_differ`: only from an entry approved by the owner under a row of the harness's own decisions the owner approved (`core_only_columns`), for the lines of the entry's `cases` on every database.
  For those lines, against the reference only, a core runner's side has exactly those columns taken out, with the key, time column and links they make: the public schema (`_without_in_schema`), each payload's tables, rows, schema and relationships, the core's call included (`_without_in_payload`), and each statement result read with one, where the statement names the table and the reference's result lacks the column (`_without_in_results`, `_core_only_out`: the reference reads a column it then leaves out).
  The reference's from_source warning of leaving them out, one `SDMWarning` per table named, the first of that class, is taken out of its side for each table whose named column the runner kept (`without_left_out_warning`); as that warning, and each kept column, is taken out only where the core has the column and a result's only where the reference's lacks it, a runner answering as the reference does (the contract tests' observations synthesized from the reference's) still passes.
  A result read with such a column has its row count compared and its row digests, which hash the column, reported, not gated.
  In a runtime cell other than the reference's, python-core is held to python-old, which leaves the column out as the reference does: the same taking out applies there (`_cell_pair`).
  A payload table's key, time column or end time column naming such a column is taken out with it.
  Everything else is compared exactly; between the core runners nothing changes.

### Fix rounds

- Round 1 (Claude, P2): the runtime cells' python-core against python-old pair compared raw observations, so the named columns stayed there; now taken out as against the reference, with one assertion in `dbr17-pandas2`. P3s taken: a payload table's time column naming a named column, and a negative control that a result read with one, its row count differing, still fails.

### Probes

- `contract/tests/test_t21f_core_only_columns.py`: the reference's `base.regression` line, observed as python-core keeping a column `a.b` of `users` (schema, request, a statement's result) with the reference's warning of leaving it out: the approved entry takes it out and nothing else differs; proposed, backed by no owner-approved row, for another case, or naming the column in another table, nothing is taken out; a second kept column the entry does not name still fails (the negative control).
- T2.18's twelve `name.*` cases other than the filter role, observed on lane e (python-core on DuckDB, SQLite and Postgres, rust-host on DuckDB and SQLite), with P2E-1 naming the six columns: no gated difference on any database; without the entry, 51, 51 and 33.
- Round 2 (Codex, P2): taking the left-out warning out of python-old's warnings left its recorded origin, so HOST-5's runtime-deprecation narrowing, which reads the two lists aligned, refused the pair; the origin is now taken out at the same place, origins that never matched left as they are. The focused test composes the two in `floor-pandas2.2`, with a client-raised deprecation as its negative control.
