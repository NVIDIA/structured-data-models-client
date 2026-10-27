# T3.10: core defaults and phase-end acceptance

The code switch implements owner decisions KO3-36 and KO3-37.
Covered S0 validation and S1/S2 source description, materialization, prediction and capability entrypoints default to the core together.
Schema edits, inspection and state held by a core-described schema continue through its existing core representation.
Unimplemented source and training paths retain their existing route.
An absent `SDM_CORE_ROUTES` uses these defaults.
An explicitly empty value selects legacy code, and an explicit path list selects only those core entrypoints for comparisons.
Unknown, unsupported and partially selected calls still fail without fallback.

## Tests and legacy controls

The tests-first change adds public DuckDB and SQLite describe/edit/materialize/predict calls in clean child processes with no routing override and both old packages unimportable.
Routing tests cover the defaults, explicit legacy selection, runtime override changes and missing-native failure on the default route.
Existing source and semantic-model routing expectations now require core defaults.
The older comparison helpers select their legacy side explicitly, preserving their existing assertions and native-side runs.
The client suite has no new global legacy override.

The following existing comparison modules retain explicit legacy controls:

- `test_the_routing_seam.py`, `test_the_schema_route.py`, `test_the_schema_edits.py`, `test_the_materialize_route.py` and `test_the_predict_route.py`.
- `test_t316_the_schema_as_it_stands.py`, `test_t316_the_schema_as_plain_data.py`, `test_t132h_keys_and_times.py`, `test_the_date_literal_range.py` and `test_the_query_long_tail.py`.
- `test_databricks_from_source.py`, `test_snowflake_from_source.py` and `test_t38_forced_routes.py`, including its pre-switch-schema recorder.
- The contract's refusal, entity-reference, forecast, pickle, old-request and legacy sampling-spy comparisons.

The relational engine's own test suite selects the legacy route through its package-local fixture until Phase 5, as KO3-37 requires.
Legacy recorder and benchmark branches also select legacy code explicitly, while preserving the caller's saved environment on exit where they previously did so.
The smoke, null-link benchmark, real-data benchmark and key-rule probe use an explicitly empty selection for their old controls.
The sampled-null recorder selects the requested route for every vector and restores the caller's setting after success or failure.
The nested-schema and whole-table-time probes explicitly select legacy in their reference subprocesses too.
Focused regressions audit the actual smoke's describe/materialize/predict calls, the other helpers' public source boundary and the recorder's consecutive vectors and environment restoration.
The Python runner selects the requested route for every case, including consecutive `python-old` cases.
No recorded expectation, observation, capture, comparison threshold or generated golden is rewritten.

## Locked acceptance

Half `3a` retains its original 309 acceptance cases and all existing steps and checks.
It appends all 79 landed S2 acceptance cases, including the final Java Databricks cases, bringing the locked total to 388.
This is the one planned T3.10 inventory extension, not a reduction of the existing acceptance floor.

## Validation status

T3.10 remains open for final acceptance; its default switch and preceding repair batches have landed on the phase branch.
The final-source CI run on `ef10270ec4f1d9c096678874148675e4fa448275` exposed dependency-lock freshness, stale isolated fixtures and an exact replay-plan mismatch.
The current bounded repair batch regenerates the dependency lock and synthetic replay through their writers, restores historical scope-fixture bytes without changing the guard, and preserves decision rows and capture-check imports in isolated fixtures.
The replay mismatch is the existing PostgreSQL UTC result-conversion metadata missing from the fixture; strict request and plan matching stays unchanged.

The remaining checkpoints are:

1. Obtain both independent code approvals, confirm the known failures and affected fixture consumers, and complete the required package handoff.
2. Land the repairs and obtain green full CI and artifacts on the final frozen phase head.
3. Complete remaining reference recordings and final-core warehouse, Java, model-matrix, original-notebook, full-data and performance comparisons.
4. Run sealed held-out validation last, complete both whole-phase reviews and the locked exit checks, then merge and push to `rust-core` under the owner's explicit instruction.

Private run receipts identify completed evidence and preserve failed attempts; preparation, recordings and a package stamp alone do not establish phase acceptance.
No final CI, complete live campaign, performance or held-out pass is claimed by this tracker update.
Phase 4 is rebased onto the resulting `rust-core` after the Phase 3 merge.
