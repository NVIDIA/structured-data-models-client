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
The Python runner selects the requested route for every case, including consecutive `python-old` cases.
No recorded expectation, observation, capture, comparison threshold or generated golden is rewritten.

## Locked acceptance

Half `3a` retains its original 309 acceptance cases and all existing steps and checks.
It appends the 77 S2 acceptance cases landed at the views phase head, bringing the locked total to 386.
This is the one planned T3.10 inventory extension, not a reduction of the existing acceptance floor.

## Validation status

Tests are written and implementation is ready for two independent reviews.
No package checks, full CI, warehouse runs, matrix jobs, timing campaign or held-out run are claimed here.
Package checks follow both approvals on the final integrated SHA.
The orchestrator owns final phase CI and live evidence on the frozen phase head.
This code switch does not mark Phase 3 complete and does not merge `rust-core`.
