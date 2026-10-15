# H11 partial counts (rust-host leg missing; owner to rule)

rust-host: runner failure on h11.key.composite-primary-key: core.describe_source.source is not a SourceSpec; no observations.
Candidate 0a8a74125d8372b9bc71363239322a785430cc5c; 42 held-out cases; register snapshot register-at-run-start.yaml (4 owner-pending entries open: P1B-80, P1B-86, P1B-87, P1B-89).

- gating DIFFERS lines from compare.py: 0 (compare: no differences; held-out cases are not baselined, so outcome differences are reported as diagnostic, counted below)
- python-core panics / INTERNAL_ERROR / missing: 0 []
- python-core new refusals of an input the reference accepts: 0 []
- python-core outcome class differs from the reference: 6
  - h11.decimal.key-38-0-large: reference `refusal INTERNAL_ERROR`, python-core `success`
  - h11.empty.entity-table: reference `refusal RuntimeError`, python-core `refusal INVALID_REQUEST`
  - h11.key.composite-primary-key: reference `refusal INVALID_REQUEST`, python-core `refusal UNSUPPORTED_REQUEST`
  - h11.nested.list-of-list: reference `refusal ValueError`, python-core `refusal UNSUPPORTED_REQUEST`
  - h11.nested.struct-on-fact-table: reference `refusal RuntimeError`, python-core `refusal UNSUPPORTED_REQUEST`
  - h11.null.nan-and-infinity-target: reference `refusal INVALID_REQUEST`, python-core `refusal UNSUPPORTED_REQUEST`
- python-old outcome class differs from the reference: 0 []
- cases with only non-outcome diagnostics (statement text/bindings/count, route, core, warnings, inputs): 36
  - h11.decimal.binary-threshold-10-2: sql.bindings, sql.text
  - h11.decimal.feature-18-18-null-heavy: sql.bindings, sql.text
  - h11.decimal.target-38-37: sql.bindings, sql.text
  - h11.decimal.target-sum-past-precision: inputs, route, sql.bindings, sql.text
  - h11.edit.clear-time-column-window: inputs, outcome, route
  - h11.edit.drop-target-column: inputs, outcome, route
  - h11.edit.set-second-time-column: sql.bindings, sql.text
  - h11.edit.stype-categorical-int-target: sql.bindings, sql.text
  - h11.empty.single-entity: sql.bindings, sql.text
  - h11.key.duplicate-primary-values: inputs, outcome, route
  - h11.key.explicit-tables-and-link: inputs, outcome, route
  - h11.key.int64-near-max: sql.bindings, sql.text
  - h11.key.negative-multiclass: sql.bindings, sql.text
  - h11.key.uint32-binary: inputs, sql.bindings, sql.text
  - h11.key.varchar-leading-zeros: sql.bindings, sql.text
  - h11.key.varchar-unicode-binary: sql.bindings, sql.text
  - h11.key.width-mismatch: inputs, sql.bindings, sql.text
  - h11.nested.json-text: sql.bindings, sql.text
  - h11.nested.list-on-fact-table: sql.bindings, sql.text
  - h11.nested.list-varchar-null-elements: sql.bindings, sql.text
  - h11.null.all-foreign-keys: sql.bindings, sql.text
  - h11.null.target-mostly-null-binary: sql.bindings, sql.text
  - h11.null.time-column-all-null: inputs, outcome, route
  - h11.option.inference-config: sql.bindings, sql.text
  - h11.task.binary-count-equals: sql.bindings, sql.text
  - h11.task.forecast-sum: inputs, payload, sql.bindings, sql.text
  - h11.task.multiclass-text-with-null-target: sql.bindings, sql.text
  - h11.task.ranking-without-candidates: inputs, outcome, route
  - h11.time.anchor-leap-day-binary: sql.bindings, sql.text
  - h11.time.date-month-ends: sql.bindings, sql.text
  - h11.time.dst-fall-back-binary: sql.bindings, sql.text
  - h11.time.dst-spring-forward-hours: sql.bindings, sql.text
  - h11.time.month-window-from-month-end: sql.bindings, sql.text
  - h11.time.pre-epoch: sql.bindings, sql.text
  - h11.time.text-iso-offset: sql.bindings, sql.text
  - h11.time.text-iso-zulu-fraction: sql.bindings, sql.text

H11 threshold (E66(e)): any panic, any new refusal, or more than 4 cases differing from the reference without an entry approved before the run.
On the python legs: 0 panics, 0 new refusals; 6 outcome-differing cases, each to be checked against register-at-run-start.yaml (reviewer). NOT a verdict: the rust-host leg is missing.
