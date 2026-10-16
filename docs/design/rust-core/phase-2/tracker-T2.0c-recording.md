# Phase 2 tracker: T2.0c's re-recording

The reference re-recorded at `6298913c`, and every recorded line traced to the `main` commit that moved it.
The merger takes these commits as they are; the register entries and the baselines are the merger's.

## How it was recorded

The reference checkout `~/code/sdmc-reference` moved from `b413e9ad` to `origin/rust-core-reference` `6298913c` by `git merge --ff-only`.
Its environment was rebuilt by `phase-0/reference-environment.md`'s commands on Python 3.12.13, with every distribution pinned to the previous environment's freeze; the old environment was kept, renamed `.venv-b413e9ad`.
The new freeze is identical to the old one, so the environment digest stays `85ec83f2`; only the platform (`linux-x86_64`, from `darwin-arm64`) and the reference commit change in `contract/reference-env.txt`.
The reference guard passed before the move, after the rebuild and after recording (24 names, and its three `--self-test` controls each time).
Every recorder ran offline, against the reference's scripted local endpoint.

Each change below was traced by recording again with one `main` commit's source changes (`packages/*/src`) reverted on a copy of `6298913c`: `26ecdd28` (inference config only when set), `eb6c68eb` (metric views, decimals read as their data type), `1fc5a686` (multi-call gathering).
The cause column names each commit whose revert brings the item back; "identity only" means no revert moves it.
One change is not explained by the three reverts: `key-cast-vectors.json`'s `int64.float.2pow63`, whose public answer stays `INVALID_REQUEST` but whose observed cast of the float 2**63 to `int64[pyarrow]` was 9223372036854775807 on the macOS arm64 recording and raises `ArrowInvalid` here, with or without any revert.
It is most likely the processor's float-to-integer conversion (arm64 against x86_64), but that was not verified on an arm64 machine.

## Summary

| Cause | Reference lines | Recorded vectors and cases |
|---|---|---|
| 26ecdd28 only | 27 | 328 |
| eb6c68eb (with 26ecdd28 in the same item's payload where it has one) | 4 | 271 |
| 1fc5a686 | 0 | 3 |
| identity only (platform, zone, ports, recorded commit) | 45 | 8, and every file's reference field |
| not explained by the three reverts | 0 | 1 (`int64.float.2pow63`'s cast, see below) |
| not the reference (the core's own side) | 0 | 17 |

The commit message of `57ab2ef7` says 41 lines change in identity only; the count is 45 (76 lines, 31 of them moved).

## The vector the reference does not finish

On `6298913c` the reference does not finish `contract/key-cast-vectors.json`'s vector `decimal128-18-0.exponent.huge.decimal`: `eb6c68eb` gives the entity column the key's integer type, and pandas converts `Decimal("1.00E+2147483649")` to an integer in `_box_pa_array`.
At `b413e9ad`'s cast the same entity is refused `INVALID_REQUEST` in about 2 s.
Measured alone on `6298913c`: `1E+200000` takes about 7 s, `1E+2000000` 302 s, and `1E+20000000` did not finish in 850 s, so the time grows about as the square of the exponent.

Under the owner's approval of 2026-10-16 for a reference timeout outcome, `key_cast_record.py` now observes each vector in a worker process with `TIME_LIMIT_SECONDS = 300` to answer.
A vector not answered in time is recorded as `{"outcome": "unfinished", "within_seconds": 300}`, with no cast, its register mark says why, and the worker is started again.
300 s is more than 30 times the slowest vector that does finish: 5.4 s in the recording (`int8.int.present`, the first vector, which pays for the imports) and 9.1 s in the slowest of the three probe recordings, run side by side (`decimal128-18-0.exponent.tiny.decimal`).
Exactly one vector got the outcome: `decimal128-18-0.exponent.huge.decimal`; with `eb6c68eb` reverted none does.
`contract/tests/test_t132a_key_cast.py`'s `test_the_vectors_are_not_vacuous` now admits that outcome for that vector alone, and asks the reference's cast of every vector that finished.

## Other findings for the merger

- `contract/tests/test_t132b_decimal_probe.py`'s `REFERENCE` (pasted from `decimal_probe.py observe python-ref` at `b413e9ad`) still matches the reference at `6298913c` except its payload's digest, which `26ecdd28` moves (`a93397…` to `6736f0…`, confirmed with `26ecdd28` reverted).
- The case files of `key.decimal-18-0`, `key.decimal-38-10`, `key.hugeint` and `decimal.class-target-38-18` expect the old `INTERNAL_ERROR`, and their `register` fields name it; `tests/test_contract_cases.py` fails on them until the merger updates the cases (which changes their digests, so their reference lines are then recorded again at the case files).
- `contract/t136-host2-vectors.json` and `contract/t136-host2-zone-vectors.json` record both routes in one run; the reference side does not move, and the core side was measured with this tree's core at `292f8d8a`.
  Their environments named each route's `sdm_client` by an absolute path, which put a home directory (a username) into the repository; `host2_values_record.py` now names it relative to its checkout or from its `site-packages`, and both files were recorded again.
  The zone file's with-pytz answers (P1B-39, P1B-89) were recorded again too, by the reference's code from `git archive 6298913c` with pytz 2026.5 on the reference's pins; none moves.
  No file these commits change holds a home directory path.
- `contract/type-set-vectors.json` is recorded under `TZ=America/Los_Angeles` (the orchestrator's decision), since on this machine's UTC zone DuckDB hands a `TIMESTAMPTZ` over as `timestamp[us, tz=Etc/UTC]`, a form no type-set row has, and `test_t133_type_set.py`'s `test_every_form_a_session_setting_gives_has_a_row` failed.
  `contract/time-inference-vectors.json` stays recorded in this machine's zone: its four `timestamptz` vectors' DuckDB minimum reads in UTC, and its tests pass.
- `key-inference`'s `nulls-1-decimal.200` and `sampled-null-key`'s `mixed-null-decimal` were the decimal refusals that `test_t132e_key_inference.py` and `test_t132e_sampled_null_keys.py` rest on (`test_the_vectors_are_not_vacuous`, `test_option_b_s_vectors_are_the_reference_s_refusals`, `test_the_review_s_reproduction_is_a_success_with_a_request`); those tests fail on the new recordings.
- `rust/crates/sdm-pql/tests/dates.json` stays the macOS recording at `b413e9ad`.
  Recorded here it is the other side of the `%Y` padding (`pads_year` false) and loses the 690 `plain` rows and 144 format guesses that only a padded year gives, which two `dates.rs` tests are built on.
  The reference's date reading does not move with `main`: the recording at `6298913c` equals the recordings with each of the three commits reverted.
  A macOS run at `6298913c` would bring its source line to the new reference.
- `t128b`'s `--cell` recordings ran the reference's code from `git archive 6298913c` on Python 3.10 with pandas 2.2.2, pyarrow 14.0.1, NumPy 1.26.4 (`main`'s pandas floor, as the brief takes it) and on Python 3.11 with pandas 3.0.6; they gave the same `dtypes_pandas2` and `by_python` entries as before.
- `tests/golden/t132g-*/python-ref.jsonl` are not re-recorded (ratchet fixtures, not held to `reference-env.txt`).

## Tests on this branch

Run with this tree's core (`292f8d8a`'s, built here) against the new recordings.
Every failure seen is the core or the old route answering as the old reference did, a case file or register entry naming the old outcome, or a test resting on a decimal refusal the reference no longer makes; each is the merger's (the `26ecdd28` port, register and bridge entries, case updates).
- `tests/test_contract_reference.py`: 55 passed.
- Client recordings (`test_labels_match_the_reference.py`, `test_t128a_payload_rules.py`, `test_t128b_predict_result.py`, `test_schema_pickles.py`): 246 passed, 6 failed (decimal labels, the two inference-config payloads, the two 1,001-entity cases).
- `contract/tests` for the re-recorded vectors: the failures are `python-core` and `python-old` payloads (`26ecdd28`), the decimal outcomes `eb6c68eb` moved, and `test_t132a_key_cast.py`'s `test_every_vector_ref_names_its_transition` (a register entry's transition, P1B-33's among them, and the new unfinished outcome).
- `test_t136_host2_values.py`, `test_t136_host2_zones.py`, `test_t133_type_set.py`: all passed.
- `cargo test -p sdm-pql -p sdm-core -p sdm-testkit`: `dates` passes on the restored file; the `sdm-testkit` failures are `rust-host` against the moved decimal, payload and 1,001-entity answers.

## Trace

### `contract/reference/6298913c….jsonl` (python-ref, 76 lines)

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `s0.accept.binary-classification` | success materialize | identity (platform, commits) | identity only |
| `s0.accept.multiclass-classification` | success materialize | identity (platform, commits) | identity only |
| `s0.accept.sum-window` | success materialize | identity (platform, commits) | identity only |
| `s0.boundary.date-long-field` | success materialize | identity (platform, commits) | identity only |
| `s0.boundary.date-packed` | success materialize | identity (platform, commits) | identity only |
| `s0.boundary.deepest` | success materialize | identity (platform, commits) | identity only |
| `s0.boundary.depth-999` | refusal materialize | identity (platform, commits) | identity only |
| `s0.boundary.digits-4300` | success materialize | identity (platform, commits) | identity only |
| `s0.boundary.digits-4301` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.boundary.escape-lone-surrogate` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.boundary.escape-surrogate-pair` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.boundary.escape-truncated` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.boundary.escapes` | success materialize | identity (platform, commits) | identity only |
| `s0.boundary.start-past-64kb` | success materialize | identity (platform, commits) | identity only |
| `s0.refuse.date-impossible` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.join-event-filter` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.model-implicit-join` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.parse-syntax` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.target-foreign-key` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.task-forecast-zero` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.time-past-window` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.type-constant` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.unknown-column` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `s0.refuse.value-int64` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `base.adventureworks` | success predict | public.payloads.parsed.inference.inference_config.category_shuffle, public.payloads.parsed.inference.inference_config.hop_shuffle, public.payloads.parsed.inference.inference_config.kind | 26ecdd28 |
| `base.binary` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `base.multiclass` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `base.regression` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `decimal.class-target-38-18` | refusal materialize INTERNAL_ERROR → success predict | outcome, driver, public | eb6c68eb (outcome); 26ecdd28 (payload) |
| `decimal.feature-38-0-wide` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `decimal.target-38-18` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `empty.entity-without-facts` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `empty.fact-table` | refusal from_source | identity (platform, commits) | identity only |
| `empty.indices` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `feature.bool` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `key.decimal-18-0` | refusal materialize INTERNAL_ERROR → success predict | outcome, driver, public | eb6c68eb (outcome); 26ecdd28 (payload) |
| `key.decimal-38-10` | refusal materialize INTERNAL_ERROR → success predict | outcome, driver, public | eb6c68eb (outcome); 26ecdd28 (payload) |
| `key.declared-int32` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `key.hugeint` | refusal materialize INTERNAL_ERROR → success predict | outcome, driver, public | eb6c68eb (outcome); 26ecdd28 (payload) |
| `key.uint64-above-int64-max` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `key.uint64-small` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `key.uint8` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `key.varchar-large-buffers` | refusal materialize INTERNAL_ERROR | identity (platform, commits) | identity only |
| `key.varchar` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `nested.fixed-array` | refusal from_source | identity (platform, commits) | identity only |
| `nested.list-int` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `nested.map` | refusal from_source | identity (platform, commits) | identity only |
| `nested.struct` | refusal from_source | identity (platform, commits) | identity only |
| `nested.union` | refusal from_source | identity (platform, commits) | identity only |
| `null.feature-all-null` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `null.some-foreign-keys` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `null.some-times` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `null.target-all-null` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `refuse.unknown-column` | refusal materialize INVALID_REQUEST | identity (platform, commits) | identity only |
| `time.date` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `time.feature-date-and-timestamp-with-null` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `time.feature-interval` | refusal from_source | identity (platform, commits) | identity only |
| `time.feature-time-of-day` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `time.text-declared` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `time.text-iso-date` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `time.text-one-bad-value` | refusal materialize INTERNAL_ERROR | identity (platform, commits) | identity only |
| `time.text-us-date` | refusal materialize INTERNAL_ERROR | identity (platform, commits) | identity only |
| `time.timestamp-ns` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `time.timestamp-s` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `time.timestamptz` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `edit.accepted-meaning-key-links` | success materialize | identity (platform, commits) | identity only |
| `edit.key-missing-column` | refusal from_source INVALID_REQUEST | identity (platform, commits) | identity only |
| `edit.key-time-column` | refusal from_source INVALID_REQUEST | identity (platform, commits) | identity only |
| `edit.leave-out-the-key` | refusal from_source INVALID_REQUEST | identity (platform, commits) | identity only |
| `edit.link-not-the-key` | refusal from_source INVALID_REQUEST | identity (platform, commits) | identity only |
| `edit.meaning-not-carried` | refusal from_source INVALID_REQUEST | identity (platform, commits) | identity only |
| `edit.meaning-unknown` | refusal from_source INVALID_REQUEST | identity (platform, commits) | identity only |
| `edit.time-column-not-a-time` | refusal from_source INVALID_REQUEST | identity (platform, commits) | identity only |
| `edit.time-column-text` | success materialize | identity (platform, commits) | identity only |
| `base.adventureworks.churn` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |
| `base.adventureworks.color` | success predict | public.payloads.parsed.inference.inference_config | 26ecdd28 |

### `packages/structured-data-models-client/tests/labels_reference.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `sum-decimal` | labels | expected.rows | eb6c68eb |
| `avg-decimal` | labels | expected.rows | eb6c68eb |
| `multiclass-static-decimal` | refused INTERNAL_ERROR → labels | expected.code, expected.rows, expected.class | eb6c68eb |

### `packages/structured-data-models-client/tests/t128a-payload-rules.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `base.regression.entities-1001` | refused UNSUPPORTED_REQUEST → refused INVALID_REQUEST | expected.code | 1fc5a686 |
| `many-1001.entities-1001` | refused UNSUPPORTED_REQUEST → payload | expected.mro, expected.code, expected.class | 1fc5a686 |
| `base.regression.inference-config` | payload | expected.call.inference.inference_config.kind, expected.call.inference.inference_config.hop_shuffle, expected.call.inference.inference_config.category_shuffle | 26ecdd28 |
| `base.binary.inference-config` | payload | expected.call.inference.inference_config.kind, expected.call.inference.inference_config.hop_shuffle, expected.call.inference.inference_config.class_shuffle | 26ecdd28 |

### `packages/structured-data-models-client/tests/t128b-predict-result.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `size.one-past-the-limit` | refused materialize INVALID_REQUEST | expected.message | 1fc5a686 |

### `contract/response-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `boundary.provenance.local-not-json` | – | reference.message | identity only (the local endpoint's port, new each run) |
| `boundary.provenance.local-array` | – | reference.message | identity only (the local endpoint's port, new each run) |
| `boundary.provenance.local-truncated` | – | reference.message | identity only (the local endpoint's port, new each run) |
| `boundary.provenance.local-null` | – | reference.message | identity only (the local endpoint's port, new each run) |

### `contract/decimal-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `(cases)` | – | digest, decimal.class-target-38-18.result, decimal.class-target-38-18.payload | eb6c68eb, through the committed reference line it projects |
| `feature.users.d4-0.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d4-2.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d4-4.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d9-0.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d9-2.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d9-9.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d18-0.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d18-2.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d18-10.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d18-18.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d19-0.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d19-2.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d19-10.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d19-19.spread` | success predict | digest, reference.columns.users.f.request.predict.cells, reference.columns.users.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d38-0.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d38-2.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d38-10.spread` | success predict | digest | 26ecdd28 |
| `feature.users.d38-38.spread` | success predict | digest, reference.columns.users.f.request.predict.cells, reference.columns.users.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d4-0.max` | success predict | digest | 26ecdd28 |
| `feature.users.d4-2.max` | success predict | digest, reference.columns.users.f.request.predict.cells, reference.columns.users.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d4-4.max` | success predict | digest | 26ecdd28 |
| `feature.users.d9-0.max` | success predict | digest | 26ecdd28 |
| `feature.users.d9-2.max` | success predict | digest | 26ecdd28 |
| `feature.users.d9-9.max` | success predict | digest, reference.columns.users.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d18-0.max` | success predict | digest | 26ecdd28 |
| `feature.users.d18-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d18-10.max` | success predict | digest | 26ecdd28 |
| `feature.users.d18-18.max` | success predict | digest | 26ecdd28 |
| `feature.users.d19-0.max` | success predict | digest | 26ecdd28 |
| `feature.users.d19-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d19-10.max` | success predict | digest | 26ecdd28 |
| `feature.users.d19-19.max` | success predict | digest | 26ecdd28 |
| `feature.users.d38-0.max` | success predict | digest | 26ecdd28 |
| `feature.users.d38-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d38-10.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d38-38.max` | success predict | digest, reference.columns.users.f.request.context.cells, reference.columns.users.f.request.predict.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d4-0.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d4-2.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d4-4.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d9-0.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d9-2.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d9-9.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d18-0.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d18-2.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d18-10.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d18-18.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d19-0.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d19-2.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d19-10.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d19-19.spread` | success predict | digest, reference.columns.orders.f.request.predict.cells, reference.columns.orders.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d38-0.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d38-2.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d38-10.spread` | success predict | digest | 26ecdd28 |
| `feature.orders.d38-38.spread` | success predict | digest, reference.columns.orders.f.request.predict.cells, reference.columns.orders.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d4-0.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d4-2.max` | success predict | digest, reference.columns.orders.f.request.predict.cells, reference.columns.orders.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d4-4.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d9-0.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d9-2.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d9-9.max` | success predict | digest, reference.columns.orders.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d18-0.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d18-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d18-10.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d18-18.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d19-0.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d19-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d19-10.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d19-19.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d38-0.max` | success predict | digest | 26ecdd28 |
| `feature.orders.d38-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d38-10.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `feature.orders.d38-38.max` | success predict | digest, reference.columns.orders.f.request.predict.cells, reference.columns.orders.f.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d4-0.zero` | success predict | digest | 26ecdd28 |
| `feature.users.d4-0.negative` | success predict | digest | 26ecdd28 |
| `feature.users.d4-0.null` | success predict | digest | 26ecdd28 |
| `feature.users.d9-2.zero` | success predict | digest | 26ecdd28 |
| `feature.users.d9-2.negative` | success predict | digest | 26ecdd28 |
| `feature.users.d9-2.null` | success predict | digest | 26ecdd28 |
| `feature.users.d19-10.zero` | success predict | digest | 26ecdd28 |
| `feature.users.d19-10.negative` | success predict | digest | 26ecdd28 |
| `feature.users.d19-10.null` | success predict | digest | 26ecdd28 |
| `feature.users.d38-0.zero` | success predict | digest | 26ecdd28 |
| `feature.users.d38-0.negative` | success predict | digest | 26ecdd28 |
| `feature.users.d38-0.null` | success predict | digest | 26ecdd28 |
| `feature.users.d38-38.zero` | success predict | digest | 26ecdd28 |
| `feature.users.d38-38.negative` | success predict | digest, reference.columns.users.f.request.context.cells, reference.columns.users.f.request.predict.cells | eb6c68eb (payload digest also 26ecdd28) |
| `feature.users.d38-38.null` | success predict | digest | 26ecdd28 |
| `wide.hugeint-max` | success predict | digest | 26ecdd28 |
| `wide.hugeint-min` | success predict | digest | 26ecdd28 |
| `wide.hugeint-39-digits` | success predict | digest | 26ecdd28 |
| `wide.uhugeint-max` | success predict | digest | 26ecdd28 |
| `wide.uhugeint-past-hugeint` | success predict | digest | 26ecdd28 |
| `wide.hugeint-small` | success predict | digest | 26ecdd28 |
| `target.sum.d4-0.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d4-2.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d4-4.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d9-0.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d9-2.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d9-9.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d18-0.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d18-2.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d18-10.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d18-18.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d19-0.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d19-2.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d19-10.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d19-19.spread` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d38-0.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d38-2.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d38-10.spread` | success predict | digest | 26ecdd28 |
| `target.sum.d38-38.spread` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d4-0.max` | success predict | digest | 26ecdd28 |
| `target.sum.d4-2.max` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d4-4.max` | success predict | digest | 26ecdd28 |
| `target.sum.d9-0.max` | success predict | digest | 26ecdd28 |
| `target.sum.d9-2.max` | success predict | digest | 26ecdd28 |
| `target.sum.d9-9.max` | success predict | digest, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d18-0.max` | success predict | digest | 26ecdd28 |
| `target.sum.d18-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d18-10.max` | success predict | digest | 26ecdd28 |
| `target.sum.d18-18.max` | success predict | digest | 26ecdd28 |
| `target.sum.d19-0.max` | success predict | digest | 26ecdd28 |
| `target.sum.d19-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d19-10.max` | success predict | digest | 26ecdd28 |
| `target.sum.d19-19.max` | success predict | digest | 26ecdd28 |
| `target.sum.d38-0.max` | success predict | digest | 26ecdd28 |
| `target.sum.d38-2.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d38-10.max` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `target.sum.d38-38.max` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.avg.d4-2.spread` | success predict | digest | 26ecdd28 |
| `target.avg.d18-0.spread` | success predict | digest | 26ecdd28 |
| `target.avg.d38-10.spread` | success predict | digest | 26ecdd28 |
| `target.avg.d38-38.spread` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.min.d4-2.spread` | success predict | digest | 26ecdd28 |
| `target.min.d18-0.spread` | success predict | digest | 26ecdd28 |
| `target.min.d38-10.spread` | success predict | digest | 26ecdd28 |
| `target.min.d38-38.spread` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.max.d4-2.spread` | success predict | digest | 26ecdd28 |
| `target.max.d18-0.spread` | success predict | digest | 26ecdd28 |
| `target.max.d38-10.spread` | success predict | digest | 26ecdd28 |
| `target.max.d38-38.spread` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `target.count.d4-2.spread` | success predict | digest | 26ecdd28 |
| `target.count.d18-0.spread` | success predict | digest | 26ecdd28 |
| `target.count.d38-10.spread` | success predict | digest | 26ecdd28 |
| `target.count.d38-38.spread` | success predict | digest, reference.columns.orders.price.request.predict.cells, reference.columns.orders.price.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d4-0.sum-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d4-0.sum-at` | success predict | digest | 26ecdd28 |
| `overflow.d4-0.sum-past-negative` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d4-0.sum-cancels` | success predict | digest | 26ecdd28 |
| `overflow.d4-0.sum-far-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d4-2.sum-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d4-2.sum-at` | success predict | digest, reference.columns.instance_table.TARGET.request.context.cells | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d9-0.sum-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d18-0.sum-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d18-2.sum-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d38-0.sum-at` | success predict | digest | 26ecdd28 |
| `overflow.d38-38.sum-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `overflow.d4-0.avg-past` | success predict | digest | 26ecdd28 |
| `overflow.d4-0.max-past` | success predict | digest | 26ecdd28 |
| `overflow.d4-0.binary-past` | refusal materialize INVALID_REQUEST → success predict | digest, reference.result, reference.payload | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-three.d4-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-three.d4-2` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-three.d9-9` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-three.d18-2` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-three.d38-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-three.d38-10` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-many.d4-2` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-many.d18-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.static-many.d38-10` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `class.binary-sum.d4-2` | success predict | digest | 26ecdd28 |
| `class.binary-sum.d18-0` | success predict | digest | 26ecdd28 |
| `class.binary-sum.d38-10` | success predict | digest | 26ecdd28 |
| `class.binary-sum-fraction.d4-2` | success predict | digest | 26ecdd28 |
| `class.binary-static.d4-2` | success predict | digest | 26ecdd28 |
| `class.binary-static.d38-0` | success predict | digest | 26ecdd28 |
| `key.both.d4-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.both.d9-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.both.d19-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.both.d38-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.both.d9-2` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.both.d18-2` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.foreign-only.d4-0` | success predict | digest | 26ecdd28 |
| `key.primary-only.d4-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.foreign-only.d18-0` | success predict | digest | 26ecdd28 |
| `key.primary-only.d18-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.fact-key.d18-0` | success predict | digest | 26ecdd28 |
| `key.fact-key.d38-0` | success predict | digest | 26ecdd28 |
| `key.negative.d9-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.at-precision.d18-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.at-precision.d19-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.at-precision.d38-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `key.null-foreign.d18-0` | refusal materialize INTERNAL_ERROR → success predict | digest, register.why, reference.result | eb6c68eb (payload digest also 26ecdd28) |
| `time.epoch-feature` | success predict | digest | 26ecdd28 |
| `time.epoch-ms-feature` | success predict | digest | 26ecdd28 |
| `time.yyyymmdd-feature` | success predict | digest | 26ecdd28 |

### `contract/empty-null-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `empty.third-table-not-named` | success predict | digest | 26ecdd28 |
| `empty.view` | success predict | digest | 26ecdd28 |
| `edge.one-order` | success predict | digest | 26ecdd28 |
| `edge.one-user` | success predict | digest | 26ecdd28 |
| `edge.one-user-of-two-with-orders` | success predict | digest | 26ecdd28 |
| `edge.one-entity` | success predict | digest | 26ecdd28 |
| `edge.every-entity` | success predict | digest | 26ecdd28 |
| `edge.entity-without-facts` | success predict | digest | 26ecdd28 |
| `edge.entities-without-facts` | success predict | digest | 26ecdd28 |
| `null.boolean` | success predict | digest | 26ecdd28 |
| `null.tinyint` | success predict | digest | 26ecdd28 |
| `null.smallint` | success predict | digest | 26ecdd28 |
| `null.integer` | success predict | digest | 26ecdd28 |
| `null.bigint` | success predict | digest | 26ecdd28 |
| `null.hugeint` | success predict | digest | 26ecdd28 |
| `null.utinyint` | success predict | digest | 26ecdd28 |
| `null.usmallint` | success predict | digest | 26ecdd28 |
| `null.uinteger` | success predict | digest | 26ecdd28 |
| `null.ubigint` | success predict | digest | 26ecdd28 |
| `null.float` | success predict | digest | 26ecdd28 |
| `null.double` | success predict | digest | 26ecdd28 |
| `null.decimal-18-2` | success predict | digest | 26ecdd28 |
| `null.decimal-38-10` | success predict | digest | 26ecdd28 |
| `null.varchar` | success predict | digest | 26ecdd28 |
| `null.uuid` | success predict | digest | 26ecdd28 |
| `null.date` | success predict | digest | 26ecdd28 |
| `null.time` | success predict | digest | 26ecdd28 |
| `null.timestamp` | success predict | digest | 26ecdd28 |
| `null.timestamp-s` | success predict | digest | 26ecdd28 |
| `null.timestamp-ms` | success predict | digest | 26ecdd28 |
| `null.timestamp-ns` | success predict | digest | 26ecdd28 |
| `null.timestamptz` | success predict | digest | 26ecdd28 |
| `null.blob` | success predict | digest | 26ecdd28 |
| `null.list-integer` | success predict | digest | 26ecdd28 |
| `null.orders-boolean` | success predict | digest | 26ecdd28 |
| `null.orders-integer` | success predict | digest | 26ecdd28 |
| `null.orders-double` | success predict | digest | 26ecdd28 |
| `null.orders-varchar` | success predict | digest | 26ecdd28 |
| `null.orders-timestamp` | success predict | digest | 26ecdd28 |
| `null.untyped` | success predict | digest | 26ecdd28 |
| `null.all-but-one-integer` | success predict | digest | 26ecdd28 |
| `null.all-but-one-varchar` | success predict | digest | 26ecdd28 |
| `null.every-feature` | success predict | digest | 26ecdd28 |
| `null.key-fact-some` | success predict | digest | 26ecdd28 |
| `null.fk-some` | success predict | digest | 26ecdd28 |
| `null.fk-most` | success predict | digest | 26ecdd28 |
| `null.fk-all` | success predict | digest | 26ecdd28 |
| `null.fk-of-an-asked-entity` | success predict | digest | 26ecdd28 |
| `null.fk-in-the-window` | success predict | digest | 26ecdd28 |
| `null.time-some` | success predict | digest | 26ecdd28 |
| `null.time-of-an-asked-entity` | success predict | digest | 26ecdd28 |
| `null.time-in-the-window` | success predict | digest | 26ecdd28 |
| `null.time-date` | success predict | digest | 26ecdd28 |
| `null.time-timestamptz` | success predict | digest | 26ecdd28 |
| `null.time-named` | success predict | digest | 26ecdd28 |
| `null.target-all` | success predict | digest | 26ecdd28 |
| `null.target-some` | success predict | digest | 26ecdd28 |
| `null.target-of-an-asked-entity` | success predict | digest | 26ecdd28 |
| `null.target-all-count` | success predict | digest | 26ecdd28 |
| `null.target-new-column` | success predict | digest | 26ecdd28 |
| `edge.every-entity-classification` | success predict | digest | 26ecdd28 |

### `contract/given-vectors.json`

No item changes; only the recorded reference commit moves (identity only).

### `contract/float-multiclass-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `double.one-to-five` | success predict | digest | 26ecdd28 |
| `double.tenths` | success predict | digest | 26ecdd28 |
| `float.one-to-five` | success predict | digest | 26ecdd28 |
| `float.tenths` | success predict | digest | 26ecdd28 |
| `float.null-in-context` | success predict | digest | 26ecdd28 |
| `double.small-spellings` | success predict | digest | 26ecdd28 |
| `double.large-spellings` | success predict | digest | 26ecdd28 |
| `double.one-float32` | success predict | digest | 26ecdd28 |
| `double.float32-limits` | success predict | digest | 26ecdd28 |
| `double.null-in-context` | success predict | digest | 26ecdd28 |
| `double.null-asked` | success predict | digest | 26ecdd28 |
| `double.negative-zero` | success predict | digest | 26ecdd28 |
| `float.negative-zero` | success predict | digest | 26ecdd28 |
| `preview.double-nan` | success predict | digest | 26ecdd28 |
| `preview.float-nan` | success predict | digest | 26ecdd28 |
| `preview.double-nan-asked` | success predict | digest | 26ecdd28 |
| `preview.double-signed-zeros` | success predict | digest | 26ecdd28 |
| `preview.float-signed-zeros` | success predict | digest | 26ecdd28 |
| `large.double-whole` | success predict | digest | 26ecdd28 |
| `large.double-nan-past-preview` | success predict | digest | 26ecdd28 |
| `large.float-nan-past-preview` | success predict | digest | 26ecdd28 |
| `large.double-negative-zero-past-preview` | success predict | digest | 26ecdd28 |
| `large.double-positive-zero-past-preview` | success predict | digest | 26ecdd28 |
| `large.double-ten-classes` | success predict | digest | 26ecdd28 |
| `large.float-ten-classes` | success predict | digest | 26ecdd28 |
| `large.double-sixty-one-values` | success predict | digest | 26ecdd28 |

### `contract/key-cast-vectors.json`

132 vectors change outcome (one of them to unfinished) and 67 only the reference's observed cast (the decimal keys' pandas dtype, from `decimal128` to `int64[pyarrow]` or `double[pyarrow]`); the register marks the recorder derives move with them.

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `int64.float.2pow63` | refusal INVALID_REQUEST | cast | not explained by the three reverts (see above) |
| `decimal128-18-0.int.present` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.absent` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.negative` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.128` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.256` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.32768` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.65536` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.2pow31` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.2pow32` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.2pow53-plus-1` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.int.2pow63-minus-1` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.int.2pow63` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-18-0.int.2pow64` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-18-0.int.minus-2pow63-minus-1` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-18-0.float.integral` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.float.half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.float.negative-zero` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.float.1e20` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.float.inf` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.float.2pow63` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.text.digits` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.text.decimal-point` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.text.half` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.text.spaced` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.text.leading-zero` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.text.letters` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.text.empty` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.text.exponent` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.text.negative` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.text.hex` | refusal INVALID_REQUEST → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.text.fullwidth-digit` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.text.2pow64` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.mixed.int-text` | refusal INVALID_REQUEST | cast, as_text, register | eb6c68eb |
| `decimal128-18-0.mixed.int-float` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.mixed.int-half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.mixed.float-text` | refusal INVALID_REQUEST | cast, as_text, register | eb6c68eb |
| `decimal128-18-0.mixed.bool-int` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.mixed.int-none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.mixed.text-none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.mixed.int-nan` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.bool.both` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.null.none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.null.nan` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.null.pd-na` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.numpy.int64` | refusal INVALID_REQUEST → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.numpy.float64-half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.decimal.integral` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.decimal.half` | refusal INVALID_REQUEST → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.time.timestamp` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.time.timestamp-1us` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.time.timestamp-utc` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-18-0.time.date` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-18-0.bytes.digit` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-18-0.exponent.0.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.1.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.2.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.3.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.4.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.5.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.6.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.7.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.8.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-18-0.exponent.huge.decimal` | refusal INVALID_REQUEST → unfinished within 300 s | public, cast, register | eb6c68eb |
| `decimal128-18-0.exponent.tiny.decimal` | refusal INVALID_REQUEST → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.present` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.absent` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.negative` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.128` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.256` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.32768` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.65536` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.2pow31` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.2pow32` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.2pow53-plus-1` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.2pow63-minus-1` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.int.2pow63` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-10.int.2pow64` | refusal INTERNAL_ERROR | cast, register | eb6c68eb |
| `decimal128-38-10.int.minus-2pow63-minus-1` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-10.float.integral` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.float.half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.float.negative-zero` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.float.1e20` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.float.inf` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.float.2pow63` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.text.digits` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.text.decimal-point` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.text.half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.text.spaced` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.text.leading-zero` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.text.letters` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.text.empty` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.text.exponent` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.text.negative` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.text.hex` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.text.fullwidth-digit` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.text.2pow64` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.mixed.int-text` | refusal INVALID_REQUEST | cast, as_text, register | eb6c68eb |
| `decimal128-38-10.mixed.int-float` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.mixed.int-half` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.mixed.float-text` | refusal INVALID_REQUEST | cast, as_text, register | eb6c68eb |
| `decimal128-38-10.mixed.bool-int` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.mixed.int-none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.mixed.text-none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.mixed.int-nan` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.bool.both` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.null.none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.null.nan` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.null.pd-na` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.numpy.int64` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.numpy.float64-half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.decimal.integral` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.decimal.half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.time.timestamp` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-10.time.timestamp-1us` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-10.time.timestamp-utc` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-10.time.date` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-10.bytes.digit` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10.own.decimal` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.own.text` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.own.float` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-10.exponent.0.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.1.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.2.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.3.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.4.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.5.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.6.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.7.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.8.text` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.huge.decimal` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-10.exponent.tiny.decimal` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-0.int.present` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.absent` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.negative` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.128` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.256` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.32768` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.65536` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.2pow31` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.2pow32` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.2pow53-plus-1` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.2pow63-minus-1` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.int.2pow63` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-0.int.2pow64` | refusal INTERNAL_ERROR | cast, register | eb6c68eb |
| `decimal128-38-0.int.minus-2pow63-minus-1` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-0.float.integral` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.float.half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.float.negative-zero` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.float.1e20` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.float.inf` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-0.float.2pow63` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.text.digits` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.text.decimal-point` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.text.half` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-0.text.spaced` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-0.text.leading-zero` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.text.letters` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-0.text.empty` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-0.text.exponent` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.text.negative` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.text.hex` | refusal INVALID_REQUEST → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.text.fullwidth-digit` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-0.text.2pow64` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.mixed.int-text` | refusal INVALID_REQUEST | cast, as_text, register | eb6c68eb |
| `decimal128-38-0.mixed.int-float` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.mixed.int-half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.mixed.float-text` | refusal INVALID_REQUEST | cast, as_text, register | eb6c68eb |
| `decimal128-38-0.mixed.bool-int` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.mixed.int-none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.mixed.text-none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.mixed.int-nan` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.bool.both` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.null.none` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.null.nan` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.null.pd-na` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.numpy.int64` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.numpy.float64-half` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.decimal.integral` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.decimal.half` | refusal INVALID_REQUEST → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.time.timestamp` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.time.timestamp-1us` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0.time.timestamp-utc` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-0.time.date` | refusal INTERNAL_ERROR | cast | eb6c68eb |
| `decimal128-38-0.bytes.digit` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-0-wide.wide.decimal` | refusal INTERNAL_ERROR → success | public, register | eb6c68eb |
| `decimal128-38-0-wide.wide.text` | refusal INTERNAL_ERROR → success | public, register | eb6c68eb |
| `decimal128-38-0-wide.wide.int` | refusal INTERNAL_ERROR → success | public, register | eb6c68eb |
| `decimal128-38-0-wide.wide.exponent` | refusal INTERNAL_ERROR → success | public, register | eb6c68eb |
| `decimal128-38-0-wide.wide.scaled-zeros` | refusal INTERNAL_ERROR → success | public, register | eb6c68eb |
| `decimal128-38-10-wide.wide.decimal` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10-wide.wide.text` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10-wide.wide.trailing-zeros` | refusal INTERNAL_ERROR → refusal INVALID_REQUEST | public, cast, register | eb6c68eb |
| `decimal128-38-10-wide.wide.inexact` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-37.wide.decimal` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-37.wide.text` | refusal INTERNAL_ERROR → success | public, cast, register | eb6c68eb |
| `decimal128-38-37.wide.out-of-range` | refusal INVALID_REQUEST | cast | eb6c68eb |
| `decimal128-38-37.wide.inexact` | refusal INVALID_REQUEST → success | public, cast, register | eb6c68eb |
| `decimal128-38-10-exponent.mixed.float-text` | refusal INVALID_REQUEST | cast, as_text, register | eb6c68eb |

### `contract/key-inference-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `nulls-1-decimal.200` | refusal materialize INTERNAL_ERROR → success materialize | register.why, register.required, reference.outcome.code | eb6c68eb |

### `contract/multicategorical-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `sep.semicolon` | success predict | digest | 26ecdd28 |
| `sep.colon` | success predict | digest | 26ecdd28 |
| `sep.pipe` | success predict | digest | 26ecdd28 |
| `sep.tab` | success predict | digest | 26ecdd28 |
| `sep.semicolon-8-rows` | success predict | digest | 26ecdd28 |
| `sep.colon-8-rows` | success predict | digest | 26ecdd28 |
| `sep.comma` | success predict | digest | 26ecdd28 |
| `sep.space` | success predict | digest | 26ecdd28 |
| `sep.slash` | success predict | digest | 26ecdd28 |
| `sep.semicolon-spaced` | success predict | digest | 26ecdd28 |
| `sep.semicolon-empty-parts` | success predict | digest | 26ecdd28 |
| `sep.ratio-at-1.5` | success predict | digest | 26ecdd28 |
| `sep.ratio-over-1.5` | success predict | digest | 26ecdd28 |
| `sep.every-part-unique` | success predict | digest | 26ecdd28 |
| `sep.parts-100` | success predict | digest | 26ecdd28 |
| `sep.parts-101` | success predict | digest | 26ecdd28 |
| `text.url-unique` | success predict | digest | 26ecdd28 |
| `text.url-few` | success predict | digest | 26ecdd28 |
| `text.url-with-port-and-query` | success predict | digest | 26ecdd28 |
| `text.free-occasional-colon` | success predict | digest | 26ecdd28 |
| `text.free-occasional-semicolon` | success predict | digest | 26ecdd28 |
| `text.free-few-occasional-colon` | success predict | digest | 26ecdd28 |
| `text.mac-address` | success predict | digest | 26ecdd28 |
| `text.ipv6` | success predict | digest | 26ecdd28 |
| `text.windows-path` | success predict | digest | 26ecdd28 |
| `text.plain-categories` | success predict | digest | 26ecdd28 |
| `text.plain-unique` | success predict | digest | 26ecdd28 |
| `mixed.semicolon-and-pipe` | success predict | digest | 26ecdd28 |
| `mixed.semicolon-pipe-tie` | success predict | digest | 26ecdd28 |
| `mixed.pipe-semicolon-tie` | success predict | digest | 26ecdd28 |
| `mixed.pipe-majority-multi` | success predict | digest | 26ecdd28 |
| `mixed.semicolon-majority-not-multi` | success predict | digest | 26ecdd28 |
| `mixed.tie-pipe-first` | success predict | digest | 26ecdd28 |
| `mixed.tie-semicolon-first` | success predict | digest | 26ecdd28 |
| `mixed.key-value-pairs` | success predict | digest | 26ecdd28 |
| `mixed.comma-and-semicolon` | success predict | digest | 26ecdd28 |
| `mixed.times-and-categories` | success predict | digest | 26ecdd28 |
| `null.some` | success predict | digest | 26ecdd28 |
| `null.most` | success predict | digest | 26ecdd28 |
| `null.all-but-one` | success predict | digest | 26ecdd28 |
| `null.all` | success predict | digest | 26ecdd28 |
| `empty.some` | success predict | digest | 26ecdd28 |
| `empty.all` | success predict | digest | 26ecdd28 |
| `empty.and-null` | success predict | digest | 26ecdd28 |
| `empty.only-separators` | success predict | digest | 26ecdd28 |
| `single.one-value` | success predict | digest | 26ecdd28 |
| `single.one-value-with-separator` | success predict | digest | 26ecdd28 |
| `single.one-part-repeated` | success predict | digest | 26ecdd28 |
| `single.one-category-or-two` | success predict | digest | 26ecdd28 |
| `single.one-value-8-rows` | success predict | digest | 26ecdd28 |
| `large.multi-throughout` | success predict | digest | 26ecdd28 |
| `large.plain-then-multi-after-500` | success predict | digest | 26ecdd28 |
| `large.plain-then-multi-after-1000` | success predict | digest | 26ecdd28 |
| `large.multi-then-plain-after-500` | success predict | digest | 26ecdd28 |
| `large.multi-then-plain-after-1000` | success predict | digest | 26ecdd28 |
| `large.multi-new-parts-after-1000` | success predict | digest | 26ecdd28 |
| `large.multi-many-parts-after-1000` | success predict | digest | 26ecdd28 |
| `large.pipe-then-semicolon-after-1000` | success predict | digest | 26ecdd28 |
| `large.nulls-first-500` | success predict | digest | 26ecdd28 |
| `large.nulls-first-1000` | success predict | digest | 26ecdd28 |
| `fact.semicolon` | success predict | digest | 26ecdd28 |
| `fact.pipe` | success predict | digest | 26ecdd28 |
| `fact.time-of-day` | success predict | digest | 26ecdd28 |
| `fact.semicolon-8-rows` | success predict | digest | 26ecdd28 |
| `fact.and-entity` | success predict | digest | 26ecdd28 |

### `contract/nested-type-vectors.json`

No item changes; only the recorded reference commit moves (identity only).

### `contract/sampled-null-key-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `singleton-null-text` | success predict | reference.request.inference.inference_config.kind, reference.request.inference.inference_config.hop_shuffle, reference.request.inference.inference_config.column_shuffle | 26ecdd28 |
| `null-text-not-sampled` | success predict | reference.request.inference.inference_config.kind, reference.request.inference.inference_config.hop_shuffle, reference.request.inference.inference_config.column_shuffle | 26ecdd28 |
| `singleton-null-decimal` | success predict | reference.request.inference.inference_config.kind, reference.request.inference.inference_config.hop_shuffle, reference.request.inference.inference_config.column_shuffle | 26ecdd28 |
| `null-decimal-not-sampled` | success predict | reference.request.inference.inference_config.kind, reference.request.inference.inference_config.hop_shuffle, reference.request.inference.inference_config.column_shuffle | 26ecdd28 |
| `mixed-null-decimal` | refusal materialize INTERNAL_ERROR → success predict | register.why, register.required, reference.request | eb6c68eb |
| `two-nulls-text` | success predict | reference.request.inference.inference_config.kind, reference.request.inference.inference_config.hop_shuffle, reference.request.inference.inference_config.column_shuffle | 26ecdd28 |

### `contract/t136-average-null-vectors.json`

No item changes; only the recorded reference commit moves (identity only).

### `contract/t136-forecast-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `sum-days-3` | success predict | digest | 26ecdd28 |
| `sum-days-1` | success predict | digest | 26ecdd28 |
| `sum-offset-days-2` | success predict | digest | 26ecdd28 |
| `count-days-4` | success predict | digest | 26ecdd28 |
| `avg-weeks-2` | success predict | digest | 26ecdd28 |
| `min-months-2` | success predict | digest | 26ecdd28 |
| `max-hours-2` | success predict | digest | 26ecdd28 |
| `sum-early-anchor-3` | success predict | digest | 26ecdd28 |
| `sum-assuming-3` | success predict | digest | 26ecdd28 |
| `count-filtered-3` | success predict | digest | 26ecdd28 |
| `sum-entity-filter-3` | success predict | digest | 26ecdd28 |

### `contract/time-inference-vectors.json`

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `timestamptz.999` | success from_source | duckdb.columns.happened_at.whole.min, duckdb.columns.happened_at.preview.min | identity only (the machine's zone: UTC here, America/Los_Angeles on the macOS recording) |
| `timestamptz.1000` | success from_source | duckdb.columns.happened_at.whole.min, duckdb.columns.happened_at.preview.min | identity only (the machine's zone: UTC here, America/Los_Angeles on the macOS recording) |
| `timestamptz.1001` | success from_source | duckdb.columns.happened_at.whole.min, duckdb.columns.happened_at.preview.min | identity only (the machine's zone: UTC here, America/Los_Angeles on the macOS recording) |
| `timestamptz.5000` | success from_source | duckdb.columns.happened_at.whole.min, duckdb.columns.happened_at.preview.min | identity only (the machine's zone: UTC here, America/Los_Angeles on the macOS recording) |

### `contract/type-set-vectors.json`

Recorded under `TZ=America/Los_Angeles`, as the macOS recording's machine zone effectively was (the orchestrator's decision), so `duckdb_arrow_by_setting.TIMESTAMPTZ` is unchanged.

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `decimal32.9-2` | refusal from_source RuntimeError → success predict | reference, reference_answer.described, reference_answer.outcome.error | eb6c68eb |
| `decimal64.18-2` | refusal from_source RuntimeError → success predict | reference, reference_answer.described, reference_answer.outcome.error | eb6c68eb |
| `decimal64.18-0` | refusal from_source RuntimeError → success predict | reference, reference_answer.described, reference_answer.outcome.error | eb6c68eb |

### `contract/t136-host2-vectors.json`

The outcome column is the reference's, which does not move; only `core_at_measurement` does.

| Item | Outcome, old → new | What moved | Cause |
|---|---|---|---|
| `first.users.TIMESTAMP_S.infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.TIMESTAMP_S.-infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.TIMESTAMP_MS.infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.far.s.past` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.far.s.before` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.far.s.just-past` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.far.ms.past` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.TIMESTAMP_S.year-10000` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.TIMESTAMP.infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.TIMESTAMP.year-10000` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.users.TIMESTAMP_S.null-then-infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.feature.TIMESTAMP_S.infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.from_source | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.time.TIMESTAMP_S.infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `first.time.TIMESTAMP_S.-infinity` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `generated.g24.users.TIMESTAMP_S` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.materialize | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `generated.g30.feature.TIMESTAMP_MS` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.from_source | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |
| `generated.g32.feature.TIMESTAMP_S` | refusal from_source OverflowError | core_at_measurement.described, core_at_measurement.outcome.stage, core_at_measurement.warnings.from_source | not the reference: the core at 292f8d8a against the core measured at 97aefc9d |

### `contract/t136-host2-zone-vectors.json`

No item changes; only the recorded reference commit moves (identity only).

### `contract/fixtures/pickles/`

`expected.json` changes in its reference commit only.
Each of the three pickles gains `Schema`'s `_conversion_notes` state and nothing else (`eb6c68eb`).

### PQL recordings

`rust/crates/sdm-core/tests/pql/{corpus,custom-profiles,journey,quirks}.json` and `rust/crates/sdm-pql/tests/validator-options.json` change in their source line only (identity only).
`rust/crates/sdm-pql/tests/dates.json` is kept as recorded on macOS at `b413e9ad` (see the findings above).
