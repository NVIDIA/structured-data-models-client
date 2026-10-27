# Phase 3 tracker: T3.2

The design branch owns `tracker.md`; this file holds T3.2's section until it folds in.
T3.1, the previous package on lane A, landed at d1e02d7c.

## Round 1

The test author's three commits (`T3.2 tests: ...`) are on the lane unchanged; every test they authored passes, and `contract/tests/t32-pending.json` lists none pending.
The implementer's own tests are `contract/tests/test_t32_recording_own.py` and `tests/test_t32_train_own.py`.

### The case format (both loaders)

- `warehouses`, `input.view` (with `trust_expressions`), `input.connect` and `materialize.num_context_samples`, held by `contract/tests/t32-case-format-vectors.json` in both loaders.
- `view` and `connect` only beside `warehouses`; a view's kind kept on each leg named; a connect path's case naming exactly its leg.
- P2-T2.1i-1: both loaders refuse a `fail_call` of `-0` and `NaN` or `Infinity` anywhere.

### The recording mode

- `run_python.py --mode python-ref --reference DIR --database <leg> --record FILE`, the live configuration in `SDM_LIVE_CONFIG`, the placeholders' names in `SDM_CAPTURE_<NAME>`.
- The capture and the statement log on each warehouse driver, per connection and per driver class, the log entered only in a `with` (P2-T2.1a-2).
- Every private name written back as its placeholder; the write fails on any private value left, tables included.
- T3.3's writer turns the output into the capture manifests `capture_check.py captures` reads; the loader interface is `databases.load(duck, leg, WarehouseTarget)`, and the stand-in's is `run_python.py`'s `Standin` over `standin_client.open(leg, case, bases, captures)`.

### The rest of the train

- `live_only` in the observation schema, `observation.rs`, `compare.py` (`--verdicts`, `--candidate`), the ratchet and `gate.sh`; the policy's `standin` codes and rules held to the owner's approval (KO3-45).
- `java` as a runner; `sdm-jni` in `l0_deps.py` and `l0_scan.py`; `lane.py start --brief`, whole IDs from Phase 3, no lane on a finished phase.
- Coverage's leg claims, manifest test claims and `--gate`; `register.py`'s owner rows with the allowlist (U4); the census's real-data mode per leg.
- P2-MID2-7, P2-T2.0f-1, P2-T2.21-1, C18's static check, handoff's cleanup on a signal, clean_tmp's listening sockets, the dry-run plan's corrections.

### Answers to the brief's appended items

- The auto/explicit claims the Snowflake and Lakebase test author chose are confirmed: a leg's default source is built with its class (`DatabricksSource`, `SnowflakeSource`, `PostgresSource`), so it claims `source.open/explicit`, and a connect case hands `Schema.from_source` the raw connection or session, found by detection, so it claims `source.open/auto` (`run_python.py`'s `warehouse_source`, as the old client reads them, CN/source.py:97-107).
- `lane.py` started from the `rust-core` checkout: it now refuses a phase `rust-core` already holds, naming it.

### Probes

- Outside JSON: deep and wide capture lines, manifests and verdict manifests are each refused by name or read (`test_deep_and_wide_capture_lines_are_refused_by_name`, `test_a_deep_or_wide_manifest_is_refused_by_name`, `test_a_live_only_item_needs_its_e1_verdict_on_the_final_candidate`).
- Seeded faults: `test_an_incomplete_capture_fails_by_name` drops or retypes each piece of evidence the recording check reads; the live-only, verdict, owner-row and census tests each fail their rule with its evidence missing.

### Not done in this round

- `gate.sh` makes the stand-in and Java observations; judging them per leg against the captures' reference lines waits for T3.3's capture reader.
- `key.past-2-53.catalog@postgres`'s reference line predates the text digest (P2-T2.21-1): the comparison names it until the orchestrator records it again.
