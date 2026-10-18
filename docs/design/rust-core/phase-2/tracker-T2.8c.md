# T2.8c: multi-call `predict`

Lane f's worktree, started on the phase head after T2.19's landing (dfdd532a), moved onto T2.4's (8006efce), T2.1f's (dfce343e), T2.1g's (1486a8f1) and, for landing, T2.18's (f42d92ea).
`main`'s 1fc5a686 (in the reference at 6298913c) gathers a request of more entities than one call scores as several calls and predicts them as one; this package writes it in the core (design, choice 6), never in the host.
The acceptance cases are the test author's (`T2.8c tests:` commits on `t28c-tests`: 72521060, 217eea1b).

## What changed

- **materialize gathers several calls** (`sdm-core` `materialize::one`).
  The context's examples and labels are gathered once; then, for each batch of at most `rows_per_call` entities, the entity lookup, the subgraph walk and the payload are built, every one before any is sent, as the reference's engine builds each batch (`_iter_task_requests`: `narrow_prediction`, then `_get_context`).
  The state's `payload` is the first call's and `RequestState.batches` holds each later call's with the rows it scores.
  The notices are given once, the tables sent and the context answering the question counted across the calls, as the client's `_tables_sent` and `_drop_context_that_answers_the_question` count them.
- **`rows_per_call`** is read with the call's values (`sdm-request` `options::rows_per_call`): a whole number of at least one, which only lowers the 10,000 one call scores (`rows_one_call_scores`, `PREDICT_ROWS`, the serving layer's `TFM_MAX_TEST_ROWS`); a bool, a float, text or a number below one is `INVALID_REQUEST` naming the field; one past `u64` reads as `u64::MAX` and lowers nothing.
  The core's old 1,000-entity refusal is gone.
- **The ceilings**: more than 100,000 entities is refused before anything is read (`refuse_entities`, rule `most_entities`), more than 100 calls once the context is gathered (rule `most_calls`), each where the reference refuses it; both are the host's to set (`Materialize.most_entities`, `most_calls`), the Python host passing the client's `MOST_ENTITIES` and `MOST_CALLS` as it passes `MOST_PARTS`.
- **The batched-forecast refusal**: a forecast over more entities than one call scores is refused before anything is gathered (rule `batched_part`), and its population is counted across its steps (`most_entities`, `gathered_times`), as the client's `_in_parts` refuses both.
- **The 30 MiB body**: the payload refusal (`sdm-request` `payload::build`) names what to change: `remedy` `smaller_calls` with `rows_per_call`, the most rows a call of the request scores (`_the_largest_batch`), where the call has more than one row; `no_smaller_call` otherwise; and `batch_too_large_to_send`, as the reference's details.
  The host words both remedies in the reference's words (`_rust/errors.py`).
- **predict** (`sdm-core` `predict`): each call carries the id the host drew for it; every call is checked and written before the first is sent; the context is held to the cap per call, not to the sum (`prepare` over all calls, `_context_rows_per_call`); each call's rows are numbered on from where the previous call's ended (`call_starts`, `numbered_from`, as `_numbered_across_calls`); the answers are merged as one (`merge`, `batches` and `batch_request_ids` in the metadata).
- **Answered parts on a failed call**: the run ends with the call's own error, carrying the answers received before it (`CoreError.answered`, laid out as `Predicted.result`, with the rows they score and the field `calls_answered`); a forecast's failed step carries the steps answered before it (`parts_answered`).
  The result's model, request id, metadata and outputs travel on the attached result's schema metadata, so the host rebuilds the reference's merged result without a contract change.
  The Python host tells the core of an endpoint's refusal by its status (`run.Refused`), so the core carries the answers, and raises the refusal itself (E47) with `details['answered']`, `rows_answered`, `calls_answered` or `parts_answered`, as `main`'s client does.
- **Hosts**: the Python bridges pass `rows_per_call` and the ceilings, hold and hand back the batches, and draw an id per call; `run_python.py` expects each later call's id; `sdm-testkit`'s host holds and hands back the batches.
  The bridge freeze's six changed entries are declared (`moved`, T2.8c; `_result` a refactor).
- **Register**: P2B-3 and P2B-4, KO2-45's temporary entries for T2.8c, are retired (`rejected`, IDs kept, each with what replaced it); P2B-9, only ever proposed, is rejected with `approved_by: reviewer` as the proposal lapses, for T2.8c's reviewers to confirm; `register.py validate` passes.
- **Older client tests** that held the core's 1,000-entity refusal (`test_t128a_payload_rules.py`, `test_the_materialize_route.py`'s P1B-20 test) now hold the reference's outcomes, and the P1B-20 test the new ceilings.

## Probes

The test author's 21 cases (`multicall.*`) on a scratch combined branch (`t28c-combined`, the lane's commit merged with `t28c-tests`, never the lane), run on python-core (the test-ops wheel built from this tree), rust-host and python-ref (the reference venv at 6298913c), with a local diff of every call's request, the result, the warnings' classes and the statements per public call:

- DuckDB, python-core against python-ref and against python-old: all 21 equal in outcome, stage and code, every call's request (100 calls in `calls.100`), the result and its `row_index` order, the warnings and the statement counts.
- SQLite, python-core against python-ref: all 21 equal.
- Postgres 17 (the pinned image, a container of this lane's), python-core against python-ref: all equal but the statement counts, which differ by KO2-58's reads only (`current_schema()` and each table's row count, constant per call, independent of the batches).
- rust-host against python-core (`compare.py`, the `core` rule) on all three databases (Postgres by replay of python-core's recording): no difference.
- Answered parts, through the public API on both routes (a scratch probe): a failure on call 0, 1 and 2 of three, and on step 3 of a three-step forecast; `details` and `details['answered']` (its rows, model, request id, metadata and outputs) equal to the old route's in each.
- The implementer's own Rust tests, `sdm-core`'s `tests/multi_call.rs` (8): the ids, the numbering, the metadata, the answered parts after a refusal and after a failed transport, none on a first-call failure, the context held per call, a forecast's answered steps, and a later call without an id refused.

## Acceptance

The test author's five commits (`t28c-tests`, 72521060 to 6b86ea60) cherry-picked onto the lane, as the orchestrator asked:

- `contract/tests/test_t28c_multicall.py` (python-core, DuckDB, SQLite and Postgres 17 in the pinned image, `SDM_REQUIRE_POSTGRES=1`): every authored `py:` and `py-carried:` key passes (99 left `pending`).
- `sdm-testkit`'s `tests/t28c_multicall.rs` (rust-host, DuckDB and SQLite): every authored `rust:` key passes (36 left `pending`).
- `t28c-pending.json`'s `pending` is empty; `authored` is unchanged.
- Register P1B-20 names the ceilings' and the batched forecast's cases on all three databases, as the test author asked.

## The judging change, T2.1g (lane h, lands first)

python-core recorded one call's body, the last, and the payload rule held it against every request received, so no case of several calls could pass.
T2.1g (48d3e0f8 and 08bf20aa, lane h, from dfce343e) records every call's body (`core.predicted.calls`) on both core runners, holds them call by call, and has the test host read `RequestState.batches` and `CoreError.answered`; single-call lines are unchanged (`contract/tests/test_t21g_calls.py`).
T2.1g landed first (1486a8f1); on the rebase onto it the two commits that carried its files byte for byte dropped out as empty, and the rebased tree equals the one before but for T2.1g's own tracker.
My own earlier edits to `run_python.py` and `host.rs` (a later call's id, the batches held) are subsumed by T2.1g's.

## What the gate would still report

`compare.py` (with T2.1g's rule) over python-core and rust-host on the committed reference lines, DuckDB:

1. **The class name** on `calls.101`, `entities.100001` and `forecast.batched`: `UnsupportedRequestError` where the reference raises `StructuredDataError`, code equal; register P1B-20 (approved), whose cases now name them; the ratchet's baselines at landing hold it.
2. **An endpoint's refusal at predict** (`fail-call-0`, `fail-call-2`, `forecast.fail-call-1`): python-core raises the reference's `NimRequestError` (no code, the answered parts carried) while its core ends `INVALID_RESPONSE`; rust-host, with no transport, reports the core's code. Settled by the orchestrator as a comparator rule in T2.1g (08bf20aa): exactly that pairing agrees, on both core runners; no register entry. With it, these cases leave no difference.

## Landing work

- **Baselines**: `ratchet.py accept --checkpoint 1486a8f1` over python-core's, python-old's and rust-host's DuckDB observations of the 21 multicall cases, its lines added to the committed files, every other line kept: 21 for python-core (the three P1B-20 cases held by `case_outcomes` added to that entry, from `none`, as a first admission), 21 for python-old, 18 for rust-host.
- **rust-host's three endpoint-refusal cases** (`fail-call-0`, `fail-call-2`, `forecast.fail-call-1`): on a first admission the ratchet, and `sdm-testkit`'s SQLite host test, hold rust-host to the reference's outcome (no code) unless an approved entry names the case, while rust-host's is its core's `INVALID_RESPONSE`. Settled by the orchestrator with register P2G-1 (approved by the reviewer under E45, both reviewers to confirm in round 1; a new lane G block): rust-host `case_outcomes` for the three cases, naming their DuckDB and SQLite keys. With it, `accept` proposes rust-host's three lines (added, every other line kept), `ratchet.py check --checkpoint f42d92ea` passes, and `rust_host_runs_every_s1_case_on_sqlite` passes.
- **Rebase onto T2.18's landing**: small conflicts resolved by keeping both sides (the Python materialize bridge, the core's imports, `test_contract_cases.py`'s case lists, the reference lines and baselines, each a union), the freeze's two shared entries declared anew for this package's change, one refusal-map citation moved, and the goldens taken from the phase head and collected again.
- **Goldens**: see the next commit's message: `sdm-data`'s DuckDB, SQLite and Postgres goldens collected with T2.4's run check over python-core's run of the multicall and T2.18's name cases on each database (each later call's key set its own attachment id); each database's render and run checks pass with `SDM_L4_OBSERVATIONS` naming that run.

## Deviations and what is left

- No new register entry; P2B-3, P2B-4 and P2B-9 retired, P1B-20's cases extended.
- Judging-path files touched on this lane: `contract/runners/run_python.py`, `scripts/rust_core/compare.py`, `contract/observation.schema.json` and `sdm-testkit`'s host (T2.1g's, mirrored); `scripts/rust_core/bridge_changes.json` and the freeze baseline; `contract/refusal-maps/` (three line citations); `docs/design/rust-core/register.yaml`.
- At landing: the cases' baselines (`ratchet.py accept`) and the SQLite and Postgres goldens for the cases' statements, as at T2.19; `tests/test_contract_cases.py`'s `_not_synthesizable` still names the six multi-call cases the test author left out before T2.1g.
- The local `contract/tests` run (4,462 passed) failed 132 tests, every one the recorded-environment guard of this machine's venv (pandas and pyarrow not a recorded cell: `test_t136_host2_*`, `test_gate_modes`, `test_gate_evidence_paths`, `test_t20_locked_set`), none of this package's behavior.
- A pre-existing finding, not this package's (the orchestrator carries it for investigation): the old-against-core differential's 50 seeded DuckDB sources show 7 unclassified "schema after the edits" differences, `from_source` reading a column's meaning (`d`) as categorical where the reference reads numerical, before any materialize, the same with one call per source or two.
