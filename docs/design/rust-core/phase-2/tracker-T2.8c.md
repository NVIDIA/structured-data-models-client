# T2.8c: multi-call `predict`

Lane f's worktree, on the phase head after T2.19's landing (dfdd532a).
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
- **Register**: P2B-3, P2B-4 and P2B-9, KO2-45's temporary entries for T2.8c, are retired (`rejected`, IDs kept, each with what replaced it); `register.py validate` passes.
- **Older client tests** that held the core's 1,000-entity refusal (`test_t128a_payload_rules.py`, `test_the_materialize_route.py`'s P1B-20 test) now hold the reference's outcomes, and the P1B-20 test the new ceilings.

## Probes

The test author's 21 cases (`multicall.*`) on a scratch combined branch (`t28c-combined`, the lane's commit merged with `t28c-tests`, never the lane), run on python-core (the test-ops wheel built from this tree), rust-host and python-ref (the reference venv at 6298913c), with a local diff of every call's request, the result, the warnings' classes and the statements per public call:

- DuckDB, python-core against python-ref and against python-old: all 21 equal in outcome, stage and code, every call's request (100 calls in `calls.100`), the result and its `row_index` order, the warnings and the statement counts.
- SQLite, python-core against python-ref: all 21 equal.
- Postgres 17 (the pinned image, a container of this lane's), python-core against python-ref: all equal but the statement counts, which differ by KO2-58's reads only (`current_schema()` and each table's row count, constant per call, independent of the batches).
- rust-host against python-core (`compare.py`, the `core` rule) on all three databases (Postgres by replay of python-core's recording): no difference.
- Answered parts, through the public API on both routes (a scratch probe): a failure on call 0, 1 and 2 of three, and on step 3 of a three-step forecast; `details` and `details['answered']` (its rows, model, request id, metadata and outputs) equal to the old route's in each.
- The implementer's own Rust tests, `sdm-core`'s `tests/multi_call.rs` (8): the ids, the numbering, the metadata, the answered parts after a refusal and after a failed transport, none on a first-call failure, the context held per call, a forecast's answered steps, and a later call without an id refused.

## What the gate would still report (needs the orchestrator)

`compare.py` with the local python-ref recording as the reference finds three classes, the same on all three databases:

1. **The class name on the ceilings and the batched-forecast refusal** (`calls.101`, `entities.100001`, `forecast.batched`): `UnsupportedRequestError` where the reference raises `StructuredDataError`, code `UNSUPPORTED_REQUEST` equal; register P1B-20's approved difference (the Python host's one class for the code).
2. **The payload rule's last call only**: python-core records `core.predicted.body` as the last call's and rust-host the same, while `compare.py` holds `[body]` against every request received (python-core) or the reference sent (rust-host), so any case of several calls fails "1 items against N" (`rows-per-call-3`, `calls.100`, `entities-per-call.10001`, and also a forecast of two steps, `forecast.one-call`). A judging gap: no earlier case made several calls.
3. **An endpoint's refusal at predict** (`fail-call-0`, `fail-call-2`, `forecast.fail-call-1`): python-core raises the reference's `NimRequestError` (code none) while its core ends `INVALID_RESPONSE` naming the status, which `compare.py`'s "outcome against core" holds equal; rust-host, which has no transport, reports `INVALID_RESPONSE`. A judging gap too: no earlier case had the endpoint refuse.

## Deviations and what is left

- No new register entry: on every case the core answers, both hosts match the reference on all three databases, but for the three classes above.
- The judging-path files touched: `contract/runners/run_python.py` (a later call's request id, a few lines) and `sdm-testkit`'s `host.rs` (the batches held and handed back), each a host's reading of the contract's `RequestState.batches`, which the test author's commit leaves to the implementer; `scripts/rust_core/bridge_changes.json` and the freeze baseline; `docs/design/rust-core/register.yaml` (the three retirements).
  `compare.py` is untouched; items 2 and 3 above need a judging change, which this package has not made.
- The python-ref lines and baselines for the cases are the test author's and the combined branch's (as at T2.19).
