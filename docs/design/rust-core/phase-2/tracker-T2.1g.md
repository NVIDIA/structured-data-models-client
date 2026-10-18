# T2.1g: the payload rule over several calls

Lane h, on the phase head at T2.1f's landing (dfce343e); a judging package that lands alone, before T2.8c (the orchestrator's split).
T2.8c's cases are the first whose predict makes several calls (`main`'s 1fc5a686), and a forecast of two steps is another; the payload rule held one body, the last call's, against every request received, so no such case could pass.

## What changed

- **The runner** (`contract/runners/run_python.py`, `_Capture`): python-core keeps every call's body in order and records them as `core.predicted.calls`, beside `body`, the last's, where predict made more than one call; a request of one call records `body` alone, as before. A later call of a request asked in several (`RequestState.batches`, contract v26) is expected to carry the id the host gave it, as a forecast's later step is.
- **The observation schema** (`contract/observation.schema.json`): `core.predicted.calls`, optional, at least two bodies.
- **The comparator** (`scripts/rust_core/compare.py`): the payload rule (`_core_payload`) holds the calls, call by call and in order, against the requests the endpoint received (python-core) or the reference sent (rust-host), a difference named by the call's place (`core.predicted.calls/<n>/...`); one call is `core.predicted.body`, judged as before. The core rule's projection (`_predicted`) masks each call's request id as the last's (decision E35).
- **The rust-host test kit** (`sdm-testkit`'s `host.rs`, `observation.rs`): holds and hands back `RequestState.batches` (each later call's payload, rows and this host's one request id); takes the answers a failed predict carried (`CoreError.answered`) off the run, kept in process (`Observed::answered`); and writes `calls` into the observation where there were several, as python-core does.

## Checks

- `contract/tests/test_t21g_calls.py`: one call judged as before (equal passes; a seeded difference fails at `core.predicted.body/0/`); several calls equal pass; a call that differs fails at `core.predicted.calls/2/`; a call missing, one too many, the calls out of order, and only the last call recorded (the rule before this package) each fail, on both core runners.
- No case before T2.8c's makes several calls, so no observation, reference line or baseline changes.
