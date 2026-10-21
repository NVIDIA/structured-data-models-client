# Phase 2 tracker: T2.16a

The design branch owns `tracker.md`; this file holds T2.16a's section until it folds in.
Lane o, on the phase head b2732c12 (after T2.13b's landing), then the test author's acceptance tests (`T2.16a tests:`, from `t216a-tests`), then this package (`T2.16a:` commits).

## T2.16a

The final gate's tooling, before the held-out one shot: the held-out record's open items and the judging gaps carried to "before the slice gate" (brief `T2.16.md`, section "T2.16a").

### H11's threshold per database leg (recorded before the run)

The held-out set runs every case on DuckDB, SQLite and Postgres.
A runner's observation of a case is keyed by the bare case id on DuckDB and `<id>@<database>` elsewhere; each key is one leg of the case.
H11 reads "more than 4 cases differing from the reference without an entry approved before the run", and it is read per case (the orchestrator's reading, 2026-10-21, found plain):
- a case counts once, however many of its legs differ;
- a case differs when any leg's outcome class, on either core runner, differs from python-ref's line of the same key;
- a case is approved only when every differing leg is (`--approved` takes a case id, every leg, or one leg's key);
- any leg's panic is a no-go, and a leg python-ref observed that a core runner did not observe is a panic;
- any leg's new refusal of an input the reference accepts, unapproved, is a no-go.

### What changed

- **Counting every leg** (`scripts/rust_core/heldout.py`): `counts` and `report` read `<id>@<database>` legs as above; `counts.md` lists each differing leg by key and the summary counts cases.
  `heldout_run.sh` runs python-ref, python-core and rust-host on each of `--databases` (all three by default), rust-host on Postgres replaying python-core's fetches; python-old runs on DuckDB alone; `compare.py` runs once per database into `compare.txt`. A Postgres leg needs `SDM_POSTGRES_TEST_DSN`, and only whether it is set is said.
- **The converter** (`heldout.py convert ORIG OUT --prefix P`): the held-out format (c8075715) into the head's case format, each case held to the head's own loader (`run_python.load_cases`).
  No field has moved since c8075715 (`FORMAT_MOVES` is empty): every field added since (T2.1m's `lag_timesteps`, `outputs` and `quantiles`, T2.1n's `explanation` and `item_metadata`, T2.1t's sampling options, `context_anchor_time` and `embeddings`) is optional, and the ranking and multi-call fields are still `input.materialize`'s. A case is therefore copied byte for byte and the digests stay the frozen ones; the command prints both.
- **Dummy bundles** (`heldout.py dummy panic|over-threshold OUT --prefix P [--threshold N]`): check 9's two synthetic run directories, with DuckDB and SQLite legs, which `counts` must find a no-go: a rust-host `INTERNAL_ERROR` on one SQLite leg, and N+1 cases each differing on one leg (neither a panic nor a new refusal).
- **The result's explanation in both runners** (`contract/runners/run_python.py`, `sdm-testkit/src/host.rs`, `contract/observation.schema.json`): `core.predicted.result.explanation`, the result schema's `explanation` metadata as the JSON text the core wrote, so `compare.py`'s whole-`core.predicted` comparison holds the hosts to the same bytes.
  An item's explanation and metadata are already the result's own `explanation` and `metadata` columns, among the rows both runners record.
- **Offline L7** (`contract/tools/l7.py`): `measure` per environment and `report` against KO2-29's limits; a one-time report, no seeded-fault suite (KO2-53, KO2-76).
  Boundary overhead is measured as the Python host's codec CPU plus one native crossing's CPU per control message; zero-copy is left to `test_arrow_lifetimes.py` in CI.
- **Coverage** (`scripts/rust_core/coverage.py`): a query's target is read behind `EVALUATE` or `EXPLAIN` and with `FOR t.c = v` or `FOR t.c IN (...)`; `float64/target` is back on the eight cases 1e89eacc took it from, and their 24 python-ref lines are re-recorded (identity only differs). `date.day-first-then-packed` predicts `COUNT(orders.*)`, which no column plays, so its claim stays dropped (the orchestrator's decision).
- **The kept rules on real data** (`docs/design/rust-core/phase-2/evidence/census/real-data.md`, `tests/test_census_real_data.py`): `text_time_rules` and `model` are hit 0 times on every database by construction (the orchestrator's option b), with a guard test.

### Not done

- No stored-NaN case and no `key_name_order` case in the held-out set (held-out record items 3 and 4): on the packet's "not done" list; nothing is added after the freeze.
- `contract/tools/l7.py` is not added to `contract/protected-paths.txt`, as the census tool is not: both are end-of-phase reports outside the gate since KO2-76.
