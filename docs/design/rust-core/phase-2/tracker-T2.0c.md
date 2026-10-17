# Phase 2 tracker: T2.0c

The design branch owns `tracker.md`; this file holds T2.0c's section until it folds in.

## T2.0c

T2.0c brings `main` at `327fda03` (`v1.0.0`) into Phase 2, at the T2.0 landing `292f8d8a`, with the reference re-recorded at `6298913c`, in one landing.
It follows the audit's merge plan (`main-audit/4-merge-plan.md` in the orchestration state) and the owner's decisions of 2026-10-16, which the design branch writes as `KO2-n` rows.
It lands from the topic branch `rust-core-phase-2-main-merge`, after T2.0e (the gate rules that let a new reference and a merge of `main` through).

### The merge and each conflict's resolution

`git merge --no-ff origin/main` gave 9 conflicting paths, the same as the audit's trial (`git merge-tree`, tree `638a4581`).
Every path outside these 9 is git's own merge: the merge commit's tree differs from `638a4581` only in them.

| # | Path | Resolution | Why |
|---|---|---|---|
| 1 | `ci.yml`, `test` job | Phase 2's steps (`fetch-depth: 0`, the `native-wheel` artifact, the weekly pip cache, the client from the native wheel), with `main`'s SHA-pinned `checkout` and `setup-python` and `main`'s `snowflake,s3` connectors extras | The bootstrap test needs the full history; `main`'s `435a0879` added the extras so the driver tests stop skipping |
| 2 | `ci.yml`, `engine` job | `main`'s SHA pins, `fetch-depth: 0` kept | The gate's checkpoint and worktree need the history |
| 3 | `client.py`, `SDMClient._answer_in_parts` | Phase 2's forecast-on-the-core early return first, then `main`'s typed `answers` and `answered` lists | The core block returns before the lists are used; `main`'s lists carry `1fc5a686`'s partial answers |
| 4 | `client.py`, module helpers | `_predicts_on_the_core` and `main`'s four multi-call helpers, all kept | Pure add/add |
| 5 | `materializers/relational.py`, `materialize` | Phase 2's split by route, with `main`'s `rows_per_call_asked_for(options)` added to the old route's checks | The core route has its own rules (T1.28a); `main`'s other lines were already in the old route's branch |
| 6 | `materializers/relational.py`, `_gather` | Phase 2's `(payloads, state)` return type, with `main`'s batching docstring appended | The merged body already returns `tuple(...), None` |
| 7–9 | `schema.py` (`__init__`, the pickle-state defaults, `copy`) | `_core_description` and `_conversion_notes` both kept | Old pickles without `_conversion_notes` still load through the `()` default |
| 10 | client `pyproject.toml` | `pandas>=2.2.2`, `numpy>=1.23.5`, `pyarrow>=14.0.1` | Owner: `main`'s pandas floor; E16's NumPy floor lowered to the 1.23.5 that DBR 14.3/15.4 LTS and serverless 1/2 ship; the core's pyarrow floor kept |
| 11 | connectors `pyproject.toml` | `main`'s `pandas>=2.2.2` and unbounded `pyarrow`, with `main`'s comments | Owner: a floor there evicts a runtime's working pyarrow; the client floors it wherever the core runs |
| 12 | relational-engine `pyproject.toml`, `snowflake` extra | `main`'s: no NumPy bound | `main`'s Snowflake bounds; Snowflake is Phase 3 |
| 13 | `examples/databricks/adventureworks_relational.ipynb` | `main`'s delete | `main` replaced it with `kumo_relational_quickstart.ipynb` (`18e26f1c`); the contract test's copy is a fixture (below) |
| 14 | `examples/databricks/lakebase_relational.ipynb` | `main`'s notebook, unchanged | With the NumPy floor at the runtimes' 1.23.5, Phase 1's hold to the 1.26 line has no reason (a deviation from the plan, which predates the owner's NumPy decision) |

### The follow-through

- Floors everywhere: `pandas>=2.2.2` and `numpy>=1.23.5` in pql and the engine's `setup.py` too.
  `tests/test_release_contract.py` holds the new set, with `main`'s unbounded connectors pyarrow as its one named allowance.
- The floor cell moves with the floors: `BINDING_FLOOR`, `contract-python (floor-pandas2.2)`'s `CELL_PINS`, `run_python.py`'s `CELLS` and `wheel_smoke.py`'s `floor` lane are Python 3.10, pandas 2.2.2, pyarrow 14.0.1, NumPy 1.23.5.
  The cell keeps its name, which the gate and the evidence tests read.
- `docs/get-started/installation.md` is `main`'s again (keep the runtime's own NumPy), and `scripts/live/dbr_install_check.py` runs that guide's cell and checks NumPy stays the runtime's.
- The deleted AdventureWorks notebook is frozen at `fa0637d4` as `contract/fixtures/adventureworks/notebook-fa0637d4.ipynb`, which `test_t133_adventureworks_cases.py` reads; no case file changes.
- `migration-plan.md` and `phase-0/databricks-jobs.md` point at `kumo_relational_quickstart.ipynb`; the latter notes that the quickstart reads through Spark only.
- `uv.lock` (`main`'s nSpect inventory, `4cfbcd18`) is regenerated with uv 0.13.1 and the pinned toolchain, and `rust-lint` checks it is current (`uv lock --check`).

### The port of `26ecdd28`

`sdm-request`'s `body` sends only the inference-config keys the caller set (`asked`), and leaves `inference_config` out when none are left, as `RelationalAdapter._inference` does on `main`.
An asked key the model does not read is still refused before the call, as before.
`rust/crates/sdm-core/tests/predict.rs` holds it: the default call carries no config, one asked key is sent alone, two in the payload's order, an asked key the payload lacks adds nothing, and an unread asked key is still `UNSUPPORTED_REQUEST`.
The gathered payloads (`contract/pinned/base.regression.core.json`, `t128w-payload-bytes.json`, `payload_lock.rs`) still carry the engine's config, as `main`'s materialize still fills it; only the sent body changes.

### Hygiene `main` enforces

- The two credential-shaped test literals are put together at run time, as `adbe935b` does.
- Every tag-pinned action in `ci.yml` is pinned by commit digest with its version: `main`'s digests for `checkout`, `setup-python`, `upload-artifact` and `download-artifact`, and `v6.1.0` (`55cc8345`) for `actions/cache`.
- `codeql.yml` has a `rust` leg (`build-mode: none`); it runs where `codeql.yml` runs, on `main`.

### The re-recording, taken unchanged

The recorder's commits are merged unchanged: `rust-core-phase-2-main-merge-recording` at `7a6658f1` (11 `T2.0c tests:` commits on `292f8d8a`, by a merge commit), `rust-core-phase-2-main-merge-recording-2` at `4f1c3247` (3 commits on this branch, by a fast-forward) and `rust-core-phase-2-main-merge-recording-3` at `da6fd15a` (5 commits on `c1d4f6e6`, by a merge commit: the class target's python-ref line, the key-cast marks for `eb6c68eb`'s casts with the `unfinished` transition, and the tests that rested on the old reference's refusals).
Its trace of every moved line to a `main` commit is [tracker-T2.0c-recording.md](tracker-T2.0c-recording.md).
After it, from this side:
- The four decimal cases (`key.decimal-18-0`, `key.decimal-38-10`, `key.hugeint`, `decimal.class-target-38-18`) expect the new reference's success at predict, their `register` field naming P2B-1; `decimal.class-target-38-18`'s core section carries predict, as `key.decimal-18-0`'s does.
- The core's own side of `contract/t136-forecast-vectors.json` is recorded again by `forecast_record.py candidate` in the `ref-pandas3` cell, since the port changes the bytes each forecast sends; the reference side is untouched.
- `contract/refusal-maps/decimal.json` is rewritten by `refusal_map.py write`, its probe following its case.
- The bridge freeze declares the 14 old-route entries the merge changes with T2.0e's `main` reason (`1fc5a686` for 12, `eb6c68eb` for 2); `_check_entity_count` leaves `COVERED`, as `main` deleted it; the baseline is written again; `selftest.py`'s entity-cap case moves `_check_limits`' batch cap instead, and its installed-wheel fixture reads the pinned checkout.

### Exact decimals

The owner moved T2.17's exact decimals into T2.0c on 2026-10-17 (KO2-35, KO2-50): the core answers every decimal input the reference at `6298913c` answers, stays exact and never merges distinct keys.
- A decimal sum past its column's 38 digits, or past `i128`, is summed exactly in 256 bits and read once at float32, and a mean is exact whatever its total (`sdm-engine` `labels.rs`, `ExactSum`).
- A decimal target is read as the reference reads it: int64 at scale 0 where every value fits, float32 otherwise, and where two distinct values meet at one float32 each keeps its own digits, so no two values become one class (`decimal_target`).
- A decimal column is compared with a constant exactly, at no scale but their own: the constant is the decimal the query wrote, the shortest decimal naming its double (the orchestrator's confirmed reading, so `d = 0.1` holds on `DECIMAL(38,18)`), a constant past the column's digits orders by its sign, an infinite one past every value (`decimal_order`; `d > 1.5` holds for d = 2, KO2-50).
- A decimal feature is sent as the nearest double to its exact value (`sdm-request` `decimal_float`), where it was refused when a double did not hold it.
- A null decimal key beside values is kept, as the reference keeps its integer twin's (`sdm-core` `subgraph.rs`); P1B-40's option (b) stays for text keys.
- The Rust test host passes the case's asked inference options to predict, which the `26ecdd28` port needs (`sdm-testkit` `host.rs`).
- The decimal family's refusal map holds no refusal any more, written by `refusal_map.py write` with rust-host's observations, and its probes' pin is regenerated from python-core's run.
- New cases from the case author (`rust-core-phase-2-main-merge-recording-5`): the comparisons around 1.5 (`compare-finer-gt`, `-eq`, `-negative`, `-same-scale`, `-past-precision`, `-past-precision-gt`), `sum-10-2`, `sum-past-38`, `feature-inexact-38-18`, `key-wide`, `class-target-collide`.

### Register entries (lane B)

| Entry | What | Approval |
|---|---|---|
| P2B-1 | Exact decimals where the reference rounds, merges or refuses: text ids for keys with decimal places, the nearest double for features, exact sums and means, a sum past 38 digits answered, distinct class values kept apart; 46 vector transitions and the class target's two core outcomes | KO2-35, the owner's (permanent) |
| P2B-2 | python-core refuses a column named with a dot, backtick or line break, which the reference leaves out (until T2.18) | KO2-45, the owner's (temporary) |
| P2B-3 | The core refuses 1,001 to 10,000 entities (until T2.8c) | KO2-45 (temporary) |
| P2B-4 | The core refuses `rows_per_call` (until T2.8c) | KO2-45 (temporary) |
| P2B-7 | python-old answers the four decimal cases as the reference does, by `main`'s `eb6c68eb` | proposed, for the reviewers |
| P2B-8 | An entity that names no key of a decimal key's type exactly is refused by name, where the reference rounds it onto a key (four key-cast vectors) | proposed, for the owner |

P2B-5 and P2B-6 (decimal null keys, `d > 1.5`), temporary entries never landed, are gone with the fix.
P1B-18's and P1B-53's refusal of `decimal.class-target-38-18` no longer exists, and their outcomes on it are taken out; P1B-33's eight refs that name no vector at `6298913c` are taken out; each for the reviewers.

### Probes

- The floor at Python 3.10, pandas 2.2.2, pyarrow 14.0.1, NumPy 1.23.5: the binding's tests pass on the native wheel (35); the engine imports; the floor cell's remaining differences are `main`'s decimal moves on python-old.
- Column names `a.b`, ``a`b``, `a\nb`, `a\rb` on DuckDB: the old route (main's) leaves the column out with one `SDMWarning`; the core route refuses `INVALID_REQUEST` (P2B-2).
- 1,001 users and `rows_per_call=3` over 8: the old route gathers 1 and 3 payloads; the core refuses `UNSUPPORTED_REQUEST` and `INVALID_REQUEST` (P2B-3, P2B-4).
- The bridge declarations judged by T2.0e's `bridge_freeze.py` at lane a `f59f0d61`: 4 of 14 pass; the 10 entries both sides changed fail its "exactly as main left it" rule, which T2.0e fix 3 changes to the entry as the merge commit left it.

### `dates.json` and the reference rule

`rust/crates/sdm-pql/tests/dates.json` stays the macOS recording at `b413e9ad` (the recorder's choice: a Linux recording loses the padded-year rows two `dates.rs` tests need, and the reference's date reading does not move with `main`).
No test holds its `source` to `contract/reference-env.txt`: `dates.rs` reads only its rows, and `tests/test_contract_reference.py` reads `contract/reference/`, so the tests pass.
T2.0e's reference rule (lane a `f59f0d61`, `ratchet.py` `check_reference`) fails every file outside `docs/` and `contract/reference/` that cites the old SHA in full: `dates.json`, and also the two ratchet fixtures `tests/golden/t132g-fixture-cases/python-ref.jsonl` and `tests/golden/t132g-later-stage/python-ref.jsonl`, which the recorder did not re-record.
The lightest honest routes, for the owner: a one-command macOS run of `dates_record.py` at `6298913c` (the rows equal, only the source line moves), or a named exemption in T2.0e's rule for recordings whose reading `main` did not move, listed by path; the two golden fixtures need the same choice (re-recorded, or exempt as ratchet fixtures that hold no reference identity).

### `int64.float.2pow63` on Linux

The reference's own cast of the float 2**63 to `int64[pyarrow]` saturated to `9223372036854775807` on the macOS arm64 recording and raises `ArrowInvalid` on this Linux x86_64 machine, with or without any revert (the recorder's notes); its public answer, `INVALID_REQUEST` at materialize, is the same on both.
Nothing of the core's or the register's rests on the cast: the vector is not marked register-required, no entry names it, the Rust host's `rust_host_casts_to_an_int64_key_as_the_reference` and python-core's per-vector check compare the public answer only and pass on Linux, and `reference.cast` is read only for the text keys' float spelling and for being present.
CI runs on Linux x86_64, as the recording was made.

### Tests run

- `cargo test --workspace`: all pass but the vector tests resting on the old reference's decimal and 1,001-entity refusals (listed under Waiting).
- `contract/tests` in the `ref-pandas3` cell: 3835 passed, 167 failed before the last fixes; each remaining failure is listed under Waiting.
- The CI `test` job's suites: 4123 passed; the failures were the 26ecdd28 recordings (now re-recorded), `engine-wheels`' timeout and the bytes test (fixed), and two environment-only ones (`SDM_PHASE_BRANCH` exported here; a sibling contract checkout).
- `scripts/rust_core/selftest.py`: the bridge freeze's controls pass once the freeze passes on the tree (after T2.0e fix 3).

### Waiting

- T2.0g: the checkpoint's bridge freeze still names `_check_entity_count`, which `main` deleted (l0).
- The case author (`recording-6`): `decimal.sum-past-38`'s and `decimal.class-target-38-18`'s cases carry predict to the core, and their python-ref lines; the test author: `t132b_float_constants.rs` against KO2-50, the option (b) tests for decimal keys, three refusal-map tests on an empty decimal map.
- The owner: P2B-8, and how compare treats KO2-35's field-level differences (the payloads of `key.decimal-38-10`, `decimal.sum-10-2`, `decimal.feature-inexact-38-18`; `decimal.sum-past-38` and `decimal.class-target-collide`, answered where the reference refuses).
- The reviewers: P2B-7 and the changed P1B-18, P1B-53 and P1B-33.
- Then `ratchet.py accept` for the baselines (the eleven new cases and the moved outcomes) and the gate.
