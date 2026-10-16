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

### Waiting

- The re-recording (the recorder agent, its own commits) and what depends on it: the moved baselines (`ratchet.py accept`), the register entries and the trace table.
- The bridge declarations for the old-route functions `main` changed, in T2.0e's `main` reason, and `_check_entity_count` leaving `COVERED` (`main` deleted it in `1fc5a686`).
- CI on the candidate, which refuses the new reference until T2.0e lands.
