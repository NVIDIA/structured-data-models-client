# Phase 2 evidence pack (T2.16)

The evidence for Phase 2's go or no-go, item by item as design.md's "The evidence pack" and the locked checks (as KO2-76 changed them) list it.
Every generated file here was made by the branch's own tools; the command that made each is named below.
The hand-written files are this index and [l3/classification.md](l3/classification.md).
Local paths in copied outputs are left out or written `<scratch>`.

## The final tree

- Phase commit: `f133c4b8f52ed5934a17218be0ca7b0f94e03436` (`Land T2.14c`), an empty landing whose tree equals its candidate `be55838a14da280e628f4d7c40f0e04f553303c8`.
  The reports below ran on be55838a's clean worktree before the landing; the tree is the same, so they stand for f133c4b8.
- Reference: `rust-core-reference` at `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1` (KO2-84), `contract/reference/43e55f4f….jsonl`.
- Exact-SHA CI run: 39577080078 on be55838a, every job green (the landing stands on it, decision E131).
- Cells: `scripts/rust_core/cells.sh --rev be55838a… OUT` (ref-pandas3, dbr17-pandas2, floor-pandas2.2), with the connectors' `sqlite` and `postgres` extras added to each cell, as ci.yml's "Install the cell" step has them and cells.sh does not (carried into Phase 3).
- Postgres: a throwaway `postgres:17` container, given by `SDM_POSTGRES_TEST_DSN`.

## Items

| # | Item (check) | Here | Made by | Not covered here |
|---|---|---|---|---|
| 1 | The locked gate, `gate.sh --locked 2a` (checks 1, 2, 7, 10, 12) | the gate's evidence JSON, outside the tree, by SHA (to be added by the phase-end run) | `phase_end.sh --phase 2 --sha f133c4b8… --execute --until 2` (the orchestrator) | – |
| 2 | Per-case table by database and host (check 1) | the phase-end run's `cases.md` and `seams.md` (to be added) | `scripts/rust_core/evidence.py` through `phase_end.sh`, with `SDM_INVENTORY=docs/design/rust-core/phase-0/phase-1b-inventory.yaml` | `evidence.py` reads a Phase 1b inventory and the local gate observes DuckDB legs only, so its table is DuckDB's, for 41 of the 52 locked cases. The SQLite and Postgres legs are judged in CI run 39577080078's per-database jobs (`contract-python (<cell>, <database>)`, `cross-host (<database>)`), all green |
| 3 | The refusal census (check 2) | [census/](census/): `duckdb.json`, `sqlite.json`, `postgres.json`, `sites.json`, [real-data.md](census/real-data.md) | `python -I contract/tools/refusal_census.py check --root .`; `... sites --root . > sites.json`; `<ref-pandas3 cell>/bin/python contract/tools/refusal_census.py run --reference <reference> --reference-python <reference>/.venv/bin/python --database D --out D.json`, for D in duckdb, sqlite, postgres | The real-data counts are zero by construction (T2.16a), held by `tests/test_census_real_data.py`; the owner's answer is KO2-90 |
| 4 | `coverage.py` (check 1) | [coverage/check.txt](coverage/check.txt), [coverage/s1.txt](coverage/s1.txt) | `python3 -I scripts/rust_core/coverage.py --check --repo . --rev be55838a… --checkpoint 9a899950…`; S1: [coverage/s1_coverage.py](coverage/s1_coverage.py), which calls coverage.py's own `generated()` over the S0 and S1 paths and every cell whose `expect` is not `unknown` | `coverage.py --gate S1` (rule 7) was never built; `ACTIVE` is still S0. The S1 listing shows 58 of 75 S0 and S1 paths and 22 of 44 in-scope cells with no acceptance case |
| 5 | The Postgres common-input comparison, H24 (check 1) | [h24/compare-verdict.txt](h24/compare-verdict.txt), [h24/diagnostics-by-rule.txt](h24/diagnostics-by-rule.txt), [h24/other-diagnostics.txt](h24/other-diagnostics.txt) | CI run 39577080078, job `cross-host (postgres)` (rust-host replaying python-core's recorded Postgres fetches, then `compare.py`): `gh run view 39577080078 --job 119158024692 --log` | No tool runs it locally; the comparison is CI's |
| 6 | L3 (check 6) | [l3/](l3/): the two runs and [classification.md](l3/classification.md) | `python3 -I contract/tools/differential.py run --reference-python <reference python> --candidate-python <ref-pandas3 env with be55838a's test-ops wheel> --seed 27 --n 200 --corpus generated --corrected --out DIR`, and the same with `--corpus selfref` | DuckDB only: the tool has no database axis, so L3 on SQLite and Postgres is not measured |
| 7 | Offline L7 (check 11) | [l7/l7.md](l7/l7.md), [l7/l7-rerun-r21.md](l7/l7-rerun-r21.md) | `contract/tools/l7.py measure --mode python-ref` (the reference's interpreter) and `--mode python-core` (an env holding CI's native wheel of be55838a), over `contract/cases/s1/*.json contract/cases/s0/*.json`, 5 repeats; `l7.py report --ref --core --observations <the reference line file> <the gate's python-core and rust-host> --wheel <CI's be55838a wheel> --baseline-wheel <CI run 38818504440's wheel of 91a00319>`; the rerun: the two scenarios over 1.1 and import time, 21 repeats | Measured under a load average of 40 to 55 on 8 vCPUs; see the note below |
| 8 | Unseeded SQLite sampling by properties (KO2-23, KO2-76) | [sampling/unseeded-pytest.txt](sampling/unseeded-pytest.txt) | `<ref-pandas3 cell>/bin/python -m pytest -v contract/tests/test_t211_unseeded.py`, with `SDM_POSTGRES_TEST_DSN` | – |
| 9 | The flip dry run, for information | [flip/](flip/): `forced-paths.txt`, `unforced-summary.txt`, `forced-summary.txt` | `SDM_CORE_ROUTES=<every path in _routing.CORE_ROUTED>` and unset, each `python -m pytest packages/structured-data-models-client/tests -m 'not live_nim and not live_warehouse' -n 2 --dist loadfile` in the ref-pandas3 cell | – |
| 10 | The held-out run (check 9) | to be added by the orchestrator's one-shot run | `phase_end.sh` step 3 (`heldout_run.sh`, `heldout.py counts`) | – |
| 11 | Conformance kit, L4, L5, cross-host (checks 3, 4, 5, 8) | CI run 39577080078 | land.py's exact-SHA rule (E131) | – |
| 12 | Reference updates (T2.0c, T2.0i), host ledger before and after (T2.14) | `tracker-T2.0c.md`, `tracker-T2.0i.md`, `tracker-T2.14.md` | the packages' own records | Not regenerated here |

## Results in brief

- Census: 41 sites, each with a probe (`census: ok`). The core refuses where the reference answers at one site, on DuckDB only: rule `aggregation`, a LIST_DISTINCT of a foreign key, kept until Phase 4 (P2E-19, KO2-75). On SQLite and Postgres, none.
- Kept rules on real data: `text_time_rules` 0, `model` 0, `aggregation` 0 hits, on all three databases (KO2-90).
- Coverage `--check`: ok. The S1 listing: 17 of 75 S0 and S1 paths and 22 of 44 in-scope cells have an acceptance case.
- H24: `compare: no differences`; 16,054 non-gating diagnostics: 9,396 `sql.text` and 6,623 `sql.parameters` (the core's Postgres SQL is its own), and 35 others ([h24/other-diagnostics.txt](h24/other-diagnostics.txt): registered outcome differences on `s0.boundary.*` cases and statement counts of registered early refusals).
- L3: generated, 200 sources: 166 agree, 26 under P1B-11, 8 classified by hand (7 under P2B-1, 1 under DT-15). Selfref, 9 sources: 4 agree, 5 under P1B-13. None unclassified.
- L7: median core-to-reference CPU 0.554 over 175 scenarios; native library 1.170 times 91a00319's (limit 1.25); boundary overhead, message sizes and no pandas in providers hold. At 5 repeats, two scenarios exceeded 1.1 (1.125 and 1.201) and import time was over (1,066 against 975 ms). At 21 repeats the same two read 1.022 and 0.989 and import time 1,131 against 1,151 ms, so we read the first figures as machine noise, not a regression. The 4 statement-count differences are registered refusals that stop before materialize's reads (P2I-4, and the two `s0.boundary.date-*` cases).
- Unseeded sampling: 9 of 9 pass on DuckDB, SQLite and Postgres.
- Flip dry run: unforced 2,789 passed; forced 118 failed of 2,789, all in three files: `test_the_time_column_is_inferred_alike.py` (61), `test_a_key_is_counted_where_its_rows_are.py` (38), `test_tables_stay_where_they_are.py` (19).
