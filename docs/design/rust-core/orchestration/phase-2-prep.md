# Phase 2: preparation, plan and owner questions

Phase 2 is S1 offline: `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, including ranking and forecasting, plus the provider conformance kit, a minimal non-SQL test provider and a swapped data layer (migration-plan.md sections 4 and 8).
Its exit is the slice gate (plan section 7).
This page is a starting plan, drafted on 2026-10-14 from Phase 1's state and updated after Phase 1's merge. The kickoff turns it into `phase-2/design.md` and `phase-2/tracker.md`.

## The short answer

- **As written, Phase 2 takes about 1.5 to 2 days**: about 18 packages. It fits in one day only with the cuts in the owner questions below, and only if the prep is done before the clock starts.
- **What makes one day possible**: every owner question answered at kickoff; the contract and harness changes land first, as one package; test authors record the reference observations before the day; 5 or 6 lanes with one Codex per lane; the slice-gate chores (the update from `main`, the reference rerun, L3) start at hour 0.
- **Java has no slot** in any phase as planned. It needs its own track of about 12 to 15 packages (owner decision).

## Prep, before the clock starts

Do these first, mostly in parallel. None is product code.

1. **T2.0 Parameterize the phase branch, and bootstrap the first checkpoint.** 18 files hard-code `rust-core-phase-1`: `.github/workflows/ci.yml` (11 places, including `cancel-in-progress`), `scripts/rust_core/prepush.sh`, ten Python scripts beside it (`land.py`, `lane.py`, `ratchet.py`, `prune_caches.py`, `replay_jobs.py`, `bridge_freeze.py`, `l0_deps.py` and three self-tests) and six tests under `tests/`. Make the phase a single setting (for example `rust-core-phase-[0-9]+` patterns in CI and one constant or flag in the scripts). `grep -rn 'rust-core-phase-1' .github scripts tests` finds them.
   Two more things in the same package. The phase branch starts at `rust-core`'s head, which is a merge, not a landing commit, so `ratchet.py` ("the checkpoint ... is not a landing commit") refuses the gate for every early candidate: accept the phase branch's fork point from `rust-core` as the first checkpoint, with a self-test. And `land.py`, `ratchet.py` and `lane.py` accept only `T<n>.<n>[a-z]` and `CI<n>` package IDs, so Phase 2 packages are named `T2.<n>`.
   **CI on `rust-core` itself is red since the Phase 1 merge** (run 38818504440 on `91a00319`): off `rust-core-phase-1` branches the gate runs without the ratchet (`gate: no ratchet off the phase branch`) and `contract/tests/test_runners.py` holds outcomes to the reference, so every approved difference fails `rust-test`, `contract-python` and `cross-host`. T2.0 also gives the epic branch a ratchet base (for example, the merged phase's last landing as its checkpoint), so `rust-core` goes green again.
   And a flake to fix while there: `sdm-testkit`'s `duck::load` can read DuckDB's extension while a parallel test is still installing it (`json.duckdb_extension.tmp-...` "is not the one recorded"); install once before the tests run.
   It touches judging paths, so it lands alone, first, with two reviewers.
2. **A new held-out set**, frozen in the current case format before implementation starts, kept where implementers can't read it. Phase 1's set is spent.
3. **The speed-ups** from the handbook's "Where Phase 1's time went": the queue-driven `land_chain.sh`, CI5 (split `rust-host` out of `rust-test`, shard `cargo test` and the contract tests; it lands alone), one Codex pane per lane, and the fast local loop. About a day of work in parallel; each one pays back within the phase.
4. **The owner questions below**, all sent at once, in plain language, each with a recommendation.
5. **Test-author batch**: SQLite and Postgres base loaders, case variants, one or two cases for each lifted rule, all with reference observations. Extending the reference capture to ADBC SQLite and psycopg is a judging-path change and lands alone.
6. **The update from `main` and the reference rerun** start in prep, not at the gate.

## Work breakdown (about 18 packages)

M is mechanical (ported, with reference vectors), N novel, R risky.

| ID | Package | Kind | Estimate | Depends on |
|---|---|---|---|---|
| T2.1 | **Contract and harness part 1**: one version bump with paging (a continuation on fetch results), multi-column keys in `Neighbors`, `Lookup` and key sets (for `composite_key`), ranking and forecast result fields, expression columns, fallback limits as capabilities, `RequestState` pickle versioning, a backend axis in the case format with `malformed-cases.json` vectors | N | 2 h | prep |
| T2.2 | SQLite dialect in `sdm-data`, the Python ADBC provider returning Arrow, a dev-only rust-host provider over `rusqlite` | M | 4 h | T2.1 |
| T2.3 | Postgres dialect, the psycopg provider (Arrow without pandas), a rust-host provider (real driver or recorded results, Q2.2) | M/R | 4–5 h | T2.1 |
| T2.4 | The full L4 (goldens on all three databases) and L5 (pandas 2 and 3; the same tables through three drivers give identical Arrow) matrices | M | 3 h | T2.2, T2.3 |
| T2.5 | Conformance kit in `sdm-testkit`: fixture dataset, multi-page results, limits, expected answers, capability failure and retryability | N | 4 h | T2.1 |
| T2.6 | In-memory fallback for optional requests: ordering, row limits, memory and transfer limits that fail loudly (Phase 4's frames reuse it) | N/R | 4 h | T2.1 |
| T2.7 | A minimal non-SQL test provider passing the kit, and a swapped `DataLayer` running the same prediction with no domain edits | M | 2–3 h | T2.5 |
| T2.8a | **Ranking, materialize side**: `top_k`, `candidates`, the decomposition moved into `sdm-request` | N/R | 4–5 h | T2.1 |
| T2.8b | **Ranking, predict side**: scores and rankings, `prediction_statistic`, NaN ranks last (PD-1), parts put back together | N/R | 4 h | T2.8a |
| T2.9 | Forecasting: `num_forecasts`, `lag_timesteps`, quantiles in the result (the decomposition is already in the core since T1.36) | N/R | 3–4 h | T2.1 |
| T2.10 | PQL targets: other aggregations, `open_window`, `condition`, `logical_target`, `target_filter`, `static_target` | M | 4 h | T2.1 |
| T2.11 | Sampling options: `anchor_entity`, `context_anchor_time`, `exclude_cols_dict`, `num_hops`, `num_neighbors`, `max_pq_iterations`, `use_prediction_time`, `return_embeddings`, unseeded, `parent_table` (real data hits it) | M/R | 4–5 h | T2.1 |
| T2.12 | Schema rules: `composite_key`, `float_nan` (the lead item of plan section 8's carried rows), `float_signed_zero`, `key_tie`, `key_name_order` | R | 4–5 h | T2.1 |
| T2.13 | The query and option long tail: `entity_filter`, `entity_ids`, `assuming`, `evaluate`, `explain`, `target_transform_null`, `inference_config_text` | mixed | 3–5 h | T2.1 |
| T2.14 | S1 host clean-up: the response-bytes handoff, reference `Request` pickles loading (H23), ledger rows deleted, the bridge freeze lowered | M | 3 h | T2.8b |
| T2.15 | Coverage `--gate S1` and L3 seed sets wired into `gate.sh` (judging path; land it early) | M | 2 h | T2.1 |
| T2.16 | The slice gate: full local gate, L3, register reconciliation, evidence pack, phase review | M | 6 h | all |
| carried | Plan section 8's "Carried into Phase 2" rows (harness hygiene, host wording, FX-1 to FX-6), bundled into the lanes they touch rather than landed one by one | M | spread | — |

**Critical path**: prep → T2.1 → T2.8a → T2.8b → T2.14 → gate, about 19 to 20 hours with one fix round each. There's no slack for a second fix round on ranking, so give lane D the strongest implementer and both reviewers every round.

## A one-day schedule (6 lanes)

| Lane | H0–H4 | H4–H8 | H8–H12 | H12–H16 | H16–H24 |
|---|---|---|---|---|---|
| A (head) | T2.1, then T2.15 | T2.14 prep | T2.14 | gate support | T2.16 gate and review |
| B dialects | T2.2 | T2.3 | T2.4 | fixes | |
| C data layer | T2.5 | T2.6 | T2.7 | fixes | |
| D ranking | T2.8a | T2.8a fixes | T2.8b | T2.8b fixes | |
| E forecast and targets | T2.9 | fixes | T2.10 | fixes | |
| F options and schema | T2.11 | T2.12 | T2.13 | fixes | |
| Reference machine | update from `main`, reference rerun, SQLite and Postgres bases | gate dry run at about H8 | | full `gate.sh --locked` and L3 | |

Every lane starts on top of T2.1's candidate. A package moved over landing commits only lands on its own run without new CI. One moved over another lane's product commits needs a fresh CI run and a confirmation, so expect about one extra CI cycle per landing until the merge-queue flow (prep item 3) stacks approved packages in landing order.

## Owner questions to settle at kickoff

Send these together, each with the recommendation, and record the answers in `phase-2/decisions.md`.

**Process**
- **G1. What does "one day" mean?** 24 hours of wall clock with agents running overnight and prep outside the clock, or one working day? A working day needs about half this scope.
- **G3. Reviewers.** (a) one Codex pane per lane; (b) one reviewer for confirmations, docs-only and tiny fixes (under about 30 lines outside protected and judging paths), with the tier chosen by a script; (c) both. Recommended: (c). Two reviewers stay on round 1 and on every product or judging-path fix round.
- **G4. Checkpoints.** One design checkpoint mid-phase and one at the end, instead of every 4 or 5 packages.
- **G6. Classes reviewers may approve without the owner**, so nothing waits mid-phase: (a) a reference crash or internal error becomes a named error or a correct answer; (b) a success with an equal outcome; (c) a payload equal to the reference's; (d) anything else is parked, refused by name, and batched for the owner at the phase end, and the gate may pass with that list pending.
- **G7. Stop rules under parallel lanes.** A check failing twice the same way is reported, not waited on. Registry or quota errors that pass on rerun don't count as live failures.
- **G8. Admission in one landing.** When the reference observations are committed first, a new case enters as acceptance in the landing where it matches, instead of a diagnostic landing and then an admission landing.
- **G9. Short documents.** A design of 3 pages or fewer, decision rows only for owner-level choices, as-built notes for deviations only. Phase 1's files are 220 to 300 KB each.
- **Keep lane build directories across landings** (disk for speed), a cache-warming workflow on `main` for topic-branch caches, and the test author working in parallel with the implementer.

**Product**
- **Q2.1 Cases per backend.** Recommended: the 4 baseline cases, AdventureWorks, and the key, decimal, time and empty-or-null families on SQLite and Postgres (where drivers differ); the rest DuckDB-only. Bases loaded from the DuckDB-built tables through Arrow.
- **Q2.2 rust-host on Postgres.** A real dev-only driver (about half a day, risky) or a recorded provider replaying captured results. Recommended: recorded.
- **Q2.3 The dataframe-engine comparison at entry.** Phase 1 already built on plain Arrow. Waive it and record why, or run a 2-hour spike with a decision rule fixed in advance. Recommended: waive.
- **Q2.4 Which of the 35 named refusals (decision E92) to lift.** All in Phase 2, or the ones real data and the notebooks reach (`parent_table`, ranking, forecasting, aggregations, conditions, options), with the long tail (`evaluate`, `explain`, `assuming`, `return_embeddings`, `max_pq_iterations`, `key_name_order`) decided separately.
- **Q2.5 SQLite sampling.** The reference uses `ORDER BY RANDOM()` and ignores the seed. Sample seeded in the core (repeatable; better than the reference, registered), or compare SQLite on common inputs only.
- **Q2.6 Approve RS-6** (warehouse sampling SQL), which covers Postgres's `setseed`.
- **Q2.7 Deferrals.** ASan and LSan to Phase 5's L8; Postgres common-input comparison as information only; expression columns to Phase 3 if only metric views need them.
- **Q2.8 L3** as `contract/tools/differential.py` plus agreed seed sets (for example 200 per backend) against the reference, with H11's threshold, rather than a new tool.
- **Q2.9 Accuracy benchmark** only when a ranking or forecast payload differs from the reference's.
- **Java.** Its own track (about 12 to 15 packages), or deferred. It doesn't block Python.

## Risks that would break the day

| Risk | Mitigation |
|---|---|
| Ranking's decomposition needs two or more fix rounds | Split into T2.8a and T2.8b, tests and vectors written in prep, the result shape fixed in T2.1 |
| `composite_key` touches key sets, the graph and the sampler | Its contract shape goes in T2.1; if it slips it stays a parked refusal under G6(d) |
| Six lanes saturate CI | The private images and skipped landing runs are in place; stack approved packages; CI5 |
| One Codex pane serializes about 60 reviews | One pane per lane (G3) |
| Contract bumps collide across lanes | All new vocabulary goes in T2.1; anything later is renumbered by the second lane to land |
| The reference machine is slow | Start the rerun and base loading in prep; dry-run the gate at about H8 |
| `evaluate` or `explain` turn out large | Q2.4 lets them be cut without blocking the gate |
| The update from `main` changes behavior | Do it in prep; each new difference must trace to a deliberate commit on `main` |
