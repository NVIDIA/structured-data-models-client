# Phase 1 tracker

Status values: not started, in progress, done, blocked.
Each row links its evidence when done.
Rows follow the execution order of [design.md](design.md), so T1.6b follows T1.11; the half column keeps the plan's 1a and 1b exits.
A task's checks are active from the moment it lands, as listed in the design's task table.

| Task | Half | Status | Evidence | Notes |
|---|---|---|---|---|
| T1.1 Phase documents and locked checks | 1a | done | Reviews R1 to R3 (subagent and Codex), design at ec37923 | Third, targeted review round, then implementation proceeds on the orchestrator decisions in [decisions.md](decisions.md); the owner confirms them later (D1) |
| T1.2 Workspace scaffold | 1a | done | c682f13, 3f38423, 8b9d5a4, e274372; CI run 37745535713 green: `rust-lint` (fmt, clippy `-D warnings`), `rust-deny` (advisories, bans, licenses, sources ok), `rust-test` (empty crates), existing jobs unchanged; hygiene covers `.rs` | The untracked root `uv.lock` was already gone and is now ignored; two pyo3 0.28 advisories ignored under E1; licences ISC, BSD-3-Clause and Zlib added (ASF category A) and the CDLA-Permissive-2.0 of `webpki-roots` allowed for that crate only (a build dependency of the test host's DuckDB); Rust jobs pin third-party actions by commit, as the organisation's allow list requires; a cold-cache `rust-lint` or `rust-test` takes about 20 minutes, mostly the bundled DuckDB build |
| T1.3 Zone and calendar vectors | 1a | done | 8aef27a; CI run 37788929024 green (`rust-test`: `sdm-host` 6 zone-vector and 6 ordered-lookup tests) | All 2,808 zone answers agree over the vendored 2026c directory and over the bundled 2026e database, each run checking its version first; 208 `DateOffset` answers agree; the probe's reading of 2,137 unit, value and PQL-literal vectors on `jiff` equals its `chrono` answers once instants are counted with civil arithmetic, because `jiff::Timestamp` stops at 9999-12-30T22:00Z (listed in dependencies.md); 4b39725 drops dependency debug info after the bundled DuckDB filled the runner's disk |
| T1.4 `sdm-contract` types, schema and lock | 1a | done | 23b0931; CI run 37788929024 green: `rust-lint` runs the regeneration and lock tests and `scripts/rust_core/contract_lock.py`; 22 `sdm-contract` tests (bounded proptest fuzz, 2,000 cases per property); `tests/test_rust_contract.py` (8 tests) in the Python jobs | Contract version 1 locked; bytes and Arrow travel as attachments named by id (E2); the three Phase 1b operations are provisional shapes that T1.11 refines with a version bump; doc comments are part of the schema, so editing one needs a bump |
| T1.5 `sdm-core` Run state machine | 1a | done | 1aae431; CI run 37788929024 green (22 `sdm-core` tests: 14 run-state, 8 dispatch, composition root, profiles and capability checks) | `validate_query` dispatches and resolves its profile, then refuses as unsupported until T1.6a puts the interim parser behind `sdm_core::pql`; the model capability checks run over `QueryFeatures` the adapter fills; the other three operations refuse as unsupported until Phase 1b |
| T1.6a Interim parser port and adapter | 1a | not started | | Only what T1.10 and T1.11 need gates; the corpus is reported; from the batch 1 review: the reference also refuses a temporal target with `MODEL_NO_TEMPORAL_TARGET` before resolution (`check_parsed_capabilities` in the PQL package's `validator/capability.py`), which `check_capabilities` in `profiles.rs` does not yet model; and the `key_columns_in_filters` and `timestamp_columns_in_filters` capability flags are parsed but consulted by no check |
| T1.7 `sdm-host` | 1a | not started | | |
| T1.8 `sdm-py`, the client wheel and CI rewiring | 1a | not started | | |
| T1.9 Python host and routing seam | 1a | not started | | |
| T1.10 First S0 case on every runner | 1a | not started | | Guard name list updated on `rust-core-reference` before any reference observation is recorded; records through `run_python.py` in `python-ref` mode until T1.14 lands `gate.sh` |
| T1.11 First DuckDB journey: `base.regression` | 1b | not started | | Must reach the result on `python-core` and `rust-host` with no old-engine delegation; diagnostic until T1.29; demonstrates the driver capture or records its infeasibility for the owner (Q2) |
| T1.12 S0 case corpus | 1a | not started | | |
| T1.13 S0 reference observations | 1a | not started | | Records through `run_python.py` in `python-ref` mode until T1.14 lands `gate.sh` |
| T1.14 Ratchet, register v2 and bootstrap checkpoint | 1a | not started | | The bootstrap checkpoint is independently reviewed once |
| T1.15 Coverage generator for S0 | 1a | not started | | |
| T1.16 L0 guards | 1a | not started | | From the batch 1 review: ban `jiff::tz::db` (the global database, which reads the environment) in clippy, as `zones.rs` already avoids it; `sdm-py`'s pyo3 bans (E1) landed early in the batch 1 review fixes |
| T1.17 Clean-container wheel smoke | 1a | not started | | |
| T1.18 L2 in CI and the status page | 1a | not started | | |
| T1.19 Local gate, seeded faults and comparator faults | 1a | not started | | |
| T1.6b Interim parser completion | 1a | not started | | Before T1.20 |
| T1.20 Phase 1a review | 1a | not started | | Completed independent review, blocking findings resolved; every Phase 1a check active |
| T1.21 DT-3 and ORD-1 findings | 1b | not started | | |
| T1.22 Random streams | 1b | not started | | |
| T1.23 `sdm-schema` and `describe_source` | 1b | not started | | |
| T1.24 `Schema.from_source` core route | 1b | not started | | |
| T1.25 Time module on `jiff` | 1b | not started | | `jiff` timestamps end at 9999-12-30T22:00Z: instants at or past it use civil arithmetic, and any refusal there needs a case and a register entry first (E6, P1B-1) |
| T1.26 Sampling | 1b | not started | | |
| T1.27 Labels | 1b | not started | | |
| T1.28 Wire and request | 1b | not started | | |
| T1.29 Three baseline cases | 1b | not started | | `base.regression` joins the acceptance corpus here at the latest |
| T1.30 DuckDB SQL goldens (L4 part) | 1b | not started | | |
| T1.31 Python conversion table (L5 part) | 1b | not started | | |
| T1.32 All 40 cases | 1b | not started | | |
| T1.33 Register entries | 1b | not started | | |
| T1.34 Type-set findings | 1b | not started | | |
| T1.35 Evidence pack | 1b | not started | | |
| T1.36 Phase 1b review, decision and merge | 1b | not started | | Completed independent review with blocking findings resolved, the owner's go or no-go, and the local merge on go |

## Batch 1 review fixes

The independent review of T1.2 to T1.5 (batch 1) found seven should-fix items and several nits, fixed on the phase branch:

- Zone names resolve only in their canonical spelling (E3, register DT-8 proposed for the owner, since the reference takes some other spellings); 4ce742a.
- The contract's encoder fails rather than write a scalar or column type its decoder refuses; column type decimals and zones are checked like scalars; finite float text that overflows is refused; the fuzz test gains unencodable scalars and a byte-damage property over the message corpus (E4); b8012ca.
- Reply and attachment ids use checked arithmetic, ending the run with `INTERNAL_ERROR`; the host clock must be in UTC; kind `cancelled` after cancel recorded (E5); 51c5e91.
- `sdm-py`'s clippy bans the pyo3 calls of the two ignored advisories, and the `deny.toml` comment no longer claims the GIL mitigates RUSTSEC-2026-0177 (E1); 270d519.
- `contract_lock.py` fails on an unreadable earlier lock and checks the lock at the merge base with `rust-core`; 418c426.
- The zone vectors tell a gap from a fold; `t06-vectors.json` records the probe's source checksums; 644b332.
- E2 says who numbers attachments in each direction, their scope and release, and how request state crosses runs; E6 records the `jiff` year-9999 rule; doc comments in the schema hash are an open decision; notes on T1.6a, T1.16 and T1.25; 7e916b6.

CI run 37793542647 on 7e916b6 green, every job; `rust-lint` ran the lock check against the merge base with `rust-core`.
