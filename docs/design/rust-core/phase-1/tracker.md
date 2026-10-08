# Phase 1 tracker

Status values: not started, in progress, done, blocked.
Each row links its evidence when done.
Rows follow the execution order of [design.md](design.md); the half column keeps the plan's 1a and 1b exits.

| Task | Half | Status | Evidence | Notes |
|---|---|---|---|---|
| T1.1 Phase documents and locked checks | 1a | not started | | Owner confirms the orchestrator decisions in [decisions.md](decisions.md), the no-go criteria and the locked checks |
| T1.2 Workspace scaffold | 1a | not started | | Resolve the untracked root `uv.lock` first |
| T1.3 Zone and calendar vectors | 1a | not started | | |
| T1.4 `sdm-contract` types, schema and lock | 1a | not started | | |
| T1.5 `sdm-core` Run state machine | 1a | not started | | |
| T1.6 `sdm-pql-interim` and the adapter | 1a | not started | | |
| T1.7 `sdm-host` | 1a | not started | | |
| T1.8 `sdm-py`, the client wheel and CI rewiring | 1a | not started | | |
| T1.9 Python host and routing seam | 1a | not started | | |
| T1.10 First S0 case on every runner | 1a | not started | | Guard name list updated on `rust-core-reference` before any reference observation is recorded |
| T1.11 First DuckDB journey: `base.regression` | 1b | not started | | Risk reduction only; no early merge, and the 1a exit is unchanged |
| T1.12 S0 case corpus | 1a | not started | | |
| T1.13 S0 reference observations | 1a | not started | | |
| T1.14 Ratchet, register v2 and bootstrap baseline | 1a | not started | | |
| T1.15 Coverage generator for S0 | 1a | not started | | |
| T1.16 L0 guards | 1a | not started | | |
| T1.17 Clean-container wheel smoke | 1a | not started | | |
| T1.18 L2 in CI and the status page | 1a | not started | | |
| T1.19 Local gate and seeded faults | 1a | not started | | |
| T1.20 Phase 1a review | 1a | not started | | Fresh-agent review, at most two rounds |
| T1.21 DT-3 and ORD-1 findings | 1b | not started | | |
| T1.22 Random streams | 1b | not started | | |
| T1.23 `sdm-schema` and `describe_source` | 1b | not started | | |
| T1.24 `Schema.from_source` core route | 1b | not started | | |
| T1.25 Time module on `jiff` | 1b | not started | | |
| T1.26 Sampling | 1b | not started | | |
| T1.27 Labels | 1b | not started | | |
| T1.28 Wire and request | 1b | not started | | |
| T1.29 Three baseline cases | 1b | not started | | |
| T1.30 DuckDB SQL goldens (L4 part) | 1b | not started | | |
| T1.31 Python conversion table (L5 part) | 1b | not started | | |
| T1.32 All 40 cases | 1b | not started | | |
| T1.33 Register entries | 1b | not started | | |
| T1.34 Type-set findings | 1b | not started | | |
| T1.35 Evidence pack | 1b | not started | | |
| T1.36 Phase 1b review, decision and merge | 1b | not started | | Fresh-agent review of the whole phase, the owner's go or no-go, and the local merge on go |
