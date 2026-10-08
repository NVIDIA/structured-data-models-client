# Phase 1 tracker

Status values: not started, in progress, done, blocked.
Each row links its evidence when done.
Rows follow the execution order of [design.md](design.md), so T1.6b follows T1.11; the half column keeps the plan's 1a and 1b exits.
A task's checks are active from the moment it lands, as listed in the design's task table.

| Task | Half | Status | Evidence | Notes |
|---|---|---|---|---|
| T1.1 Phase documents and locked checks | 1a | not started | | Third, targeted review round, then implementation proceeds on the orchestrator decisions in [decisions.md](decisions.md); the owner confirms them later (D1) |
| T1.2 Workspace scaffold | 1a | not started | | Resolve the untracked root `uv.lock` first |
| T1.3 Zone and calendar vectors | 1a | not started | | |
| T1.4 `sdm-contract` types, schema and lock | 1a | not started | | |
| T1.5 `sdm-core` Run state machine | 1a | not started | | |
| T1.6a Interim parser port and adapter | 1a | not started | | Only what T1.10 and T1.11 need gates; the corpus is reported |
| T1.7 `sdm-host` | 1a | not started | | |
| T1.8 `sdm-py`, the client wheel and CI rewiring | 1a | not started | | |
| T1.9 Python host and routing seam | 1a | not started | | |
| T1.10 First S0 case on every runner | 1a | not started | | Guard name list updated on `rust-core-reference` before any reference observation is recorded |
| T1.11 First DuckDB journey: `base.regression` | 1b | not started | | Must reach the result on `python-core` and `rust-host` with no old-engine delegation; diagnostic until T1.29; demonstrates the driver capture or records its infeasibility for the owner (Q2) |
| T1.12 S0 case corpus | 1a | not started | | |
| T1.13 S0 reference observations | 1a | not started | | |
| T1.14 Ratchet, register v2 and bootstrap checkpoint | 1a | not started | | The bootstrap checkpoint is independently reviewed once |
| T1.15 Coverage generator for S0 | 1a | not started | | |
| T1.16 L0 guards | 1a | not started | | |
| T1.17 Clean-container wheel smoke | 1a | not started | | |
| T1.18 L2 in CI and the status page | 1a | not started | | |
| T1.19 Local gate, seeded faults and comparator faults | 1a | not started | | |
| T1.6b Interim parser completion | 1a | not started | | Before T1.20 |
| T1.20 Phase 1a review | 1a | not started | | Completed independent review, blocking findings resolved; every Phase 1a check active |
| T1.21 DT-3 and ORD-1 findings | 1b | not started | | |
| T1.22 Random streams | 1b | not started | | |
| T1.23 `sdm-schema` and `describe_source` | 1b | not started | | |
| T1.24 `Schema.from_source` core route | 1b | not started | | |
| T1.25 Time module on `jiff` | 1b | not started | | |
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
