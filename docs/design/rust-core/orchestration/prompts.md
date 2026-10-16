# Prompts for a new orchestrator session

Paste these into Claude Code, one at a time, from `~/code/sdmc-rust-core` (a `rust-core` worktree).
Each waits for its predecessor to finish. Replace `<...>` before pasting.

## 1. Setup and verify

```text
You're taking over orchestration of the Rust core migration on a new machine. First, check the machine is ready; don't start any phase work yet.

Read docs/design/rust-core/orchestration/setup.md. Then check each item and report a short table of pass or fail with the fix for each failure:
- gh auth status (repo, workflow and package scopes), ssh -T git@github.com, and a dry-run push: git push --dry-run origin HEAD:refs/heads/rust-core-orchestration-check
- the Rust toolchain from rust/rust-toolchain.toml, uv, Python 3.10 to 3.14, Docker, jq, perl, codex and herdr
- the worktrees ~/code/sdmc-rust-core (rust-core) and ~/code/sdmc-reference (rust-core-reference), and the reference guard passing
- SDM_ORCH_STATE exists and SDM_PHASE_BRANCH is set
- one cold build: cd rust && cargo build --workspace (in the background; report the time)
- free disk space (at least 200 GB) and whether something keeps the machine awake

Things only I can do (logins, SSO, Databricks), list for me as "! <command>" lines I can run. Never print a credential, token, hostname or account name.
```

## 2. Take over

```text
Read, in this order: docs/design/rust-core/orchestration/README.md (the handbook), its briefs/ and phase-1-log.md, then migration-plan.md (all of it), learnings.md, phase-1/design.md (its sections and its "Explicitly out" list), phase-1/decisions.md's rows marked approved or confirmed by the owner, and the headings of phase-1/as-built.md and register.yaml.

Then:
1. Save memories for what must survive this session: the owner's preferences and the hard rules from the handbook's "The owner" and "Hard rules" sections, one memory per rule with its reason; and a project memory saying the migration's state (Phase 1 merged into rust-core as 91a00319 on 2026-10-15; Phase 2 not started) and that the handbook is the starting point.
2. Report back in under 15 lines, in plain language: what the next phase is, the five rules you consider most important, and anything in the handbook that looks out of date or contradicts the repository (check claims against the code; cite file:line).
```

## 3. Kick off Phase 2

```text
Start Phase 2. Read docs/design/rust-core/orchestration/phase-2-prep.md and the handbook's "Every phase starts with its design".

1. Create rust-core-phase-2 from rust-core. Write phase-2/design.md, decisions.md and tracker.md from the inputs the handbook lists (the plan, learnings.md, Phase 1's design, decisions, as-built, register and not-yet-refusals, the carried rows, and phase-2-prep.md). Keep it short and point at Phase 1 for everything unchanged.
2. Have both reviewers review the design until both approve. In the same turn, send me every owner question at once (the prep file's plus any the design raises), in plain language, each with your recommendation and what it costs either way. Record my answers in decisions.md and lock the acceptance checks. No product package starts before that.
3. Meanwhile, T2.0: parameterize the phase branch across ci.yml and scripts/rust_core, and bootstrap the first checkpoint (a separate test author, an implementer, two reviewers; it lands alone).
4. In parallel: the new held-out set, frozen in the current case format, where implementers can't read it; the test-author batch; the update from main and the reference rerun; the speed-ups (queue-driven land_chain, CI5, one Codex pane per lane, the fast local loop).
5. Then run the one-day plan: T2.1 alone first, then the lanes stacked on it.

Never sit idle waiting on CI or on me: when a question is mine, mark it owner-pending in the tracker and keep going on everything else. Give me a five-line check-in twice a day, and a plain-language note whenever you need a decision.
```

## 4. Watchdogs

```text
Set up the watchdogs for this phase (session-only, nothing that outlives the session):
1. A silent background loop around scripts/rust_core/orchestration/stuck_agents.py and rule_watch.py on this session's task-output directory, which only notifies me when an agent is blocked or breaks a rule. When an agent is flagged, stop it and resume it with the blocking command named and an alternative.
2. A heartbeat every 3 hours: check for stuck agents, red CI on the phase and lane branches, queued landings and idle lanes, and act on what you find. If nothing needs doing, say so in one line.
3. A daily cleanup: scripts/rust_core/orchestration/clean_tmp.sh as a dry run first. Rerun it with --apply only if nothing it lists is still needed; otherwise remove the listed items you're sure of one by one, with the owner's approval for anything outside the scratchpad.
```
