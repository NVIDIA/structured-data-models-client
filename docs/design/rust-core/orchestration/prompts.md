# Prompts for a new orchestrator session

Paste these into Claude Code one at a time, from `~/code/sdmc-rust-core` (a `rust-core` worktree), each after the one before has finished.
Replace `<...>` before pasting.

1. **Setup**: check the machine.
2. **Take over**: read the light set, save memories, report back.
3. **Watchdogs**: start them before any agent runs.
4. **Phase 2 design**: bring learnings.md up to date, write the design, review it until both reviewers approve, send the owner's questions. It stops there.
5. **Run Phase 2**: only after the owner has seen the design and answered.

The main session's context is the scarce resource. It reads the handbook, the plan and learnings.md itself, and gives everything bigger to helper agents that return short digests.

## 1. Setup

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
Read these in full: docs/design/rust-core/orchestration/README.md (the handbook) and its briefs/, then docs/design/rust-core/migration-plan.md and docs/design/rust-core/learnings.md. Skim orchestration/phase-1-log.md. Don't read Phase 1's own design, decisions or as-built notes yourself; they are 220 to 330 KB each, and the design step gives them to helper agents.

Then:
1. Save memories for what must survive this session: the owner's preferences and the hard rules from the handbook's "The owner" and "Hard rules" sections, one memory per rule with its reason; how a package runs and lands; and a project memory with the migration's state (Phase 1 merged into rust-core as 91a00319 on 2026-10-15; rust-core's CI red until T2.0; Phase 2 not started) and that the handbook is the starting point.
2. Report back in under 15 lines, in plain language: what the next phase is, how a phase starts, the five rules you consider most important, and anything in the handbook that looks out of date or contradicts the repository (check claims against the code; cite file:line).
```

## 3. Watchdogs

```text
Set up the watchdogs for this phase (session-only, nothing that outlives the session):
1. A silent background loop around scripts/rust_core/orchestration/stuck_agents.py and rule_watch.py on this session's task-output directory, which only notifies me when an agent is blocked or breaks a rule. When an agent is flagged, stop it and resume it with the blocking command named and an alternative.
2. A heartbeat every 3 hours: check for stuck agents, red CI on the phase and lane branches, queued landings and idle lanes, and act on what you find. If nothing needs doing, say so in one line.
3. A daily cleanup: scripts/rust_core/orchestration/clean_tmp.sh as a dry run first. Rerun it with --apply only if nothing it lists is still needed (the owner's standing approval, see the handbook's "The owner"); otherwise ask me before removing anything.
```

## 4. Phase 2 design

```text
Start Phase 2 with its design. Follow the handbook's "Every phase starts with its design" exactly; getting this design right matters more than speed. Read docs/design/rust-core/orchestration/phase-2-prep.md as starting material.

1. Create rust-core-phase-2 from rust-core, and the topic branch rust-core-phase-2-design from it for the design documents (never commit them directly to the phase branch).
2. Digests, in parallel, one helper agent each, every claim citing section, row or file:line: Phase 1's design.md (mechanisms to reuse, the "Explicitly out" list, the Host-held semantics ledger), decisions.md (rows in force for Phase 2, owner-approved rows, rules an agent must not break), as-built.md (what exists and where it differs from the design), register.yaml and not-yet-refusals.md (entries and refusals Phase 2 owns, E92's list), and the plan's "Carried into Phase 2" rows. One page each.
3. learnings.md (on the design branch): a helper agent drafts the lessons Phase 1 taught after checkpoint 8, its last, as lessons 17 onward in the format of 12 to 16, from the sources the handbook's step 0 lists (the final whole-phase reviews, the FR and FR2 rows, the held-out run, fix-round-history.md, phase-1-log.md and the handbook). Product and design lessons only.
4. Write phase-2/design.md, decisions.md and tracker.md on the design branch from the plan, learnings.md (with the new lessons) and the digests. Short; point at Phase 1 for everything unchanged.
5. Review: each round, a fresh Claude subagent and Codex (one pane per reviewer) review the design and the new lessons independently with briefs/design-reviewer.md. Fix every P1 and P2 and run another full round with both. Repeat until both approve the same SHA with no P1 or P2, then one closure check. Keep the rounds table in the tracker. Push the design branch after each round; its CI may be red until T2.0 lands, and the review doesn't wait on it.
6. While the design is in review: T2.0 (a separate test author, an implementer, two reviewers; it lands alone, first), the update from main and the reference rerun, and the speed-ups. Nothing that depends on Phase 2's scope; the new held-out set waits for the locked scope and T2.1's case format.
7. When the design is approved, stop and send me, in plain language: a one-page summary of the design, the review rounds and what they changed, the new lessons, and every owner question at once (the prep file's and any the design raised), each with your recommendation and what it costs either way. Keep step 6's work running while I answer; only product packages, the test-author batch and the held-out set wait for my answers.
```

## 5. Run Phase 2

```text
Record my answers in phase-2/decisions.md. If any changes the design, run one more review round with both reviewers on the change. Then lock the acceptance checks and land the design and the learnings.md update with land.py as a docs package. Run Phase 2 as the design and the handbook say: the test-author batch, T2.1 alone first, the new held-out set frozen once T2.1 has settled the case format, then the lanes stacked on T2.1.

Never sit idle waiting on CI or on me: when a question is mine, mark it owner-pending in the tracker and keep going on everything else. Give me a five-line check-in twice a day, and a plain-language note whenever you need a decision. Run design checkpoints against learnings.md as the design says, and add new lessons there as they come up.
```
