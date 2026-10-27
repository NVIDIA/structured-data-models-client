# Running a phase of the Rust core migration

This is the operating handbook for the remaining migration work.
Updated 2026-10-24 from the owner's current instructions and the October 22–24 decisions.
It replaces the older process descriptions in this folder where they disagree.
The owner's latest instruction takes precedence over this handbook, older briefs and scripts.
A script that still implements an older policy is a defect to reconcile, not authority to restore that policy.
This migration-only material is removed when the migration ends.

## Current rules at a glance

- Write the package's tests before its implementation.
- Push the candidate and start both independent code reviews immediately.
- Run package tests, lint and cheap pre-push checks only after both reviewers approve the same SHA.
- If those checks fail, fix the cause, obtain both approvals on the new SHA, then run the checks again.
- Do not run full suites or full CI within a topic package.
- Full CI runs nightly on the phase head and at phase end.
- P0 and P1 findings block; P2 and P3 do not automatically block.
- Up to three product packages may run at once, subject to the machine's resource limits.
- Use short itemized designs and the existing tools; do not add process machinery or prose without a concrete need or the owner's request.

These rules supersede the old pre-review stamp, pre-review package tests, affected-suite checks, separate test-author requirement, tooling train, P1/P2 blocking bar and one- or two-package limits.
Writing tests first and executing the package checks after review are separate requirements.

## Where everything lives

| Material | Location |
|---|---|
| Architecture, phases and inherited decisions | [migration-plan.md](../migration-plan.md) |
| Product and process lessons | [learnings.md](../learnings.md) |
| Scope and approved phase decisions | `phase-N/` |
| Every measured difference from the reference | [register.yaml](../register.yaml) |
| Product and crate boundaries | [repository README](../../../../README.md), [Rust README](../../../../rust/README.md) |
| Landing, lanes, stamps and comparison tools | `scripts/rust_core/` |
| Agent and resource helpers | [orchestration tools](../../../../scripts/rust_core/orchestration/) |
| Machine setup | [setup.md](setup.md) |
| Historical Phase 1 account | [phase-1-log.md](phase-1-log.md) |
| Historical preparation and prompts | [phase-2-prep.md](phase-2-prep.md), [later-phases-prep.md](later-phases-prep.md), [prompts.md](prompts.md) |

Branch ancestry, current review reports and actual worktree changes establish what has landed or remains unfinished.
A tracker or old preparation document is not proof that work is still pending.
Keep account-specific settings, credentials and private run artifacts outside the repository under the configured state directory.
The private owner-answer record preserves the original words behind decisions; an agent summary or automated notification is not an owner answer.

## Branches and roles

`rust-core` is the final migration branch.
Package topics and lanes land into `rust-core-phase-N` through `land.py`.
The owner decides when a verified phase merges into `rust-core` and when the finished migration merges into `main`.
`rust-core-reference` is the separately installed Python comparison baseline; do not substitute moving `main` for it.
No pull requests, direct epic commits, tags or protected-branch pushes are part of ordinary package work.

The orchestrator coordinates scope, agents, reviews, resources, decisions and landings.
An implementer writes tests first, then the implementation, and reports a fixed candidate SHA for review.
A separate test author is optional under the current process; already independent acceptance tests and recorded reference values must not be weakened to make a change pass.
Two fresh Codex reviewers use `gpt-6.1-sol` with high reasoning and inspect the same fixed SHA independently.
One reviews the whole change, prioritizing design, contracts, duplication, boundaries and general correctness.
The other starts from the package's particular failure risks and relevant existing evidence.
These are different starting points, not limits on what either reviewer may flag.
Both receive the approved requirements and decisions, and neither sees the other's report before writing its own initial findings.
Do not prime either reviewer with the implementer's conclusions or a desired verdict.
The shared restrictions on test execution, live work and review scope still apply to both.
They use throwaway worktrees or `git show`, never the changing implementation worktree.
The owner answers user-visible behavior questions and gives the phase's go or no-go and merge decision.

## Design and repeated findings

Before a new phase, read the plan, current learnings, prior phase's as-built evidence and unresolved carried items.
Update lessons from completed work before designing the next phase.
Use helper digests for large historical documents instead of repeatedly copying them into every brief.
Keep the plan itemized, and each package design about a page: behavior, crate/module ownership, interfaces, tests, dependencies and unresolved owner choices.
Do not recreate the former large Phase 3 design or new tracker/decision prose unless requested.

Review designs independently against the language-neutral contract, DRY, separation of concerns, modularity and interfaces.
For the remaining Phase 3 work, a reviewed short design is sufficient to proceed under the owner's overnight delegation.
User-visible changes stay proposed until their authorized approver decides.
Locked acceptance changes and comparison relaxations remain owner decisions.

After two rounds on the same theme, or by round four, stop patching that theme and reassess its design.
Write down the old client's actual behavior, identify the missing rule or state assumption, choose a simpler design and resume.
The owner delegated that reassessment to the orchestrator on October 24.
Do not discard real findings merely to reduce the round count.

## A package, start to finish

1. **Brief and lane.** Name the package, its bounded scope, dependencies, approved design and known findings.
   Reserve its lane with `lane.py start`; every commit subject starts with that package ID.
2. **Tests first.** Commit the new tests before product implementation.
   Tests exercise public behavior, use both routes where applicable, and retain authentic reference observations.
   Reviewers verify the tests-first history.
3. **Implement and push.** Finish the code, push its candidate topic and report the exact SHA.
   Do not run package tests, lint or cheap pre-push checks before the reviewers approve it.
   Keep protected-branch and ownership enforcement active; reconcile an outdated hook rather than bypassing it with `--no-verify`.
4. **Review immediately.** Start both independent reviewers as soon as the SHA is available, with no stamp or CI prerequisite.
   Reviewers read the code, including tests, design boundaries and judging changes.
   A targeted investigation of a specific suspected defect is not a reason to launch package or broad suites during review.
5. **Fix and re-review.** Resolve P0/P1 findings and have both reviewers approve the same final SHA.
   Later rounds inspect the fixes and the earlier findings, rather than restarting the whole review.
   Address useful P2/P3 findings where practical and carry unresolved ones explicitly.
6. **Check once after approval.** Coordinate the cheap pre-push checks and `handoff.py run` as one bounded check of the approved SHA, restricted to lint and this package's own tests.
   Do not run overlapping checks twice or let the current pre-push script expand this into tool self-tests or broad clippy runs.
   Do not add affected suites, the full contract corpus, workspace tests, a reference rerun, a wheel build or full CI to the package check.
   If a check fails, fix it, return to both reviewers, and only then test the newly approved SHA.
7. **Land.** A clean approved SHA with its passing stamp lands through `land.py`, alone or in a stack.
   `handoff.py wait SHA` reports success or failure; the orchestrator consumes that result promptly.
8. **Continue.** Start the next package on the actual landing commit, preserving work still needed by dependent packages.

The automatic pre-push hook may still implement both older timing and wider scope, including tool self-tests and broad clippy checks.
Some standing briefs may also retain older severity rules.
Check their behavior before use and make a narrow reviewed correction when necessary.
Do not claim this handbook change alone changed those tools.

## Reviews and behavior differences

P0/P1 cover serious defects and normal-use failures, including wrong answers, crashes, lost support, misleading errors and checks that accept invalid evidence.
Rare sequences can be P2 under the owner's October 24 guidance; severity follows realistic use and impact.
Reviewers also inspect the contract, duplication, module ownership and language-neutral interfaces.

The new core should match successful old-client behavior or improve it.
A clearer error instead of an internal crash, or support for an input the old client rejects, can be an improvement.
Every measured difference still needs a register entry with its exact scope and approval source.
An agent must not silently approve its own change to accepted answers, model inputs, exclusions or comparison rules.
A recommendation, silence, a background event or a proposal marked "owner to confirm" is not approval.

Each review report names its fixed candidate SHA, verdict, findings and observed differences.
Do not wait for CI to begin or finish a package review.
A reviewer does not rerun suites as a routine review step.
When code suggests a defect, investigate only enough to establish that finding and explain why it was investigated.

## The handoff stamp

`handoff.py run --package <ID> --python <lane python>` runs after both approvals, in a clean worktree at that exact SHA.
Its intended scope is lint and the package's own added or changed test files, nothing broader.
Separate Python test roots as CI does; the engine suite cannot be collected with the other packages.
Use a named Rust test target rather than workspace-wide tests.
The stamp is valid only while the candidate and tree remain unchanged.
`handoff.py check SHA` reads it; `handoff.py wait SHA` reports completion.

A failed check leads to a fix and renewed review before the next check run.
Do not repeat checks between review rounds, or broaden testing merely because the candidate is waiting.
On an unchanged candidate already approved by both reviewers, continue only required groups that have not executed to collect remaining failures before editing.
Preserve their outputs, do not repeat groups that passed during collection, and never use a failed run as `--previous` stamp evidence.
After fixes receive both exact-SHA approvals, confirm the fast targets that failed before running the normal package handoff.
This changes neither the required checks nor the passing-stamp requirement.
The general setup and test commands in [CONTRIBUTING.md](../../../../CONTRIBUTING.md) describe available suites, not permission to run them all within a topic.

### Owner-approved T3.10 fixture repair scope

The default remains whole changed test files.
On 2026-10-25 the owner approved the one-time `--fixture-repair-scope t310-fixtures-7ab261a6d` exception for the exact T3.10 fixture sources pinned in `handoff.py`.
Decision row `T310-OWNER-2026-10-25-FIXTURE-SCOPE-1` records the actual approval of that scope and its source commit, which the tool requires before use.
Both exact-source reviews and required controls still precede execution.
The exception keeps the ten real factory controls, four lifecycle outcomes and two actual gate fixture consumers instead of repeating unchanged semantic scenarios in the pinned fixture-only files.
The complete changed coverage-checker tests, all other substantive own tests, normal lint and every final-source CI scenario remain required.
The flag is explicit, limited to T3.10, refuses any changed pinned source and records its exact suites, source hashes and approval-row digest in the log and passing stamp.
This exception supplies no failed-stamp reuse, automatic fixture classification, global pytest filter or change to full phase-end evidence.

## Landing

`land.py` owns phase-branch landings and their review, protected-path, base and checkpoint trailers.
It requires two approvals and the candidate's handoff stamp, not an exact-candidate CI run.
Judging changes land with the package that needs them and receive both reviews.
The separate tooling train and its train-only machinery were removed at the owner's request.

Up to four approved packages may stack, each with its own stamp, using `land.py --stack` and `land.py stack-land`.
This stack limit is distinct from the three active product-package limit.
Start dependent work on a landing commit, not on the unlanded candidate below it.
Use the existing mechanical confirmation tools only when they prove a move preserves the approved changes.
A conflict, changed patch or substantive merge needs the appropriate review and stamp on its final candidate.
Preserve recording provenance when integrating branches; do not rewrite evidence to conceal a changed source tree.

## Environments and resource limits

Use one isolated environment per lane through `cells.sh`; do not substitute the shared `.venv` for a recorded-build environment.
Export the current `SDM_PHASE_BRANCH`, `SDM_LANE_ENV` and kit interpreter explicitly.
A shell profile left on an old phase is not a reliable default.
Reuse environments while their installed sources and dependency key remain unchanged.

Heavy commands go through the existing `heavy` wrapper and its CPU, memory and disk admission checks.
Three product packages may run at once under the October 24 authorization.
The separate recording campaign does not create another product lane, but still consumes the same machine resources.
Do not raise concurrency or resource thresholds silently to get around a queue.

Rust builds share sccache but need distinct target directories for concurrent package/review work.
Two worktrees reusing incremental output can falsely reuse another worktree's binary.
A reviewer or package must use its assigned target and installed build.
Refresh the tools checkout only when it is clean; never overwrite another agent's pending edits or merge.

## Live work and reference recordings

The owner selected B-lite: no offline warehouse stand-in or new local warehouse emulation.
Provider tests use fakes fed recorded results; the real warehouse runs establish actual behavior.
Databricks, Snowflake and Lakebase comparisons use each leg's own reference observations.
Do not copy a DuckDB expectation over a different genuine warehouse outcome.

Live setup writes are limited to the authorized test area.
The client's actual query logic is read-only.
The write guard protects against accidental writes from our code, not arbitrary attack-style SQL; crafted bypasses outside plausible client behavior are not automatic P0 findings.
Restricted warehouse roles provide protection independently of the text guard.
Keep connection settings, private schema names and credentials outside committed artifacts.

A successful statement execution does not prove its results were recorded.
Capture fetch-time failures as well as execution failures, and reject result-producing queries with neither results nor an error.
Validate each new recording with the full `capture_check.py recording --locked` check before committing it; manifest-only validation does not establish query completeness.
State-changing reads such as `setseed` need an explicit narrow rule, not a broad unread-query exemption.
Do not save connection, capacity or other environment failures as the expected product answer.
Record clean source trees or preserve the explicitly permitted provenance explanation consistently.
Never edit recorded results or verified outputs by hand.

For Databricks, upload only to the authorized volume, use one-time jobs, and leave existing notebooks and clusters alone.
Run the scripted model endpoint where the job can actually reach it.
Verify access before leaving a long recording unattended.
The owner approved keeping Lakebase and the model endpoint running until the phase finishes.

## CI and phase-end evidence

Full CI runs nightly on the phase head with fresh dependencies, and at phase end.
It does not run on topic pushes and is not a package landing prerequisite.
The forced-core client suite belongs in that nightly matrix.
The orchestrator watches nightly results and turns failures into concrete fix items; a missing nightly also needs investigation.
Do not report a nightly as running or green without checking it.
Watch in the background with bounded `gh run list` or `gh run view`, never `gh run watch`.

At phase end:

1. Close or explicitly drop remaining packages and reconcile the register against actual owner and reviewer decisions.
2. Run `packet_checks.sh N` and inspect its findings; do not treat an obsolete per-package CI expectation as a new landing rule.
3. Build and run the final phase's required evidence from one named SHA using `phase_end.sh`, the locked gates and the runtime cells.
   This is the phase-end exception to the no-full-suites-in-topics rule, not another check on every package.
4. Cover the full applicable contract corpus, differential generator, SQL/provider checks, conversion tables, platform packaging and performance.
   For S2 and the switch, include live warehouses, the Databricks job matrix and required live timing.
5. Use a fresh held-out set once, after the ordinary evidence is ready.
   Implementers never open it, and a partially started one-shot run is not silently restarted.
6. Have both independent reviewers review the whole phase, map locked checks to actual evidence, and resolve blocking findings.
7. Give the owner a concise go/no-go packet: evidence, differences, limitations, remaining questions and the exact merge candidate.
8. Complete the final exact-tree CI and `final_gate.sh` evidence required by the phase's locked exit, then report ready to merge.
   Only the owner decides the phase merge into `rust-core`; do not push it on ordinary package authority.
9. After the owner's merge, inspect CI on `rust-core` and carry genuine remaining items into the next phase.

Record commands and artifact SHAs for evidence rather than hand-written claims of success.
If a required check was not run, say so plainly.
The old Phase 1 and Phase 2 held-out sets are spent.

## Watchdogs, approvals and cleanup

Subagents must not wait unattended on user approval prompts.
The orchestrator coordinates a permitted command form or surfaces the actual decision promptly.
Claude's existing permission hook rejects subagent prompts immediately and tells the agent to rewrite or report the command; a stall watcher catches commands with no active process.
Those session-specific hooks are not assumed to apply to another agent runtime.
Verify active processes and pending tool results before saying an agent is still working.

Use bounded commands with stdin closed and noninteractive Git/GitHub settings for probes.
A finished build, review or background task needs a continuation or report, not another idle status message.
Check stalls frequently and investigate a missing command after five minutes.
Never modify a shell script while it is running.

Cleanup starts with a dry run and preserves active processes, worktrees, recordings and unfinished merges.
The owner gave standing approval to remove old unused test temp files carefully, and separately approved the listed leftover cleanup.
Do not ask again for that same authorized scope.
Unrelated destructive cleanup still needs authorization.
Use the existing cleanup tooling where applicable and release a lane only after its artifacts are no longer needed.

## Hard rules and communication

- Sign off every commit and never add an agent co-author.
- Regenerate generated files with their generators; never edit them manually.
- Keep SPDX, repository hygiene, license and observable API/wire snapshot requirements.
- Preserve protected-branch enforcement and do not bypass hooks or policy with `--no-verify`.
- No Slack or other outbound messages without explicit authorization.
- Never expose credentials or private infrastructure in repository files or reports.
- Respect the existing command guard and owner-authorized exceptions; do not assume another runtime inherits Claude's hooks.
- Keep each full sentence on its own physical line in long Markdown and use en dashes, never em dashes.

Explain decisions in plain language, with a short example and recommendation when needed.
Carry forward existing authorization instead of asking the same question again.
Separate questions that genuinely need the owner from routine implementation choices and delegated design fixes.
Give estimates from remaining work and known review/recording risks, not from stale tracker dates.
