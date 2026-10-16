# Running a phase of the Rust core migration

This folder is the handbook for whoever orchestrates the remaining phases, person or agent.
It records how Phase 1 was run, what slowed it down and how to avoid it.
It's process material only and is deleted at the end of the migration, together with `scripts/rust_core/orchestration/`.

- [setup.md](setup.md): getting a new machine ready.
- [prompts.md](prompts.md): the prompts that start an orchestrator session.
- [phase-2-prep.md](phase-2-prep.md): Phase 2's work breakdown, schedule and the owner questions to settle at kickoff.
- [phase-1-log.md](phase-1-log.md): a short log of Phase 1.
- [briefs/](briefs/): the standing rules appended to every design reviewer, implementer, test author and reviewer brief.
- [`scripts/rust_core/orchestration/`](../../../../scripts/rust_core/orchestration/): the watchdog and landing helpers.

## Where everything else lives

| What | Where |
|---|---|
| The plan, its standing decisions and process rules | [migration-plan.md](../migration-plan.md) (sections 2, 8 and 9) |
| What went wrong in the first attempt | [learnings.md](../learnings.md); design checkpoints review against it |
| A phase's design, decisions, tracker and as-built notes | `phase-N/` (Phase 1: [phase-1/](../phase-1/)) |
| Every measured difference from the reference | [register.yaml](../register.yaml), checked by `scripts/rust_core/register.py` |
| Phase 1's leftovers for Phase 2 | migration-plan.md section 8, "Carried into Phase 2" |
| Landing, lanes, the gate, the ratchet | `scripts/rust_core/` (`land.py`, `lane.py`, `gate.sh`, `ratchet.py`, `prepush.sh`) |

Branches: `rust-core` is the epic branch. Each phase works on `rust-core-phase-N`, cut from `rust-core`, with lane branches `rust-core-phase-N-lane-<x>` and topic branches `rust-core-phase-N-<name>`.
`rust-core-reference` holds the reference (the old Python client with determinism fixes), checked out at `~/code/sdmc-reference`.
Nothing merges into `main`; the owner decides on the epic merge at the end of Phase 5.

## Roles

- **Orchestrator**: the main session. It plans, writes briefs, starts agents, reads reports and lands packages. It never writes product code itself.
- **Test author**: a fresh agent per package. It writes the acceptance tests and reference-recorded vectors first, in `<ID> tests: ...` commits. Owner rule: the agent that writes the code never writes its own acceptance tests.
- **Implementer**: a fresh agent per package. It makes the tests pass without editing them, pushes and reports its SHA. It never watches CI.
- **Two reviewers per substantive round**: a fresh Claude subagent and Codex (in a herdr pane, or headless `codex exec`). Both get the same brief and review the exact SHA.
- **The owner**: answers product questions, approves register entries reviewers can't, and merges each phase into `rust-core`.

## Every phase starts with its design

No package starts until the phase has a reviewed design and locked acceptance checks (migration-plan.md section 9). Phase 1 did this as its first task, T1.1 ("Phase documents and locked checks"), and so does every later phase.

**0. Bring learnings.md up to date first.** It is what every design and design checkpoint is judged against. It holds the first attempt's lessons (1 to 11) and Phase 1's up to checkpoint 8 (12 to 16). Before writing the next design, add what the finished phase taught that isn't there yet: for Phase 2, from Phase 1's final whole-phase reviews ([phase-1/tracker.md](../phase-1/tracker.md)'s T1.36 section, [as-built.md](../phase-1/as-built.md)'s T1.36 notes and [evidence/reviews.md](../phase-1/evidence/reviews.md)), the FR and FR2 rows of migration-plan.md section 8, the held-out run ([evidence/heldout.md](../phase-1/evidence/heldout.md) and [h11-classification.md](../phase-1/evidence/heldout/h11-classification.md)), [fix-round-history.md](../phase-1/fix-round-history.md), [phase-1-log.md](phase-1-log.md) and this handbook. (Phase 1 had no design checkpoint after 8.) Same format as lessons 12 to 16: what happened, what it cost, the rule that prevents it. Product and design lessons go there; process lessons stay in this handbook. Both reviewers review it.

**1. Read what the phase inherits.** The design is written from these, not from memory. Phase 1's design, decisions and as-built notes are 220 to 330 KB each, too much for the orchestrator to read whole: give each big source to a helper agent that returns a one-page digest of what the new phase inherits from it (rules in force, mechanisms to reuse, open items, traps), citing section or row numbers, and write the design from the digests. Read the plan and learnings.md in full yourself.

| Source | What to take from it |
|---|---|
| [migration-plan.md](../migration-plan.md) | The phase's row and exit (section 8), the slice it delivers (section 4), the gate (section 7), the standing decisions (section 2), the architecture (section 3), the process and stop rules (section 9) |
| [learnings.md](../learnings.md) | The first attempt's failure modes. Each risk in the design names the lesson it guards against, and design checkpoints review against this file |
| [phase-1/design.md](../phase-1/design.md) | The mechanisms the phase reuses (contract, runners, comparator, ratchet, gate, lanes, landing): cite them, don't restate them. Its "Explicitly out" list names what Phase 1 handed to later phases, and its "Host-held semantics" ledger lists the rules still to move into the core |
| [phase-1/decisions.md](../phase-1/decisions.md) | The standing rows the phase runs under (for example E25 and E29 on reviews, E43 on lanes, E49 on tooling, E92 on the named refusals to lift, E130 and E131 on landing and CI). Rows marked approved or confirmed by the owner stay in force unless the owner changes them |
| [phase-1/as-built.md](../phase-1/as-built.md) | What actually exists, and where it differs from the Phase 1 design |
| [register.yaml](../register.yaml) | One register for the whole migration. Phase 1's entries stay in force; the phase adds its own and never re-approves old ones |
| [phase-1/not-yet-refusals.md](../phase-1/not-yet-refusals.md) | Every named "not yet" refusal and which phase owns it |
| migration-plan.md section 8, "Carried into Phase 2" | Phase 1's deferred review findings and the owner's follow-ups (FX rows) |
| [phase-1/evidence/](../phase-1/evidence/) | The shape of the go or no-go evidence pack |
| This folder | The process lessons, [phase-1-log.md](phase-1-log.md), and the phase's prep file (for Phase 2, [phase-2-prep.md](phase-2-prep.md)) |

**2. Write the phase's documents** in `phase-N/`, on a topic branch (for Phase 2, `rust-core-phase-2-design`), never directly on the phase branch: every commit on the phase branch must be a landing, or the ratchet refuses later gates. Push the topic branch after each review round; its CI may be red until the phase's tooling package (T2.0) lands, and the review doesn't wait on it. The design is a package with an ID in land.py's pattern (`T<n>.<n>[a-z]`; for Phase 2, `T2.0a`), and its commit subjects start with that ID: the ratchet recognises a landing only by `Land T<n>.<n>...:` or `Land CI<n>:`, and a phase-branch landing under any other title breaks every later gate and can't be undone without a force push. Once the design is approved and locked: rebase it onto the phase head (T2.0 will have landed under it), get two APPROVEs on that exact SHA (`mech_confirm.py` for the clean rebase, plus a confirmation of the owner's-answers and lock commits), wait for green CI, and land it with `land.py` as a docs package (`--title 'T2.0a: ...'`, `--protected-paths "none (docs and evidence only)"`), together with the learnings.md update.
- `design.md`, with Phase 1's sections but only what's new: goal; scope and "explicitly out"; architecture choices new in this phase; the task table (each package with its output, exit check, dependencies, lane, and whether it changes a judging path); owner check-ins; the evidence pack; gates (merge rules, locked acceptance checks, process rules that differ from Phase 1); risks, each with the learnings lesson it guards. Phase 1's design grew to 220 KB; aim for a few pages that point at Phase 1 for everything unchanged.
- `decisions.md`, empty except for the owner's kickoff answers. New rows only; cite Phase 1's rows instead of copying them.
- `tracker.md`, one row per package, plus the stop-rule counters.

**3. Review it, and iterate until it's right.** Getting the design right matters more than any other step: Phase 1's design needed three review rounds and a closure check, and the first attempt's failures were mostly in its approach (learnings.md), not in individual lines of code. Each round, a fresh Claude subagent and Codex review independently with [briefs/design-reviewer.md](briefs/design-reviewer.md), which walks the design against the plan, every learnings.md lesson one by one, Phase 1 as built, and the locked checks. Fix every P1 and P2, then run another full round with both. Stop only when both approve the same SHA with no P1 or P2; then one closure check confirms the last round's findings are applied. Keep a short table of rounds and findings in the tracker. Don't shortcut these rounds to save time: a wrong design costs days, a review round costs an hour.

**4. Settle the owner's questions and lock the checks.** Send the open questions all at once, with a recommendation each. The owner's answers go into `decisions.md`. If an answer changes the design, run another review round on the change. Then the locked acceptance checks are marked locked, and any later change to them goes to the owner.

Work that doesn't depend on the design's scope may run while it's in review and while the owner answers: for Phase 2, T2.0, the update from `main` and the reference rerun, and the speed-ups. Anything that depends on the locked scope waits: for Phase 2, the test-author batch and the new held-out set.

## A package, start to finish

1. **Brief.** One file per package: scope, exits, the design row, decision rows, known findings to skip, and the standing rules from [briefs/implementer.md](briefs/implementer.md). Name every judging-path file it touches (see Landing).
2. **Tests first.** Start the test author with [briefs/test-author.md](briefs/test-author.md). The implementer may start from the brief in parallel; it must not read the test author's work in progress, and the tests arrive before its first push.
3. **Lane.** `scripts/rust_core/lane.py start --lane <x> --package <ID> --worktree <path> --wait`. Every commit subject starts with the package ID (`<ID>: ...`, `<ID> fix <n>: ...`, `<ID> tracker: ...`); `prepush.sh` and `land.py` refuse others.
4. **Push.** The implementer runs `scripts/rust_core/prepush.sh` (the cheap tier only, never `--full`), pushes, reports the SHA and ends its turn.
5. **Review and CI in parallel.** At push, start both reviewers with [briefs/reviewer.md](briefs/reviewer.md) and a background check of the exact SHA's CI run. Don't wait for CI before reviewing. Run `cancel_superseded.sh` after each push, since lane CI doesn't cancel superseded runs.
6. **Fix rounds** until both reviewers approve the same SHA. Reviews block only on P1 and P2; P3s go to a carry-forward list. Later rounds review only the fix commits plus the earlier repros.
7. **Land** with `land.py` (or queue it for `auto_land.py`), only on two `APPROVE` reports naming the SHA and a green exact-SHA CI run.
8. **Start the lane's next package** on a local branch stacked on the approved candidate as soon as both approvals are in. Never sit idle on CI.

## Ending a phase

1. **Close the open work**: every package landed or explicitly dropped, and dropped items closed out (pending entries retired, the decision recorded).
2. **Reconcile the register**: every measured difference has an entry, none is left proposed (unless the owner accepted a parked list at kickoff, as Phase 2's question G6(d) proposes), and `scripts/rust_core/register.py validate` passes.
3. **Run the locked checks and the held-out set** on the exact final SHA, and write the evidence pack to `phase-N/evidence/`.
4. **A fresh whole-phase review** by both reviewers. Its P1 and P2 findings are fixed, and its P3s are carried forward.
5. **Carry forward what's left**: unfixed P3s and owner follow-ups become named rows for the next phase in migration-plan.md section 8, as "Carried into Phase 2" does; `as-built.md` records what differs from the design; process lessons go into this handbook.
6. **The go or no-go packet**: a plain-language summary for the owner, with the evidence and anything that needs their decision. The owner decides, and the merge into `rust-core` is a `--no-ff` merge with a DCO sign-off.
7. **After the merge, check CI on `rust-core`** and the next phase's first run. Phase 1's merge left `rust-core` red, noticed only hours later while preparing Phase 2 (see [phase-2-prep.md](phase-2-prep.md), T2.0).

## Landing

- **land.py does all landings.** It makes an empty `Land <ID>: ...` commit with `Reviewed-by`, `Protected-paths`, `Base-SHA` and `Checkpoint-SHA` trailers and fast-forwards the phase branch in one atomic push. It refuses an empty `--protected-paths`; pass `"none (docs and evidence only)"` when nothing protected changed.
- **CI skips the landing commit** when the candidate's tree has green exact-SHA evidence (decision E131). `land.py verify <landing SHA>` stands on the candidate's run.
- **Moved landings (`--moved-from X`)**: an approved, green X rebased over landing commits only (same tree, the same patches one to one) lands on X's reviews and run, with no re-review. A move over another package's product commits is not a moved landing: it needs its own CI run and a confirmation.
- **Stack on the landing commit**, never on a landed candidate. Phase 1 lost a cycle when the ratchet refused a merge base that wasn't a landing commit.
- **Clean rebases are confirmed by script** (`mech_confirm.py`; decision E130). Reviewers look again only when the move changed something.
- **`land_chain.sh` queues one package at a time** and lands each only after the run of the package below it has completed.
- **Judging-path changes land alone, first.** If a package changes a file `gate.sh` takes from the checkpoint (`compare.py`, `run_python.py`, `ratchet.py`, `coverage.py`, `gate.sh`, `wheel_smoke.py`, the guards `bridge_freeze.py` and `l0_*.py`, `ci.yml`) and relies on that change for its own gate, it lands in two parts: the judging change alone, then the rest. Every guard or gate change must also pass in its post-landing state.

## CI

- **The four container images are private copies on ghcr.io** (`ghcr.io/nvidia/structured-data-models-client/...`, same digests), pulled with the workflow token. Anonymous registry pulls caused about 57% of Phase 1's failed runs before the move.
- **Every job has `timeout-minutes`.** A hung `rust-test` once ran 2h24m.
- **Dependencies build at opt-level 1**, so a cold `rust-test` takes about 34 minutes (it was over 2 hours).
- **Caches are branch-scoped.** A push restores caches only from its own branch and `main`, so a new topic branch starts cold. Prefer lanes. The Actions cache holds 10 GiB; `scripts/rust_core/prune_caches.py` prunes it, and pip caches save only on `main`.
- **External dependencies drift.** snowflake-connector-python 5.0.0 turned CI red mid-phase. When an unrelated job fails, check the dependency versions first.
- **Watch CI from the orchestrator only**, in the background: poll `gh run view <id> --json status,conclusion` under a timeout. Never `gh run watch`; it has hung agents.
- **The binding tests hard-code the contract version** in `rust/bindings/sdm-py/tests/messages.py`. Phase 1 missed it at two contract bumps (Phase 2 row FX-6 derives it). Until FX-6 lands, every contract bump updates it.

## Watchdogs and stalls

Phase 1 lost more time to stalled agents than to any single bug. The stall patterns and their fixes:

| Stall | Cost in Phase 1 | Fix |
|---|---|---|
| An approval prompt on `rm -rf`, `--force` or `sudo` | 9.5 h and 9 h | Briefs ban them; `rule_watch.py` flags them; the only cleanup allowed is `git worktree prune` |
| A turn that ended "waiting for CI" | 10.6 h overnight | Implementers report and end their turn; the orchestrator owns CI waits |
| A probe or fake `gh` that never returned | 17 min | Every probe runs `perl -e 'alarm shift; exec @ARGV' 600 <cmd> </dev/null` with `GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0` |
| A finished build that was never continued | hours | `stuck_agents.py` flags a tool call with no result after N minutes |
| A legitimately long build or gate | none, but false alarms | Treat `gate.sh`, `until` and `sleep` as expected, with a 60–90 minute ceiling |
| Codex finished and nobody noticed | 30 min+ | Every Codex prompt gets a background waiter in the same turn |
| The laptop slept | about 1 h per agent | Keep the machine awake (see setup.md) |
| Disk full, then a cleanup deleted a reviewer's scratch directory | a lost review | `clean_tmp.sh` is dry-run by default and never touches worktrees in use |

Run `stuck_agents.py` as a silent background loop that notifies only when something is blocked. A frequent cron that reports "all fine" just floods the orchestrator's context. When an agent is flagged, stop it and resume it with the blocking command named and an alternative.

`rule_watch.py` matches command text, so commands that merely mention `sudo` or `-D` (a grep, a brief) raise false alarms. Add acknowledged events to its ignore file rather than to the script.

Never edit a bash script while it's running. Bash reads scripts incrementally, so the running copy executes the edit half-way. Stop it, edit, restart.

## What the data says about reviews

- **Two reviewers catch real bugs.** In 111 rounds where both reviewed the same SHA, 40 had a blocker, and in 31 of those 40 (78%) one reviewer alone would have approved a P1 or P2. Codex alone would have missed 17; the subagent alone 22. Keep both on round 1 and on every fix round touching product or judging-path code.
- **Confirmation rounds almost never find anything.** All 20 paired confirmation, stack and tiny-fix rounds were approved by both. Use the script for clean moves and one reviewer for confirmations.
- **One Codex pane is a bottleneck.** It served three lanes and was queued 26 times. Run one per lane, or headless.
- **Fix rounds introduce new bugs.** The forecast move (T1.36 fixes 21 to 28) took four rounds. Review each fix round's own diff and rerun the earlier repros.
- **A check that fails twice the same way changes method**; it goes to the owner rather than getting a third patch.

## Where Phase 1's time went

82 landings over about a week, 6 to 14 a day. A package's median first-commit-to-landing time was about 4.5 hours; the tail was 8 to 11.
Implementers spent about 2 of their 3.3 active hours in local pytest, cargo and gate runs.
The day-killers were the tails: rebases across other lanes' product commits (about 0.6 per landing, 45 to 120 minutes each), red or cancelled CI (on 2026-10-14 only 35 of 106 runs were green), owner waits (one 6.7 hours, two overnight) and stalls.

The fixes, best first. Items 1, 3, 5 and 6 aren't built yet; they are Phase 2 prep (see [phase-2-prep.md](phase-2-prep.md)):

1. **A merge-queue flow.** Stack each approved, green package on the queue at approval time, so every later move is over landing commits only and lands without new CI. `land_chain.sh` reads its queue from a file.
2. **The CI fixes above** (private images, timeouts, opt-level 1, no CI on landing commits). Done.
3. **Split the slow CI jobs**: a separate `rust-host` job so `cross-host` stops waiting for `cargo test`, and shard `cargo test` and the contract tests, each shard checking that the shards add up to the whole. Target: a push in about 20 to 25 minutes instead of 43 to 88.
4. **Settle owner questions at kickoff**, all at once, plus two fixed check-ins a day. New questions found mid-phase are marked owner-pending and work continues around them (the owner's rule from Phase 1).
5. **A fast local loop**: keep each lane's Cargo target across landings, a shared sccache, one prebuilt wheel per phase head, warm venvs, `pytest -n auto`.
6. **One Codex per lane**, re-prompted automatically after capacity errors.
7. **Fewer, bigger landings.** One early judging-path landing, case families admitted in one landing when the reference observations are committed first. Splits cost 7 to 12% of lane time and caught nothing.

One day per phase holds about 15 to 20 landings on 3 to 6 lanes. Phase 1's size (70) doesn't fit in a day under any of these changes.

## The owner

How the owner works:
- **Plain language.** Start with a short summary that assumes no context. Explain internal IDs or leave them out, and give concrete examples. ID-heavy lists were hard to follow.
- **Process is never the blocker.** If a rule makes the right thing awkward, propose the lightest compliant route or a rule change, and say so plainly.
- **Honest, small estimates**, sized from the actual diff. "4–5 hours" for a small change drew "ridiculously long".
- **Parallelize, and never sit idle waiting on CI.**
- **Match the old client by default when it succeeds.** Be better than it where it crashes or errors: a named error beats an internal error. Every measured difference still gets a register entry, and changes to accepted answers or model inputs go to the owner one by one, with a recommendation.
- **One day per phase is the goal.** "We CANNOT take a full month for this migration."
- **Deletions only after a dry run**, with a backup and the owner's explicit approval. The one standing exception: `clean_tmp.sh`'s removal of stale agent scratch and build output, which the owner approved in Phase 1 (2026-10-12 and 13), may run with `--apply` after a dry run shows nothing still needed.
- **Close out dropped items** (retire pending entries, record the decision) so they make no noise later.

What only the owner decides: a new refusal of input the reference accepts, relaxing a comparison rule, adding an exclusion, a new acceptance capability, register entries the reviewers can't approve, a third change to a phase's locked checks, lifting a stop rule, and each phase's go and merge.
Never approve your own register entry or exception.

## Hard rules for every agent

- Never print, log, store or commit credentials, hostnames, account or workspace names, or usernames. Keep internal infrastructure names out of repo files. This repository is public.
- DCO sign-off only (`git commit -s`), with no `Co-Authored-By`. Never `git config` in a shared repo.
- Push only to phase, lane, topic and archive branches. No pull requests. `rust-core` moves only by the owner's merge (or the orchestrator's, on the owner's explicit instruction).
- No `rm -rf` or `rm -r` outside the session scratchpad, no `--force` or `-D` removals, no `sudo`, no `prepush.sh --full`, no `gh run watch`.
- Live Databricks work: upload only to the volume the owner authorized, run one-time jobs only, and never touch existing notebooks or clusters. Use only the database instance the owner names.
- Slack is read-only.
- Implementers never open held-out data. A held-out set is used once; Phase 1's is spent.

## Gotchas from Phase 1

- **The held-out bundle froze in an older case format.** Phase 1 converted it with `--pre-given-specs` (owner item 49). Phase 2 needs a new held-out set, frozen once the scope is locked and T2.1 has settled the case format, with a tested converter and a dry run of the held-out runner on a dummy bundle before the one-shot, and never opened by implementers.
- **New register and decision IDs collide across lanes.** Give each lane its own ID block (for example P2B-1..19 and E1..9 for lane a) and have lanes write only in their own sub-table or section of shared docs.
- **Shared docs conflict on rebase** (17 conflicts, mostly `tracker.md` and `decisions.md`). Each package writes only in its pre-created section.
- **A lane venv that imports the lane checkout** gives false "passes now" results. Use non-editable installs.
- **Pushes need an SSH key authorized for the organisation's SSO.** A push failing with "has enabled or enforced SAML SSO" means the key needs re-authorizing on GitHub.
- **The held-out H11 rule has no approved-entry exception for new refusals.** A new refusal found there fails the check, whatever the register says.
