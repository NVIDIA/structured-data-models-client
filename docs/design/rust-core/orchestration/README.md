# Running a phase of the Rust core migration

This folder is the handbook for whoever orchestrates the remaining phases, person or agent.
It records how Phase 1 was run, what slowed it down and how to avoid it.
It's process material only and is deleted at the end of the migration, together with `scripts/rust_core/orchestration/`.

- [setup.md](setup.md): getting a new machine ready.
- [prompts.md](prompts.md): the prompts that start an orchestrator session.
- [phase-2-prep.md](phase-2-prep.md): Phase 2's work breakdown, schedule and the owner questions to settle at kickoff.
- [later-phases-prep.md](later-phases-prep.md): the same for Phases 3 to 5, plus the proposed changes to the plan's checks and the Java track.
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
- **Two reviewers on every round**, confirmations and docs included, and a round passes only when both approve the same SHA (Phase 2 decision KO2-9, the owner's rule): a fresh Claude subagent and Codex (in a herdr pane, or headless `codex exec`). Both get the same brief and review the exact SHA.
  Codex in a herdr pane doesn't notify the orchestrator when it finishes, so every prompt gets a background waiter in the same turn:
  `herdr --session <s> agent prompt <pane> "<brief, naming the report file>"`, then, run in the background, `sleep 5; herdr --session <s> agent wait <pane> --timeout 3000000 >/dev/null 2>&1; ls -la <report file>`.
  If the wait returns with no report (it can return before Codex starts working), check `herdr --session <s> pane list` and re-arm it. Re-prompt after a capacity error.
- **The owner**: answers product questions, approves register entries reviewers can't, and merges each phase into `rust-core`.

## Every phase starts with its design

No package starts until the phase has a reviewed design and locked acceptance checks (migration-plan.md section 9). Phase 1 did this as its first task, T1.1 ("Phase documents and locked checks"), and so does every later phase.

**0. Bring learnings.md up to date first.** It is what every design and design checkpoint is judged against. Since Phase 3's T3.1 it holds 20 lessons, product and process together (Phase 3 decision KO3-10), each with an "Enforced by" line that `tests/test_learnings_enforced.py` checks; its last table maps the lesson numbers used before then. Before writing the next design, add what the finished phase taught that isn't there yet; Phase 2's design followed this list: for Phase 2, from Phase 1's final whole-phase reviews ([phase-1/tracker.md](../phase-1/tracker.md)'s T1.36 section, [as-built.md](../phase-1/as-built.md)'s T1.36 notes and [evidence/reviews.md](../phase-1/evidence/reviews.md)), the FR and FR2 rows of migration-plan.md section 8, the held-out run ([evidence/heldout.md](../phase-1/evidence/heldout.md) and [h11-classification.md](../phase-1/evidence/heldout/h11-classification.md)), [fix-round-history.md](../phase-1/fix-round-history.md), [phase-1-log.md](phase-1-log.md) and this handbook. (Phase 1 had no design checkpoint after 8.) Same format as the lessons there: what happened, what it cost, the rule that prevents it, and what enforces it. Process lessons go there too; this handbook keeps the operating detail. Both reviewers review it.

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
4. **Push.** The implementer runs its package's own tests and `scripts/rust_core/prepush.sh` (the cheap tier only), pushes, reports the SHA and ends its turn.
5. **Review at push.** Start both reviewers with [briefs/reviewer.md](briefs/reviewer.md) the moment the implementer reports the SHA; a review needs no handoff stamp (the owner's "2026-10-23 stamp only at landing"). No CI runs on a lane or topic push (the owner's "2026-10-22 evening: CI off the landing path"); the full matrix runs nightly on the phase head and at the phase's end.
6. **Fix rounds** until both reviewers approve the same SHA. Reviews block only on P1 and P2; P3s go to a carry-forward list. Later rounds review only the fix commits plus the earlier repros.
7. **Stamp and stack it.** As soon as both reviewers approve the same SHA, run `handoff.py run` on it once (see "The handoff stamp" below), then stack it with `land.py --stack` when `handoff.py wait <SHA>` returns 0 (or queue a `"stack": true` entry for `auto_land.py`), and land the stack with `land.py stack-land` on every package's own handoff stamp (see "Stacked landings" below).
8. **Start the lane's next package** on the package's landing commit as soon as it is stacked.

## Ending a phase

Three scripts in `scripts/rust_core/orchestration/` carry the mechanical part. Each prints `--help`, and the two that change anything are dry runs unless given `--execute`.

- `packet_checks.sh N`: read-only. It lists landings with no local gate evidence for their exact SHA, migration-plan.md markers that disagree with `phase-N/decisions.md`, rows still "owner to confirm", the register's statuses (proposed entries by id) and the owner approvals since the last check-in. Run it when the phase-end work starts and again on the final head.
- `phase_end.sh --phase N --sha SHA`: the evidence runs from one final SHA. In order: preflight (read-only), build (the local client wheel into every runtime cell, plus CI's native wheel), evidence (`gate.sh --locked` with `--out`, the other cells' runs, `cargo tree`, `tokei`, and `evidence.py` without the held-out bundle, so a refused input shows up before the one shot), held-out (`scripts/rust_core/heldout_run.sh`, once), an optional live install check (the command in `SDM_LIVE_CHECK_CMD`, skipped when unset), and pack (staged for `phase-N/evidence/`). Each step is stamped when done, so a rerun resumes at the failed step. A one-shot step that started and didn't finish stops every rerun: a person decides. Cells, tools, the held-out bundle and anything account-specific come from environment variables, never from the script.
- `final_gate.sh WORKTREE PYTHON CI_RUN HALF...`: fetches the exact-SHA CI wheels, runs `gate.sh --locked` for each half named, and checks that the one evidence file (`<git-common-dir>/sdm-gate-evidence/gate/<sha>.json`) shows them all passed.

1. **Close the open work**: every package landed or explicitly dropped, and dropped items closed out (pending entries retired, the decision recorded). `scripts/rust_core/lane.py status` shows no lane holding an unlanded package. The tracker's stop-rule counters are current, and any counter at 2 has gone to the owner. Send the phase's check-in reports that are still drafts, or fold them into the packet, so the owner doesn't see them for the first time at the decision.
2. **Reconcile the register**: every measured difference has an entry, none is left proposed (unless the owner accepted a parked list at kickoff, as Phase 2's question G6(d) proposes), and `scripts/rust_core/register.py validate` passes. `packet_checks.sh N` section D must show 0 proposed: the held-out run counts only entries approved before it starts.
3. **Run the locked checks and the held-out set** on the exact final SHA, and write the evidence pack to `phase-N/evidence/`:
   - `phase_end.sh --phase N --sha SHA` (dry) until preflight is clean, then `--execute --until 2`: everything before the one shot.
   - Then `--execute` again for the held-out run, the live check and the pack. If preflight says the gate's runner files differ from the checkpoint's, land an empty commit E on SHA and run on E with `--base SHA --landing SHA`.
   - By hand afterwards: fill the held-out report's "approved before the run" column only from the register snapshot taken at run start; write `evidence/README.md`, naming the command that made each item; copy the staged pack into `phase-N/evidence/` in the phase-end package. The gate's evidence JSON stays outside the tree, and the pack cites it by SHA.
   - Check every pack item against the design's evidence list, one at a time. Each must be generated by the branch's own runs, not written by hand. Anything not run (an architecture with no CI wheel, a runtime not tried, a column not measured) goes on the "not done" list.
4. **A fresh whole-phase review** by both reviewers (`origin/rust-core..HEAD`, emphasis on the work since the last design checkpoint). Its P1 and P2 findings are fixed, and its P3s are carried forward. Both briefs cover:
   - each locked check mapped to evidence on the merge SHA;
   - the held-out verdict computed as the design wrote it;
   - every register entry approved, rejected or owner-pending with a named question, with no implementer approving its own entry and every `approved_by: owner` citing the owner's written answer;
   - the boundary (L0 report clean, no `pyo3` in the pure-Rust host's `cargo tree`, emulation allowlist changed only through owner rows);
   - landing commits trailer-only, with `Protected-paths` naming every protected change;
   - test-author files untouched by implementers;
   - docs whose statuses are true.
5. **Carry forward what's left**: unfixed P3s and owner follow-ups become named rows for the next phase in migration-plan.md section 8, as "Carried into Phase 2" does; `as-built.md` records what differs from the design; process lessons go into this handbook. Fix stale "pending owner confirmation" markers in migration-plan.md that `packet_checks.sh` section B flags.
6. **The go or no-go packet**: a plain-language summary for the owner, with the evidence and anything that needs their decision. Keep it to one page plus links:
   - the decision asked for (go, no-go, or re-plan);
   - the executor's recommendation, labelled as such;
   - each no-go criterion with its result;
   - the held-out counts against the threshold;
   - the per-case table summary;
   - the register totals and what the owner approved;
   - what wasn't done or isn't proven;
   - the rows confirmed by default, with at most five flagged where the owner might choose differently;
   - landings with no local gate evidence (`packet_checks.sh` section A);
   - how to merge.

   The owner's go or no-go goes into `phase-N/decisions.md` in their own words, as a docs commit before the final candidate. Then:
   - Push the candidate and wait for green CI on every job. Rerun a registry rate-limit failure; never skip it.
   - Run `final_gate.sh <clean worktree at the candidate> <python with this SHA's local build> <candidate's run> --base <phase tip at start> --execute <halves>`.
   - Land with `land.py` once both reviewers confirm.
   - Run `final_gate.sh` again on a clean worktree at the landing commit, with the candidate's run (same tree). That evidence file is the merge-SHA evidence the packet cites.
   - Check CI's runtime-cell jobs on the landing's tree.

   Then stop and report "ready to merge" with the SHA, both evidence files and the CI run IDs. Agents never merge, push `rust-core` or tag. The owner decides, and the merge into `rust-core` is a `--no-ff` merge with a DCO sign-off.
7. **After the merge, check CI on `rust-core`** and the next phase's first run. Phase 1's merge left `rust-core` red, noticed only hours later while preparing Phase 2 (see [phase-2-prep.md](phase-2-prep.md), T2.0).

## Landing

- **land.py does all landings.** It makes an empty `Land <ID>: ...` commit with `Reviewed-by`, `Protected-paths`, `Base-SHA` and `Checkpoint-SHA` trailers and fast-forwards the phase branch in one atomic push. It refuses an empty `--protected-paths`; pass `"none (docs and evidence only)"` when nothing protected changed.
- **A landing stands on its handoff stamp and two APPROVE reports** (the owner's "2026-10-22 evening: CI off the landing path"); no CI run on the candidate is required. The landing commit's `Handoff-stamp: <SHA> <time>` trailer records the stamp, and `land.py verify <landing SHA>` checks it; a landing made before the rule, with no such trailer, is still verified on its CI run.
- **Moved landings (`--moved-from X`)**: an approved, stamped X rebased over landing commits only (same tree, the same patches one to one) lands on X's reviews and stamp, with no re-review. A move over another package's product commits is not a moved landing: it needs its own stamp and a confirmation.
- **Stack on the landing commit**, never on a landed candidate. Phase 1 lost a cycle when the ratchet refused a merge base that wasn't a landing commit.
- **Clean rebases are confirmed by script** (`mech_confirm.py`; decision E130). Reviewers look again only when the move changed something.
- **Stacks, not chains.** Up to 4 approved packages land together, each on its own stamp (`land.py --stack`, then `stack-land`; "Stacked landings" below). `land_chain.sh` lands a chain one package at a time, each on its own stamp.
- **Judging-path changes land with their package.** A package may change a path in `scripts/rust_core/judging-paths.txt` and product code together and land normally; both reviewers review all of it (T3.24, the owner, 2026-10-23: "The train rule - sure delete it, this will save time."). See "The judging paths" below.
  Every guard or gate change must also pass in its post-landing state.

## The Phase 3 process mechanisms (T3.0)

Phase 2 took 131 hours against a 50-hour plan; the root-cause analysis found that the rules which held were the ones a script enforced.
T3.0 turns the owner-approved process changes (2026-10-21, evening) into tools.
Nothing here stops work because a time or count was reached.

### Stacked landings (item 2)

Up to 4 approved packages land together, each on its own handoff stamp, made for that package (the owner's "2026-10-22 evening: CI off the landing path" replaced the green CI run on the top).

1. When both reviewers approve package P1's SHA, stack it from its worktree: `land.py --stack --candidate SHA --topic-branch <its branch> --review A --review B --title 'Land P1: ...' --protected-paths '...'`.
   It makes P1's empty landing commit locally: `Base-SHA` and `Checkpoint-SHA` the landing below it (or the phase head), `Stack-base` the phase head, and `Topic-branch`.
   It needs the package's own stamp, recorded as its `Handoff-stamp` trailer, and no CI run; it pushes nothing, and prints `STACKED <sha>`.
2. The next package is built (or rebased) on that landing commit, reviewed, and stacked the same way.
   `lane.py check` and the pre-push hook accept the stack below a lane's package.
   `--max-stack N` or `SDM_STACK_MAX` lowers the limit of 4, which CI's ratchet holds.
3. Run `land.py stack-land --tip <the top landing> [--topic-branch <the top's branch>]`.
   It checks the whole chain again, each package's stamp included, and fast-forwards the phase branch over every landing in one atomic push.
   Each landing's checkpoint stays the landing before it.
4. If a package has no stamp, or a stamp made for another package, `stack-land` refuses naming it and `handoff.py run`; a prefix whose packages are stamped lands with `land.py stack-land --tip <that landing>`.
5. If the phase head moved under a stack, `land.py restack --tip <top landing> [--move-branches]` replays it on the new head in a scratch worktree. Each patch must be the same (patch-id, `move_confirm.py`'s rule), so the reviews carry over; each moved package needs a new stamp, since `restack` drops the old trailer. A conflict or a changed patch refuses, and that package needs a review round. A single approved package moved before stacking is still confirmed with `move_confirm.py`.

`land.py verify` of a stacked landing stands on its own stamp. `auto_land.py` does steps 1 and 3 from queue entries: `"stack": true` and `{"stack_tip": SHA, ...}`.

### The judging paths (item 1)

`scripts/rust_core/judging-paths.txt` lists the judging paths: the files that decide pass or fail (the gate, the ratchet, the comparator, coverage, the census, `land.py` and `lane.py`, the case and observation readers and their schemas).
What product packages change as normal work is left out: the bridge freeze and its seeds, `wheel_smoke.py`, the recorders and probes under `contract/tools/`, vectors, cases and `ci.yml`.
`land.py` reads it from the phase head, so a package cannot take its own change off the list.
Everything outside the list (and outside tests, docs and the orchestration scripts) is product code, which the commit-msg rule's `Rule:` line covers in a fix commit (item 4).
A package may change judging paths and product code together and land normally, alone or in a stack; both reviewers review everything it changes.
The tooling train (a separate `--train` package, first in its stack, for every judging change after the phase's first package) is gone: it forced three mechanical splits in Phase 3 that caught nothing (T3.24, the owner, 2026-10-23).

### The handoff stamp (item 4)

Implementers and test authors run their package's own tests before pushing; reviews start at push, with no stamp.
Once both reviewers approve the same SHA, `scripts/rust_core/handoff.py run --package <ID> [--python <lane environment python>] [--previous <last round's SHA>]` runs on it once, in the lane worktree, with the tools' interpreter (the owner's "2026-10-23 stamp only at landing").
On a clean worktree it runs two steps, printing each as it starts and ends with its time, and stops at the first that fails (the owner's "2026-10-23 cut the tooling back"):

1. lint: `ruff`, `cargo fmt`, `mypy --python-executable <lane python>` on changed packages, `l0_scan.py`, the hygiene check, and `cargo deny --locked check` when `rust/` changed;
2. the package's own tests: each test file it adds or changes, one pytest session per test root as CI runs them (a crate test by its own `--test`).

The stamp, with the two APPROVE reports, is what a landing needs; no CI run on the candidate is required (the owner's "2026-10-22 evening: CI off the landing path").
The full CI matrix runs nightly on the phase head with fresh dependencies, the client's suite with every route forced onto the core among it, its failures becoming fix items, and at the phase's end as its final gate, debugged from the lowest module up.

In a fix round whose `--previous` SHA has a stamp, step 2 takes only what changed since it.
Every command it starts reads `--python` as `SDM_PYTHON`.
It writes `<git common dir>/sdm-handoff/<SHA>.json` only when both pass and HEAD and the tree are unchanged; `handoff.py check SHA` reads it.
Each start, stamp, failure and stop is a line of `<git common dir>/sdm-handoff/events.jsonl`, and `handoff.py wait SHA` returns the moment the SHA is stamped (exit 0) or its run fails or stops (exit 1): stack the package on its exit 0; a failure is a fix round.
While a run lives it holds `sdm-handoff/running/<ID>.json`; the pre-push hook refuses a push from the package's worktree until the run ends (`handoff.py running --package ID` names it), and a second run of the package is refused.
New findings meanwhile go into a local next fix, pushed after the round's report.
`land.py`, `land.py stack-land` and `auto_land.py` refuse a SHA without a stamp; `codex_pane.py review` and Claude reviewer launches need none.
Reviewers read the code, and list every difference from the old client under `differences:`, each better or worse; only a worse one blocks (the owner, G2; [briefs/reviewer.md](briefs/reviewer.md)).
`affected.py --unmapped` fails when a tracked path under `contract/`, `scripts/rust_core/` or `rust/crates/` maps to no suite; add a row to its `MAP`.

A fix commit (`<ID> fix <n>: ...`) that changes product code carries a line `Rule: <the old client's rule, one sentence>`.
`scripts/rust_core/commit_msg.py` checks it as the commit-msg hook (`prepush.sh --install-hook`, run by `lane.py start`), and `land.py` checks every commit it lands.

### The command hook (item 6)

`.claude/settings.json` runs `scripts/rust_core/orchestration/guard_cmd.py` before every Bash call of a Claude agent started in the repository.
It denies the banned commands, naming the rule and the alternative: `rm -r` (and `find -delete`) outside your scratch (the session's directory under `/tmp/claude-*`, a directory in `SDM_SCRATCH`, or a directory under `/mnt/localssd/tmp/` or `/mnt/localssd/target/` the session made itself with `mkdir`; another agent's needs the owner's token), a `git push` that updates or deletes `rust-core`, `main` or `rust-core-reference` however it is spelled (the owner, Q14; the pre-push hook refuses it too, in every worktree of the clone, registered or not), `--force`, `-f`, `--force-with-lease` and `+` refspecs on `git push`, `git push --delete`, `git clean -f`, `cargo clean`, `git worktree remove --force`, `docker rm -f`, `--no-verify` and `git -c core.hooksPath`, `sudo`, `git config` (reads allowed), `git stash`, `git branch -D`, `gh run watch`, `git fetch --prune`, and `prepush.sh --full`.
An owner-approved one-off runs as `SDM_APPROVED=<id> <command>`, `<id>` the slug of the owner-answers heading that approves it (`guard_cmd.py --tokens` lists them); it is logged in `$SDM_ORCH_STATE/logs/approved-commands.log`, and every refusal in `guard-denied.log`.
Codex panes keep `rule_watch.py`: the owner declined a Codex hook (2026-10-16) and kept the watcher for Codex (2026-10-21, decision 7).
The orchestrator's own session gets the hook when it is started in a checkout holding `.claude/settings.json`.

### CI's split, timing guard and base (item 3)

These replace rules Phase 2 kept by hand; each is enforced by the workflow and held by `tests/test_ci_workflow.py`.

- **The checking tools' own tests run once, and only when they can change.**
  A test that copies the checkout into a temporary repository and reruns the gate, the ratchet or the comparator carries the pytest marker `tooling` (`pytestmark = pytest.mark.tooling`, or on its copying tests alone when the file also checks the real tree).
  `scripts/rust_core/tooling_tests.py check` lists any such test without the marker, and `list` shows what it finds.
  `test (3.x)`, `engine` and `contract-python (ref-pandas3)` deselect them; `tooling (contract)` and `tooling (root)` run them, on Python 3.12, only when the landing job's plan says a judging path (`scripts/rust_core/judging-paths.txt`), anything under `scripts/rust_core/`, the workflow, or a marked test or a module one loads changed (`tooling_tests.py must-run`).
  A product package that adds a case does not run them; a tooling change does.
  Known gap, carried as a Phase 3 row: a change to the cases alone does not run them, so a case a tooling test reads can break it unseen until the next tooling change; a small fixed corpus for those tests would close it.
- **The landing job plans the run once, as it starts.**
  It records the phase head the candidate was pushed on (`ratchet.py push-base`), and every judging job reads it as `SDM_PUSH_BASE` through `ratchet.py where` (and `rust-lint`'s bridge freeze), so a landing while the run goes on, the candidate's own included, cannot turn a comparison into a self-comparison.
  A rerun of failed jobs keeps it; a rerun of every job after the candidate landed takes the head it landed on.
- **Test authors' branches run a short matrix.**
  A branch named `<topic>-tests` or `<topic>-tests-<suffix>` runs the client tests on 3.12 alone and builds no aarch64 or engine wheels; its tests fail by design until the code exists.
- **A newer push cancels the run in progress on the same topic or lane branch.**
  A lane carries one package, so its older run is evidence for nothing; the phase branch's runs are never cancelled.
- **The timing guard** (`scripts/rust_core/orchestration/ci_timing.py`, the `timing guard` job) records every job's minutes in the run's `ci-timing` artifact and names a job that used more than 80% of its `timeout-minutes` on two of its last three runs, as an error annotation and a fix item.
  It never fails the run (no hard budgets).
  The heartbeat runs `ci_timing.py --pull`, which writes the newest run's items to `$SDM_ORCH_STATE/fix-items/ci-timing.md`; a named job gets a fix at its cause, not a higher limit.

### Environments within the machine (item 5)

The machine has 8 vCPUs, 30 GB of memory and 412 GB of fast disk at `/mnt/localssd`.

- **One environment per lane, reused until what it installs changes.**
  `scripts/rust_core/cells.sh /mnt/localssd/cells/<lane>` builds the lane's cells into `env-<key>`, keyed by `uv.lock`, `rust/`, the packages' sources, `native_wheel.py` and the cells in `ci.yml` (`cells.sh --print-key` prints it).
  A docs, cases or tests commit reuses the environment in under a second; a new key builds a new one (about 3 minutes, most of it from the shared cache) and removes the lane's older one.
  Export `SDM_LANE_ENV=/mnt/localssd/cells/<lane>` and `SDM_KIT_PYTHON=$SDM_LANE_ENV/current/kit/bin/python` (the provider kit's test-ops build, as CI's contract jobs have it), and run with the cell python it prints (`$SDM_LANE_ENV/current/cell-ref-pandas3/bin/python`).
- **The runners refuse the shared `.venv`.**
  On this machine `run_python.py` and the provider kit refuse a checkout's `.venv`, and with `SDM_LANE_ENV` set any interpreter outside it, naming the right one; Phase 2 lost twelve hours to a stale shared environment read as an environment fault.
- **One Rust build cache.**
  `~/.cargo/config.toml` routes every build through sccache at `/mnt/localssd/sccache`.
  An agent's `CARGO_TARGET_DIR` (`/mnt/localssd/target/<agent>`) holds incremental state only; `clean_tmp.sh` removes it when stale or over `CAP_GIB`.
- **`heavy` admits a job only when the machine can take it.**
  `$SDM_ORCH_STATE/bin/heavy` calls `scripts/rust_core/orchestration/heavy`, which starts a heavy command when a CPU slot, free memory and free disk all allow it (thresholds in `scripts/rust_core/orchestration/heavy.conf`, or `SDM_HEAVY_CONFIG`), runs `clean_tmp.sh --apply` first when the disk is short, and otherwise queues, saying what it waits for.
  Heavy means any `gate.sh` leg, a full `pytest` of `contract/tests` or `tests`, a `tooling` test file, `cargo test`, `build` or `clippy` of the workspace, and every wheel build.
  The command hook refuses `cargo test`, `build` and `clippy`, a pytest over the whole `tests` or `contract/tests`, and `gate.sh` unless run behind `heavy` (the owner, 2026-10-22).
  `heavy --priority` (or `SDM_HEAVY_PRIORITY=1`), for the critical path's jobs as the orchestrator grants it, takes the next free slot before any job without it; every job admitted after a wait appends who, whether it had priority, how long it waited and the command to `$SDM_ORCH_STATE/heavy-waits.log`.
- **Two packages at once** (the owner, "2026-10-23 overnight plan"; one from "2026-10-23 cut the tooling back", three from 2026-10-22).
  `lane.py start` refuses a third active lane, naming the active ones: a lane worktree whose recorded package has not landed and is not paused; the limit is `MAX_LANES` in `heavy.conf`.
  `lane.py pause --lane <x>` frees a slot (`--resume` takes it back); a test author counts inside its package's lane.
- **The watchdog.** The heartbeat runs `scripts/rust_core/orchestration/watchdog.py --phase <phase branch> --dispatch` every 15 minutes.
  It prints one line per problem and nothing when all is well: a lane worktree with no commit and no running process for 45 minutes (`IDLE`), the failed jobs of the newest finished CI run of each branch under the phase (`RED CI`), the failed jobs of the newest nightly run on the phase head (`NIGHTLY`, also appended once per run to `$SDM_ORCH_STATE/phase-<N>/fix-items.md`), and a package more than 2 hours past `forecast.yaml` and not landed (`SLIP`).
  With `--dispatch` it starts the phase head's nightly run when there has been none for 24 hours, since GitHub reads the schedule from `main`'s copy of the workflow.
- **A lane's directories go when it ends**: `clean_tmp.sh --release <lane>` (dry run first, then `--apply`) removes its environment, target and tmp directories.

### The dry run and the slip report (item 1)

- **Before the design locks**, run `scripts/rust_core/gate_dryrun.sh kinds.yaml --python $SDM_LANE_ENV/current/cell-ref-pandas3/bin/python --gate <half>` over the design's planned case kinds (`kinds: [{name, case}]`, one stub case each).
  It judges each stub with the checkpoint's own case reader, `coverage.py`, `ratchet.py`, Python runner and `compare.py` (a worktree of the checkpoint, as the gate's harness judges) (and with `--rust` the test kit, with `--gate HALF` the locked gate) on a throwaway commit, and prints a `GAP <kind> (<case>): <tool>: ...` line for every refusal.
  Those lines are the tooling work to do up front, instead of one judging change at a time mid-phase.
  A tool that fails on no line naming a kind prints `INCOMPLETE` and the dry run exits 1: what it judged is unknown.
  Without `--gate <half>` (the locked half the planned kinds fall in) the run is `INCOMPLETE` too, since the locked gate's refusals are part of the list.
- **The slip report**: the heartbeat runs `scripts/rust_core/orchestration/phase_stats.py --phase-branch origin/$SDM_PHASE_BRANCH --no-ci --forecast <the design's forecast>`, a `{start, packages: [{id, hours}]}` file, and reads how many hours the phase is behind and the package or unplanned landings that caused most of it.
  It never blocks anything.

### The checking tools for Phase 3's legs (T3.2)

T3.2 taught the checking tools Phase 3's warehouse and Java legs.

- **Legs.** A case's `warehouses` (S2) names the legs it runs on, and only there; an S1 case runs on the three databases and every warehouse leg, but the two 100,001-row batching cases (`run_python.py`'s `case_legs`, `case.rs`'s `Case::warehouses`).
  Observations are keyed `<case>@<leg>`.
- **The recording mode, for C-1 and E-0.** `run_python.py --mode python-ref --reference DIR --database <leg> --record FILE` runs the old client live on the leg the live configuration opens (`SDM_LIVE_CONFIG`, a JSON file outside the repository: each leg's connect arguments, any value read from the environment as `{"env": NAME}`), with each placeholder's private name read from `SDM_CAPTURE_<NAME>` (`{catalog}` from `SDM_CAPTURE_CATALOG`).
  Each case's tables are loaded by T3.3's `databases.load(duck, leg, WarehouseTarget)`; every driver call is captured, and the driver's statement log runs beside it as the capture's query log.
  The output has each private name written back as its placeholder, and the write fails on any private value left; T3.3's writer turns it into the capture manifests.
- **Captures.** `capture_check.py captures --cases DIR [--locked] MANIFEST...` lists each case and leg with no capture of its current input as `pending live: <case>@<leg>` (failing with `--locked`); `capture_check.py commit --candidate SHA [REV]` holds a captures-only commit to the final candidate; `contract/captures/**` is protected.
- **Warehouse legs.** No offline stand-in (the owner's decision B-lite): a warehouse leg's observations are compared like any database's, by every rule, against the reference lines C-1 recorded on it; `gate.sh --locked` runs `captures --locked` once `contract/captures/` exists.
- **Runners.** `gate.sh --run` takes `java` (`run_java.py`, T3.11's).
- **Lanes.** `lane.py start --brief BRIEF` refuses a brief `brief_check.py` fails; from Phase 3 on it refuses a package ID with a letter, and it never starts a lane on a phase `rust-core` already holds.
- **Coverage.** `coverage.py --check --gate LANGUAGE:SLICE` gates a slice's paths and cells (`python:S1`, `python:S2`, `java:S1`); each leg claims its own path, and a manifest path's `tests` (pytest node IDs) may claim a path the case format cannot express.
- **Option C's sampled case.** `compare-policy.yaml`'s `sampled` section names the one sampled Snowflake case (`{leg, stage, statement, decision}`); on that leg its comparison holds only the sampling statement and the result's row count, besides the outcome, route, inputs, schema and warnings, and `capture_check.py` refuses a second case, another leg or another decision.
- **The census's real-data mode.** `refusal_census.py real-data --database LEG --cases DIR --control CASE --out FILE`, one leg at a time, a warehouse leg live, failing when its seeded control is not refused.

### After T3.0 lands: the orchestrator's one-time steps

The private copies outside the repository would bypass the mechanisms above, so they become wrappers of the repository's tools.
Until the phase merges into `rust-core`, the tools live on the phase branch, so the wrappers point at a worktree kept at the phase head: `git worktree add --detach ~/code/sdmc-phase-head origin/rust-core-phase-3` once, and `git -C ~/code/sdmc-phase-head checkout --detach origin/rust-core-phase-3` after each landing (`~/code/sdmc-rust-core` once the phase has merged).

- `$SDM_ORCH_STATE/bin/heavy` becomes `exec ~/code/sdmc-phase-head/scripts/rust_core/orchestration/heavy "$@"`.
- `$SDM_ORCH_STATE/bin/codex_pane.py` becomes `exec python3 ~/code/sdmc-phase-head/scripts/rust_core/orchestration/codex_pane.py "$@"`, so the panes run the repository's version.
- Agents export `SDM_LANE_ENV` and `SDM_KIT_PYTHON` from `cells.sh`, since the runners refuse the shared `.venv` from then on, and pass the lane's cell python to `handoff.py run --python`.
- Agents, the orchestrator included, start in a checkout holding `.claude/settings.json`, so the command hook runs for them.
- A shell profile that exports an older phase's `SDM_PHASE_BRANCH` points every tool at that phase; export the current one, or none.

## CI

- **The four container images are private copies on ghcr.io** (`ghcr.io/nvidia/structured-data-models-client/...`, same digests), pulled with the workflow token. Anonymous registry pulls caused about 57% of Phase 1's failed runs before the move.
- **Every job has `timeout-minutes`.** A hung `rust-test` once ran 2h24m. The timing guard names a job that comes near it (see "CI's split, timing guard and base" above).
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
- **Confirmation rounds almost never find anything.** All 20 paired confirmation, stack and tiny-fix rounds were approved by both. Use the script for clean moves; since Phase 2 (decision KO2-9), confirmations still have both reviewers, because the owner wants both to sign off.
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
- The banned commands are blocked by the command hook (see "The command hook" below): `rm -r` outside your scratch, `--force` and `-D` removals, `--no-verify`, `sudo`, `git config`, `git stash`, `prepush.sh --full`, `gh run watch`, `git fetch --prune`. Codex panes have no hook; `rule_watch.py` watches them.
- Live Databricks work: upload only to the volume the owner authorized, run one-time jobs only, and never touch existing notebooks or clusters. Use only the database instance the owner names.
- Slack is read-only.
- Implementers never open held-out data. A held-out set is used once; Phase 1's is spent.

## Gotchas from Phase 1

- **The held-out bundle froze in an older case format.** Phase 1 converted it with `--pre-given-specs` (owner item 49). Phase 2 needs a new held-out set, frozen once the scope is locked and T2.1 has settled the case format, with a tested converter and a dry run of the held-out runner on a dummy bundle before the one-shot, and never opened by implementers.
- **New register and decision IDs collide across lanes.** Give each lane its own ID block (for example P2B-1..19 and E1..9 for lane a) and have lanes write only in their own sub-table or section of shared docs.
- **Shared docs conflict on rebase** (17 conflicts, mostly `tracker.md` and `decisions.md`). Each package writes only in its pre-created section.
- **A lane venv that imports the lane checkout** gives false "passes now" results. `cells.sh` installs every first-party package as a wheel.
- **Pushes need an SSH key authorized for the organisation's SSO.** A push failing with "has enabled or enforced SAML SSO" means the key needs re-authorizing on GitHub.
- **The held-out H11 rule has no approved-entry exception for new refusals.** A new refusal found there fails the check, whatever the register says.
