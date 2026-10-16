# Phase 1 run log

A short account of how Phase 1 of the Rust-core migration actually ran, for whoever runs Phase 2.
Times are PDT. Hours are rough; several lanes ran at once, so lane-hours add up to more than the elapsed time.

## Summary

- Ran from 2026-10-08 00:00 (design locked at 00:34) to 2026-10-15 15:38. About 7.5 days, or about 184 hours.
- Outcome: Phase 1 merged into `rust-core` on 2026-10-15 as `91a00319` (a `--no-ff` merge of phase head `bd44b342`).
- Locked gates 1a and 1b both passed on the final tree. The held-out check H11 passed on its one-shot run.
- 82 landings and about 1,130 commits on the phase branch. Through 10-14, only 32 of 70 landings were product work; the rest were harness, tooling and corrections.
- The owner gave a conditional go on 10-14; the go record landed as `bd44b342` just before the merge.

## Timeline

One line per day. Landing counts come from the phase branch's `Land` commits.

- **10-07, 23:38.** Phase 0 merged (`2e9d4f4d`). Phase 1 starts on top of it.
- **10-08, 6 landings.** Design lock, T1.7b through T1.9, and the checkpoint 1 corrections (T1.8b). Overnight, the T1.10 implementer ended its turn "waiting for CI". CI failed and nobody noticed for 10.6 hours.
- **10-09, 7 landings.** T1.10 lands at 10:36. The guard packages T1.10c and T1.10d (T1.10c took 4 fix rounds). T1.11a to T1.11c. CI1 adds `prepush.sh` and `land.py`.
- **10-10, 14 landings.** CI2 adds lane tooling, so two lanes run at once. Checkpoints 3 and 4 both ask for a bounded course correction. The owner is away for about 36 hours from that evening.
- **10-11, 7 landings.** T1.14 (ratchet, register v2, gate) and T1.18 (L2 in CI) land. Checkpoints 5 and 6. T1.18 sits about 9 hours behind an approval prompt.
- **10-12, 13 landings.** The owner drops the local `prepush --full`, so CI is the full check. New rules: a separate test author writes acceptance tests first; always two reviewers; auto-land once both approve and CI is green. Checkpoint 7.
- **10-13, 13 landings.** T1.20 lands at 02:52 and both reviewers accept Phase 1a. Checkpoint 8. The first accuracy benchmark ran on random data and was thrown out; the real-data rerun passed. A third lane starts. The disk fills and two lanes pause.
- **10-14, 13 landings.** New rustls advisories turn every commit red. A cold `rust-test` takes 2 h 24 min. The registry hits data limits on most pushes. The stack is rebuilt five times. CI3 lands at 15:43, so a candidate moved over landing commits can land on its own evidence. The owner gives a conditional go.
- **10-14 to 10-15, overnight.** CI4 lands at 23:29: CI images on ghcr.io, timeouts, and no CI on landing commits. The bundled admissions (T1.32g part 2b) and T1.33 part 3 land just after midnight.
- **10-15, 9 landings.** The final review's findings are closed in T1.36 (`c505788f`, `0a8a7412`). By 06:49 every planned package has landed. Then the held-out one-shot runs, a forecast fix lands (`852e1e26`), the final gates pass, the evidence pack lands (`625b8009`), the go record lands (`bd44b342`), and the merge lands at 15:38.

## Where the time went

From a read-only time analysis made on 10-14. The rows overlap: one stalled CI run is also a late landing. Don't add them up.

**Writing code was a small share.** Implementers were active about 200 minutes per package, test authors about 60. A package's 2 to 12 hours went mostly to review, landing, stalls and waiting.

1. **Review and fix rounds: about 70 to 110 lane-hours.**
   About 360 review reports and about 92 fix rounds, at about 1 hour each end to end.
   About two-thirds of blockers came in round 1. About 13 later blockers were caused by the previous fix.
   It was worth it: about 54 product bugs were caught with tests green, several of them silent wrong answers.
   In 31 of 40 blocking rounds where both reviewed the same SHA, only one reviewer blocked, so keep two reviewers on anything that changes product or judging code.
2. **Stuck or idle agents: about 35 to 45 lane-hours, at least 25 on the critical path.**
   T1.10 waited on CI overnight (10.6 h). T1.18 sat behind an approval prompt for `rm -rf` / `--force` (about 9 h). A T1.14 reviewer stalled (about 5 h).
   A laptop host also slept, filled its disk and lost CPU to antivirus scans.
   A watchdog that flags and restarts stuck agents helped from 10-12 on.
3. **CI failures and slow CI: about 15 to 25 hours of critical path, and growing.**
   487 CI runs, about 129 run-hours. 84 of 152 failing attempts were infrastructure only.
   In the last 48 hours, about 57% of failed attempts were container registry pulls (rate or data limits). They used about a third of runner time.
   The median green run went from 8 minutes on 10-10 to 43 minutes on 10-14.
4. **Cold Rust builds.**
   A cold `rust-lint` or `rust-test` took about 20 minutes on day one, mostly building the bundled DuckDB.
   Later, with an over-full cache and DuckDB tests in debug builds, one cold `rust-test` took 2 h 24 min.
   Building dependencies at `opt-level` 1 (CI4) cut a cold run to about 34 minutes.
5. **Owner decisions and reversals: about 25 to 40 lane-hours of churn.**
   About 55 owner entries in the run log. Answers came in batches, often 8 to 10 hours apart.
   One package (T1.29) waited 6.7 hours on answers and a sign-in.
   Late answers reversed earlier work: no pandas emulation, the benchmark protocol, several register entries.
6. **Landing overhead and restacks: about 30 to 45 lane-hours.**
   The tail from last commit to landing was about 0.7 hours at first and about 1.2 hours by 10-14.
   About 0.6 rebases per landing, 45 to 120 minutes each. About 10 ID or contract-version collisions between lanes.
   About 70 confirmation reports after clean moves found 2 defects. CI3 and mechanical confirmation replaced most of them.
7. **One Codex pane.** Every Codex review queued behind the last one. The log has 25 "queued" entries and 3 reports lost to capacity errors.
8. **Judging-path splits: about 11 extra landings.** T1.32g alone needed five judging landings in about 22 hours. Admitting new data shapes kept exposing gaps in the comparator.

## Incidents worth remembering

**Agent ended its turn "waiting for CI" (10-08).**
What happened: the T1.10 implementer pushed, said it would wait for CI, and ended its turn. CI failed and sat for 10.6 hours.
Fix: at first, implementers watched CI in the foreground. From 10-12 on, implementers report their SHA and end the turn, and the orchestrator owns every CI wait, in the background.
Lesson: an agent that says "waiting" is idle. Something else has to wake it.

**Approval prompt stall (10-11).**
What happened: an agent ran a command that needed the owner's approval, during the night. It sat for about 9 hours.
Fix: agents never run `rm -rf`, `--force` or `sudo`. A rule watcher stops the landing driver if one does.
Lesson: anything that prompts the owner is a stall when the owner is asleep.

**Ratchet merge-base race (10-14, twice).**
What happened: the ratchet takes its baseline from the merge base with the phase branch, which must be a landing commit.
First, CI3 landed while a stack's CI was running; a late job read the moved branch and got a merge base that was not a landing. Every gate refused.
Then a package was stacked on another package's candidate after that candidate had already landed under a new SHA. Same result.
Fix: CI4 hardened the CI base to the newest landing ancestor. Stack on the landing commit, never on the candidate it came from.
Lesson: after any landing, restack onto the landing commit before pushing.

**Hard-coded CONTRACT in the binding tests (missed twice).**
What happened: `rust/bindings/sdm-py/tests/messages.py` hard-codes the contract version. Two fixes bumped the contract (23 to 24, then 24 to 25) and forgot this file. CI went red both times.
Fix: bumped by hand each time.
Lesson for Phase 2: read the contract version from the crate rather than writing the number in a test.

**snowflake-connector-python 5.0.0 (10-15).**
What happened: version 5.0.0 came out on the last day. It raises a DeprecationWarning, and our tests treat warnings as errors. Every CI run went red.
Fix: support 4.x and 5.x by using the public parameter style when present; the allow-list now comes from the driver's own config, including its legacy aliases. It landed in `0a8a7412`.
Lesson: an outside release can break CI on the day you need it. The rustls advisories of 10-14 were the same story. Check advisories and new releases before the final day.

**Held-out bundle frozen in an old case format (10-15).**
What happened: the held-out bundle was sealed at T1.20 in the contract-21 case format. T1.32h later changed the source spec format, and the Rust host refuses old-format specs on purpose. The one-shot run recorded the Python legs, but the Rust-host leg failed on its first case.
Fix: an opt-in flag that converts old-format specs (`a9464730`). The owner ruled that the Rust-host leg could complete with it.
Lesson: when a held-out set is sealed early, keep a tested converter for its format, and dry-run the held-out runner on a dummy bundle of the same format before the one-shot.

**H11 found a real gap (10-15).**
What happened: on the Rust-host leg, one held-out case got a new refusal. One task type was still split up by the Python host before it reached the core, so the pure-Rust host could not run it. Both reviewers failed H11 on that alone.
Fix: the owner chose not to grant an exception. The decomposition moved into the core (`852e1e26`), the legs were rerun, and both reviewers passed H11.
Lesson: logic that only the Python host does will show up in the Rust host's held-out leg. List it before the run.

**Pushes blocked by SSO (10-15).**
What happened: the last two landings could not be pushed. The organisation enforces SAML SSO and the SSH key was no longer authorised for it.
Fix: the owner re-authorised the key. Landing commits that failed to push were undone locally and requeued.
Lesson: check push access with a dry-run push before the final landings, not during them.

**Editing a running bash script.**
What happened: bash reads a script as it runs, so editing a long-running driver script in place can make the running copy execute a half-changed file.
Fix: stop the driver, edit, then restart it.
Lesson: never edit a script while it is running. Write a new file and move it into place, or stop first.

**Benchmark on random data (10-13).**
What happened: the accuracy benchmark compared old and new on generated data with no signal. It "failed", and nobody but the owner noticed that the comparison meant nothing.
Fix: parity benchmarks use real datasets with a sanity floor. The real-data rerun passed.
Lesson: generated data can test plumbing, never model quality.

**Disk full (10-13).**
What happened: Cargo build output from many worktrees filled the disk. Two lanes paused, and one cleanup also deleted a reviewer's live scratch directory.
Fix: a disk audit, then pruning landed worktrees and old build output, with a check that nothing in flight is touched.
Lesson: one `target/` per worktree adds up fast. Prune as packages land.

## What to change for Phase 2

These come straight from the sinks above.

- Plan fewer, larger landings, with the judging path settled in the first one.
- Ask the owner the known policy questions in one sheet at the start of the phase, each claim checked at file:line.
- Keep CI green and bring a push to about 20 to 25 minutes: ghcr images, `opt-level` 1 and no CI on landing commits (done), then split and shard the slow jobs.
- Run more than one Codex pane, or have a fallback when it is at capacity.
- Run on a host that does not sleep and has disk headroom.
- Start from a short summary of Phase 1, not the full 1.35 MB of Phase 1 docs.

## More detail

- [phase-1/tracker.md](../phase-1/tracker.md): every task, its status, landing SHA, CI run and notes.
- [phase-1/fix-round-history.md](../phase-1/fix-round-history.md): what each fix round changed, per package.
- [phase-1/decisions.md](../phase-1/decisions.md) and [learnings.md](../learnings.md): the decisions and lessons as recorded during the run.
