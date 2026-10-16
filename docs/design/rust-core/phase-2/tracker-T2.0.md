# Phase 2 tracker: T2.0

The design branch owns `tracker.md`; this file holds T2.0's section until it folds in.

## T2.0

T2.0 makes the tooling phase-aware and bootstraps Phase 2 and `rust-core`'s CI.
It lands alone, first, on `rust-core-phase-2`, with two reviewers.

### What changed, by exit

1. **The phase is one setting.**
   `scripts/rust_core/phases.py` names the branches once: the epic branch, the phase branch pattern `rust-core-phase-N`, the lane, topic and archive names under it, the package ID pattern, and `CURRENT`, the phase in progress.
   Every tool resolves the phase branch the same way: its own `--phase-branch` flag, else `SDM_PHASE_BRANCH`, else the phase of the branch it works on, else `CURRENT`.
   That covers `land.py`, `lane.py`, `ratchet.py`, `prepush.sh` (in shell, with the same order), `prune_caches.py`, `replay_jobs.py` and `bridge_freeze.py` (and so `l0_deps.py`).
   A name that is no phase branch is refused by name, never replaced by the default.
   `ci.yml` names no phase.
   Its branch filters are patterns, and its step texts match `^rust-core-phase-(0|[1-9][0-9]*)` in bash.
   The two expressions that can't match a pattern (the concurrency and the phase check's condition) build the list `rust-core-phase-0` to `-9` with `format()`.
   The landing step fails a push to a phase branch past that list, so a later phase can't go unchecked.
   `selftest.py`'s Phase 1 fixture and `test_t133_register.py` stay on Phase 1, as the brief asks.
   The `*-pending.json` files are left as they are.
   They record the test IDs their test authors wrote in Phase 1, every `pending` list is empty, and no check reads their `authored` lists against live tests.
2. **Every phase's decisions count.**
   `capture_check.phase_decisions` reads every `docs/design/rust-core/phase-N/decisions.md` in phase order.
   `ratchet.py` (rule `decisions`, read at the checkpoint), `coverage.py` (rule 8) and `capture_check.policy` (the exclusions) all read through it.
   It fails closed, naming the file, when any of these holds:
   - Phase 1's file is missing.
   - A phase below the highest phase with a decisions file has none.
   - A phase with a `design.md` has no decisions file.
   - A file is unreadable or not UTF-8.
   - A row has no status.
   - A row ID appears twice, in one file or across two.
3. **The first checkpoint.**
   `ratchet.py` accepts one non-landing merge base: the phase branch's fork point from the epic branch, `git merge-base origin/<phase> origin/rust-core`.
   Any other non-landing commit is still refused, by name: an unlanded commit on the phase branch, an older `rust-core` commit, or the Phase 1 merge.
4. **`rust-core`'s CI goes green.**
   `ratchet.py where` on a push to `rust-core` prints `--base <the merged phase's last landing>`.
   That landing is the second parent of the newest merge on `rust-core`'s first-parent history whose second parent is a landing commit (`bd44b342`, `Land T1.36`, today).
   So `rust-core` is judged like a moved landing of that phase's head, by that landing's harness, with every comparison as it was.
   `ci.yml`'s gate steps must run on `rust-core` too: an empty `where` there fails.
   `contract/tests/test_runners.py` reads `where` the same way.
5. **`gate.sh --locked` for any phase.**
   `gate.sh --locked` reads its locked set as data from the judging harness's tree.
   Each phase has `docs/design/rust-core/phase-N/locked-set.yaml`, a protected path.
   A half of the locked set gives either:
   - its full shape: its own steps, its numbered checks, its reference checks, its "every" check and its inventory (Phase 1's file, which reproduces `1a` and `1b` exactly);
   - or only its `cases`: one check holding every common step and the cases' inventory step.
   A half that no file names, or a file not of the format, is a usage error.
   `phase_end.sh` derives the half and the inventory from the phase's locked set at the SHA, and otherwise asks for `--half` and `SDM_INVENTORY`.
6. **Package IDs.**
   `T2.0`, `T2.0a`, `T2.15` and `CI5` pass `land.py` (end to end, self-tests), `lane.py` and `ratchet.py`'s landing pattern.
   All three read `phases.PACKAGE_ID`.
7. **The DuckDB install race.** See its commit for the mechanism and the stress probe.
8. **Watchdogs.**
   `stuck_agents.py` and `rule_watch.py` print `WATCHDOG-ERROR` and exit 1 for a tasks directory that is missing and for a transcript that can't be read or parsed.
   The loop's greps in `prompts.md` keep that line.
9. **The Codex pane helper** is now in `scripts/rust_core/orchestration/`, with its self-test and a pytest that runs it.

### The bootstrap route

Phase 2's fork point is `9604074a`, `rust-core`'s head.
It is no landing commit, and its own `ratchet.py` (Phase 1's) refuses any non-landing checkpoint, including itself.
Its harness therefore can't judge T2.0.
The route:

- **The gate.**
  The candidate's `ratchet.py` resolves the checkpoint to the fork point.
  `gate.sh` sees that the fork point's own `ratchet.py` refuses it.
  It then judges with the merged phase's last landing (`bd44b342`, `ratchet.py stand-in`) and prints that it did so.
  That landing's harness is Phase 1's final reviewed harness.
  Its judging data equals the fork point's: the 13 handoff commits change no contract file, register, baseline or decisions file.
  It judges T2.0 against Phase 1's baselines with its own ratchet, comparator, coverage and L0 scripts.
  This runs in CI on T2.0's exact SHA, so the exact-SHA CI run can be green with every check meaningful.
  The stand-in applies only when the checkpoint is no landing commit (the fork point is the only one the ratchet accepts) and its own `ratchet.py` refuses it.
  After T2.0 lands, every checkpoint is a landing commit holding T2.0's ratchet, and the path is never taken.
- **The landing.**
  `rust-core`'s `land.py` lands only on Phase 1, so T2.0 lands with its own `land.py`: `land.py ... --phase-branch rust-core-phase-2` from T2.0's worktree.
  `land.py` refuses a lane whose worktree records no package (`git config --worktree sdm.package`).
  The orchestrator made the lane by hand, so it either runs T2.0's `lane.py start` for `T2.0` there or records the package itself.
  The landing push's phase check runs T2.0's `ratchet.py checkpoint --phase-push 9604074a`, which accepts the fork point.
  That push's other jobs are skipped on the candidate's green run (E131).
- **The post-landing state.**
  Once landed, the phase branch's head is T2.0's landing commit, which holds the new harness.
  The next candidate is judged against it (see the probes).

Whether this route needs an owner rule is the owner's question: see the report.

### Probes

These were run on T2.0's tree before the first push.

- **Exit 4, on the real history.**
  With `GITHUB_EVENT_NAME=push GITHUB_REF_NAME=rust-core`, `ratchet.py where` prints `--base bd44b342...`, Phase 1's last landing.
  With the base it prints, these all pass on T2.0's tree over `rust-core`'s head, judged by `bd44b342`'s harness:
  - `ratchet.py check --checkpoint bd44b342`: ok.
  - `gate.sh --only coverage`: ok.
  - `gate.sh --only l0`: ok.

  The ratchet lists 43 protected paths changed since that landing (the handoff commits and T2.0).
  They are notes, not failures: CI on `rust-core` gives no `--protected-paths`, and its head is no landing commit, so the rule has no trailer to hold them to.
  `rust-core` therefore needs no owner row for the handoff commits.
  The comparisons themselves are the acceptance tests' (`contract/tests/test_t20_checkpoints.py`):
  - raw against the reference, the real `python-core` observations fail (the red runs reproduced);
  - held to `where`'s effective expectations, they pass;
  - a seeded unapproved outcome still fails.
- **Exit 3 and the bootstrap, on the real history.**
  `ratchet.py checkpoint --base origin/rust-core-phase-2` resolves to the fork point `9604074a` (`harness`).
  Then `gate.sh --base origin/rust-core-phase-2 --only ratchet`, `--only coverage` and `--only l0` each pass.
  Each says the fork point's own ratchet refuses it and `bd44b342` stands in, judged by that landing's harness.
  On a local landing commit `Land T2.0` over T2.0, `ratchet.py checkpoint --phase-push 9604074a` resolves to the fork point.
  `tests/test_t20_bootstrap.py` holds the stand-in route:
  - the stand-in judges when the fork point's ratchet refuses it;
  - the fork point judges itself when its ratchet accepts it;
  - an unlanded commit on the phase branch is refused before any harness.
- **The post-landing state.**
  A local landing commit `Land T2.0: ...` (Base-SHA the fork point) is the phase head, with a probe candidate on it.
  `gate.sh --base <that landing>` with `--only ratchet`, `--only coverage` and `--only l0` each pass, judged by the landing's own harness, T2.0's.
- **Outside input.**
  A locked set nested 20,000 levels deep is refused by name ("not a locked set it reads", a usage error).
  One holding 100,000 cases is read.
  Decisions files are read as text lines, with no nesting.
- **Exit 7.**
  The DuckDB stress probes are in the fix's commit message.
  - Before the fix (`9604074a`): 70 of 80 processes failed the implementer's probe, and 44 of 80 failed the test author's `duckdb_install_race_probe.sh`.
  - After: 0 of 80 on both.
  - The race had a second form: DuckDB's own "Could not set lock on file ...duckdb_extension.info" during concurrent installs.
- **Local runs.**
  - `pytest contract/tests -n 4 --dist loadfile` ran in a reference-cell environment (pandas 3.0.6, pyarrow 25.0.1, numpy 2.5.3, this tree's native wheel): every test passed once `ratchet.py`'s bootstrap print kept its Phase 1 form.
  - In the development environment, which is no runtime cell, `test_gate_modes.py`'s `--locked` tests fail on `9604074a` too ("reported, not gated").
  - `prepush.sh`'s cheap tier passes but for `lane.py check`, which refuses this lane because the orchestrator made it by hand and the worktree records no package.

### Deviations

- The `*-pending.json` files keep their Phase 1 test IDs.
  They are records of what each test author wrote, every `pending` list is empty, and nothing reads `authored` against live tests.
- `ci.yml`'s concurrency and phase-check condition can't use a pattern, so they list phases 0 to 9, built with `format()`, and the landing step fails a push past them.
- The orchestration README's tool table changes in the Codex pane helper's commit, together with the watchdog rows.
