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
   The two expressions that can't match a pattern (the concurrency and the phase check's condition) read a phase branch as a name under `rust-core-phase-` that holds no digit followed by `-`.
   The prefix holds no digit, and every branch under a phase (`rust-core-phase-N-<name>`) holds one, so this works for any N, multi-digit included (fix 1).
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
   On the epic branch no landing will name a protected change made after the merged phase's last landing, so the ratchet's rule `epic` (fix 1) fails each such change unless an owner-approved row names its commit.
   The rule applies when the candidate is the tip of `origin/rust-core` and is no landing commit.
   It covers every commit on the head's first-parent history after the merge that brought the checkpoint, and that merge for what it holds beyond the checkpoint.
   The row is read from the epic head's own phase decisions files, since a commit made on the epic branch after the last landing can be named nowhere else.
   It names a commit by a SHA prefix of 8 or more digits, or by a range `A..B` (git's meaning, first-parent), in backticks.
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
   `stuck_agents.py` and `rule_watch.py` print `WATCHDOG-ERROR` and exit 1 when no tasks directory is given at all (fix 1), when it is missing, and when a transcript can't be read or parsed.
   The loop's greps in `prompts.md` keep that line.
9. **The Codex pane helper** is now in `scripts/rust_core/orchestration/`, with its self-test and a pytest that runs it.
   Its code is the old helper's unchanged, `SDM_REPO`'s default included (an AST comparison is identical), with the SPDX header added and `ruff format` applied (fix 1).
   Its self-test now sets the throwaway repository's commit identity through the environment instead of `git config`, and one lambda became a function.

### The bootstrap route

Phase 2's fork point is `9604074a`, `rust-core`'s head.
It is no landing commit, and its own `ratchet.py` (Phase 1's) refuses any non-landing checkpoint, including itself.
Its harness therefore can't judge T2.0.

**The substitution, for the owner to authorize (decision H2: the checkpoint's harness judges).**
For this one checkpoint, `gate.sh` judges with the harness of Phase 1's last landing, `bd44b342` (`Land T1.36`), in place of the fork point's own.
It hands that harness `--base bd44b342`, so the candidate is held to that landing's baselines, register, decisions, cases and protected list.
Those equal the fork point's: the 13 handoff commits only add files, and none of them judging data.
The gate prints that the stand-in judged.
The substitution is pinned in `gate.sh` (`rust-core-phase-2`, `9604074a`, `bd44b342`), and `gate.sh` reads every condition from git itself, not from the candidate's `ratchet.py` (`bootstrap_problem`).
It applies only when all of these hold:

- The candidate's checkpoint is `9604074a`.
- `9604074a` is `git merge-base origin/rust-core-phase-2 origin/rust-core`.
- The candidate holds no landing commit after `9604074a`, but for itself as the first landing: an empty commit with one parent whose Base-SHA and Checkpoint-SHA are `9604074a`. That landing is judged as its parent's tree under its own trailers (`--protected-paths` from Reviewed-by and Protected-paths), which covers its phase push and every replay of it.
- `bd44b342` is a landing commit and the merged phase's last landing `9604074a` holds: the second parent of the newest merge on its first-parent history whose second parent is a landing.
- `9604074a` only adds files to `bd44b342`'s tree, and none under the judging data: `contract/`, the register, every phase's decisions and locked set, `phase-0/`, `rust/`, and the harness's scripts.

Any other non-landing checkpoint takes the normal path: its own harness resolves it again and refuses any other commit.
Once T2.0 lands, every candidate holds T2.0's landing after `9604074a`, so the substitution can't apply again.
`SDM_GATE_BOOTSTRAP` replaces the pin for the gate's own tests only, and `ci.yml` never sets it (a test holds it to that).
The negative controls are in `tests/test_t20_bootstrap.py`, and each fails if the conditions are removed:

- a candidate above a landing given the fork point as its base (the reviewer's repro);
- a candidate whose `ratchet.py` names another non-landing checkpoint (Codex's repro: that commit's own harness judges and refuses);
- the same candidate naming the pinned fork point from above a landing;
- a fork point that differs from the stand-in in a judging file;
- a remote epic branch whose fork point is another;
- a first landing recording another Checkpoint-SHA.

**The landing.**
`rust-core`'s `land.py` lands only on Phase 1, so T2.0 lands with its own `land.py`: `land.py ... --phase-branch rust-core-phase-2` from T2.0's worktree.
The orchestrator records the lane worktree's package (`sdm.package T2.0`) at landing.
The landing push's phase check runs T2.0's `ratchet.py checkpoint --phase-push 9604074a`, which accepts the fork point.
The push's other jobs are skipped on the candidate's green run (E131).
Without that evidence they run, and the gate takes the first-landing case above.

**The post-landing state.**
Once landed, the phase branch's head is T2.0's landing commit, which holds the new harness.
The next candidate is judged against it (see the probes).

**For `rust-core` to go green**, the owner merges T2.0's landing commit into `rust-core` with `--no-ff`.
`where` then names that landing as `rust-core`'s checkpoint, its harness T2.0's.
The handoff commits are its ancestors, so they need no row.
If `rust-core` is instead judged against `bd44b342` with T2.0's harness (T2.0 reaching it otherwise), rule `epic` needs an owner-approved row in the epic head's phase decisions file naming the 13 handoff commits, for example `` `91a00319..9604074a` `` (the first-parent commits after the Phase 1 merge, `98ae9bd2` to `9604074a`).
On the real history, without such a row, rule `epic` fails on the 5 of those commits that change a protected path: `98ae9bd2`, `beef2082`, `345af5b2`, `6dc7b9f3` and `9604074a`.

### Probes

These were run on T2.0's tree before the first push.

- **Exit 4, on the real history.**
  With `GITHUB_EVENT_NAME=push GITHUB_REF_NAME=rust-core`, `ratchet.py where` prints `--base bd44b342...`, Phase 1's last landing.
  With the base it prints, these all pass on T2.0's tree over `rust-core`'s head, judged by `bd44b342`'s harness:
  - `ratchet.py check --checkpoint bd44b342`: ok.
  - `gate.sh --only coverage`: ok.
  - `gate.sh --only l0`: ok.

  The first round passed here because protected changes on the epic branch were only notes, which round 1 found to be a hole: fix 1's rule `epic` now fails each unnamed one (see the bootstrap route for the row it needs).
  The comparisons themselves are the acceptance tests' (`contract/tests/test_t20_checkpoints.py`):
  - raw against the reference, the real `python-core` observations fail (the red runs reproduced);
  - held to `where`'s effective expectations, they pass;
  - a seeded unapproved outcome still fails.
- **Exit 3 and the bootstrap, on the real history.**
  `ratchet.py checkpoint --base origin/rust-core-phase-2` resolves to the fork point `9604074a` (`harness`).
  Then `gate.sh --base origin/rust-core-phase-2 --only ratchet`, `--only coverage` and `--only l0` each pass.
  Each says the fork point's own ratchet refuses it and `bd44b342` stands in, judged by that landing's harness.
  On a local landing commit `Land T2.0` over T2.0, `ratchet.py checkpoint --phase-push 9604074a` resolves to the fork point.
  `tests/test_t20_bootstrap.py` holds the stand-in route, with its negative controls (above), and runs the real pin on the real history.
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
- `land.py verify` given no phase (no flag, no `SDM_PHASE_BRANCH`) takes the lowest-numbered phase branch on the remote that holds the landing, and refuses by name when none does (fix 1).
- A malformed `SDM_PHASE_BRANCH` is a named error in every tool, `bridge_freeze.py` and `replay_jobs.py` included (fix 1).
- The orchestration README's tool table changes in the Codex pane helper's commit, together with the watchdog rows.

### Fix rounds

- Round 1 (both reviewers):
  - The stand-in route was candidate-chosen and stayed open after T2.0 landed. It is now pinned, with every condition read from git, plus negative controls.
  - It now gates T2.0's own landing commit and its phase push.
  - Protected changes on `rust-core` beyond the base landing were advisory. They now fail unless an owner row names their commits (rule `epic`).
  - `ci.yml`'s cap on phases 0 to 9 is gone.
  - P3s fixed: `land.py verify` infers the landing's phase; the phase error is named in `bridge_freeze.py` and `replay_jobs.py`; the Codex pane helper's default is restored; the watchdogs fail loudly with no tasks directory at all.
