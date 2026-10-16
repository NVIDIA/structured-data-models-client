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
   The two expressions that can't match a pattern (the concurrency and the phase check's condition) read a phase branch as a name under `rust-core-phase-` that ends in a digit and holds no digit followed by `-`.
   The prefix holds no digit, and every branch under a phase (`rust-core-phase-N-<name>`) holds one, so this works for any N, multi-digit included (fix 1).
   A name these let through that is no phase branch (a letter before the last digit, or a leading zero) fails the phase check by name, since `where` names no base for it (fix 2).
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
   `ratchet.py where` on a push to `rust-core` prints `--base <the merged phase's last landing> --epic`.
   That landing is the second parent of the newest phase merge on `rust-core`'s first-parent history whose second parent is a landing commit (`bd44b342`, `Land T1.36`, today).
   Only a merge's second parent counts, never a landing-shaped commit elsewhere, and the landing's record must carry Reviewed-by, or `where` fails by name (fix 4).
   `--epic` takes that base as given and refuses any other, never the head's own trailers, so a landing-shaped head can't choose its own checkpoint (fix 4).
   So `rust-core` is judged like a moved landing of that phase's head, by that landing's harness, with every comparison as it was.
   `ci.yml`'s gate steps must run on `rust-core` too: an empty `where` there fails.
   `contract/tests/test_runners.py` reads `where` the same way.
   On the epic branch no landing will name a protected change made after the checkpoint L (the merged phase's last landing), so the ratchet's rule `epic` (fixes 1 and 2) fails each such change unless an owner-approved row names its commit.
   - **When it applies:** with `--epic`, which `where` prints for a push to `rust-core` and `gate.sh` hands the harness. So a replay of an older push, or a local replay (`gate.sh --base L --epic`), judges the rule, whatever the remote tip is (fix 4).
   - **What it covers:**
     - What the phase merge M that brought L in holds beyond L. Each such path counts as the change of the epic-side commits that made it (`L..M^1`), or else of M itself.
     - Every commit after M (`M..HEAD`), each against its first parent, a merge included. So a merge that takes a protected file from an older parent is seen, and a change a later commit undoes still counts (fix 4).
     - Paths are read NUL-separated, so a name past ASCII or holding a control character is seen (fix 4).
     - Nothing after L counts as a landing, whatever its subject or trailers: a later landing reaches `rust-core` only by the owner's merge, which makes it the next checkpoint.
   - **Where the rows come from:** only L's own phase decisions files, reviewed and landed, never the head being judged, so no push approves itself. A row added after L counts once a later reviewed landing carries it and the owner merges that landing.
   - **How a row names a commit:** a SHA prefix of 8 or more digits, or a range `A..B` (git's meaning, first-parent), in backticks.
   - **Threat model:** honest drift, as decision E56(b) states it for the L0 guards. CI on `rust-core` runs the pushed head's own `ci.yml` and `gate.sh`, so a push crafted to evade the rule (by changing either) can. Only the owner pushes `rust-core`. The rule catches an honest protected change that no reviewed landing names; it is a tripwire, not a control against a deliberate push.
   - **What runs it:** the rule is in T2.0's `ratchet.py`, and the gate runs the checkpoint's harness.
     - Today `rust-core`'s head holds Phase 1's `ci.yml`, which names no ratchet base, so `rust-core` stays red on its approved differences, as since the Phase 1 merge, and runs no rule `epic`.
     - If T2.0's tooling reached `rust-core` with `bd44b342` still its checkpoint, `bd44b342`'s `gate.sh` would refuse the `--epic` flag ("unknown argument"), so the gate fails closed rather than passing without the rule.
     - Once the owner merges T2.0's landing with `--no-ff`, T2.0's landing is the checkpoint and its harness runs the rule.
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
The substitution is pinned on one line of `gate.sh` that nothing overrides (package `T2.0`, `rust-core-phase-2`, `9604074a`, `bd44b342`).
`gate.sh` reads every condition from git itself, not from the candidate's `ratchet.py` (`bootstrap_problem`).
It applies only when all of these hold:

- The candidate's checkpoint is `9604074a`.
- `9604074a` is `git merge-base origin/rust-core-phase-2 origin/rust-core`.
- Every commit after `9604074a` is T2.0's: `T2.0: `, `T2.0 tests: `, `T2.0 fix <n>: ` or `T2.0 tracker: `.
- The candidate holds no landing commit after `9604074a`, but for itself as T2.0's first landing: an empty `Land T2.0: ...` commit with one parent, a Reviewed-by trailer, and Base-SHA and Checkpoint-SHA `9604074a`. That landing is judged as its parent's tree under its own trailers (`--protected-paths` from Reviewed-by and Protected-paths), which covers its phase push and every replay of it.
- `bd44b342` is a landing commit and the merged phase's last landing `9604074a` holds: the second parent of the newest merge on its first-parent history whose second parent is a landing.
- `9604074a` only adds files to `bd44b342`'s tree, and none under the judging data: `contract/`, the register, every phase's decisions and locked set, `phase-0/`, `rust/`, and the harness's scripts.

Any other non-landing checkpoint takes the normal path: its own harness resolves it again and refuses any other commit.
Once T2.0 lands, every candidate holds T2.0's landing after `9604074a`, so the substitution can't apply again.
The gate's tests run disposable copies of `gate.sh` with the pin line rewritten to their fixture, and a test holds the real line to the real pin.
The negative controls are in `tests/test_t20_bootstrap.py`, and each fails if the conditions are removed:

- a candidate above a landing given the fork point as its base (the reviewer's repro);
- a candidate whose `ratchet.py` names another non-landing checkpoint (Codex's repro: that commit's own harness judges and refuses);
- the same candidate naming the pinned fork point from above a landing;
- a fork point that differs from the stand-in in a judging file;
- a remote epic branch whose fork point is another;
- a first landing recording another Checkpoint-SHA;
- another package's candidate, first landing and phase push (fix 2);
- a first landing with no Reviewed-by (fix 2).

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
That is the only route.
`where` then names that landing as `rust-core`'s checkpoint, its harness T2.0's, with rule `epic`.
The handoff commits are its ancestors, so they need no row.
Before that merge, `rust-core` stays red on its approved differences, as since the Phase 1 merge, and has no rule `epic`.

### Probes

These were run on T2.0's tree before the first push.

- **Exit 4, on the real history.**
  With `GITHUB_EVENT_NAME=push GITHUB_REF_NAME=rust-core`, `ratchet.py where` prints `--base bd44b342...`, Phase 1's last landing.
  With the base it prints, these all pass on T2.0's tree over `rust-core`'s head, judged by `bd44b342`'s harness:
  - `ratchet.py check --checkpoint bd44b342`: ok.
  - `gate.sh --only coverage`: ok.
  - `gate.sh --only l0`: ok.

  The first round passed here because protected changes on the epic branch were only notes, which round 1 found to be a hole: rule `epic` (fixes 1 and 2) now fails each unnamed one once `rust-core`'s checkpoint is T2.0's landing or later.
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

### For the owner

These questions are deliberately left open, not fixed:

- **H2:** authorize the one-time substitution of `bd44b342`'s harness for the fork point's, as described under "The bootstrap route".
- **The window before T2.0's landing is merged into `rust-core`:** `rust-core` has no rule `epic` until the owner's `--no-ff` merge of T2.0's landing. The owner is asked to make that merge as soon as T2.0 lands (round 3, Codex P1-2).
- **Where epic approvals come from:** rows count only from the checkpoint, so a protected change made directly on `rust-core` needs its approving row landed through a phase, and then merged, first.
- **The bootstrap recognises T2.0 by commit subject:** another package's commits titled as T2.0's would pass it. That window closes when T2.0 lands or `rust-core` moves, and `land.py`'s lane registration stands behind subjects (round 3, reviewer P3-1).
- **A push to `rust-core` editing `ci.yml` or `gate.sh` evades every check by construction:** whether `rust-core` gets branch protection (owner-only pushes) is the owner's call (round 3, reviewer P3-3).
- **`ci.yml`'s phase predicate:** GitHub expressions can't express "digits only". The concurrency and the phase check's condition still let a name with a letter before its last digit, a leading zero or a dot (`rust-core-phase-x1`, `-02`, `-2.0`) through as a phase branch. Such a run isn't cancelled, and its phase check fails by name.
  - Computing the classification once in a first step with a shell regex can't serve the concurrency, which is evaluated before any step.
  - It also can't serve the phase check's condition, whose acceptance tests read GitHub's contexts only.
  - So both stay as they are (round 3, Codex P3-1).

### Fix rounds

- Round 1 (both reviewers):
  - The stand-in route was candidate-chosen and stayed open after T2.0 landed. It is now pinned, with every condition read from git, plus negative controls.
  - It now gates T2.0's own landing commit and its phase push.
  - Protected changes on `rust-core` beyond the base landing were advisory. They now fail unless an owner row names their commits (rule `epic`).
  - `ci.yml`'s cap on phases 0 to 9 is gone.
  - P3s fixed: `land.py verify` infers the landing's phase; the phase error is named in `bridge_freeze.py` and `replay_jobs.py`; the Codex pane helper's default is restored; the watchdogs fail loudly with no tasks directory at all.
- Round 2 (both reviewers):
  - Rule `epic` now takes its rows only from the checkpoint, exempts no landing-shaped commit after it, and follows the run rather than the remote tip.
  - The bootstrap's override variable is gone, and the bootstrap is restricted to T2.0's own commits and first landing, which must carry Reviewed-by.
  - `ci.yml`'s phase predicate requires a final digit, and fails by name on what it still lets through.
  - The tracker no longer claims rule `epic` runs before T2.0's landing is on `rust-core`.
- Round 3 (both reviewers), fix 4:
  - Rule `epic` reads paths NUL-separated.
  - It judges every commit after the phase merge against its first parent, and attributes what the merge brings to the commits that made it.
  - It runs on the explicit `--epic` mode, a local replay included.
  - `rust-core`'s base comes only from a phase merge's reviewed landing, never from the head's trailers.
  - The threat model and the open owner questions are recorded.
