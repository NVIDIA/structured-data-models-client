# Orchestration helpers for the Rust core migration

Small tools the orchestrator runs around `../land.py` and `../lane.py` while
many agents work in parallel lanes. They land approved packages, cancel CI
runs nobody needs, and watch agent transcripts for stuck calls and rule
breaks. None of them land anything on their own authority: `land.py` makes
every check.

## Settings shared by all tools

| Variable | Meaning | Default |
|---|---|---|
| `SDM_ORCH_STATE` | State directory: `land-queue/`, `logs/`, `lanes_done`, `rule_watch.ignore`, the lander's lock | `~/.local/share/sdm-orchestration` |
| `SDM_LAND_QUEUE` | The landing queue | `$SDM_ORCH_STATE/land-queue` |
| `SDM_PHASE_BRANCH` | The phase branch, e.g. `rust-core-phase-3`. Lanes are `<phase>-lane-<x>`, topic branches `<phase>-<name>`, archives `<phase>-archive-...` | none; required by `auto_land.py`, `land_chain.sh`, `cancel_superseded.sh` and `lane_idle.py`, which exit 2 when it is unset (the two Python tools also take `--phase`). The tools beside them read it too (`../phases.py`, T2.0): `land.py` and `lane.py` (whose `--phase-branch` overrides it), `ratchet.py`, `prepush.sh`, `prune_caches.py`, `replay_jobs.py` and `bridge_freeze.py`; unset, they take the phase of the branch they work on, else `phases.py`'s `CURRENT` |
| `SDM_REPO` | The local clone | the git toplevel of this directory |
| `SDM_TASKS_DIR` | The agents' task-output directory (`a*.output` transcripts) | none; required by the transcript tools |

## Tools

- **`auto_land.py`**: for each `land-queue/<pkg>.json`, waits for green CI on
  the exact candidate, reruns Docker rate-limit failures, and calls `land.py`
  with `--phase-branch <phase>`. When the candidate is behind the phase head,
  it rebases it automatically if the rebase is clean, and never resolves
  conflicts. A queue entry it cannot read (bad JSON, a missing key) prints
  `FAILED <file>: <error>` and becomes `<pkg>.failed`; the other entries are
  still processed. Also reads `SDM_TOOLS_PYTHON` (the Python for
  `ratchet.py`, default: the Python running it) and `SDM_CLEAN_AFTER_LAND`
  (see below). Typical use: `python3 auto_land.py` every few minutes. It
  prints `LANDED`, `FAILED`, `REBASED`, `MOVED`, `CONFLICT` or
  `NEEDS-CONFIRM` lines.
  - After a landing it leaves the lane's build directory alone, so lanes
    keep their Cargo target across landings. Set `SDM_CLEAN_AFTER_LAND=1` to
    have it run `clean_tmp.sh --apply --after-land <lane worktree>` after
    each landing instead, which deletes that lane's Cargo target unless a
    process is using it (logged to `SDM_AFTER_LAND_LOG`, default
    `$SDM_ORCH_STATE/logs/after-land.log`).
- **`auto_land_selftest.py`**: tests `auto_land.py` and `cancel_superseded.sh`
  against a temporary git origin with a fake `gh` and a fake `land.py`, and
  also checks that `SDM_PHASE_BRANCH` is required, that `land_chain.sh`
  refuses an unqueued package, and that `clean_tmp.sh` never lists a
  directory holding a git worktree. It needs no network. Run it after
  changing any of these tools: `python3 auto_land_selftest.py`.
- **`land_chain.sh`**: lands a stack of packages in order. It first checks
  that every package in `ORDER` has a `<pkg>.next` or `<pkg>.json` (or
  `<pkg>.landed`, when restarting a chain) in the queue, and exits 1 naming
  the ones that do not. Then it waits until every stack CI run has finished
  and is green, queues **one package at a time** (`<pkg>.next` to
  `<pkg>.json`) and runs the watchdog checks every 2 minutes. Needs `ORDER`,
  `SHAS`, `SDM_TASKS_DIR` and `SDM_PHASE_BRANCH`. Example:
  `ORDER="t2.1-p1 t2.2-p1" SHAS="<sha1> <sha2>" SDM_TASKS_DIR=... ./land_chain.sh`.
- **`cancel_superseded.sh`**: cancels unfinished CI runs on lane and queued
  topic branches whose commit is no longer the branch head. It never cancels
  a landing run (any run titled `Land ...`, `Land T<n>...` and
  `Land CI<n>...` alike) or a moved candidate's evidence run. Example:
  `./cancel_superseded.sh --dry-run`.
- **`mech_confirm.py`**: confirms that a rebased tip carries exactly the
  approved commits, then writes confirmation files that cite the original
  reviews. Example: `python3 mech_confirm.py --repo . --old-base A --old-tip X
  --new-base B --new-tip C --review r1.md --review r2.md --out-dir DIR --name PKG`.
- **`stuck_agents.py`**: lists agent tool calls that have had no result for a
  given number of minutes. A blocked call usually means the agent is waiting
  for approval. Waits that are expected (`gate.sh`, `until` and `sleep`
  loops, `herdr`, ...) are labelled `expected-wait`, until they are older
  than `SDM_EXPECTED_WAIT_CEILING_MIN` minutes (default 90): then they are
  reported as `POSSIBLY-BLOCKED` too, since a hung loop or gate run looks the
  same. Example: `python3 stuck_agents.py "$SDM_TASKS_DIR" 40`.
- **Both watchdogs fail loudly** when they cannot see (`transcripts.py`,
  shared by the two): a tasks directory that does not exist, a transcript
  that cannot be read, or a line in one that is not a JSON object prints
  `WATCHDOG-ERROR <tool>: <what>` on stdout and exits 1, so the loop's grep
  (prompts.md, "Watchdog texts", and `land_chain.sh`) keeps it. A partial
  last line, which the agent is still writing, is skipped, not an error.
  `tests/test_watchdogs.py` tests both, and runs the loop text as written.
- **`rule_watch.py`**: flags process-rule violations in recent agent
  commands: a local `prepush.sh --full`, `gh run watch`, `rm -r`, forced
  removals and `sudo`. To silence a violation you have already dealt with,
  add a regular expression to `$SDM_ORCH_STATE/rule_watch.ignore`, one per
  line (an agent id alone silences that agent). Example:
  `python3 rule_watch.py "$SDM_TASKS_DIR" 6`.
- **`codex_pane.py`**: runs Codex reviews in herdr panes, one pane per
  package, kept alive across rounds. `open --package ID` creates the
  package's tab in the `codex-reviews` workspace, a review worktree under
  `$SDM_ORCH_STATE/review-worktrees/`, and starts Codex there; `review
  --package ID --sha SHA --brief FILE` checks the worktree out at the exact
  SHA and sends the round's prompt, naming a fresh report file (at most
  `SDM_CODEX_CAP`, default 4, at once; `--queue` waits for a slot);
  `wait --package ID` blocks until the round settles and prints one line
  (`REPORT`, `BLOCKED`, `IDLE-NO-REPORT`, `CAPACITY`, `GONE` or `TIMEOUT`),
  retrying a capacity error with backoff first; run it in the background;
  `list` shows the registry (`$SDM_ORCH_STATE/codex-panes.json`). It never
  prints herdr's raw JSON, which carries terminal titles with user and host
  names. Also reads `SDM_HERDR`, `SDM_CODEX_MODEL`, `SDM_CODEX_EFFORT` and
  the `SDM_CODEX_*` timings. `codex_pane_selftest.py` tests it against a
  fake herdr and a throwaway repository, with no network;
  `tests/test_codex_pane.py` runs it.
- **`lane_idle.py`**: lists lanes no agent has touched recently. Lanes are
  found from `origin/<phase>-lane-*` and from `<repo>-lane-<x>` worktree
  directories (`SDM_LANE_WORKTREE_PREFIX` overrides the prefix). Lanes named
  in `$SDM_ORCH_STATE/lanes_done` are skipped. Example:
  `python3 lane_idle.py "$SDM_TASKS_DIR" 25`.
- **`clean_tmp.sh`**: removes stale agent scratch from `/tmp`, from
  `$SDM_SCRATCH` (if set), and Cargo `target/` directories. It is a **dry run
  unless you pass `--apply`**. It never removes tracked files, a git
  worktree, `~/code`, the repo or the state directory, nor a directory that
  holds any of them: one with a `.git` up to 4 levels below it, or one under
  which a worktree that `git -C "$SDM_REPO" worktree list` names lies (so a
  reviewer's throwaway worktree in its scratch directory is safe).
  Examples: `./clean_tmp.sh` to see what would go, and
  `./clean_tmp.sh --apply` to remove it.

- **`guard_cmd.py`** (T3.0): the Claude Code PreToolUse hook the
  repository's `.claude/settings.json` runs before every Bash and Agent
  call. It denies the banned commands with the rule and the alternative,
  lets an owner-approved one-off through as `SDM_APPROVED=<id> <command>`
  (logged; `--tokens` lists the ids), and refuses a reviewer launch on a SHA
  without a handoff stamp (`../handoff.py`). The handbook's "The command
  hook" section has the list. `rule_watch.py` still watches Codex panes.
- **`auto_land.py`'s stack entries** (T3.0): `"stack": true` stacks an
  approved package with `land.py --stack`; `{"stack_tip": SHA, ...}` lands
  the stack with `land.py stack-land` once its top's run is green.

## Phase-end, review and housekeeping tools

Each has `--help`; anything that changes state is a dry run unless told otherwise.

- **`phase_end.sh`**: the end-of-phase driver: preflight, the CI wheel and cells, the evidence pack (`evidence.py`), the one-shot held-out run (`../heldout_run.sh`), an optional live install check (`SDM_LIVE_CHECK_CMD`), and the pack. Stamps let a rerun resume without repeating the one-shot. `--execute` to act.
- **`final_gate.sh`**: fetches the exact-SHA CI wheels and runs `gate.sh --locked` for each half named, into one evidence file.
- **`packet_checks.sh PHASE`**: read-only go or no-go checks: landings without gate evidence, plan markers against decisions, rows awaiting the owner, proposed register entries.
- **`register_tools.py`**: `proposed` lists proposed register entries and rows awaiting the owner; `resolve` merges `register.yaml` by entry and pending JSON files by group, usable as a git merge driver (setup lines in `--help`).
- **`branch_cleanup.py`**: end-of-phase branch and worktree cleanup: `audit`, `plan` (with a keep-list), `execute` (dry run unless `--apply`; writes and verifies a backup bundle and a restore manifest first, re-checks each item before deleting, stops at the first failure, never deletes anything not merged into the base).
- **`phase_stats.py`**: read-only retrospective numbers: landings per day, package stage times, CI per day with failures split into infrastructure and real, and review verdicts with same-SHA pairing of the two reviewers.

Outside this folder: `../cells.sh` builds warm, non-editable cell venvs keyed by tree hash; `../heldout.py` and `../heldout_run.sh` validate, freeze, run and count a held-out set; `scripts/live/dbx_runs.py` and `scripts/live/dbr_install_check*` run one-time Databricks checks, with every workspace-specific value from the environment.

## Rules these tools follow, and probes should too

- The tools never run `gh run watch`. They poll with `gh run list` and
  return, so a stalled watch cannot hang the orchestrator.
- Run any probe or one-off command with a time limit and with stdin from
  `/dev/null`, so it can neither hang nor wait for input:
  `perl -e 'alarm shift; exec @ARGV' 600 cmd </dev/null`.
- `land_chain.sh` queues one package at a time. Landing a package while later
  stack runs are still in flight breaks their checkpoint.
