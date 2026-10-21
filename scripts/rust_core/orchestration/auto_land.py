#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Land queued packages once CI on their exact candidate is green (owner-approved 2026-10-12).

Each file in <state>/land-queue/*.json describes one approved candidate:
  {"lane": "1a", "worktree": "...", "candidate": SHA, "reviews": [path, path],
   "title": "Land T1.x: ...", "checkpoint": SHA, "protected_paths": "a; b", "summary": "..."}
land.py itself enforces every rule (both reviews approve the SHA, CI green, descends from
the phase head, protected paths). This script only waits for CI, reruns jobs that failed on
a Docker pull rate limit, and calls land.py (with --phase-branch <phase>). Prints one line
per event:
  LANDED <title> <landing sha> | FAILED <title>: <reason> | (nothing while waiting)
An entry it cannot read or process (bad JSON, a missing key) prints  FAILED <file>: <error>,
becomes <pkg>.failed and the other entries are still processed.

Clean rebase (added 2026-10-13). When land.py refuses because the candidate does not
descend from the phase head, and the lane worktree is clean with HEAD == the candidate,
this script runs `git rebase <phase head>` there. It never resolves a conflict:
  - replays with no conflict: pushes (atomically) the old candidate to
    <phase>-archive-<lane>-<Pkg>-<old8> and the rebased tip to the lane branch
    with --force-with-lease=<lane ref>:<old candidate>; writes
    land-queue/<pkg>.rebased.rangediff; renames the queue file to <pkg>.needs-confirm
    (candidate and checkpoint updated, the old candidate as rebased_from) and prints
      REBASED <pkg> <old> -> <new> (needs confirmation reviews)
    It never lands the rebased SHA: the orchestrator runs the two confirmation reviews on
    the new SHA, puts their paths in "reviews" and renames the file back to <pkg>.json.
  - conflicts: `git rebase --abort`, the queue file becomes <pkg>.failed with a
    "conflicts" field, and it prints  CONFLICT <pkg> <files>
  - dirty worktree / HEAD is not the candidate: no rebase; .failed as before, with the
    reason in "rebase_refused".
Every run also prints  NEEDS-CONFIRM <pkg> ...  for each <pkg>.needs-confirm still waiting.

Topic branches (added 2026-10-14). An entry may carry "branch": "<phase>-<name>"
(a topic branch, not a lane; "lane" may then be absent). It is used wherever a lane entry
uses <phase>-lane-<lane>: the CI lookup, the auto-rebase's lease and push ref, and
land.py's --topic-branch; its archive branch is <phase>-archive-<name>-<Pkg>-<old8>.
Entries without "branch" behave as before.

Moved over landing commits (CI3, land.py --moved-from; owner's decision E130, 2026-10-14:
no reviewer re-run and no serial re-CI after a clean move). Opt-in per entry, "moved_ok": true.
  - A clean auto-rebase of a moved_ok entry archives and lease-pushes as above, then keeps the
    file as <pkg>.json with "moved_from" = X (the entry's existing moved_from, else the old
    candidate: the commit with the green run and the reviews; "reviews" is not changed),
    "candidate" = the new tip and "checkpoint" unchanged (the range-diff is written against X),
    and prints  MOVED <pkg> <old8> -> <new8> (lands on <X8>'s evidence)
  - An entry with "moved_from" X is judged on X's CI run (runs_for(X, branch); the same
    success / Docker-limit rerun / failure handling), and land.py gets --moved-from X, the
    reviews as they are, and --checkpoint = git merge-base <candidate> <phase head> (land.py
    refuses any other checkpoint; the queued one, X's, is kept in the file and not passed).
    The ratchet's protected paths are read against that same merge base.
  - land.py refusing the move itself ("--moved-from <X>: not the same code / patches /
    moved over landing commits / only landing commits", CI3's check_moved) means the move was
    not over landing commits only: the file becomes <pkg>.needs-confirm without moved_from,
    with "moved_refused" (the reason), rebased_from = X and checkpoint = that merge base, and
    the candidate's own CI run is rerun if it was cancelled (cancel_superseded.sh cancels it
    while X's run is the evidence). Any other refusal fails the entry as before.

Environment:
  SDM_ORCH_STATE      state directory (default ~/.local/share/sdm-orchestration): holds
                      land-queue/, logs/ and the lock that keeps one lander at a time
  SDM_LAND_QUEUE      the queue directory (default <state>/land-queue)
  SDM_PHASE_BRANCH    the phase branch, e.g. rust-core-phase-3 (required unless --phase
                      gives it; exits 2 when neither does); lanes are <phase>-lane-<x>,
                      topic branches <phase>-<name>
  SDM_TOOLS_PYTHON    the Python that runs `ratchet.py check` first (default: the Python
                      running this script; the worktree's .venv* and python3 are the fallbacks)
  SDM_CLEAN_AFTER_LAND  set to 1 to run `clean_tmp.sh --apply --after-land <worktree>` after
                      each landing, which deletes the lane's Cargo target (default: off, so
                      lanes keep their build directories across landings)
  SDM_AFTER_LAND_LOG  that cleanup's log (default <state>/logs/after-land.log)
SDM_LAND_QUEUE and SDM_AFTER_LAND_LOG are used by auto_land_selftest.py.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

R = 'NVIDIA/structured-data-models-client'
STATE = pathlib.Path(
    os.environ.get(
        'SDM_ORCH_STATE',
        pathlib.Path.home() / '.local/share/sdm-orchestration',
    )
).expanduser()
Q = pathlib.Path(
    os.environ.get('SDM_LAND_QUEUE', STATE / 'land-queue')
).expanduser()
REMOTE = 'origin'
PHASE = (
    os.environ.get('SDM_PHASE_BRANCH') or ''
)  # required: checked in __main__
NOT_DESCENDED = 'does not descend from the head of'


def topic_branch_re(phase):
    # land.py's own refusals of a topic branch name (scripts/rust_core/land.py TOPIC_BRANCH).
    return re.compile(
        re.escape(phase) + r'-[A-Za-z0-9](?:[A-Za-z0-9_-]|\.(?!\.))*'
    )


TOPIC_BRANCH = None  # topic_branch_re(PHASE), set in __main__
# CI3's check_moved refusals: the candidate is not X moved over landing commits only.
MOVE_REFUSED = re.compile(
    r'--moved-from [0-9a-f]{7,40}: not (?:the same code|the same patches'
    r'|moved over landing commits|only landing commits)'
)


def sh(*a, cwd=None):
    return subprocess.run(a, capture_output=True, text=True, cwd=cwd)


def git(wt, *a):
    # rerere off and no ref updates, so a rebase can neither auto-resolve a conflict
    # from a recorded resolution nor move any other branch.
    return sh(
        'git',
        '-c',
        'rerere.enabled=false',
        '-c',
        'rebase.updateRefs=false',
        '-c',
        'rebase.autoStash=false',
        *a,
        cwd=wt,
    )


def runs_for(sha, branch):
    out = (
        sh(
            'gh',
            'run',
            'list',
            '-R',
            R,
            '--branch',
            branch,
            '--limit',
            '20',
            '--json',
            'databaseId,headSha,status,conclusion',
        ).stdout
        or '[]'
    )
    return [r for r in json.loads(out) if r['headSha'] == sha]


def infra_only(run_id):
    jobs = json.loads(
        sh('gh', 'run', 'view', str(run_id), '-R', R, '--json', 'jobs').stdout
        or '{}'
    ).get('jobs', [])
    bad = [j for j in jobs if j.get('conclusion') == 'failure']
    if not bad:
        return False
    for j in bad:
        log = sh(
            'gh', 'api', f'repos/{R}/actions/jobs/{j["databaseId"]}/logs'
        ).stdout
        if 'toomanyrequests' not in log:
            return False
    return True


def write_as(f, q, suffix):
    """Rename the queue file to <stem><suffix> holding q."""
    dest = f.with_suffix(suffix)
    f.rename(dest)
    dest.write_text(json.dumps(q, indent=1) + '\n')
    return dest


def branch_of(q):
    """The entry's topic branch: its "branch" when given, else its lane's branch."""
    b = q.get('branch')
    if b is None:
        return f'{PHASE}-lane-{q["lane"]}'
    if (
        not isinstance(b, str)
        or not TOPIC_BRANCH.fullmatch(b)
        or b.endswith(('.', '.lock'))
        or b.startswith(f'{PHASE}-archive-')
    ):
        raise ValueError(f'"branch" {b!r} is not a {PHASE}-<name> topic branch')
    return b


def archive_name(branch, pkg, sha):
    # t2.1e-p1 -> T2.1e-p1, as the existing archive branches
    p = pkg[:1].upper() + pkg[1:]
    name = branch[len(PHASE) + 1 :]
    # A lane keeps its old archive names: archive-<lane>-...
    if name.startswith('lane-'):
        name = name[len('lane-') :]
    return f'{PHASE}-archive-{name}-{p}-{sha[:8]}'


def auto_rebase(f, q, branch):
    """Called after land.py refused a non-descended candidate. Returns the line to print."""
    wt, old, pkg = q['worktree'], q['candidate'], f.stem
    lane_ref = f'refs/heads/{branch}'
    git(wt, 'fetch', '-q', REMOTE)
    base = git(wt, 'rev-parse', f'refs/remotes/{REMOTE}/{PHASE}').stdout.strip()
    head = git(wt, 'rev-parse', 'HEAD').stdout.strip()
    status = git(wt, 'status', '--porcelain', '--untracked-files=all')
    why = None
    if status.returncode != 0:
        why = f'git status failed in {wt}: {status.stderr.strip()[:200]}'
    elif status.stdout.strip():
        why = f'worktree {wt} is not clean'
    elif head != old:
        why = f'worktree HEAD {head[:12]} is not the candidate {old[:12]}'
    elif not base:
        why = f'cannot resolve {REMOTE}/{PHASE}'
    elif git(wt, 'merge-base', '--is-ancestor', base, old).returncode == 0:
        why = f'the candidate already descends from {PHASE} ({base[:12]}); land.py refused for another reason'
    elif any(
        os.path.exists(
            os.path.join(
                wt, git(wt, 'rev-parse', '--git-path', d).stdout.strip()
            )
        )
        for d in ('rebase-merge', 'rebase-apply')
    ):
        why = 'a rebase is already in progress in the worktree'
    if why:
        q['rebase_refused'] = why
        write_as(f, q, '.failed')
        return f'FAILED {q["title"]}: not descended from {PHASE}; no automatic rebase: {why}'

    old_base = git(wt, 'merge-base', old, base).stdout.strip()
    rb = git(wt, 'rebase', '--no-autosquash', base)
    if rb.returncode != 0:
        conflicts = git(
            wt, 'diff', '--name-only', '--diff-filter=U'
        ).stdout.split()
        git(wt, 'rebase', '--abort')
        after = git(wt, 'rev-parse', 'HEAD').stdout.strip()
        q['conflicts'] = conflicts
        q['rebase_onto'] = base
        if after != old:
            q['rebase_abort_warning'] = (
                f'HEAD is {after} after --abort, expected {old}'
            )
        if not conflicts:
            q['rebase_error'] = ' '.join((rb.stdout + rb.stderr).split())[-500:]
        write_as(f, q, '.failed')
        if conflicts:
            return f'CONFLICT {pkg} {" ".join(conflicts)}'
        return f'FAILED {q["title"]}: git rebase onto {base[:12]} failed without conflicts: {q["rebase_error"][-200:]}'

    new = git(wt, 'rev-parse', 'HEAD').stdout.strip()
    archive = archive_name(branch, pkg, old)
    push = git(
        wt,
        'push',
        '--atomic',
        '--porcelain',
        f'--force-with-lease={lane_ref}:{old}',
        REMOTE,
        f'{old}:refs/heads/{archive}',
        f'{new}:{lane_ref}',
    )
    if push.returncode != 0:
        # Put the lane worktree back on the old candidate; --keep refuses to drop local changes.
        reset = git(wt, 'reset', '--keep', old)
        q['rebase_refused'] = (
            f'rebased cleanly to {new} but the push failed: '
            f'{" ".join((push.stdout + push.stderr).split())[-300:]}'
            + (
                ''
                if reset.returncode == 0
                else f'; reset --keep to the old candidate FAILED: {" ".join(reset.stderr.split())[:200]}'
            )
        )
        write_as(f, q, '.failed')
        return f'FAILED {q["title"]}: {q["rebase_refused"]}'

    rd_path = Q / f'{pkg}.rebased.rangediff'
    if q.get('moved_ok') is True:
        # X keeps the green run and the reviews; land.py --moved-from decides the move was clean.
        x = q.get('moved_from') or old
        x_base = (
            git(wt, 'merge-base', x, base).stdout.strip()
            if x != old
            else old_base
        )
        rd = git(
            wt, 'range-diff', '--no-color', f'{x_base}..{x}', f'{base}..{new}'
        )
        rd_path.write_text(
            f'# git range-diff {x_base}..{x} {base}..{new}\n'
            + rd.stdout
            + rd.stderr
        )
        q.update(
            candidate=new,
            moved_from=x,
            rebased_from=old,
            archive_branch=archive,
            rangediff=str(rd_path),
        )
        f.write_text(json.dumps(q, indent=1) + '\n')
        return (
            f"MOVED {pkg} {old[:8]} -> {new[:8]} (lands on {x[:8]}'s evidence)"
        )
    # Not moved_ok: the confirmation reviews and the new SHA's own run are the evidence now.
    if 'moved_from' in q:
        q['moved_from_dropped'] = q.pop('moved_from')
    rd = git(
        wt, 'range-diff', '--no-color', f'{old_base}..{old}', f'{base}..{new}'
    )
    rd_path.write_text(
        f'# git range-diff {old_base}..{old} {base}..{new}\n'
        + rd.stdout
        + rd.stderr
    )
    q.update(
        candidate=new,
        checkpoint=base,
        rebased_from=old,
        archive_branch=archive,
        rangediff=str(rd_path),
    )
    write_as(f, q, '.needs-confirm')
    return f'REBASED {pkg} {old} -> {new} (needs confirmation reviews)'


def move_refused(f, q, out, branch, checkpoint):
    """land.py refused --moved-from X: the move was not over landing commits only."""
    x = q.pop('moved_from')
    lines = out.splitlines()
    at = next((i for i, ln in enumerate(lines) if MOVE_REFUSED.search(ln)), 0)
    reason = ' '.join(' '.join(lines[at:]).split())[:600]
    q.update(rebased_from=x, moved_refused=reason, checkpoint=checkpoint)
    # The candidate's own run is now needed; cancel_superseded.sh cancels it while X is evidence.
    own = runs_for(q['candidate'], branch)
    if (
        own
        and own[0]['status'] == 'completed'
        and own[0]['conclusion'] == 'cancelled'
    ):
        rr = sh('gh', 'run', 'rerun', str(own[0]['databaseId']), '-R', R)
        q['ci_rerun'] = (
            own[0]['databaseId']
            if rr.returncode == 0
            else f'rerun failed: {rr.stderr.strip()[:200]}'
        )
    write_as(f, q, '.needs-confirm')


def report_needs_confirm():
    for f in sorted(Q.glob('*.needs-confirm')):
        try:
            q = json.loads(f.read_text())
            why = (
                f' (--moved-from refused: {q["moved_refused"][:300]})'
                if q.get('moved_refused')
                else ''
            )
            print(
                f'NEEDS-CONFIRM {f.stem} {q.get("candidate", "?")} rebased from '
                f'{q.get("rebased_from", "?")}: two confirmation reviews on the new SHA '
                f'(range-diff {q.get("rangediff", "?")}), then re-queue as {f.stem}.json{why}'
            )
        except Exception as e:
            print(f'NEEDS-CONFIRM {f.stem} (unreadable: {e})')


PROTECTED_LINE = 'protected: changed, for the landing to name: '


def protected_from_ratchet(worktree, checkpoint):
    """The protected paths `ratchet.py check` says the landing must name, or None."""
    wt = pathlib.Path(worktree)
    tools = pathlib.Path(os.environ.get('SDM_TOOLS_PYTHON') or sys.executable)
    for py in (
        tools,
        wt / '.venv-ref-pandas3/bin/python',
        wt / '.venv/bin/python',
        pathlib.Path('/usr/bin/env'),
    ):
        cmd = (
            [str(py)]
            + (['python3'] if py.name == 'env' else [])
            + [
                'scripts/rust_core/ratchet.py',
                'check',
                '--checkpoint',
                checkpoint,
            ]
        )
        if py.name != 'env' and not py.exists():
            continue
        res = sh(*cmd, cwd=worktree)
        out = res.stdout + res.stderr
        if 'ModuleNotFoundError' in out or 'Traceback' in out:
            continue
        return [
            ln.split(PROTECTED_LINE, 1)[1].strip()
            for ln in out.splitlines()
            if PROTECTED_LINE in ln
        ]
    return None


def process(f):
    q = json.loads(f.read_text())
    if not isinstance(q, dict):
        raise ValueError('the entry is not a JSON object')
    try:
        branch = branch_of(q)
    except ValueError as e:
        print(f'FAILED {q.get("title", f.stem)}: {e}')
        f.rename(f.with_suffix('.failed'))
        return
    # A moved candidate is judged on X's run (E130: no re-CI after a clean move).
    x = q.get('moved_from')
    runs = runs_for(x or q['candidate'], branch)
    if not runs:
        return
    run = runs[0]
    if run['status'] != 'completed':
        return
    if run['conclusion'] != 'success':
        if infra_only(run['databaseId']):
            marker = f.with_suffix('.reruns')
            n = int(marker.read_text()) if marker.exists() else 0
            if n < 4:
                sh(
                    'gh',
                    'run',
                    'rerun',
                    str(run['databaseId']),
                    '-R',
                    R,
                    '--failed',
                )
                marker.write_text(str(n + 1))
                return
        on = f' on the moved-from commit {x[:12]}' if x else ''
        print(
            f'FAILED {q["title"]}: CI run {run["databaseId"]} {run["conclusion"]}{on} (not only a Docker limit)'
        )
        f.rename(f.with_suffix('.failed'))
        return
    # The ratchet's own list of protected paths is authoritative (2026-10-14: two landings
    # went red because a hand-built list missed protected files). Union it with the queued one;
    # refuse to land if the ratchet can't be run.
    checkpoint = q['checkpoint']
    if x:
        # land.py refuses a --checkpoint other than merge-base(candidate, phase head); the
        # queued one stays X's. Read the ratchet against the same merge base.
        sh('git', 'fetch', '-q', cwd=q['worktree'])
        checkpoint = sh(
            'git',
            'merge-base',
            q['candidate'],
            f'refs/remotes/{REMOTE}/{PHASE}',
            cwd=q['worktree'],
        ).stdout.strip()
        if not checkpoint:
            print(
                f'FAILED {q["title"]}: cannot compute the merge base of {q["candidate"][:12]} '
                f'and {REMOTE}/{PHASE} in {q["worktree"]}'
            )
            f.rename(f.with_suffix('.failed'))
            return
    named = protected_from_ratchet(q['worktree'], checkpoint)
    if named is None:
        print(
            f"FAILED {q['title']}: could not read the ratchet's protected paths"
        )
        f.rename(f.with_suffix('.failed'))
        return
    queued = [p.strip() for p in q['protected_paths'].split(';') if p.strip()]
    paths = '; '.join(sorted(set(queued) | set(named)))
    args = [
        'python3',
        'scripts/rust_core/land.py',
        '--candidate',
        q['candidate'],
        '--phase-branch',
        PHASE,
        '--topic-branch',
        branch,
        '--title',
        q['title'],
        '--checkpoint',
        checkpoint,
        '--protected-paths',
        paths,
        '--summary',
        q['summary'],
    ]
    if x:
        args += ['--moved-from', x]
    for r in q['reviews']:
        args += ['--review', r]
    sh('git', 'fetch', '-q', cwd=q['worktree'])
    res = sh(*args, cwd=q['worktree'])
    out = res.stdout + res.stderr
    landed = [ln for ln in out.splitlines() if ln.startswith('LANDED ')]
    if landed:
        print(f'LANDED {q["title"]} {landed[0].split()[1]}')
        f.rename(f.with_suffix('.landed'))
        if os.environ.get('SDM_CLEAN_AFTER_LAND') == '1':
            # Opt-in (owner-approved 2026-10-13, step 4): drop the landed lane's build output
            # unless in use. Off by default: lanes keep their Cargo target across landings.
            cl = sh(
                str(pathlib.Path(__file__).with_name('clean_tmp.sh')),
                '--apply',
                '--after-land',
                q['worktree'],
            )
            log_path = pathlib.Path(
                os.environ.get('SDM_AFTER_LAND_LOG')
                or STATE / 'logs' / 'after-land.log'
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a') as log:
                log.write(f'{q["title"]}: {cl.stdout.strip()}\n')
    elif NOT_DESCENDED in out:
        print(auto_rebase(f, q, branch))
    elif x and MOVE_REFUSED.search(out):
        # reported by report_needs_confirm() in main()
        move_refused(f, q, out, branch, checkpoint)
    else:
        tail = ' | '.join(out.strip().splitlines()[-3:])
        print(f'FAILED {q["title"]}: land.py refused: {tail[:300]}')
        f.rename(f.with_suffix('.failed'))


def main():
    for f in sorted(Q.glob('*.json')):
        # A malformed entry (bad JSON, a missing key) fails alone; the others still run.
        try:
            process(f)
        except Exception as e:
            print(f'FAILED {f.name}: {type(e).__name__}: {e}')
            if f.exists():
                f.rename(f.with_suffix('.failed'))
    report_needs_confirm()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--phase',
        default=PHASE,
        help='the phase branch, e.g. rust-core-phase-3 (default $SDM_PHASE_BRANCH; one is required)',
    )
    PHASE = ap.parse_args().phase
    if not PHASE:
        ap.error(
            'set SDM_PHASE_BRANCH or pass --phase: the phase branch, e.g. rust-core-phase-3'
        )
    TOPIC_BRANCH = topic_branch_re(PHASE)
    import fcntl

    STATE.mkdir(parents=True, exist_ok=True)
    _lock = open(STATE / 'auto-land.lock', 'w')
    try:  # one lander at a time: a second concurrent run exits quietly
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(0)
    main()
