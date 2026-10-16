#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-test for auto_land.py's automatic clean rebase. No network, no real repo.

Each case builds a fresh bare "origin" in a temp dir under /tmp, a lane worktree cloned from
it, a fake `gh` (reports a green run on the candidate) and a fake scripts/rust_core/land.py
(refuses with land.py's "does not descend" message when the candidate is behind the phase
head), then runs auto_land.py as a subprocess with SDM_LAND_QUEUE pointing at a temp queue.

Cases: clean replay, conflict, dirty worktree, HEAD != candidate, lease rejected (lane moved);
(2026-10-14) a topic-branch entry ("branch"), an invalid "branch", and moved_ok entries (CI3,
land.py --moved-from): a clean move lands on X's evidence, a second move keeps X, X's run
decides, a move CI3 refuses goes to .needs-confirm, another refusal fails as before; and
cancel_superseded.sh (topic branches, moved_from runs protected, a moved head's run cancelled,
every "Land ..." run spared, CI<n> landings included). Also (2026-10-15): malformed queue entries
fail alone; SDM_PHASE_BRANCH is required by auto_land.py, cancel_superseded.sh, lane_idle.py and
land_chain.sh; land_chain.sh refuses an ORDER entry with nothing queued; clean_tmp.sh (dry run)
never lists a scratch directory holding a git worktree or clone.
The fake gh can also serve runs per branch from $FAKE_GH_RUNS (a JSON file, newest first),
applies -q with jq, logs to $FAKE_GH_LOG; the fake land.py logs its argv to $FAKE_LAND_LOG,
refuses a --checkpoint that is not the merge base and checks --moved-from like CI3 (same tree,
only empty 'Land ' commits in X..H); a fake ratchet.py logs its --checkpoint.
The phase branch is $SDM_PHASE_BRANCH (default rust-core-phase-2 here; the tools themselves
require it) and is set for every tool run; every other SDM_* variable is cleared and
SDM_ORCH_STATE points into the temp dir, so the real queue, logs and lock are never touched.
Usage: auto_land_selftest.py [--keep]   (--keep leaves the temp dir and prints its path)
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
AUTO_LAND = HERE / 'auto_land.py'
PHASE = os.environ.get('SDM_PHASE_BRANCH') or 'rust-core-phase-2'
LANE = 'zz'  # a lane name no real branch uses
LANE_BR = f'{PHASE}-lane-{LANE}'
PKG = 't9.99-p1'
TOPIC = 'zztopic'  # a topic branch no real branch uses
TOPIC_BR = f'{PHASE}-{TOPIC}'
REVIEWS = ['/nonexistent/a.md', '/nonexistent/b.md']
ENV = dict(
    {k: v for k, v in os.environ.items() if not k.startswith('SDM_')},
    SDM_PHASE_BRANCH=PHASE,
    GIT_AUTHOR_NAME='selftest',
    GIT_AUTHOR_EMAIL='selftest@example.invalid',
    GIT_COMMITTER_NAME='selftest',
    GIT_COMMITTER_EMAIL='selftest@example.invalid',
    GIT_CONFIG_NOSYSTEM='1',
    GIT_TERMINAL_PROMPT='0',
)

FAKE_LAND = """import json, os, subprocess, sys
a = sys.argv; cand = a[a.index('--candidate') + 1]
PHASE = os.environ['SDM_PHASE_BRANCH']
if os.environ.get('FAKE_LAND_LOG'):
    with open(os.environ['FAKE_LAND_LOG'], 'a') as fh: fh.write(json.dumps(a[1:]) + '\\n')
opt = lambda n: a[a.index(n) + 1] if n in a else None
git = lambda *x: subprocess.run(['git', *x], capture_output=True, text=True).stdout.strip()
subprocess.run(['git', 'fetch', '-q', 'origin'])
head = subprocess.run(['git', 'rev-parse', 'refs/remotes/origin/' + PHASE],
                      capture_output=True, text=True).stdout.strip()
if subprocess.run(['git', 'merge-base', '--is-ancestor', head, cand]).returncode:
    print(f'refused: the candidate {cand[:12]} does not descend from the head of '
          f'{PHASE} ({head[:12]}); rebase it onto that head and gate it again')
    sys.exit(1)
mb = git('merge-base', cand, head)
if opt('--checkpoint') and git('rev-parse', opt('--checkpoint')) != mb:
    print(f'REFUSED: the checkpoint {opt("--checkpoint")[:12]} is not the merge base of the '
          f'candidate and the phase head, {mb[:12]}'); sys.exit(2)
if os.environ.get('FAKE_LAND_REFUSE'):
    print('REFUSED: ' + os.environ['FAKE_LAND_REFUSE']); sys.exit(2)
x = opt('--moved-from')
if x:
    label = f'--moved-from {x[:12]}'
    if git('rev-parse', x + '^{tree}') != git('rev-parse', cand + '^{tree}'):
        print(f"REFUSED: {label}: not the same code: the candidate's tree is not the moved-from "
              "commit's; a candidate whose code differs needs its own reviews and CI"); sys.exit(2)
    for line in git('log', '--format=%H %s', f'{x}..{head}').splitlines():
        sha, subj = line.split(' ', 1)
        if not subj.startswith('Land ') or git('rev-parse', sha + '^{tree}') != git('rev-parse', sha + '^^{tree}'):
            print(f'REFUSED: {label}: not only landing commits between the moved-from commit and '
                  f'the head of {PHASE}:\\n- {sha[:12]} {subj}'); sys.exit(2)
print('LANDED fake' + cand); sys.exit(0)
"""
FAKE_RATCHET = """import os, sys
if os.environ.get('FAKE_RATCHET_LOG'):
    with open(os.environ['FAKE_RATCHET_LOG'], 'a') as fh:
        fh.write(sys.argv[sys.argv.index('--checkpoint') + 1] + '\\n')
"""
FAKE_GH = """#!/usr/bin/env python3
# fake gh. Without $FAKE_GH_RUNS, "run list" reports one green completed run on $FAKE_GH_SHA.
# With it, "run list --branch B" reports that file's runs whose headBranch is B, "run cancel ID"
# marks the run cancelled there. -q is applied with jq -r. Every call is logged to $FAKE_GH_LOG.
import json, os, subprocess, sys
a = sys.argv[1:]
if os.environ.get('FAKE_GH_LOG'):
    with open(os.environ['FAKE_GH_LOG'], 'a') as fh: fh.write(json.dumps(a) + '\\n')
opt = lambda n: a[a.index(n) + 1] if n in a else None
rf = os.environ.get('FAKE_GH_RUNS')
cmd = ' '.join(a[:2])
if cmd == 'run list':
    if rf:
        runs = [r for r in json.load(open(rf)) if opt('--branch') in (None, r.get('headBranch'))]
    else:
        runs = [{"databaseId": 1, "headSha": os.environ.get('FAKE_GH_SHA', ''),
                 "status": "completed", "conclusion": "success"}]
    out = json.dumps(runs)
    if opt('-q'):
        out = subprocess.run(['jq', '-r', opt('-q')], input=out, capture_output=True, text=True).stdout
    sys.stdout.write(out if out.endswith('\\n') or not out else out + '\\n')
elif cmd == 'run cancel' and rf:
    runs = json.load(open(rf))
    for r in runs:
        if str(r['databaseId']) == a[2]:
            r.update(status='completed', conclusion='cancelled')
    json.dump(runs, open(rf, 'w'))
else:
    print('{}')
"""
TRACKER = '| Package | State |\n|---|---|\n| T1.1 | landed |\n| T9.99 | in progress |\n'


def run(*a, cwd=None, check=True, env=ENV):
    r = subprocess.run(a, cwd=cwd, capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise RuntimeError(f'{a} failed: {r.stdout}{r.stderr}')
    return r


def run_limited(*a, env, timeout=60):
    """Run a tool that might loop (land_chain.sh sleeps between polls): (returncode, output).
    On timeout the whole process group is killed and the return code is -9."""
    p = subprocess.Popen(
        a,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return -9, f'{out}\n(timed out after {timeout}s)'
    return p.returncode, out


def g(cwd, *a, check=True):
    return run('git', *a, cwd=cwd, check=check)


def rev(cwd, ref):
    r = g(cwd, 'rev-parse', '--verify', '-q', ref, check=False)
    return r.stdout.strip() or None


def remote_ref(origin, br):
    return rev(origin, f'refs/heads/{br}')


def commit_file(wt, path, text, msg):
    p = pathlib.Path(wt, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    g(wt, 'add', path)
    g(wt, 'commit', '-q', '-s', '-m', msg)
    return rev(wt, 'HEAD')


def make_origin(root):
    """A bare origin and a seed clone on the phase branch at a base commit; queue, fakes, logs."""
    origin = root / 'origin.git'
    seed = root / 'seed'
    q = root / 'queue'
    fake = root / 'fakebin'
    for d in (q, fake):
        d.mkdir()
    (fake / 'gh').write_text(FAKE_GH)
    (fake / 'gh').chmod(0o755)
    g(root, 'init', '-q', '--bare', '-b', PHASE, str(origin))
    g(root, 'init', '-q', '-b', PHASE, str(seed))
    pathlib.Path(seed, 'scripts/rust_core').mkdir(parents=True)
    pathlib.Path(seed, 'scripts/rust_core/land.py').write_text(FAKE_LAND)
    pathlib.Path(seed, 'scripts/rust_core/ratchet.py').write_text(FAKE_RATCHET)
    pathlib.Path(seed, 'docs').mkdir()
    pathlib.Path(seed, 'docs/tracker.md').write_text(TRACKER)
    pathlib.Path(seed, 'src.txt').write_text('a\nb\nc\n')
    g(seed, 'add', '.')
    g(seed, 'commit', '-q', '-m', 'base')
    base = rev(seed, 'HEAD')
    g(seed, 'remote', 'add', 'origin', str(origin))
    return dict(
        origin=origin,
        seed=seed,
        q=q,
        fake=fake,
        base=base,
        root=root,
        gh_log=root / 'gh.log',
        land_log=root / 'land.log',
        ratchet_log=root / 'ratchet.log',
    )


def setup(root, phase_change):
    """origin with phase + lane at base; lane candidate commit; the phase head moved on."""
    s = make_origin(root)
    origin, seed, q, base = s['origin'], s['seed'], s['q'], s['base']
    wt = root / 'lane-wt'
    g(seed, 'push', '-q', 'origin', f'{PHASE}', f'{PHASE}:refs/heads/{LANE_BR}')
    g(root, 'clone', '-q', '-b', LANE_BR, str(origin), str(wt))
    cand = commit_file(
        wt,
        'docs/tracker.md',
        TRACKER.replace('T9.99 | in progress', 'T9.99 | approved'),
        'T9.99: candidate edits the tracker row',
    )
    cand = commit_file(wt, 'feature.txt', 'new feature\n', 'T9.99: the feature')
    g(wt, 'push', '-q', 'origin', f'HEAD:refs/heads/{LANE_BR}')
    # Another lane lands: the phase head moves.
    if phase_change == 'clean':
        commit_file(seed, 'other.txt', 'other lane\n', 'Land T9.98: other lane')
    else:  # the same tracker row, edited differently
        commit_file(
            seed,
            'docs/tracker.md',
            TRACKER.replace('T9.99 | in progress', 'T9.99 | blocked'),
            'Land T9.98: other lane edits the same row',
        )
    g(seed, 'push', '-q', 'origin', PHASE)
    head = rev(seed, 'HEAD')
    qf = q / f'{PKG}.json'
    qf.write_text(
        json.dumps(
            {
                'lane': LANE,
                'worktree': str(wt),
                'candidate': cand,
                'reviews': REVIEWS,
                'title': 'Land T9.99: self-test',
                'checkpoint': base,
                'protected_paths': '',
                'summary': 'self-test',
            }
        )
    )
    s.update(wt=wt, cand=cand, head=head, qf=qf)
    return s


def setup_moved(root, phase_change='landing', moved_ok=True):
    """A topic branch whose candidate X sits on T9.98's candidate Y, both on the topic branch.
    T9.98 lands: the phase head becomes Y plus an empty 'Land T9.98' commit, so X..H is that
    landing commit only. phase_change='code' adds a 'Land' commit that changes a file."""
    s = make_origin(root)
    origin, seed, q, base = s['origin'], s['seed'], s['q'], s['base']
    wt = root / 'topic-wt'
    g(seed, 'push', '-q', 'origin', PHASE, f'{PHASE}:refs/heads/{TOPIC_BR}')
    g(root, 'clone', '-q', '-b', TOPIC_BR, str(origin), str(wt))
    y = commit_file(wt, 'y.txt', 'other package\n', 'T9.98: the other package')
    x = commit_file(wt, 'feature.txt', 'new feature\n', 'T9.99: the feature')
    g(wt, 'push', '-q', 'origin', f'HEAD:refs/heads/{TOPIC_BR}')
    g(seed, 'fetch', '-q', 'origin', TOPIC_BR)
    g(seed, 'merge', '-q', '--ff-only', y)
    g(
        seed,
        'commit',
        '-q',
        '--allow-empty',
        '-m',
        'Land T9.98: the other package',
    )
    if phase_change == 'code':
        commit_file(
            seed,
            'other.txt',
            'other lane\n',
            'Land T9.97: a landing that changes a file',
        )
    g(seed, 'push', '-q', 'origin', PHASE)
    qf = q / f'{PKG}.json'
    entry = {
        'branch': TOPIC_BR,
        'worktree': str(wt),
        'candidate': x,
        'reviews': REVIEWS,
        'title': 'Land T9.99: self-test',
        'checkpoint': base,
        'protected_paths': '',
        'summary': 'self-test',
    }
    if moved_ok:
        entry['moved_ok'] = True
    qf.write_text(json.dumps(entry))
    s.update(
        wt=wt,
        cand=x,
        y=y,
        head=rev(seed, 'HEAD'),
        qf=qf,
        runs=root / 'runs.json',
    )
    set_runs(s, [ci(11, x)])
    return s


def ci(
    i,
    sha,
    status='completed',
    conclusion='success',
    branch=TOPIC_BR,
    title='T9.99: the feature',
):
    return dict(
        databaseId=i,
        headSha=sha,
        headBranch=branch,
        status=status,
        conclusion=conclusion if status == 'completed' else '',
        displayTitle=title,
    )


def set_runs(s, runs):
    s['runs'].write_text(json.dumps(runs))


def logged(path):
    return (
        [json.loads(ln) for ln in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def auto_land(s, sha=None, **extra):
    env = dict(
        ENV,
        SDM_ORCH_STATE=str(s['root'] / 'state'),
        SDM_LAND_QUEUE=str(s['q']),
        FAKE_GH_SHA=sha or '',
        PATH=f'{s["fake"]}{os.pathsep}{ENV["PATH"]}',
        FAKE_GH_LOG=str(s['gh_log']),
        FAKE_LAND_LOG=str(s['land_log']),
        FAKE_RATCHET_LOG=str(s['ratchet_log']),
        SDM_AFTER_LAND_LOG=str(s['root'] / 'after-land.log'),
        **extra,
    )
    if sha is None:
        env['FAKE_GH_RUNS'] = str(s['runs'])
    r = run(sys.executable, str(AUTO_LAND), env=env, check=False)
    return r.stdout + r.stderr


def show(out):
    print('  | ' + out.strip().replace('\n', '\n  | '))


results = []


def check(case, cond, what):
    results.append((case, bool(cond), what))
    print(f'  {"ok  " if cond else "FAIL"} {what}')


def unchanged_remote(case, s):
    check(
        case,
        remote_ref(s['origin'], LANE_BR) == s['cand'],
        'remote lane branch unchanged',
    )
    archives = g(
        s['origin'],
        'for-each-ref',
        '--format=%(refname)',
        f'refs/heads/{PHASE}-archive-*',
    ).stdout.strip()
    check(case, not archives, 'no archive branch pushed')


def case_clean(root):
    c = 'clean'
    print(c)
    s = setup(root, 'clean')
    out = auto_land(s, s['cand'])
    print('  | ' + out.strip().replace('\n', '\n  | '))
    new = rev(s['wt'], 'HEAD')
    check(
        c,
        out.startswith(
            f'REBASED {PKG} {s["cand"]} -> {new} (needs confirmation reviews)'
        ),
        'REBASED line',
    )
    check(
        c,
        new != s['cand']
        and g(
            s['wt'], 'merge-base', '--is-ancestor', s['head'], new, check=False
        ).returncode
        == 0,
        'new tip descends from the phase head',
    )
    check(
        c,
        remote_ref(s['origin'], LANE_BR) == new,
        'lane branch pushed to the rebased tip',
    )
    arch = f'{PHASE}-archive-{LANE}-T9.99-p1-{s["cand"][:8]}'
    check(
        c,
        remote_ref(s['origin'], arch) == s['cand'],
        f'old candidate archived as {arch}',
    )
    nc = s['q'] / f'{PKG}.needs-confirm'
    d = json.loads(nc.read_text()) if nc.exists() else {}
    check(
        c,
        not s['qf'].exists() and nc.exists(),
        '.json renamed to .needs-confirm',
    )
    check(
        c,
        d.get('candidate') == new
        and d.get('checkpoint') == s['head']
        and d.get('rebased_from') == s['cand'],
        'needs-confirm fields: candidate=new, checkpoint=phase head, rebased_from=old',
    )
    rd = s['q'] / f'{PKG}.rebased.rangediff'
    rdt = rd.read_text() if rd.exists() else ''
    check(
        c,
        rd.exists() and ' = ' in rdt and 'T9.99: the feature' in rdt,
        'range-diff written (commits shown "=")',
    )
    check(
        c,
        not g(s['wt'], 'status', '--porcelain', '--untracked-files=all').stdout,
        'worktree clean',
    )
    check(
        c, 'NEEDS-CONFIRM' in out, 'NEEDS-CONFIRM line printed in the same run'
    )
    la = logged(s['land_log'])
    check(
        c,
        len(la) == 1
        and opt(la[0], '--topic-branch') == LANE_BR
        and opt(la[0], '--phase-branch') == PHASE
        and opt(la[0], '--checkpoint') == s['base']
        and '--moved-from' not in la[0],
        'lane entry: land.py --phase-branch PHASE, --topic-branch is the lane, queued checkpoint, no --moved-from',
    )
    out2 = auto_land(s, new)
    check(
        c,
        out2.startswith(f'NEEDS-CONFIRM {PKG} {new}')
        and 'LANDED' not in out2
        and 'REBASED' not in out2,
        'next run only prints NEEDS-CONFIRM (never lands the rebased SHA itself)',
    )


def case_conflict(root):
    c = 'conflict'
    print(c)
    s = setup(root, 'conflict')
    out = auto_land(s, s['cand'])
    print('  | ' + out.strip().replace('\n', '\n  | '))
    check(
        c,
        out.startswith(f'CONFLICT {PKG} docs/tracker.md'),
        'CONFLICT line names docs/tracker.md',
    )
    f = s['q'] / f'{PKG}.failed'
    d = json.loads(f.read_text()) if f.exists() else {}
    check(
        c,
        d.get('conflicts') == ['docs/tracker.md'],
        '.failed has conflicts field',
    )
    check(
        c,
        rev(s['wt'], 'HEAD') == s['cand'],
        'HEAD back on the old candidate after --abort',
    )
    check(
        c,
        g(s['wt'], 'status', '--porcelain', '--untracked-files=all').stdout
        == '',
        'worktree clean, no rebase in progress',
    )
    check(
        c,
        not pathlib.Path(s['wt'], '.git/rebase-merge').exists(),
        'no rebase-merge state left',
    )
    unchanged_remote(c, s)


def case_dirty(root):
    c = 'dirty'
    print(c)
    s = setup(root, 'clean')
    pathlib.Path(s['wt'], 'scratch.txt').write_text('untracked\n')
    out = auto_land(s, s['cand'])
    print('  | ' + out.strip().replace('\n', '\n  | '))
    check(
        c,
        out.startswith('FAILED Land T9.99: self-test: not descended')
        and 'not clean' in out,
        'FAILED, worktree not clean',
    )
    f = s['q'] / f'{PKG}.failed'
    d = json.loads(f.read_text()) if f.exists() else {}
    check(
        c,
        'not clean' in d.get('rebase_refused', ''),
        '.failed has rebase_refused',
    )
    check(c, rev(s['wt'], 'HEAD') == s['cand'], 'HEAD untouched')
    unchanged_remote(c, s)


def case_head_moved(root):
    c = 'head!=candidate'
    print(c)
    s = setup(root, 'clean')
    commit_file(s['wt'], 'later.txt', 'later\n', 'T9.99: a later local commit')
    out = auto_land(s, s['cand'])
    print('  | ' + out.strip().replace('\n', '\n  | '))
    check(
        c,
        out.startswith('FAILED') and 'is not the candidate' in out,
        'FAILED, HEAD is not the candidate',
    )
    unchanged_remote(c, s)


def case_lease(root):
    c = 'lease'
    print(c)
    s = setup(root, 'clean')
    other = root / 'other'
    g(root, 'clone', '-q', '-b', LANE_BR, str(s['origin']), str(other))
    moved = commit_file(other, 'x.txt', 'x\n', 'T9.99: pushed from elsewhere')
    g(other, 'push', '-q', 'origin', f'HEAD:refs/heads/{LANE_BR}')
    out = auto_land(s, s['cand'])
    print('  | ' + out.strip().replace('\n', '\n  | '))
    check(
        c,
        out.startswith('FAILED') and 'push failed' in out,
        'FAILED, the lease rejected the push',
    )
    check(
        c,
        remote_ref(s['origin'], LANE_BR) == moved,
        'remote lane keeps the other push',
    )
    check(
        c,
        rev(s['wt'], 'HEAD') == s['cand'],
        'lane worktree reset (--keep) to the old candidate',
    )
    archives = g(
        s['origin'],
        'for-each-ref',
        '--format=%(refname)',
        f'refs/heads/{PHASE}-archive-*',
    ).stdout.strip()
    check(c, not archives, 'atomic push: no archive branch either')


def topic_entry_checks(c, s, new):
    check(
        c,
        remote_ref(s['origin'], TOPIC_BR) == new,
        'topic branch lease-pushed to the rebased tip',
    )
    arch = f'{PHASE}-archive-{TOPIC}-T9.99-p1-{s["cand"][:8]}'
    check(
        c,
        remote_ref(s['origin'], arch) == s['cand'],
        f'old candidate archived as {arch}',
    )
    lanes = g(
        s['origin'],
        'for-each-ref',
        '--format=%(refname)',
        f'refs/heads/{PHASE}-lane-*',
    ).stdout.strip()
    check(c, not lanes, 'no lane branch created or pushed')
    la = logged(s['land_log'])
    check(
        c,
        la and all(opt(a, '--topic-branch') == TOPIC_BR for a in la),
        'land.py --topic-branch is the topic branch',
    )
    gl = [a for a in logged(s['gh_log']) if a[:2] == ['run', 'list']]
    check(
        c,
        gl and all(opt(a, '--branch') == TOPIC_BR for a in gl),
        'CI looked up on the topic branch',
    )


def case_topic(root):
    c = 'topic branch, not moved_ok'
    print(c)
    s = setup_moved(root, moved_ok=False)
    out = auto_land(s)
    show(out)
    new = rev(s['wt'], 'HEAD')
    check(
        c,
        out.startswith(
            f'REBASED {PKG} {s["cand"]} -> {new} (needs confirmation reviews)'
        ),
        'REBASED line (not MOVED)',
    )
    topic_entry_checks(c, s, new)
    nc = s['q'] / f'{PKG}.needs-confirm'
    d = json.loads(nc.read_text()) if nc.exists() else {}
    check(
        c,
        d.get('candidate') == new
        and d.get('checkpoint') == s['head']
        and 'moved_from' not in d,
        '.needs-confirm: candidate=new, checkpoint=phase head, no moved_from',
    )


def case_bad_branch(root):
    c = 'invalid branch'
    print(c)
    s = setup(root, 'clean')
    d = json.loads(s['qf'].read_text())
    d['branch'] = 'main'
    s['qf'].write_text(json.dumps(d))
    out = auto_land(s, s['cand'])
    show(out)
    check(
        c,
        out.startswith(
            'FAILED Land T9.99: self-test: "branch" \'main\' is not'
        ),
        'FAILED, branch is not a topic branch',
    )
    check(
        c,
        (s['q'] / f'{PKG}.failed').exists() and not logged(s['land_log']),
        '.failed, land.py never called',
    )
    unchanged_remote(c, s)


def move_once(c, s, same_tree=True):
    out = auto_land(s)
    show(out)
    new = rev(s['wt'], 'HEAD')
    x = s['cand']
    check(
        c,
        out.strip()
        == f"MOVED {PKG} {x[:8]} -> {new[:8]} (lands on {x[:8]}'s evidence)",
        'MOVED line, nothing else',
    )
    d = json.loads(s['qf'].read_text()) if s['qf'].exists() else {}
    check(
        c,
        s['qf'].exists()
        and not list(s['q'].glob('*.needs-confirm'))
        and not list(s['q'].glob('*.failed')),
        'queue file stays .json',
    )
    check(
        c,
        d.get('candidate') == new
        and d.get('moved_from') == x
        and d.get('checkpoint') == s['base']
        and d.get('reviews') == REVIEWS,
        'entry: candidate=new, moved_from=X, checkpoint and reviews unchanged',
    )
    if same_tree:
        check(
            c,
            rev(s['wt'], f'{new}^{{tree}}') == rev(s['wt'], f'{x}^{{tree}}'),
            'same tree as X',
        )
    topic_entry_checks(c, s, new)
    return new


def case_moved(root):
    c = 'moved: clean move lands on X'
    print(c)
    s = setup_moved(root)
    new = move_once(c, s)
    set_runs(s, [ci(12, new, status='in_progress'), ci(11, s['cand'])])
    out = auto_land(s)
    show(out)
    check(
        c,
        out.startswith('LANDED Land T9.99: self-test')
        and (s['q'] / f'{PKG}.landed').exists(),
        "LANDED on X's green run though the new tip's run is unfinished",
    )
    check(
        c,
        not (s['root'] / 'after-land.log').exists(),
        'no after-landing cleanup unless SDM_CLEAN_AFTER_LAND=1',
    )
    la = logged(s['land_log'])[-1]
    check(
        c,
        opt(la, '--candidate') == new
        and opt(la, '--moved-from') == s['cand']
        and [la[i + 1] for i, v in enumerate(la) if v == '--review'] == REVIEWS,
        'land.py --candidate new --moved-from X with the reviews unchanged',
    )
    check(
        c,
        opt(la, '--checkpoint') == s['head']
        and s['ratchet_log'].read_text().split()[-1] == s['head'],
        'land.py and the ratchet get the merge base (the phase head), not the queued checkpoint',
    )


def case_moved_twice(root):
    c = 'moved: twice keeps X'
    print(c)
    s = setup_moved(root)
    first = move_once(c, s)
    g(
        s['seed'],
        'commit',
        '-q',
        '--allow-empty',
        '-m',
        'Land T9.96: another landing',
    )
    g(s['seed'], 'push', '-q', 'origin', PHASE)
    head2 = rev(s['seed'], 'HEAD')
    out = auto_land(s)
    show(out)
    second = rev(s['wt'], 'HEAD')
    check(
        c,
        out.strip()
        == f"MOVED {PKG} {first[:8]} -> {second[:8]} (lands on {s['cand'][:8]}'s evidence)",
        "second MOVED line still names X's evidence",
    )
    d = json.loads(s['qf'].read_text()) if s['qf'].exists() else {}
    check(
        c,
        d.get('moved_from') == s['cand']
        and d.get('candidate') == second
        and d.get('checkpoint') == s['base'],
        'moved_from stays X, checkpoint unchanged',
    )
    arch = f'{PHASE}-archive-{TOPIC}-T9.99-p1-{first[:8]}'
    check(
        c, remote_ref(s['origin'], arch) == first, 'the first move archived too'
    )
    out = auto_land(s)
    show(out)
    la = logged(s['land_log'])[-1]
    check(
        c,
        out.startswith('LANDED')
        and opt(la, '--candidate') == second
        and opt(la, '--moved-from') == s['cand']
        and opt(la, '--checkpoint') == head2,
        'LANDED: --candidate second --moved-from X',
    )


def case_moved_ci(root):
    c = "moved: X's run decides"
    print(c)
    s = setup_moved(root)
    new = move_once(c, s)
    n = len(logged(s['land_log']))
    set_runs(s, [ci(12, new), ci(11, s['cand'], status='in_progress')])
    out = auto_land(s)
    show(out)
    check(
        c,
        not out.strip()
        and s['qf'].exists()
        and len(logged(s['land_log'])) == n,
        "X's run unfinished: waits (the new tip's green run does not count)",
    )
    set_runs(s, [ci(12, new), ci(11, s['cand'], conclusion='failure')])
    out = auto_land(s)
    show(out)
    check(
        c,
        out.startswith(
            f'FAILED Land T9.99: self-test: CI run 11 failure on the moved-from commit {s["cand"][:12]}'
        )
        and (s['q'] / f'{PKG}.failed').exists()
        and len(logged(s['land_log'])) == n,
        "X's run failed: FAILED, .failed, land.py not called",
    )


def case_moved_refused(root):
    c = 'moved: CI3 refuses the move'
    print(c)
    s = setup_moved(root, 'code')
    # the rebase is clean; the tree is not X's
    new = move_once(c, s, same_tree=False)
    set_runs(s, [ci(12, new, conclusion='cancelled'), ci(11, s['cand'])])
    out = auto_land(s)
    show(out)
    nc = s['q'] / f'{PKG}.needs-confirm'
    d = json.loads(nc.read_text()) if nc.exists() else {}
    check(
        c,
        out.startswith(f'NEEDS-CONFIRM {PKG} {new} rebased from {s["cand"]}')
        and 'not the same code' in out
        and 'LANDED' not in out
        and 'FAILED' not in out,
        'NEEDS-CONFIRM line with the reason',
    )
    check(
        c,
        nc.exists() and not s['qf'].exists(),
        '.json renamed to .needs-confirm',
    )
    check(
        c,
        'moved_from' not in d
        and 'not the same code' in d.get('moved_refused', '')
        and d.get('rebased_from') == s['cand']
        and d.get('checkpoint') == s['head'],
        'moved_from dropped, moved_refused recorded, checkpoint = merge base',
    )
    check(
        c,
        ['run', 'rerun', '12', '-R', 'NVIDIA/structured-data-models-client']
        in logged(s['gh_log'])
        and d.get('ci_rerun') == 12,
        "the new tip's cancelled run is rerun",
    )
    check(
        c,
        opt(logged(s['land_log'])[-1], '--moved-from') == s['cand'],
        'land.py was asked with --moved-from X',
    )


def case_moved_other(root):
    c = 'moved: another refusal'
    print(c)
    s = setup_moved(root)
    move_once(c, s)
    out = auto_land(
        s,
        FAKE_LAND_REFUSE='the newest CI run on the moved-from commit is not green',
    )
    show(out)
    f = s['q'] / f'{PKG}.failed'
    check(
        c,
        out.startswith('FAILED Land T9.99: self-test: land.py refused')
        and f.exists()
        and not list(s['q'].glob('*.needs-confirm')),
        'FAILED and .failed as before, not .needs-confirm',
    )


CANCEL = HERE / 'cancel_superseded.sh'


def case_cancel(root):
    c = 'cancel_superseded'
    print(c)
    s = make_origin(root)
    seed = s['seed']
    heads = {}
    br = {
        'lane': LANE_BR,
        'topic': TOPIC_BR,
        'topic2': f'{PHASE}-zztopic2',
        'topic3': f'{PHASE}-zztopic3',
        'other': f'{PHASE}-zzother',
    }
    for k, b in br.items():
        heads[k] = commit_file(seed, f'{k}.txt', k, f'T9.99: {k} head')
        g(seed, 'push', '-q', 'origin', f'HEAD:refs/heads/{b}')
    X, X2, X3, X4, OLD = 'a' * 40, 'd' * 40, 'b' * 40, '9' * 40, 'e' * 40

    def ent(**k):
        return json.dumps(
            dict(worktree='/nonexistent', reviews=REVIEWS, title='t', **k)
        )

    (s['q'] / 'a.json').write_text(
        ent(
            branch=br['topic'],
            candidate=heads['topic'],
            moved_from=X,
            moved_ok=True,
        )
    )
    (s['q'] / 'b.json').write_text(
        ent(
            branch=br['topic2'],
            candidate=heads['topic2'],
            moved_from=X3,
            moved_ok=True,
        )
    )
    (s['q'] / 'c.json').write_text(
        ent(lane=LANE, candidate='c' * 40, moved_from=X2)
    )
    (s['q'] / 'd.json').write_text(
        ent(
            branch=br['topic3'],
            candidate=heads['topic3'],
            moved_from=X4,
            moved_ok=True,
        )
    )
    (s['q'] / 'e.needs-confirm').write_text(
        ent(branch=br['other'], candidate=heads['other'])
    )
    s['runs'] = root / 'runs.json'
    set_runs(
        s,
        [
            ci(21, OLD, 'in_progress', branch=br['lane']),  # superseded: cancel
            ci(
                22, heads['lane'], 'in_progress', branch=br['lane']
            ),  # head: keep
            ci(
                23, X2, 'in_progress', branch=br['lane']
            ),  # an entry's moved_from: keep
            ci(
                24, 'f' * 40, 'queued', branch=br['lane'], title='Land T9.1: x'
            ),  # landing: keep
            ci(
                25, '8' * 40, 'queued', branch=br['lane'], title='Land CI9: x'
            ),  # a CI<n> landing: keep
            ci(
                32, heads['topic'], 'in_progress', branch=br['topic']
            ),  # moved head, X running: cancel
            ci(
                31, X, 'in_progress', branch=br['topic']
            ),  # X, the evidence: keep
            ci(
                52, heads['topic2'], 'in_progress', branch=br['topic2']
            ),  # X3 failed: keep the head's
            ci(51, X3, branch=br['topic2'], conclusion='failure'),
            ci(
                62, heads['topic3'], 'queued', branch=br['topic3']
            ),  # X4 green: cancel
            ci(61, X4, branch=br['topic3']),
            ci(
                41, OLD, 'in_progress', branch=br['other']
            ),  # not in a .json entry: not scanned
        ],
    )
    env = dict(
        ENV,
        SDM_LAND_QUEUE=str(s['q']),
        SDM_ORCH_STATE=str(root / 'state'),
        SDM_REPO=str(seed),
        FAKE_GH_RUNS=str(s['runs']),
        FAKE_GH_LOG=str(s['gh_log']),
        PATH=f'{s["fake"]}{os.pathsep}{ENV["PATH"]}',
    )
    dry = run('/bin/bash', str(CANCEL), '--dry-run', env=env, check=False)
    out = dry.stdout + dry.stderr
    show(out)
    would = sorted(
        ln.split()[2]
        for ln in out.splitlines()
        if ln.startswith('would cancel')
    )
    check(
        c,
        would == ['21', '32', '62'],
        'dry run: would cancel 21 (superseded), 32 and 62 (moved heads) only; never 24/25 (Land T/CI runs)',
    )
    check(
        c,
        f'topic {br["topic"]}:' in out and f'topic {br["other"]}' not in out,
        "scans the queue's topic branches, not other branches",
    )
    check(
        c,
        not [a for a in logged(s['gh_log']) if a[:2] == ['run', 'cancel']],
        'dry run cancels nothing',
    )
    real = run('/bin/bash', str(CANCEL), env=env, check=False)
    out = real.stdout + real.stderr
    show(out)
    cancelled = sorted(
        a[2] for a in logged(s['gh_log']) if a[:2] == ['run', 'cancel']
    )
    check(
        c,
        cancelled == ['21', '32', '62'],
        'cancels 21, 32, 62; never 31/23 (moved_from runs) or 52 (X3 failed)',
    )


def case_malformed(root):
    c = 'malformed queue entries'
    print(c)
    s = setup(root, 'clean')
    (s['q'] / 'a-bad.json').write_text('{not json')
    (s['q'] / 'b-nokey.json').write_text(
        json.dumps({'lane': LANE, 'title': 'Land T9.97: no candidate'})
    )
    out = auto_land(s, s['cand'])
    show(out)
    lines = out.splitlines()
    check(
        c,
        any(ln.startswith('FAILED a-bad.json: JSONDecodeError') for ln in lines)
        and any(
            ln.startswith("FAILED b-nokey.json: KeyError: 'candidate'")
            for ln in lines
        )
        and 'Traceback' not in out,
        'FAILED <file>: <error> for bad JSON and a missing key, no traceback',
    )
    check(
        c,
        (s['q'] / 'a-bad.failed').exists()
        and (s['q'] / 'b-nokey.failed').exists(),
        'both malformed entries become .failed',
    )
    check(
        c,
        any(ln.startswith(f'REBASED {PKG} ') for ln in lines),
        'the well-formed entry after them is still processed',
    )


def no_phase_env(root, **extra):
    env = {k: v for k, v in ENV.items() if k != 'SDM_PHASE_BRANCH'}
    fake = root / 'fakebin'
    if not fake.exists():
        fake.mkdir()
        (fake / 'gh').write_text(FAKE_GH)
        (fake / 'gh').chmod(0o755)
    env.update(
        SDM_ORCH_STATE=str(root / 'state'),
        SDM_LAND_QUEUE=str(root / 'queue'),
        FAKE_GH_LOG=str(root / 'gh.log'),
        PATH=f'{fake}{os.pathsep}{ENV["PATH"]}',
        **extra,
    )
    return env


def case_phase_required(root):
    c = 'SDM_PHASE_BRANCH required'
    print(c)
    (root / 'queue').mkdir()
    (root / 'tasks').mkdir()
    env = no_phase_env(
        root,
        ORDER='t9.99-p1',
        SHAS='0' * 40,
        SDM_TASKS_DIR=str(root / 'tasks'),
    )
    for cmd in (
        (sys.executable, str(AUTO_LAND)),
        ('/bin/bash', str(CANCEL), '--dry-run'),
        (sys.executable, str(HERE / 'lane_idle.py'), str(root / 'tasks')),
        ('/bin/bash', str(HERE / 'land_chain.sh')),
    ):
        rc, out = run_limited(*cmd, env=env)
        name = pathlib.Path(cmd[1]).name
        check(
            c,
            rc == 2 and 'SDM_PHASE_BRANCH' in out,
            f'{name} without it: exit 2 naming SDM_PHASE_BRANCH (got {rc})',
        )
    check(c, not logged(root / 'gh.log'), 'no gh call was made')


def case_chain_missing(root):
    c = 'land_chain.sh: ORDER entries must be queued'
    print(c)
    q = root / 'queue'
    q.mkdir()
    (root / 'tasks').mkdir()
    (q / 't9.1-p1.next').write_text('{}')
    (q / 't9.3-p1.json').write_text('{}')
    env = no_phase_env(
        root,
        SDM_PHASE_BRANCH=PHASE,
        ORDER='t9.1-p1 t9.2-p1 t9.3-p1 t9.4-p1',
        SHAS='0' * 40,
        SDM_TASKS_DIR=str(root / 'tasks'),
    )
    rc, out = run_limited('/bin/bash', str(HERE / 'land_chain.sh'), env=env)
    show(out)
    check(
        c,
        rc == 1
        and 'NOT QUEUED' in out
        and ' t9.2-p1 t9.4-p1' in out
        and 't9.1-p1' not in out
        and 't9.3-p1' not in out,
        'exit 1 naming t9.2-p1 and t9.4-p1 only',
    )
    check(
        c,
        not logged(root / 'gh.log') and (q / 't9.1-p1.next').exists(),
        'checked before anything else: no gh call, nothing renamed',
    )


def case_clean_tmp(root):
    c = 'clean_tmp.sh: scratch holding a worktree'
    print(c)
    repo, scratch, home = root / 'repo', root / 'scratch', root / 'home'
    home.mkdir()
    g(root, 'init', '-q', '-b', 'main', str(repo))
    commit_file(repo, 'f.txt', 'f\n', 'base')
    live = scratch / 'sess1' / 'wt-live'
    deep = (
        scratch / 'sess3' / 'a' / 'b' / 'c' / 'wt-deep'
    )  # .git beyond find's depth 4
    for wt in (live, deep):
        wt.parent.mkdir(parents=True)
        g(repo, 'worktree', 'add', '-q', '--detach', str(wt))
    (live / 'uncommitted.txt').write_text('work in progress\n')
    (scratch / 'sess2').mkdir()
    (scratch / 'sess2' / 'notes.txt').write_text('stale\n')
    (scratch / 'sess4').mkdir()
    g(root, 'clone', '-q', str(repo), str(scratch / 'sess4' / 'clone'))
    env = dict(
        ENV,
        HOME=str(home),
        SDM_SCRATCH=str(scratch),
        SDM_REPO=str(repo),
        SDM_ORCH_STATE=str(root / 'state'),
        AGE='0',
    )
    r = run('/bin/bash', str(HERE / 'clean_tmp.sh'), env=env, check=False)
    out = r.stdout + r.stderr
    show('\n'.join(ln for ln in out.splitlines() if str(scratch) in ln))
    would = [ln for ln in out.splitlines() if ln.startswith('would remove ')]

    def listed(d):
        return any(
            ln.startswith(f'would remove {scratch / d} ') for ln in would
        )

    check(
        c,
        not listed('sess1'),
        'sess1 (holds a worktree with an uncommitted file) not listed',
    )
    check(
        c,
        f'refuse {scratch / "sess1"} (holds a git worktree or clone' in out,
        'sess1 refused: holds a git worktree',
    )
    check(
        c,
        not listed('sess3')
        and f'refuse {scratch / "sess3"} (holds the worktree' in out,
        "sess3 (a worktree below find's depth) refused from git worktree list",
    )
    check(c, not listed('sess4'), 'sess4 (holds a clone) not listed')
    check(c, listed('sess2'), 'sess2 (plain stale scratch) is listed')
    check(
        c,
        r.returncode == 0
        and not any(ln.startswith('remove ') for ln in out.splitlines())
        and (live / 'uncommitted.txt').exists(),
        'dry run removes nothing',
    )
    (live / 'uncommitted.txt').unlink()
    for wt in (live, deep):
        g(repo, 'worktree', 'remove', str(wt))


def main():
    keep = '--keep' in sys.argv
    tmp = tempfile.TemporaryDirectory(
        prefix='sdm-autoland-selftest-', dir='/tmp'
    )
    root = pathlib.Path(tmp.name).resolve()
    try:
        for i, case in enumerate(
            (
                case_clean,
                case_conflict,
                case_dirty,
                case_head_moved,
                case_lease,
                case_topic,
                case_bad_branch,
                case_moved,
                case_moved_twice,
                case_moved_ci,
                case_moved_refused,
                case_moved_other,
                case_cancel,
                case_malformed,
                case_phase_required,
                case_chain_missing,
                case_clean_tmp,
            )
        ):
            d = root / f'c{i}'
            d.mkdir()
            case(d)
    finally:
        if keep:
            tmp._finalizer.detach()
            print(f'kept {root}')
        else:
            tmp.cleanup()
    bad = [r for r in results if not r[1]]
    print(f'{len(results) - len(bad)}/{len(results)} checks passed')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
