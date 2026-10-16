# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`scripts/rust_core/orchestration/branch_cleanup.py` deletes only what is
merged, clean and unused, on a throwaway repository with a bare origin: a
dry run changes nothing, and --apply writes the backup bundle first, keeps
everything unsafe, uses -D only for an approved branch, and stops at the
first item that changed since the plan."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'scripts/rust_core/orchestration/branch_cleanup.py'

ENV = {
    **os.environ,
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_AUTHOR_NAME': 'Test',
    'GIT_AUTHOR_EMAIL': 'test@example.com',
    'GIT_COMMITTER_NAME': 'Test',
    'GIT_COMMITTER_EMAIL': 'test@example.com',
}

pytestmark = pytest.mark.skipif(
    shutil.which('git') is None
    or (shutil.which('lsof') is None and not Path('/proc').is_dir()),
    reason='needs git and lsof or /proc',
)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args],
        cwd=cwd,
        env=ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(work: Path, name: str) -> str:
    (work / f'{name}.txt').write_text(name + '\n')
    git(work, 'add', f'{name}.txt')
    git(work, 'commit', '-q', '-m', name)
    return git(work, 'rev-parse', 'HEAD')


def tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=ENV,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def has_ref(repo: Path, ref: str) -> bool:
    return (
        subprocess.run(
            ['git', 'rev-parse', '--verify', '--quiet', ref],
            cwd=repo,
            env=ENV,
            capture_output=True,
        ).returncode
        == 0
    )


def remote_heads(origin: Path) -> set[str]:
    out = git(origin, 'for-each-ref', '--format=%(refname:short)')
    return set(out.split())


def worktree_paths(work: Path) -> set[str]:
    out = git(work, 'worktree', 'list', '--porcelain')
    return {
        os.path.realpath(line[9:])
        for line in out.splitlines()
        if line.startswith('worktree ')
    }


@pytest.fixture()
def repo(tmp_path: Path):
    """main = A-B-C, release (the base) = C-D, unmerged E on top of C.

    Local: main (checked out, at C), merged-d (B), needs-big-d (D, merged
    into release but not into HEAD), unmerged (E), keep-me (B), wt-clean-br
    (B, in a clean worktree). Worktrees: clean (merged), dirty (merged, an
    untracked file), busy (merged, a process inside), unmerged (E).
    Origin: main, release, done-remote (B), open-remote (E),
    archive-old (B).
    """
    origin = tmp_path / 'origin.git'
    work = tmp_path / 'work'
    git(tmp_path, 'init', '-q', '--bare', '-b', 'main', str(origin))
    git(tmp_path, 'clone', '-q', str(origin), str(work))
    git(work, 'checkout', '-q', '-b', 'main')
    a = commit(work, 'a')
    b = commit(work, 'b')
    c = commit(work, 'c')
    git(work, 'push', '-q', 'origin', 'main')
    git(work, 'checkout', '-q', '-b', 'release')
    d = commit(work, 'd')
    git(work, 'checkout', '-q', '-b', 'unmerged', c)
    e = commit(work, 'e')
    git(work, 'checkout', '-q', 'main')
    git(work, 'branch', '-q', '-D', 'release')
    git(work, 'push', '-q', 'origin', f'{d}:refs/heads/release')
    git(work, 'push', '-q', 'origin', f'{b}:refs/heads/done-remote')
    git(work, 'push', '-q', 'origin', f'{e}:refs/heads/open-remote')
    git(work, 'push', '-q', 'origin', f'{b}:refs/heads/archive-old')
    git(work, 'fetch', '-q', 'origin')
    git(work, 'branch', 'merged-d', b)
    git(work, 'branch', 'needs-big-d', d)
    git(work, 'branch', 'keep-me', b)
    wts = tmp_path / 'wts'
    git(
        work,
        'worktree',
        'add',
        '-q',
        '-b',
        'wt-clean-br',
        str(wts / 'clean'),
        b,
    )
    git(work, 'worktree', 'add', '-q', '--detach', str(wts / 'dirty'), b)
    (wts / 'dirty' / 'scratch.txt').write_text('uncommitted\n')
    git(work, 'worktree', 'add', '-q', '--detach', str(wts / 'busy'), b)
    git(work, 'worktree', 'add', '-q', '--detach', str(wts / 'unmerged'), e)
    busy = subprocess.Popen(['sleep', '600'], cwd=wts / 'busy')
    keep = tmp_path / 'keep.txt'
    keep.write_text('# kept on purpose\nkeep-me\n')
    try:
        yield {
            'origin': origin,
            'work': work,
            'wts': wts,
            'keep': keep,
            'tmp': tmp_path,
            'sha': {'a': a, 'b': b, 'c': c, 'd': d, 'e': e},
        }
    finally:
        busy.kill()
        busy.wait()


def plan(r, *extra: str) -> Path:
    out = r['tmp'] / 'plan.json'
    proc = tool(
        'plan',
        '--repo',
        str(r['work']),
        '--base',
        'origin/release',
        '--keep-list',
        str(r['keep']),
        '--out',
        str(out),
        '--no-gh',
        *extra,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def targets(plan_path: Path) -> set[str]:
    items = json.loads(plan_path.read_text())['items']
    return {
        f'{i["kind"]}:{i.get("name") or os.path.realpath(i["path"])}'
        for i in items
    }


def test_audit_classifies_every_case(repo) -> None:
    r = repo
    out = r['tmp'] / 'audit.json'
    proc = tool(
        'audit',
        '--repo',
        str(r['work']),
        '--base',
        'origin/release',
        '--keep-list',
        str(r['keep']),
        '--no-gh',
        '--json',
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out.read_text())
    local = {i['name']: i for i in res['local_branches']}
    remote = {i['name']: i for i in res['remote_branches']}
    wts = {Path(i['path']).name: i for i in res['worktrees'] if not i['main']}
    assert local['merged-d']['deletable']
    assert 'unmerged' in local['unmerged']['classes']
    assert local['unmerged']['unmatched'] == 1
    assert 'kept' in local['keep-me']['classes']
    assert not local['keep-me']['deletable']
    assert 'checked-out' in local['main']['classes']
    assert 'default-branch' in local['main']['classes']
    assert not local['needs-big-d']['deletable']
    assert local['needs-big-d']['flag'] == '-D'
    assert remote['done-remote']['deletable']
    assert 'default-branch' in remote['main']['classes']
    assert 'base' in remote['release']['classes']
    assert 'archive' in remote['archive-old']['classes']
    assert 'unmerged' in remote['open-remote']['classes']
    assert wts['clean']['deletable']
    assert 'dirty' in wts['dirty']['classes']
    assert 'in-use' in wts['busy']['classes']
    assert 'unmerged' in wts['unmerged']['classes']


def test_plan_holds_only_safe_items(repo) -> None:
    r = repo
    got = targets(plan(r))
    assert got == {
        'worktree:' + os.path.realpath(r['wts'] / 'clean'),
        'remote:done-remote',
        'local:merged-d',
        'local:wt-clean-br',
    }


def test_dry_run_deletes_nothing(repo) -> None:
    r = repo
    p = plan(r)
    before = (
        git(r['work'], 'for-each-ref'),
        remote_heads(r['origin']),
        worktree_paths(r['work']),
    )
    backups = r['tmp'] / 'backups'
    proc = tool('execute', str(p), '--backup-dir', str(backups))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'WOULD git' in proc.stdout
    after = (
        git(r['work'], 'for-each-ref'),
        remote_heads(r['origin']),
        worktree_paths(r['work']),
    )
    assert after == before
    assert not backups.exists()


def test_apply_deletes_only_safe_items_and_bundles_them(repo) -> None:
    r = repo
    p = plan(r)
    backups = r['tmp'] / 'backups'
    proc = tool('execute', str(p), '--apply', '--backup-dir', str(backups))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    work, origin, wts = r['work'], r['origin'], r['wts']
    for gone in ('merged-d', 'wt-clean-br'):
        assert not has_ref(work, f'refs/heads/{gone}')
    for kept in ('main', 'unmerged', 'keep-me', 'needs-big-d'):
        assert has_ref(work, f'refs/heads/{kept}')
    assert remote_heads(origin) == {
        'main',
        'release',
        'open-remote',
        'archive-old',
    }
    paths = worktree_paths(work)
    assert os.path.realpath(wts / 'clean') not in paths
    for kept in ('dirty', 'busy', 'unmerged'):
        assert os.path.realpath(wts / kept) in paths
    assert (wts / 'dirty' / 'scratch.txt').exists()

    bundles = list(backups.glob('*.bundle'))
    assert len(bundles) == 1
    git(work, 'bundle', 'verify', str(bundles[0]))
    heads = git(work, 'bundle', 'list-heads', str(bundles[0]))
    sha = r['sha']
    assert f'{sha["b"]} refs/heads/merged-d' in heads
    assert f'{sha["b"]} refs/heads/wt-clean-br' in heads
    assert f'{sha["b"]} refs/remotes/origin/done-remote' in heads
    manifests = list(backups.glob('*.restore.json'))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())
    assert {i['status'] for i in m['items']} == {'deleted'}
    assert all(i['restore'] for i in m['items'])
    # The bundle restores a deleted branch.
    git(
        work,
        'fetch',
        '-q',
        str(bundles[0]),
        'refs/heads/merged-d:refs/heads/merged-d',
    )
    assert git(work, 'rev-parse', 'merged-d') == sha['b']


def test_big_d_only_for_an_approved_branch(repo) -> None:
    r = repo
    approved = r['tmp'] / 'approved.txt'
    approved.write_text('needs-big-d\n')
    p = plan(r, '--approved', str(approved))
    items = json.loads(p.read_text())['items']
    big = [i for i in items if i.get('name') == 'needs-big-d']
    assert big and big[0]['flag'] == '-D'
    backups = r['tmp'] / 'backups'
    # The approval must be given to execute itself too.
    proc = tool('execute', str(p), '--apply', '--backup-dir', str(backups))
    assert proc.returncode == 1
    assert '-D needs the branch in the approved list' in proc.stdout
    assert has_ref(r['work'], 'refs/heads/needs-big-d')
    p = plan(r, '--approved', str(approved))
    proc = tool(
        'execute',
        str(p),
        '--apply',
        '--backup-dir',
        str(backups),
        '--approved',
        str(approved),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not has_ref(r['work'], 'refs/heads/needs-big-d')
    assert 'branch -D needs-big-d' in proc.stdout


def test_apply_stops_at_the_first_changed_item(repo) -> None:
    r = repo
    p = plan(r)
    # The remote branch moves on (a fast-forward) after the plan: it is
    # refused, and the local branches planned after it are left alone.
    git(
        r['work'],
        'push',
        '-q',
        'origin',
        f'{r["sha"]["e"]}:refs/heads/done-remote',
    )
    backups = r['tmp'] / 'backups'
    proc = tool('execute', str(p), '--apply', '--backup-dir', str(backups))
    assert proc.returncode == 1
    assert 'REFUSED remote branch done-remote' in proc.stdout
    assert 'STOPPED' in proc.stdout
    assert 'done-remote' in remote_heads(r['origin'])
    assert has_ref(r['work'], 'refs/heads/merged-d')
    m = json.loads(next(backups.glob('*.restore.json')).read_text())
    status = {i.get('name') or 'wt': i['status'] for i in m['items']}
    assert status['wt'] == 'deleted'
    assert status['done-remote'].startswith('refused')
    assert status['merged-d'] == 'planned'


def test_base_must_be_a_branch(repo) -> None:
    # A commit, HEAD or tag as the base would leave the base branch itself
    # deletable once its tip is merged.
    r = repo
    git(r['work'], 'tag', 'v1', r['sha']['d'])
    for base in (r['sha']['d'], 'HEAD', 'v1', 'origin/release~1', 'nope'):
        proc = tool(
            'audit', '--repo', str(r['work']), '--base', base, '--no-gh'
        )
        assert proc.returncode == 2, (base, proc.stdout)
        assert 'is not a branch here' in proc.stderr, base
    for base in ('origin/release', 'refs/remotes/origin/release', 'main'):
        proc = tool(
            'audit', '--repo', str(r['work']), '--base', base, '--no-gh'
        )
        assert proc.returncode == 0, (base, proc.stderr)


def test_ignored_files_outside_caches_keep_a_worktree(repo) -> None:
    # git worktree remove deletes ignored files without asking; build caches
    # are rebuilt, anything else (notes, .env, run outputs) is kept.
    r = repo
    common = Path(git(r['work'], 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = r['work'] / common
    (common / 'info').mkdir(exist_ok=True)
    (common / 'info' / 'exclude').write_text('*.env\ntarget/\n')
    wts = r['wts']
    for name in ('notes', 'cache'):
        git(r['work'], 'worktree', 'add', '-q', '--detach', str(wts / name),
            r['sha']['b'])  # fmt: skip
        (wts / name / 'target' / 'debug').mkdir(parents=True)
        (wts / name / 'target' / 'debug' / 'out.bin').write_text('x\n')
    (wts / 'notes' / 'local.env').write_text('SECRET=1\n')
    out = r['tmp'] / 'audit.json'
    proc = tool(
        'audit', '--repo', str(r['work']), '--base', 'origin/release',
        '--keep-list', str(r['keep']), '--no-gh', '--json', str(out),
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out.read_text())
    items = {Path(i['path']).name: i for i in res['worktrees']}
    notes = items['notes']
    assert not notes['deletable']
    assert 'dirty' in notes['classes']
    assert any(
        'ignored files outside caches: 1' in why and 'local.env' in why
        for why in notes['reasons']
    ), notes['reasons']
    assert items['cache']['deletable'], items['cache']['reasons']
    assert 'worktree:' + os.path.realpath(wts / 'notes') not in targets(plan(r))
