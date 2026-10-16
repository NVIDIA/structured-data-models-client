#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-of-phase branch and worktree cleanup, guarded at every step.

    branch_cleanup.py audit   --base REF [--remote NAME] [--keep-list FILE]
                              [--archive-glob GLOB]... [--no-gh] [--json OUT]
    branch_cleanup.py plan    --base REF --keep-list FILE --out PLAN.json
                              [--remote NAME] [--approved FILE]
                              [--archive-glob GLOB]... [--no-gh]
    branch_cleanup.py execute PLAN.json [--apply] [--approved FILE]
                              [--backup-dir DIR] [--keep-list FILE]

Every subcommand takes ``--repo PATH`` (default: the current directory's
repository).

``audit`` classifies, without changing anything, every local branch, every
branch on the remote (read live with ``git ls-remote``) and every linked
worktree: merged into the base (the tip is an ancestor of ``--base``),
unmerged (with the count of commits ``git cherry`` finds no equivalent
for), archive (``--archive-glob``, default ``*archive*``), kept (the
keep-list), and the reasons an item cannot go: a dirty worktree, one
holding ignored files outside build caches (``git worktree remove`` deletes
them without asking), one a running process has open, a locked one, a
rebase or merge in progress, a branch checked out in a worktree that stays,
the remote's default branch (local or remote), the base itself, an open
pull request or a protected branch (both read from GitHub with ``gh``;
``--no-gh`` skips that check). ``--base`` must name a branch (``NAME``,
``REMOTE/NAME`` or a full ``refs/`` name), never a commit, tag or ``HEAD``,
so the base branch itself is always known and kept.

``plan`` writes the deletion plan: only merged items with no blocking
reason, in the order worktrees, remote branches, local branches. A local
branch that ``git branch -d`` would refuse (merged into the base but not
into its upstream or HEAD) needs ``-D``, and a worktree with submodules
needs ``worktree remove --force``; such an item is planned only when its
exact name or path is in the owner-approved file passed with
``--approved``, and is otherwise listed as skipped with the reason.

``execute`` is a dry run unless ``--apply`` is given: it re-checks every
item and prints what it would run. With ``--apply`` it first writes a git
bundle holding every ref it will delete and a restore manifest (JSON plus
the restore commands as a shell file) to ``--backup-dir`` (default
``<git common dir>/branch-cleanup-backups``), then, item by item, re-checks
that the item is unchanged since the plan, not on the keep-list, clean, not
in use, and an ancestor of the base as it is now, and runs the one git
command. It stops at the first failed check or command. ``-D`` and
``--force`` are used only for items in the approved file passed to
``execute`` itself.

Keep-list and approved-list files hold one entry per line; ``#`` starts a
comment. ``branch:NAME`` names a branch, ``path:PATH`` a worktree; a bare
entry starting with ``/`` or ``~`` is a path, anything else a branch. In
the keep-list, entries are shell globs and a branch entry protects both
the local and the remote branch of that name; in the approved list they
are exact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# Ignored paths that are rebuilt, not lost, when a worktree is removed.
CACHE_PARTS = frozenset(
    {
        'target',
        '.venv',
        'venv',
        '__pycache__',
        '.pytest_cache',
        '.ruff_cache',
        '.mypy_cache',
        'build',
        'dist',
        '_sibling_wheels',
        'htmlcov',
    }
)

# Ignored build outputs matched by name pattern (wheels, native modules,
# package metadata, coverage data), also rebuilt rather than lost.
CACHE_GLOBS = (
    '*.egg-info',
    '*.so',
    '*.whl',
    '.venv*',
    'native*',
    'dist-native*',
    '.coverage*',
)

OP_FILES = (
    'rebase-merge',
    'rebase-apply',
    'MERGE_HEAD',
    'BISECT_LOG',
    'CHERRY_PICK_HEAD',
    'REVERT_HEAD',
)


class CleanupError(Exception):
    pass


# Reads never take optional locks (git status refreshing the index), so an
# audit cannot get in the way of an agent working in a worktree.
GIT_ENV = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}


def git(repo, *args, check=True, input=None, cwd=None):
    proc = subprocess.run(
        ['git', *args],
        cwd=cwd or repo,
        capture_output=True,
        text=True,
        input=input if input is not None else '',
        env=GIT_ENV,
    )
    if check and proc.returncode != 0:
        raise CleanupError(
            f'git {" ".join(args)} failed: {proc.stderr.strip()}'
        )
    return proc


def rev(repo, ref):
    out = git(
        repo,
        'rev-parse',
        '--verify',
        '--quiet',
        ref + '^{commit}',
        check=False,
    ).stdout.strip()
    return out or None


def is_ancestor(repo, sha, base_sha):
    """True, False, or None when the commit is not in the local store."""
    found = git(repo, 'cat-file', '-e', sha + '^{commit}', check=False)
    if found.returncode != 0:
        return None
    rc = git(
        repo, 'merge-base', '--is-ancestor', sha, base_sha, check=False
    ).returncode
    return rc == 0


def real(path):
    return os.path.realpath(os.path.expanduser(path))


# --- keep-list and approved-list ------------------------------------------


def load_list(path):
    entries = {'branch': [], 'path': []}
    if not path:
        return entries
    for raw in Path(path).read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('branch:'):
            entries['branch'].append(line[len('branch:') :].strip())
        elif line.startswith('path:'):
            entries['path'].append(line[len('path:') :].strip())
        elif line.startswith(('/', '~')):
            entries['path'].append(line)
        else:
            entries['branch'].append(line)
    return entries


def keeps_branch(keep, name):
    return any(fnmatch.fnmatchcase(name, g) for g in keep['branch'])


def keeps_path(keep, path):
    cands = {path, real(path)}
    for g in keep['path']:
        pats = {os.path.expanduser(g), real(g)}
        if any(fnmatch.fnmatchcase(c, p) for c in cands for p in pats):
            return True
    return False


def approves_branch(approved, name):
    return name in approved['branch']


def approves_path(approved, path):
    return real(path) in {real(p) for p in approved['path']}


# --- state readers ---------------------------------------------------------


def worktrees(repo):
    out, cur = [], None
    lines = git(repo, 'worktree', 'list', '--porcelain').stdout.splitlines()
    for line in [*lines, '']:
        if not line:
            if cur:
                out.append(cur)
            cur = None
            continue
        key, _, val = line.partition(' ')
        if key == 'worktree':
            cur = {
                'path': val,
                'head': None,
                'branch': None,
                'locked': False,
                'prunable': False,
                'bare': False,
            }
        elif cur is None:
            continue
        elif key == 'HEAD':
            cur['head'] = val
        elif key == 'branch':
            cur['branch'] = val.removeprefix('refs/heads/')
        elif key in ('locked', 'prunable', 'bare'):
            cur[key] = True
    for i, w in enumerate(out):
        w['main'] = i == 0
    return out


def local_branches(repo):
    fmt = '%(refname:short)%09%(objectname)%09%(upstream)'
    out = {}
    for line in git(
        repo, 'for-each-ref', f'--format={fmt}', 'refs/heads'
    ).stdout.splitlines():
        name, sha, up = line.split('\t')
        out[name] = {'sha': sha, 'upstream': up or None}
    return out


def remote_branches(repo, remote):
    out = {}
    for line in git(repo, 'ls-remote', '--heads', remote).stdout.splitlines():
        sha, ref = line.split('\t')
        out[ref.removeprefix('refs/heads/')] = sha
    return out


def remote_default(repo, remote):
    proc = git(repo, 'ls-remote', '--symref', remote, 'HEAD', check=False)
    for line in proc.stdout.splitlines():
        if line.startswith('ref: '):
            return line[5:].split('\t')[0].removeprefix('refs/heads/')
    return None


def open_paths():
    """(path, who) for every open file and cwd, or None if unreadable."""
    proc = None
    try:
        proc = subprocess.run(
            ['lsof', '-n', '-P', '-F', 'pcn'],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        proc = None
    if proc is not None and proc.stdout:
        hits, pid, cmd = [], None, None
        for line in proc.stdout.splitlines():
            if line.startswith('p'):
                pid = line[1:]
            elif line.startswith('c'):
                cmd = line[1:]
            elif line.startswith('n/'):
                hits.append((line[1:], f'pid {pid} {cmd}'))
        return hits
    proc_dir = Path('/proc')
    if not proc_dir.is_dir():
        return None
    hits = []
    for p in proc_dir.iterdir():
        if not p.name.isdigit():
            continue
        links = [p / 'cwd']
        try:
            links += list((p / 'fd').iterdir())
        except OSError:
            pass
        for link in links:
            try:
                hits.append((os.readlink(link), f'pid {p.name}'))
            except OSError:
                continue
    return hits


def users_of(path, snapshot):
    """Processes with a cwd or an open file under path, or None."""
    if snapshot is None:
        return None
    roots = {path.rstrip('/'), real(path).rstrip('/')}
    found = set()
    for p, who in snapshot:
        for r in roots:
            if p == r or p.startswith(r + '/'):
                found.add(f'{who}: {p}')
    here = os.getcwd()
    for r in roots:
        if here == r or here.startswith(r + '/'):
            found.add(f'this process: {here}')
    return sorted(found)


def worktree_problems(repo, wt, snapshot):
    """Reasons a worktree cannot be removed now (beyond merge state)."""
    reasons = []
    path = wt['path']
    if wt['main']:
        return ['main worktree']
    if wt['bare']:
        return ['bare repository entry']
    if real(path) == real(repo):
        reasons.append('the worktree this tool runs in')
    if wt['locked']:
        reasons.append('locked')
    if wt['prunable'] or not os.path.isdir(path):
        reasons.append('directory missing (prunable)')
        return reasons
    st = git(
        repo,
        '-C',
        path,
        'status',
        '--porcelain',
        '--untracked-files=all',
        '--ignore-submodules=none',
        check=False,
    )
    if st.returncode != 0:
        reasons.append('git status failed: ' + st.stderr.strip()[:200])
    elif st.stdout.strip():
        lines = st.stdout.splitlines()
        reasons.append(
            f'dirty: {len(lines)} status line(s), e.g. {lines[0].strip()}'
        )
    ignored = ignored_files(repo, path)
    if ignored:
        reasons.append(
            f'ignored files outside caches: {len(ignored)}, which worktree '
            f'remove would delete, e.g. {ignored[0]}'
        )
    gd = git(
        repo, '-C', path, 'rev-parse', '--absolute-git-dir', check=False
    ).stdout.strip()
    ops = [f for f in OP_FILES if gd and os.path.exists(os.path.join(gd, f))]
    if ops:
        reasons.append('operation in progress: ' + ', '.join(ops))
    users = users_of(path, snapshot)
    if users is None:
        reasons.append('in-use check unavailable (no lsof and no /proc)')
    elif users:
        reasons.append(
            f'in use by a running process ({len(users)}), e.g. {users[0]}'
        )
    return reasons


def ignored_files(repo, path):
    """Ignored paths in a worktree, less build caches (a directory is
    listed once, as DIR/)."""
    proc = git(
        repo,
        '-C',
        path,
        'ls-files',
        '--others',
        '--ignored',
        '--exclude-standard',
        '--directory',
        check=False,
    )
    if proc.returncode != 0:
        return ['(git ls-files failed: ' + proc.stderr.strip()[:200] + ')']
    out = []
    for entry in proc.stdout.splitlines():
        parts = entry.rstrip('/').split('/')
        if (
            CACHE_PARTS.intersection(parts)
            or entry.endswith('.pyc')
            or any(
                fnmatch.fnmatch(part, g) for part in parts for g in CACHE_GLOBS
            )
        ):
            continue
        out.append(entry)
    return out


def needs_force(path):
    return os.path.exists(os.path.join(path, '.gitmodules'))


def github_slug(repo, remote):
    url = git(repo, 'remote', 'get-url', remote, check=False).stdout.strip()
    m = re.search(r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$', url)
    return m.group(1) if m else None


def github_state(repo, remote):
    """(open-PR head branches, protected branches) or raise CleanupError."""
    slug = github_slug(repo, remote)
    if not slug:
        raise CleanupError(
            f'remote {remote} is not a GitHub URL; pass --no-gh to skip '
            'the open-PR and protection check'
        )
    pr = subprocess.run(
        [
            'gh',
            'pr',
            'list',
            '-R',
            slug,
            '--state',
            'open',
            '--limit',
            '1000',
            '--json',
            'headRefName',
        ],
        capture_output=True,
        text=True,
    )
    if pr.returncode != 0:
        raise CleanupError('gh pr list failed: ' + pr.stderr.strip())
    prot = subprocess.run(
        [
            'gh',
            'api',
            '--paginate',
            f'repos/{slug}/branches?protected=true&per_page=100',
            '--jq',
            '.[].name',
        ],
        capture_output=True,
        text=True,
    )
    if prot.returncode != 0:
        raise CleanupError('gh api branches failed: ' + prot.stderr.strip())
    heads = {p['headRefName'] for p in json.loads(pr.stdout or '[]')}
    return heads, set(prot.stdout.split())


def base_name(base, remote):
    for pre in (f'refs/remotes/{remote}/', f'{remote}/', 'refs/heads/'):
        if base.startswith(pre):
            return base[len(pre) :]
    return base


def base_ref(repo, base, remote):
    """The full branch ref ``--base`` names, or raise CleanupError.

    A commit, tag or HEAD would leave the base branch protected by nothing
    but its name, which is then unknown.
    """
    name = base_name(base, remote)
    if base.startswith(('refs/heads/', f'refs/remotes/{remote}/')):
        full = base
    elif base.startswith(f'{remote}/'):
        full = f'refs/remotes/{remote}/{name}'
    else:
        full = f'refs/heads/{name}'
    found = git(repo, 'show-ref', '--verify', '--quiet', full, check=False)
    if not name or found.returncode != 0:
        raise CleanupError(
            f'--base {base} is not a branch here: give NAME, {remote}/NAME '
            'or a full refs/heads/ or refs/remotes/ name (a commit, tag or '
            'HEAD would leave the base branch itself unprotected)'
        )
    return full


def cherry_unmatched(repo, base_sha, sha):
    out = git(repo, 'cherry', base_sha, sha, check=False).stdout
    return sum(1 for line in out.splitlines() if line.startswith('+ '))


# --- audit -----------------------------------------------------------------


def audit(repo, base, remote, keep, archive_globs, use_gh, approved=None):
    approved = approved or load_list(None)
    base_sha = rev(repo, base_ref(repo, base, remote))
    if not base_sha:
        raise CleanupError(f'base {base} does not resolve to a commit')
    bname = base_name(base, remote)
    default = remote_default(repo, remote)
    head_sha = rev(repo, 'HEAD')
    snapshot = open_paths()
    warnings = []
    pr_heads = protected = None
    if use_gh:
        try:
            pr_heads, protected = github_state(repo, remote)
        except CleanupError as e:
            warnings.append(str(e))

    def merge_state(sha, item):
        anc = is_ancestor(repo, sha, base_sha)
        if anc is None:
            item['classes'].append('unknown')
            item['reasons'].append(
                f'commit {sha[:12]} is not in the local store (fetch first)'
            )
        elif anc:
            item['classes'].append('merged')
        else:
            item['classes'].append('unmerged')
            n = cherry_unmatched(repo, base_sha, sha)
            item['unmatched'] = n
            item['reasons'].append(
                f'not an ancestor of {base} ({n} commit(s) with no '
                'equivalent there)'
                if n
                else f'not an ancestor of {base} (patch-equivalent only)'
            )

    def is_archive(name):
        return any(fnmatch.fnmatchcase(name, g) for g in archive_globs)

    # remote branches
    live = remote_branches(repo, remote)
    remote_items = []
    for name, sha in sorted(live.items()):
        item = {'name': name, 'sha': sha, 'classes': [], 'reasons': []}
        merge_state(sha, item)
        if name == bname:
            item['classes'].append('base')
            item['reasons'].append('the base branch')
        if name == default:
            item['classes'].append('default-branch')
            item['reasons'].append(f'default branch of {remote}')
        if keeps_branch(keep, name):
            item['classes'].append('kept')
            item['reasons'].append('on the keep-list')
        if is_archive(name):
            item['classes'].append('archive')
            item['reasons'].append('archive branch')
        if use_gh:
            if pr_heads is None:
                item['reasons'].append('GitHub check failed (see warnings)')
            else:
                if name in pr_heads:
                    item['classes'].append('open-pr')
                    item['reasons'].append('an open pull request uses it')
                if name in protected:
                    item['classes'].append('protected')
                    item['reasons'].append('protected branch')
        item['deletable'] = 'merged' in item['classes'] and not item['reasons']
        remote_items.append(item)
    remote_deleted = {i['name'] for i in remote_items if i['deletable']}

    # worktrees
    branch_flags = {}
    wt_items = []
    for wt in worktrees(repo):
        item = {
            'path': wt['path'],
            'sha': wt['head'],
            'branch': wt['branch'],
            'main': wt['main'],
            'classes': [],
            'reasons': [],
        }
        problems = worktree_problems(repo, wt, snapshot)
        for p in problems:
            cls = (
                'dirty'
                if p.startswith(('dirty', 'ignored'))
                else 'in-use'
                if p.startswith('in use')
                else 'blocked'
            )
            if cls not in item['classes']:
                item['classes'].append(cls)
        item['reasons'] += problems
        if wt['head'] and not wt['main']:
            merge_state(wt['head'], item)
        if keeps_path(keep, wt['path']):
            item['classes'].append('kept')
            item['reasons'].append('path on the keep-list')
        if wt['branch'] and (
            keeps_branch(keep, wt['branch']) or wt['branch'] in (bname, default)
        ):
            item['classes'].append('kept')
            item['reasons'].append(f'checks out kept branch {wt["branch"]}')
        item['force'] = bool(
            os.path.isdir(wt['path']) and needs_force(wt['path'])
        )
        if item['force'] and not approves_path(approved, wt['path']):
            item['reasons'].append(
                'has submodules: removal needs --force and the path is not '
                'in the approved list'
            )
        item['deletable'] = 'merged' in item['classes'] and not item['reasons']
        wt_items.append(item)
        if wt['branch']:
            branch_flags.setdefault(wt['branch'], []).append(item)

    # local branches
    local = local_branches(repo)
    local_items = []
    for name, info in sorted(local.items()):
        item = {
            'name': name,
            'sha': info['sha'],
            'upstream': info['upstream'],
            'classes': [],
            'reasons': [],
        }
        merge_state(info['sha'], item)
        if name == bname:
            item['classes'].append('base')
            item['reasons'].append('the base branch')
        if name == default:
            item['classes'].append('default-branch')
            item['reasons'].append(f"the local copy of {remote}'s default")
        if keeps_branch(keep, name):
            item['classes'].append('kept')
            item['reasons'].append('on the keep-list')
        if is_archive(name):
            item['classes'].append('archive')
            item['reasons'].append('archive branch')
        for w in branch_flags.get(name, []):
            if not w['deletable']:
                item['classes'].append('checked-out')
                item['reasons'].append(
                    f'checked out in worktree {w["path"]}, which stays'
                )
        # Will `git branch -d` accept it? git compares against the upstream
        # when it is set and still exists, otherwise against HEAD.
        up = info['upstream']
        ref_sha = None
        pre = f'refs/remotes/{remote}/'
        if up and not (up.startswith(pre) and up[len(pre) :] in remote_deleted):
            ref_sha = rev(repo, up)
        ref_sha = ref_sha or head_sha
        d_ok = bool(ref_sha) and bool(is_ancestor(repo, info['sha'], ref_sha))
        item['flag'] = '-d' if d_ok else '-D'
        if not d_ok and 'merged' in item['classes']:
            if approves_branch(approved, name):
                item['classes'].append('approved-D')
            else:
                item['reasons'].append(
                    'git branch -d would refuse it (not merged into its '
                    'upstream or HEAD); -D needs the approved list'
                )
        item['deletable'] = 'merged' in item['classes'] and not item['reasons']
        local_items.append(item)

    return {
        'generated': dt.datetime.now(dt.timezone.utc).isoformat(),
        'repo': str(Path(repo).resolve()),
        'base': base,
        'base_sha': base_sha,
        'remote': remote,
        'head_sha': head_sha,
        'gh_checked': bool(use_gh and pr_heads is not None),
        'warnings': warnings,
        'worktrees': wt_items,
        'remote_branches': remote_items,
        'local_branches': local_items,
    }


def print_audit(res):
    print(f'base {res["base"]} = {res["base_sha"][:12]}')
    for w in res['warnings']:
        print(f'WARNING {w}')
    sections = (
        ('worktrees', 'path'),
        ('remote_branches', 'name'),
        ('local_branches', 'name'),
    )
    for key, label in sections:
        items = res[key]
        n = sum(i['deletable'] for i in items)
        print(f'\n{key}: {len(items)}, {n} deletable')
        for i in items:
            mark = 'DELETE' if i['deletable'] else 'keep  '
            sha = (i.get('sha') or '')[:10]
            cls = ','.join(i['classes'])
            why = '; '.join(i['reasons'])
            print(
                f'  {mark} {i[label]} {sha} [{cls}]'
                + (f' {why}' if why else '')
            )


# --- plan ------------------------------------------------------------------


def make_plan(res, keep_list_path, approved_path):
    items, skipped = [], []
    for w in res['worktrees']:
        if w['main']:
            continue
        if w['deletable']:
            items.append(
                {
                    'kind': 'worktree',
                    'path': w['path'],
                    'sha': w['sha'],
                    'branch': w['branch'],
                    'force': w['force'],
                }
            )
        else:
            skipped.append(
                {'kind': 'worktree', 'path': w['path'], 'reasons': w['reasons']}
            )
    for r in res['remote_branches']:
        if r['deletable']:
            items.append({'kind': 'remote', 'name': r['name'], 'sha': r['sha']})
        else:
            skipped.append(
                {'kind': 'remote', 'name': r['name'], 'reasons': r['reasons']}
            )
    for b in res['local_branches']:
        if b['deletable']:
            items.append(
                {
                    'kind': 'local',
                    'name': b['name'],
                    'sha': b['sha'],
                    'flag': b['flag'],
                }
            )
        else:
            skipped.append(
                {'kind': 'local', 'name': b['name'], 'reasons': b['reasons']}
            )
    return {
        'created': res['generated'],
        'repo': res['repo'],
        'base': res['base'],
        'base_sha': res['base_sha'],
        'remote': res['remote'],
        'gh_checked': res['gh_checked'],
        'keep_list': str(Path(keep_list_path).resolve()),
        'approved': str(Path(approved_path).resolve())
        if approved_path
        else None,
        'items': items,
        'skipped': skipped,
    }


def describe(item):
    if item['kind'] == 'worktree':
        return f'worktree {item["path"]}'
    return f'{item["kind"]} branch {item["name"]}'


def action(repo, remote, item):
    if item['kind'] == 'worktree':
        cmd = ['git', '-C', repo, 'worktree', 'remove']
        if item.get('force'):
            cmd.append('--force')
        return cmd + [item['path']]
    if item['kind'] == 'remote':
        return ['git', '-C', repo, 'push', remote, '--delete', item['name']]
    return ['git', '-C', repo, 'branch', item['flag'], item['name']]


def print_plan(plan):
    print(
        f'plan: {len(plan["items"])} deletion(s), '
        f'{len(plan["skipped"])} kept; base {plan["base"]} = '
        f'{plan["base_sha"][:12]}'
    )
    for i in plan['items']:
        print('  ' + shlex.join(action(plan['repo'], plan['remote'], i)))


# --- execute ---------------------------------------------------------------


def guard(
    repo,
    plan,
    item,
    keep,
    approved,
    base_sha,
    gh_state,
    gone=(),
    default=None,
):
    """Return the reasons the item may not be deleted now ([] = go).

    ``gone`` holds the real paths of worktrees this run has removed, or in
    a dry run would have removed, by the time it reaches the item;
    ``default`` is the remote's default branch.
    """
    reasons = []
    remote = plan['remote']
    kind = item['kind']
    if kind == 'worktree':
        wt = next(
            (
                w
                for w in worktrees(repo)
                if real(w['path']) == real(item['path'])
            ),
            None,
        )
        if wt is None:
            return ['no longer a registered worktree']
        if wt['head'] != item['sha']:
            reasons.append(f'HEAD moved to {wt["head"]}')
        if wt['branch'] != item['branch']:
            reasons.append(f'branch changed to {wt["branch"]}')
        reasons += worktree_problems(repo, wt, open_paths())
        if keeps_path(keep, item['path']) or (
            wt['branch'] and keeps_branch(keep, wt['branch'])
        ):
            reasons.append('on the keep-list')
        if wt['branch'] and wt['branch'] in (
            base_name(plan['base'], remote),
            default,
        ):
            reasons.append(f'checks out kept branch {wt["branch"]}')
        if item.get('force') and not approves_path(approved, item['path']):
            reasons.append('--force needs the path in the approved list')
        if (
            not item.get('force')
            and os.path.isdir(item['path'])
            and needs_force(item['path'])
        ):
            reasons.append('has submodules now; would need --force')
    elif kind == 'remote':
        live = remote_branches(repo, remote).get(item['name'])
        if live != item['sha']:
            reasons.append(f'remote tip is now {live}')
        if keeps_branch(keep, item['name']):
            reasons.append('on the keep-list')
        if item['name'] == base_name(plan['base'], remote):
            reasons.append('the base branch')
        if default and item['name'] == default:
            reasons.append(f'default branch of {remote}')
        if gh_state is not None:
            heads, protected = gh_state
            if item['name'] in heads:
                reasons.append('an open pull request uses it')
            if item['name'] in protected:
                reasons.append('protected branch')
    else:
        cur = rev(repo, f'refs/heads/{item["name"]}')
        if cur != item['sha']:
            reasons.append(f'tip is now {cur}')
        if keeps_branch(keep, item['name']):
            reasons.append('on the keep-list')
        if item['name'] == base_name(plan['base'], remote):
            reasons.append('the base branch')
        if default and item['name'] == default:
            reasons.append(f"the local copy of {remote}'s default")
        users = [
            w['path']
            for w in worktrees(repo)
            if w['branch'] == item['name'] and real(w['path']) not in gone
        ]
        if users:
            reasons.append(f'checked out in {users[0]}')
        if item['flag'] == '-D' and not approves_branch(approved, item['name']):
            reasons.append('-D needs the branch in the approved list')
    if not reasons:
        anc = is_ancestor(repo, item['sha'], base_sha)
        if not anc:
            reasons.append(
                f'{item["sha"][:12]} is not an ancestor of {plan["base"]} '
                f'({base_sha[:12]})'
            )
    return reasons


def bundle_ref(item, remote, index):
    if item['kind'] == 'local':
        return f'refs/heads/{item["name"]}'
    if item['kind'] == 'remote':
        return f'refs/remotes/{remote}/{item["name"]}'
    stem = re.sub(r'[^A-Za-z0-9._-]+', '-', Path(item['path']).name)
    return f'refs/worktree-heads/{index:03d}-{stem or "worktree"}'


def write_backup(repo, plan, backup_dir, stamp):
    """Bundle every planned ref; return (bundle path, manifest)."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    bundle = backup_dir / f'branch-cleanup-{stamp}.bundle'
    common = git(
        repo, 'rev-parse', '--path-format=absolute', '--git-common-dir'
    ).stdout.strip()
    remote = plan['remote']
    refs = []
    with tempfile.TemporaryDirectory(prefix='branch-cleanup-') as tmp:
        # A scratch repository that borrows the objects, so the bundle's
        # refs never touch the repository being cleaned.
        git(tmp, 'init', '--bare', '-q', tmp)
        alt = Path(tmp) / 'objects' / 'info' / 'alternates'
        alt.write_text(os.path.join(common, 'objects') + '\n')
        for n, item in enumerate(plan['items']):
            ref = bundle_ref(item, remote, n)
            git(tmp, 'update-ref', ref, item['sha'])
            refs.append(ref)
        git(tmp, 'bundle', 'create', str(bundle), '--all')
    git(repo, 'bundle', 'verify', str(bundle))
    rq = shlex.quote
    entries = []
    for ref, item in zip(refs, plan['items'], strict=True):
        fetch = (
            f'git -C {rq(plan["repo"])} fetch {rq(str(bundle))} '
            f'{rq(ref)}:{rq(ref)}'
        )
        if item['kind'] == 'local':
            restore = [fetch]
        elif item['kind'] == 'remote':
            restore = [
                fetch,
                f'git -C {rq(plan["repo"])} push {rq(remote)} '
                f'{item["sha"]}:refs/heads/{rq(item["name"])}',
            ]
        else:
            repo_q = rq(plan['repo'])
            if item['branch']:
                restore = [
                    fetch,
                    '# restore the local branch first if it was deleted too',
                    f'git -C {repo_q} worktree add {rq(item["path"])} '
                    f'{rq(item["branch"])}',
                ]
            else:
                restore = [
                    fetch,
                    f'git -C {repo_q} worktree add --detach '
                    f'{rq(item["path"])} {item["sha"]}',
                ]
        entries.append(
            {
                **item,
                'bundle_ref': ref,
                'restore': restore,
                'status': 'planned',
            }
        )
    manifest = {
        'created': stamp,
        'repo': plan['repo'],
        'base': plan['base'],
        'base_sha': plan['base_sha'],
        'remote': remote,
        'bundle': str(bundle),
        'items': entries,
    }
    return bundle, manifest


def save_manifest(backup_dir, stamp, manifest):
    path = backup_dir / f'branch-cleanup-{stamp}.restore.json'
    path.write_text(json.dumps(manifest, indent=1) + '\n')
    sh = backup_dir / f'branch-cleanup-{stamp}.restore.sh'
    lines = [
        '#!/bin/sh',
        '# Restore commands for the branch cleanup recorded in',
        f'# {path.name}. Run only the lines you need; status per item is',
        '# in the JSON manifest.',
    ]
    for e in manifest['items']:
        lines.append(f'\n# {describe(e)} ({e["sha"]}): {e["status"]}')
        lines += e['restore']
    sh.write_text('\n'.join(lines) + '\n')
    return path


def execute(args):
    plan = json.loads(Path(args.plan).read_text())
    repo = args.repo or plan['repo']
    remote = plan['remote']
    keep_path = args.keep_list or plan['keep_list']
    if not Path(keep_path).is_file():
        raise CleanupError(f'keep-list {keep_path} not found')
    keep = load_list(keep_path)
    approved = load_list(args.approved)
    base_sha = rev(repo, base_ref(repo, plan['base'], remote))
    if not base_sha:
        raise CleanupError(f'base {plan["base"]} does not resolve')
    default = remote_default(repo, remote)
    gh_state = None
    if plan.get('gh_checked'):
        gh_state = github_state(repo, remote)
    items = plan['items']
    mode = 'APPLY' if args.apply else 'DRY RUN'
    print(
        f'{mode}: {len(items)} item(s); base {plan["base"]} = '
        f'{base_sha[:12]} (plan saw {plan["base_sha"][:12]})'
    )
    if not items:
        return 0
    manifest = backup_dir = stamp = None
    if args.apply:
        stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        if args.backup_dir:
            backup_dir = Path(args.backup_dir).expanduser().resolve()
        else:
            common = git(
                repo,
                'rev-parse',
                '--path-format=absolute',
                '--git-common-dir',
            ).stdout.strip()
            backup_dir = Path(common) / 'branch-cleanup-backups'
        bundle, manifest = write_backup(repo, plan, backup_dir, stamp)
        mpath = save_manifest(backup_dir, stamp, manifest)
        print(f'backup bundle {bundle}')
        print(f'restore manifest {mpath}')
    failed = 0
    gone = set()
    for n, item in enumerate(items):
        reasons = guard(
            repo,
            plan,
            item,
            keep,
            approved,
            base_sha,
            gh_state,
            gone,
            default,
        )
        cmd = action(repo, remote, item)
        if reasons:
            failed += 1
            print(f'REFUSED {describe(item)}: ' + '; '.join(reasons))
            if args.apply:
                manifest['items'][n]['status'] = 'refused: ' + '; '.join(
                    reasons
                )
                save_manifest(backup_dir, stamp, manifest)
                print('STOPPED at the first failure; later items untouched')
                return 1
            continue
        if item['kind'] == 'worktree':
            gone.add(real(item['path']))
        if not args.apply:
            print('WOULD ' + shlex.join(cmd))
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            manifest['items'][n]['status'] = f'failed: {out}'
            save_manifest(backup_dir, stamp, manifest)
            print(f'FAILED {shlex.join(cmd)}: {out}')
            print('STOPPED at the first failure; later items untouched')
            return 1
        manifest['items'][n]['status'] = 'deleted'
        save_manifest(backup_dir, stamp, manifest)
        print('RAN ' + shlex.join(cmd))
    if failed:
        print(f'{failed} item(s) would be refused')
        return 1
    print('done' if args.apply else 'dry run: nothing was changed')
    return 0


# --- CLI -------------------------------------------------------------------


def toplevel(path):
    out = git(path or '.', 'rev-parse', '--show-toplevel', check=False)
    if out.returncode != 0:
        raise CleanupError(f'{path or "."} is not inside a git repository')
    return out.stdout.strip()


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--repo', help='repository (default: cwd)')
    scan = argparse.ArgumentParser(add_help=False)
    scan.add_argument(
        '--base', required=True, help='merged means an ancestor of this ref'
    )
    scan.add_argument('--remote', default='origin')
    scan.add_argument(
        '--archive-glob',
        action='append',
        help='archive branch glob (default *archive*)',
    )
    scan.add_argument(
        '--no-gh',
        action='store_true',
        help='skip the open-PR and protected-branch check',
    )
    scan.add_argument('--approved', help='owner-approved list for -D/--force')

    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n\n', 1)[1],
    )
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser(
        'audit',
        parents=[common, scan],
        help='classify branches and worktrees (read-only)',
    )
    a.add_argument('--keep-list')
    a.add_argument('--json', help='also write the audit as JSON')
    p = sub.add_parser(
        'plan', parents=[common, scan], help='write a guarded deletion plan'
    )
    p.add_argument('--keep-list', required=True)
    p.add_argument('--out', required=True, help='plan JSON to write')
    e = sub.add_parser(
        'execute', parents=[common], help='run a plan (dry run unless --apply)'
    )
    e.add_argument('plan')
    e.add_argument(
        '--apply', action='store_true', help='really delete (default: dry run)'
    )
    e.add_argument('--approved', help='owner-approved list for -D/--force')
    e.add_argument('--backup-dir', help='where the bundle and manifest go')
    e.add_argument('--keep-list', help="override the plan's keep-list")
    args = ap.parse_args(argv)
    try:
        if args.cmd == 'execute':
            if args.repo:
                args.repo = toplevel(args.repo)
            return execute(args)
        repo = toplevel(args.repo)
        keep = load_list(args.keep_list)
        approved = load_list(args.approved)
        res = audit(
            repo,
            args.base,
            args.remote,
            keep,
            args.archive_glob or ['*archive*'],
            not args.no_gh,
            approved,
        )
        if args.cmd == 'audit':
            print_audit(res)
            if args.json:
                Path(args.json).write_text(json.dumps(res, indent=1) + '\n')
            return 0
        plan = make_plan(res, args.keep_list, args.approved)
        Path(args.out).write_text(json.dumps(plan, indent=1) + '\n')
        for w in res['warnings']:
            print(f'WARNING {w}')
        print_plan(plan)
        print(f'wrote {args.out}')
        return 0
    except CleanupError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
