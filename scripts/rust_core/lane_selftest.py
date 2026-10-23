#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The self-tests of lane.py, and of the lane rules in land.py, on real git.

Each case makes a temporary bare remote holding ``main``, ``rust-core`` and
``rust-core-phase-1``, a clone of it with a second worktree, and a fake
``gh`` first on ``PATH`` that answers from a JSON file the case writes (the
CI runs on each branch and commit) and records every call. ``lane.py`` and
``land.py`` run as processes against that clone, as a person runs them: no
mock stands between them and git, and the remote refuses a push that is not
a fast-forward or that deletes a branch (``receive.denyNonFastForwards``,
``receive.denyDeletes``), so a forced or deleting push would fail the case.
Every case also holds that ``main``, ``rust-core`` and the phase branch on
the remote are moved by nothing but the case itself or a landing. Git runs
with a configuration of the case's own, never the developer's. Run it as

    python3 scripts/rust_core/lane_selftest.py

Decision E43 and package CI2.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
LANE_PY = SCRIPTS / 'lane.py'
LAND_PY = SCRIPTS / 'land.py'
PHASE = 'rust-core-phase-1'
sys.path.insert(0, str(SCRIPTS))
from phases import CURRENT  # noqa: E402

BRANCH = f'{PHASE}-lane-1b'
PROTECTED = ('main', 'rust-core', PHASE)

FAKE_GH = r'''#!/usr/bin/env python3
"""A fake gh: answers run list, watch and view from $FAKE_GH_STATE."""
import json
import os
import sys

path = os.environ['FAKE_GH_STATE']
with open(path) as handle:
    state = json.load(handle)
args = sys.argv[1:]
state['calls'].append(args)


def value(flag):
    return args[args.index(flag) + 1] if flag in args else None


code, out = 0, ''
runs = state['runs']
if args[:2] == ['run', 'list'] and '--branch' in args:
    out = json.dumps([r for r in runs if r['headBranch'] == value('--branch')])
elif args[:2] == ['run', 'list'] and '--commit' in args:
    sha = value('--commit')
    found = [r for r in runs if r['headSha'] == sha]
    if not found:
        # Runs appear on each branch in green_on, concluding as the state
        # says at the time, and are kept, as GitHub keeps them.
        found = [
            {'databaseId': 900 + len(runs) + i, 'headSha': sha,
             'headBranch': branch, 'status': 'completed',
             'conclusion': state.get('conclusion', 'success'),
             'event': 'push',
             'url': 'https://example.invalid/runs/%d' % (900 + len(runs) + i),
             'createdAt': '2026-10-09T00:00:0%dZ' % i}
            for i, branch in enumerate(state['green_on'])
        ]
        runs.extend(found)
    out = json.dumps(found)
elif args[:2] == ['run', 'watch'] and state.get('watch_fails'):
    code = 1
    sys.stderr.write('fake gh: network down\n')
elif args[:2] in (['run', 'watch'], ['run', 'view']):
    found = [r for r in runs if str(r['databaseId']) == args[2]]
    if not found:
        code = 1
    elif args[1] == 'watch' and found[0]['status'] != 'completed':
        found[0].update(status='completed', conclusion='success')
    elif args[1] == 'watch':
        code = 0 if found[0]['conclusion'] == 'success' else 1
    else:
        out = json.dumps(found[0])
else:
    code = 1
    sys.stderr.write('fake gh: unexpected %r\n' % (args,))
with open(path, 'w') as handle:
    json.dump(state, handle)
sys.stdout.write(out)
sys.exit(code)
'''


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


@dataclass
class Ran:
    code: int
    text: str


def run(argv: list[str], cwd: Path, env: dict[str, str]) -> Ran:
    done = subprocess.run(
        argv, cwd=cwd, env=env, capture_output=True, text=True
    )
    return Ran(done.returncode, done.stdout + done.stderr)


class World:
    """A bare remote, a seed clone that plays everyone else, and the clone
    lane.py runs in, with a second worktree beside it."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp.resolve()
        (self.tmp / 'bin').mkdir()
        gh = self.tmp / 'bin' / 'gh'
        gh.write_text(f'#!{sys.executable}\n' + FAKE_GH.split('\n', 1)[1])
        gh.chmod(0o755)
        config = self.tmp / 'gitconfig'
        config.write_text(
            '[user]\n\tname = Test Person\n\temail = test@example.com\n'
            '[init]\n\tdefaultBranch = main\n'
            '[advice]\n\tdetachedHead = false\n'
        )
        self.env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(('GIT_', 'GH_'))
        }
        self.env.update(
            GIT_CONFIG_GLOBAL=str(config),
            GIT_CONFIG_NOSYSTEM='1',
            FAKE_GH_STATE=str(self.tmp / 'gh.json'),
            PATH=f'{self.tmp / "bin"}{os.pathsep}{os.environ["PATH"]}',
            # The phase, as the orchestration tools name it (T2.0); the
            # phase cases below change or drop it.
            SDM_PHASE_BRANCH=PHASE,
        )
        # The cases' own lane limit, not the machine's (heavy.conf's
        # MAX_LANES, one package at a time since 2026-10-23): several cases
        # start a second lane; the limit's own case sets its own.
        limits = self.tmp / 'heavy.conf'
        limits.write_text('MAX_LANES=4\n')
        self.env['SDM_HEAVY_CONFIG'] = str(limits)
        self.gh_state({'runs': [], 'green_on': [], 'calls': []})
        self.remote = self.tmp / 'remote.git'
        self.git('init', '--quiet', '--bare', str(self.remote), cwd=self.tmp)
        self.git(
            'config', 'receive.denyNonFastForwards', 'true', cwd=self.remote
        )
        self.git('config', 'receive.denyDeletes', 'true', cwd=self.remote)
        self.seed = self.tmp / 'seed'
        self.git(
            'clone', '--quiet', str(self.remote), str(self.seed), cwd=self.tmp
        )
        self.base = self.commit(self.seed, 'Base')
        self.git('push', '--quiet', 'origin', 'HEAD:refs/heads/main', 'HEAD:refs/heads/rust-core', cwd=self.seed)  # fmt: skip
        self.phase = self.commit(self.seed, 'Land T1.11d: the landed package')
        self.git('push', '--quiet', 'origin', f'HEAD:refs/heads/{PHASE}', cwd=self.seed)  # fmt: skip
        self.clone = self.tmp / 'clone'
        self.git(
            'clone', '--quiet', str(self.remote), str(self.clone), cwd=self.tmp
        )
        self.other = self.tmp / 'other'
        self.git('worktree', 'add', '--quiet', '-b', 'other-work', str(self.other), f'origin/{PHASE}')  # fmt: skip
        self.worktree = self.tmp / 'lane-1b'

    # --- git and the scripts ---

    def git(self, *args: str, cwd: Path | None = None) -> str:
        done = run(['git', *args], cwd or self.clone, self.env)
        if done.code != 0:
            raise AssertionError(f'git {" ".join(args)}: {done.text}')
        return done.text

    def git_ok(self, *args: str, cwd: Path | None = None) -> Ran:
        return run(['git', *args], cwd or self.clone, self.env)

    def commit(self, where: Path, message: str) -> str:
        self.git('commit', '--quiet', '--allow-empty', '-m', message, cwd=where)
        return self.git('rev-parse', 'HEAD', cwd=where).strip()

    def lane(self, *args: str, cwd: Path | None = None) -> Ran:
        return run(
            [sys.executable, str(LANE_PY), *args], cwd or self.clone, self.env
        )

    def start(self, package: str = 'T1.11e', *extra: str) -> Ran:
        return self.lane(
            'start', '--lane', '1b', '--package', package,
            '--worktree', str(self.worktree), *extra,
        )  # fmt: skip

    def land(self, *args: str, cwd: Path | None = None) -> Ran:
        return run(
            [sys.executable, str(LAND_PY), *args],
            cwd or self.worktree,
            self.env,
        )

    # --- the remote, from someone else's clone ---

    def refs(self) -> dict[str, str]:
        out = self.git(
            'for-each-ref', '--format=%(refname) %(objectname)', cwd=self.remote
        )
        return dict(line.split(' ', 1) for line in out.splitlines())

    def tip(self, branch: str) -> str | None:
        return self.refs().get(f'refs/heads/{branch}')

    def advance_phase(self, message: str = 'Land T1.11x: another lane') -> str:
        self.git('fetch', '--quiet', 'origin', cwd=self.seed)
        self.git('checkout', '--quiet', '-B', 'work', f'origin/{PHASE}', cwd=self.seed)  # fmt: skip
        sha = self.commit(self.seed, message)
        self.git('push', '--quiet', 'origin', f'HEAD:refs/heads/{PHASE}', cwd=self.seed)  # fmt: skip
        return sha

    def put_lane(self, sha: str) -> None:
        self.git('push', '--quiet', 'origin', f'{sha}:refs/heads/{BRANCH}', cwd=self.seed)  # fmt: skip

    def unlanded(self, *messages: str) -> list[str]:
        """Commits on the remote lane, on top of the phase head."""
        self.git('fetch', '--quiet', 'origin', cwd=self.seed)
        self.git('checkout', '--quiet', '-B', 'pkg', f'origin/{PHASE}', cwd=self.seed)  # fmt: skip
        shas = [self.commit(self.seed, m) for m in messages]
        self.put_lane(shas[-1])
        return shas

    # --- the fake gh ---

    def gh_state(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self.tmp / 'gh.json'
        if state is not None:
            path.write_text(json.dumps(state))
        return json.loads(path.read_text())

    def set_runs(
        self, *runs: dict[str, Any], green_on: list[str] | None = None
    ) -> None:
        state = self.gh_state()
        state['runs'] = list(runs)
        if green_on is not None:
            state['green_on'] = green_on
        self.gh_state(state)

    def gh_calls(self) -> list[list[str]]:
        return self.gh_state()['calls']

    # --- the local state ---

    def local_state(self) -> dict[str, str]:
        """Everything a dry run must leave as it was."""
        return {
            'refs': self.git(
                'for-each-ref', '--format=%(refname) %(objectname)'
            ),
            'worktrees': self.git('worktree', 'list', '--porcelain'),
            'config': self.git('config', '--list', '--local'),
            'remote': json.dumps(self.refs()),
            'lane worktree': str(self.worktree.exists()),
            'hooks': ' '.join(
                sorted(
                    p.name for p in (self.clone / '.git' / 'hooks').iterdir()
                )
            ),
        }

    def package(self, where: Path) -> str | None:
        done = self.git_ok(
            'config', '--worktree', '--get', 'sdm.package', cwd=where
        )
        return done.text.strip() if done.code == 0 else None

    def other_config(self) -> str:
        """The other worktree's configuration, but for what lane.py sets
        on purpose."""
        return '\n'.join(
            line
            for line in self.git(
                'config', '--list', cwd=self.other
            ).splitlines()
            # The extension itself, and the lane's upstream, which is the
            # clone's shared configuration by design.
            if line != 'extensions.worktreeconfig=true'
            and not line.startswith(f'branch.{BRANCH}.')
        )


def ci_run(run_id: int, sha: str, status: str) -> dict[str, Any]:
    return {
        'databaseId': run_id, 'headSha': sha, 'headBranch': BRANCH,
        'status': status, 'conclusion': '', 'event': 'push',
        'url': f'https://example.invalid/runs/{run_id}',
        'createdAt': f'2026-10-09T00:00:{run_id % 60:02d}Z',
    }  # fmt: skip


Check = Callable[[World], tuple[bool, str]]
CASES: list[tuple[str, Check]] = []


def case(name: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        CASES.append((name, check))
        return check

    return register


def verdict(ok: bool, good: str, ran: Ran | str) -> tuple[bool, str]:
    text = ran if isinstance(ran, str) else f'exit {ran.code}: {ran.text}'
    return ok, good if ok else text[-900:]


def protected(w: World, before: dict[str, str]) -> bool:
    now = w.refs()
    return all(
        now.get(f'refs/heads/{b}') == before.get(f'refs/heads/{b}')
        for b in PROTECTED
    )


def refused_untouched(
    w: World, ran: Ran, fragment: str, before: dict[str, str]
) -> bool:
    return (
        ran.code == 2
        and 'REFUSED' in ran.text
        and fragment in ran.text
        and w.refs() == before
        and not w.worktree.exists()
    )


# --- start: create and fast-forward -------------------------------------------


@case('start: creates the lane at the phase head, its worktree and package')
def _(w: World) -> tuple[bool, str]:
    before, other = w.refs(), w.other_config()
    default = w.worktree
    ran = w.start()
    hooks = w.clone / '.git' / 'hooks'
    listed = (hooks / 'prepush-worktrees').read_text().splitlines()
    ok = (
        str(default) in listed
        and 'prepush.sh --install-hook' in (hooks / 'pre-push').read_text()
        and ran.code == 0
        and w.tip(BRANCH) == w.phase
        and protected(w, before)
        and w.git('rev-parse', 'HEAD', cwd=default).strip() == w.phase
        and w.git('symbolic-ref', '--short', 'HEAD', cwd=default).strip()
        == BRANCH
        and w.git('rev-parse', '--abbrev-ref', '@{u}', cwd=default).strip()
        == f'origin/{BRANCH}'
        and w.package(default) == 'T1.11e'
        and w.package(w.clone) is None
        and w.package(w.other) is None
        and w.other_config() == other
        and f'STARTED T1.11e on {BRANCH} at {w.phase}' in ran.text
        and f'next: cd {default}' in ran.text
    )
    return verdict(ok, 'created, recorded, nothing else changed', ran)


@case('start: fast-forwards a lane behind the phase head, by a normal push')
def _(w: World) -> tuple[bool, str]:
    w.put_lane(w.base)
    before = w.refs()
    ran = w.start()
    ok = (
        ran.code == 0
        and w.tip(BRANCH) == w.phase
        and protected(w, before)
        and 'fast-forward' in ran.text
        and w.package(w.worktree) == 'T1.11e'
    )
    return verdict(ok, 'fast-forwarded', ran)


@case('start: the next package fast-forwards the lane and its worktree')
def _(w: World) -> tuple[bool, str]:
    first = w.start()
    moved = w.advance_phase()
    before = w.refs()
    ran = w.start('T1.11f')
    ok = (
        first.code == 0
        and ran.code == 0
        and w.tip(BRANCH) == moved
        and protected(w, before)
        and w.git('rev-parse', 'HEAD', cwd=w.worktree).strip() == moved
        and w.package(w.worktree) == 'T1.11f'
        and '(was T1.11e)' in ran.text
    )
    return verdict(ok, 'lane and worktree at the new head, T1.11f', ran)


@case('start: a lane at the phase head is not pushed again')
def _(w: World) -> tuple[bool, str]:
    w.put_lane(w.phase)
    before = w.refs()
    ran = w.start()
    pushes = 'fast-forward' in ran.text or 'is at the phase head (' in ran.text
    ok = ran.code == 0 and w.refs() == before and not pushes
    return verdict(ok, 'nothing pushed', ran)


# --- start: refusals ----------------------------------------------------------


@case('refused: the lane holds unlanded commits; the steps are printed only')
def _(w: World) -> tuple[bool, str]:
    shas = w.unlanded('T1.11e: unfinished work', 'T1.11e fix 1: more')
    before = w.refs()
    ran = w.start('T1.11f')
    tip = shas[-1]
    archive = f'{PHASE}-archive-1b-T1.11e-{tip[:7]}'
    ok = (
        refused_untouched(w, ran, 'has not landed', before)
        and all(sha[:12] in ran.text for sha in shas)
        and 'T1.11e (2 of 2)' in ran.text
        and f'git push origin {tip}:refs/heads/{archive}' in ran.text
        and f'--force-with-lease=refs/heads/{BRANCH}:{tip}' in ran.text
        and "owner's decision" in ran.text
        and not any(ref.startswith('refs/tags/') for ref in w.refs())
    )
    return verdict(ok, 'refused, named, nothing moved or archived', ran)


@case('refused: the local lane branch holds commits not pushed or landed')
def _(w: World) -> tuple[bool, str]:
    first = w.start()
    local = w.commit(w.worktree, 'T1.11e: local only')
    before = w.refs()
    ran = w.start('T1.11f')
    ok = (
        first.code == 0
        and ran.code == 2
        and local[:12] in ran.text
        and 'the local' in ran.text
        and w.refs() == before
        and w.git('rev-parse', 'HEAD', cwd=w.worktree).strip() == local
        and w.package(w.worktree) == 'T1.11e'
    )
    return verdict(ok, 'refused, the local work kept', ran)


for state in ('in_progress', 'queued'):

    @case(f'refused: a CI run {state} on the lane')
    def _(w: World, state: str = state) -> tuple[bool, str]:
        w.put_lane(w.base)
        w.set_runs(ci_run(77, w.base, state))
        before = w.refs()
        ran = w.start()
        ok = (
            refused_untouched(w, ran, 'run 77', before)
            and 'has not finished' in ran.text
        )
        return verdict(ok, 'refused, nothing moved', ran)


@case('start --wait: watches the run, then starts')
def _(w: World) -> tuple[bool, str]:
    w.put_lane(w.base)
    w.set_runs(ci_run(77, w.base, 'in_progress'))
    ran = w.start('T1.11e', '--wait')
    watched = any(call[:3] == ['run', 'watch', '77'] for call in w.gh_calls())
    ok = ran.code == 0 and watched and w.tip(BRANCH) == w.phase
    return verdict(ok, 'watched, then fast-forwarded', ran)


@case('refused: enabling per-worktree configuration would change a worktree')
def _(w: World) -> tuple[bool, str]:
    dormant = w.git(
        'rev-parse', '--path-format=absolute', '--git-path', 'config.worktree',
        cwd=w.other,
    ).strip()  # fmt: skip
    Path(dormant).write_text('[sdm]\n\tpackage = T1.11a\n')
    before = w.refs()
    ran = w.start()
    off = w.git_ok('config', '--get', 'extensions.worktreeConfig').code != 0
    ok = refused_untouched(w, ran, dormant, before) and off
    return verdict(ok, 'refused before any change', ran)


@case('refused: the worktree path is taken')
def _(w: World) -> tuple[bool, str]:
    w.worktree.mkdir()
    (w.worktree / 'file').write_text('x')
    before = w.refs()
    ran = w.start()
    ok = ran.code == 2 and 'exists' in ran.text and w.refs() == before
    return verdict(ok, 'refused', ran)


@case('refused: untracked files would carry into the next package')
def _(w: World) -> tuple[bool, str]:
    w.start()
    (w.worktree / 'pending.py').write_text('x = 1\n')
    w.advance_phase()
    before = w.refs()
    ran = w.start('T1.11f')
    ok = (
        ran.code == 2
        and 'pending.py' in ran.text
        and w.refs() == before
        and w.package(w.worktree) == 'T1.11e'
    )
    return verdict(ok, 'refused, the package unchanged', ran)


@case('start: ignored build files stay, and do not block the next package')
def _(w: World) -> tuple[bool, str]:
    w.start()
    exclude = w.clone / '.git' / 'info' / 'exclude'
    exclude.parent.mkdir(exist_ok=True)
    exclude.write_text('target/\n')
    (w.worktree / 'target').mkdir()
    (w.worktree / 'target' / 'build.o').write_text('o')
    w.advance_phase()
    ran = w.start('T1.11f')
    ok = (
        ran.code == 0
        and (w.worktree / 'target' / 'build.o').exists()
        and w.package(w.worktree) == 'T1.11f'
    )
    return verdict(ok, 'started, the build kept', ran)


@case('start: a foreign pre-push hook is left alone, and the push goes')
def _(w: World) -> tuple[bool, str]:
    hook = w.clone / '.git' / 'hooks' / 'pre-push'
    hook.parent.mkdir(exist_ok=True)
    hook.write_text('#!/bin/sh\nexit 1\n')
    hook.chmod(0o755)
    ran = w.start()
    ok = (
        ran.code == 0
        and w.tip(BRANCH) == w.phase
        and hook.read_text() == '#!/bin/sh\nexit 1\n'
        and 'pre-push hook was not installed' in ran.text
    )
    return verdict(ok, 'pushed; the hook untouched, the gap reported', ran)


@case('refused: start without --worktree')
def _(w: World) -> tuple[bool, str]:
    before = w.refs()
    ran = w.lane('start', '--lane', '1b', '--package', 'T1.11e')
    ok = (
        ran.code == 2
        and '--worktree' in ran.text
        and w.refs() == before
        and not w.gh_calls()
    )
    return verdict(ok, 'refused', ran)


@case('start --wait: a failing gh watch backs off, then gives up')
def _(w: World) -> tuple[bool, str]:
    w.put_lane(w.base)
    w.set_runs(ci_run(77, w.base, 'in_progress'))
    state = w.gh_state()
    state['watch_fails'] = True
    w.gh_state(state)
    before = w.refs()
    ran = w.start('T1.11e', '--wait', '--wait-seconds', '0.001')
    watches = [c for c in w.gh_calls() if c[:2] == ['run', 'watch']]
    ok = (
        ran.code == 2
        and len(watches) == 10
        and ran.text.count('gh run watch failed (fake gh: network down)') == 10
        and 'again in 0.001s' in ran.text
        and 'again in 0.002s' in ran.text
        and 'again in 0.512s' in ran.text
        and 'after 10 rounds' in ran.text
        and w.refs() == before
    )
    return verdict(ok, 'ten paced rounds, then refused', ran)


# --- dry run ------------------------------------------------------------------


for setup in ('create', 'fast-forward', 'an existing worktree'):

    @case(f'dry run: {setup}, with no side effect')
    def _(w: World, setup: str = setup) -> tuple[bool, str]:
        if setup == 'fast-forward':
            w.put_lane(w.base)
        package = 'T1.11e'
        if setup == 'an existing worktree':
            # The worktree, the extension and the hook are there already;
            # the phase moves on, so the next package would fast-forward.
            w.start()
            w.advance_phase()
            package = 'T1.11f'
        before = w.local_state()
        ran = w.start(package, '--dry-run')
        after = w.local_state()
        changed = [k for k in before if before[k] != after[k]]
        ok = (
            ran.code == 0
            and not changed
            and 'nothing pushed, no worktree made' in ran.text
            and f'record {package}' in ran.text
            and (setup != 'an existing worktree' or 'fast-forward' in ran.text)
        )
        return verdict(
            ok, 'checked, nothing changed', f'changed {changed}; {ran.text}'
        )


# --- names --------------------------------------------------------------------


BAD_LANES = ('B1', '1-b', 'lane-1b', '', '1b/x', 'a' * 17, '1b ', '1.b')
BAD_PACKAGES = ('T1', 't1.11e', 'CI', 'T1.11e;x', '1.11e', 'T1.11ee', 'CI3a')

for lane in BAD_LANES:

    @case(f'refused: the lane name {lane!r}')
    def _(w: World, lane: str = lane) -> tuple[bool, str]:
        before = w.refs()
        ran = w.lane(
            'start', '--lane', lane, '--package', 'T1.11e',
            '--worktree', str(w.worktree),
        )  # fmt: skip
        ok = (
            refused_untouched(w, ran, 'not a lane name', before)
            and not w.gh_calls()
        )
        return verdict(ok, 'refused before git or gh', ran)


for package in BAD_PACKAGES:

    @case(f'refused: the package {package!r}')
    def _(w: World, package: str = package) -> tuple[bool, str]:
        before = w.refs()
        ran = w.start(package)
        ok = (
            refused_untouched(w, ran, 'not a package ID', before)
            and not w.gh_calls()
        )
        return verdict(ok, 'refused before git or gh', ran)


# --- prepush.sh's check -------------------------------------------------------


@case('check: passes when the package owns every lane commit')
def _(w: World) -> tuple[bool, str]:
    w.start()
    w.commit(w.worktree, 'T1.11e: the work')
    w.commit(
        w.worktree,
        'T1.11e tracker: the previous package landed\n\n'
        'T1.11d landed at abc1234.\n\nRefs: T1.11d',
    )
    ran = w.lane('check', cwd=w.worktree)
    ok = ran.code == 0 and 'carries T1.11e: 2 commit(s)' in ran.text
    return verdict(ok, 'passed', ran)


@case('check: fails on a commit naming another package')
def _(w: World) -> tuple[bool, str]:
    w.start()
    w.commit(w.worktree, 'T1.11e: the work')
    stray = w.commit(w.worktree, 'T1.11f: the next package, too early')
    ran = w.lane('check', cwd=w.worktree)
    ok = (
        ran.code == 2
        and f'{stray[:12]} T1.11f: the next package' in ran.text
        and 'T1.11e: the work' not in ran.text
    )
    return verdict(ok, 'refused, naming the commit', ran)


@case('check: fails on a commit naming only a longer ID')
def _(w: World) -> tuple[bool, str]:
    w.start()
    w.commit(w.worktree, 'T1.11ee and T1.11e1 and xT1.11e')
    ran = w.lane('check', cwd=w.worktree)
    return verdict(
        ran.code == 2 and 'not owned by it' in ran.text, 'refused', ran
    )


@case('check: fails on a lane whose package is unset')
def _(w: World) -> tuple[bool, str]:
    w.start()
    w.git('config', '--worktree', '--unset', 'sdm.package', cwd=w.worktree)
    ran = w.lane('check', cwd=w.worktree)
    ok = ran.code == 2 and 'records no package' in ran.text
    return verdict(ok, 'refused', ran)


@case('check: fails on a lane checked out where per-worktree config is off')
def _(w: World) -> tuple[bool, str]:
    w.git('checkout', '--quiet', '-b', BRANCH, f'origin/{PHASE}')
    ran = w.lane('check')
    ok = ran.code == 2 and 'sdm.package is unset' in ran.text
    return verdict(ok, 'refused', ran)


@case('check: does nothing on a branch that is not a lane')
def _(w: World) -> tuple[bool, str]:
    w.commit(w.clone, 'no package named')
    ran = w.lane('check')
    ok = ran.code == 0 and 'not a lane branch (main)' in ran.text
    return verdict(ok, 'nothing to check', ran)


#: The two ways a commit could pass as the recorded package's while being
#: another's (CI2 review round 1): a subject naming another package that
#: mentions this one, and a subject naming none with this one in a trailer.
REPROS = (
    ('T1.11f: the next package, which builds on T1.11e', "T1.11f's"),
    ('Unrelated change\n\nRefs: T1.11e', 'does not start with T1.11e'),
)

for message, why in REPROS:

    @case(f'check: refuses {message.splitlines()[0]!r}')
    def _(w: World, message: str = message, why: str = why) -> tuple[bool, str]:
        w.start()
        w.commit(w.worktree, 'T1.11e: the work')
        stray = w.commit(w.worktree, message)
        ran = w.lane('check', cwd=w.worktree)
        ok = ran.code == 2 and stray[:12] in ran.text and why in ran.text
        return verdict(ok, 'refused, naming the commit', ran)


@case('check: a fresh phase head after start (another lane landed)')
def _(w: World) -> tuple[bool, str]:
    # The clone has not fetched since the phase moved; start must leave its
    # tracking ref at the head it started from, or check counts the other
    # lane's landed commit as this lane's.
    moved = w.advance_phase('Land T1.11x: another lane')
    started = w.start()
    w.commit(w.worktree, 'T1.11e: the work')
    ran = w.lane('check', cwd=w.worktree)
    tracking = w.git('rev-parse', f'refs/remotes/origin/{PHASE}').strip()
    local_phase = w.git_ok('rev-parse', '--verify', f'refs/heads/{PHASE}')
    ok = (
        started.code == 0
        and ran.code == 0
        and tracking == moved
        and local_phase.code != 0
        and 'carries T1.11e: 1 commit(s)' in ran.text
    )
    return verdict(ok, 'the tracking ref refreshed; check passes', ran)


def pushed(w: World, *lines: str, cwd: Path | None = None) -> Ran:
    refs = w.tmp / 'pushed-refs'
    refs.write_text(''.join(line + '\n' for line in lines))
    return w.lane('check', '--pushed-refs', str(refs), cwd=cwd)


@case('check --pushed-refs: a lane push of owned commits passes')
def _(w: World) -> tuple[bool, str]:
    w.start()
    sha = w.commit(w.worktree, 'T1.11e: the work')
    ran = pushed(
        w,
        f'refs/heads/{BRANCH} {sha} refs/heads/{BRANCH} {w.phase}',
        f'refs/heads/x {sha} refs/heads/other {"0" * 40}',
        cwd=w.worktree,
    )
    ok = ran.code == 0 and 'carries T1.11e: 1 commit(s)' in ran.text
    return verdict(ok, 'passed', ran)


@case('check --pushed-refs: a stray commit pushed to the lane is refused')
def _(w: World) -> tuple[bool, str]:
    w.start()
    stray = w.commit(w.worktree, REPROS[0][0])
    ran = pushed(
        w, f'HEAD {stray} refs/heads/{BRANCH} {w.phase}', cwd=w.worktree
    )
    return verdict(ran.code == 2 and stray[:12] in ran.text, 'refused', ran)


@case('check --pushed-refs: a lane pushed from another branch is refused')
def _(w: World) -> tuple[bool, str]:
    w.start()
    sha = w.commit(w.clone, 'T1.11e: from the wrong worktree')
    ran = pushed(w, f'refs/heads/main {sha} refs/heads/{BRANCH} {w.phase}')
    ok = ran.code == 2 and 'from main' in ran.text
    return verdict(ok, 'refused', ran)


@case('check --pushed-refs: deleting a lane is refused')
def _(w: World) -> tuple[bool, str]:
    w.start()
    ran = pushed(
        w, f'(delete) {"0" * 40} refs/heads/{BRANCH} {w.phase}', cwd=w.worktree
    )
    return verdict(ran.code == 2 and 'deletes' in ran.text, 'refused', ran)


@case('check --pushed-refs: a push naming no lane is not checked')
def _(w: World) -> tuple[bool, str]:
    sha = w.commit(w.clone, 'no package named')
    # A topic branch: main is protected since T3.0 (the owner, Q14).
    ran = pushed(w, f'refs/heads/topic {sha} refs/heads/topic {w.base}')
    ok = ran.code == 0 and 'no lane branch in this push' in ran.text
    return verdict(ok, 'nothing to check', ran)


# --- land.py on a lane --------------------------------------------------------


def candidate(w: World, message: str = 'T1.11e: the work') -> str:
    w.start()
    sha = w.commit(w.worktree, message)
    w.git('push', '--quiet', 'origin', f'HEAD:{BRANCH}', cwd=w.worktree)
    w.set_runs(green_on=[BRANCH, PHASE])
    return sha


def land_args(w: World, sha: str, title: str) -> list[str]:
    # A landing stands on the candidate's handoff stamp (the owner's
    # "2026-10-22 evening: CI off the landing path"; T3.2 fix 4).
    common = Path(
        w.git(
            'rev-parse',
            '--path-format=absolute',
            '--git-common-dir',
            cwd=w.worktree,
        ).strip()
    )
    stamp = common / 'sdm-handoff' / f'{sha}.json'
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({'sha': sha, 'package': 'T1.11e', 'ok': True}))
    reviews = []
    for name in ('a', 'b'):
        path = w.tmp / f'review-{name}.txt'
        path.write_text(f'APPROVE\nReviewer {name} approves {sha}.\n')
        reviews += ['--review', str(path)]
    return [
        '--candidate', sha, '--topic-branch', BRANCH, '--title', title,
        '--protected-paths', 'none', '--checkpoint', w.phase, *reviews,
    ]  # fmt: skip


@case('land.py: refuses a lane whose recorded package is not the title')
def _(w: World) -> tuple[bool, str]:
    sha = candidate(w)
    before = w.refs()
    ran = w.land(*land_args(w, sha, 'Land T1.11f: the wrong package'))
    ok = (
        ran.code == 2
        and 'records T1.11e' in ran.text
        and w.refs() == before
        and w.git('rev-parse', 'HEAD', cwd=w.worktree).strip() == sha
    )
    return verdict(ok, 'refused, nothing committed or pushed', ran)


@case('land.py: refuses a lane commit that does not name the package')
def _(w: World) -> tuple[bool, str]:
    sha = candidate(w, 'An unnamed change')
    before = w.refs()
    ran = w.land(*land_args(w, sha, 'Land T1.11e: the work'))
    ok = (
        ran.code == 2 and 'An unnamed change' in ran.text and w.refs() == before
    )
    return verdict(ok, 'refused', ran)


@case('land.py: lands a matching lane, leaving it equal to the phase head')
def _(w: World) -> tuple[bool, str]:
    sha = candidate(w)
    ran = w.land(*land_args(w, sha, 'Land T1.11e: the work'))
    landing = w.git('rev-parse', 'HEAD', cwd=w.worktree).strip()
    after = w.refs()
    nxt = w.start('T1.11f')
    ok = (
        ran.code == 0
        and landing != sha
        and w.tip(BRANCH) == w.tip(PHASE) == landing
        and f'ok   {BRANCH} equals {PHASE} at {landing}' in ran.text
        and nxt.code == 0
        and w.refs() == after
        and w.package(w.worktree) == 'T1.11f'
    )
    return verdict(
        ok,
        'landed; the next package starts with no push',
        f'{ran.text}\n{nxt.text}',
    )


for message, why in REPROS:

    @case(f'land.py: refuses {message.splitlines()[0]!r} under T1.11e')
    def _(w: World, message: str = message, why: str = why) -> tuple[bool, str]:
        sha = candidate(w, message)
        before = w.refs()
        ran = w.land(*land_args(w, sha, 'Land T1.11e: the work'))
        ok = ran.code == 2 and why in ran.text and w.refs() == before
        return verdict(ok, 'refused, nothing landed', ran)


@case('land.py: lands past a pre-push hook that would refuse it')
def _(w: World) -> tuple[bool, str]:
    sha = candidate(w)
    hook = w.clone / '.git' / 'hooks' / 'pre-push'
    hook.write_text('#!/bin/sh\necho hook refused >&2\nexit 1\n')
    hook.chmod(0o755)
    ran = w.land(*land_args(w, sha, 'Land T1.11e: the work'))
    landing = w.git('rev-parse', 'HEAD', cwd=w.worktree).strip()
    ok = ran.code == 0 and w.tip(PHASE) == landing
    return verdict(ok, "landed; land.py's checks stand in", ran)


def snapshot(w: World, *trees: Path) -> str:
    """The local and remote state a refused landing must leave as it was."""
    heads = [
        w.git_ok('rev-parse', 'HEAD', cwd=tree).text
        + w.git_ok('symbolic-ref', '--quiet', 'HEAD', cwd=tree).text
        for tree in trees
    ]
    return json.dumps(
        [w.git('for-each-ref', '--format=%(refname) %(objectname)'),
         w.refs(), heads]
    )  # fmt: skip


for how in ('another branch', 'a detached HEAD'):

    @case(f'land.py: refuses a lane landed from {how}')
    def _(w: World, how: str = how) -> tuple[bool, str]:
        sha = candidate(w)
        if how == 'another branch':
            w.git('checkout', '--quiet', '-b', 'scratch', cwd=w.worktree)
        else:
            w.git('checkout', '--quiet', '--detach', cwd=w.worktree)
        before = snapshot(w, w.worktree)
        ran = w.land(*land_args(w, sha, 'Land T1.11e: the work'))
        expect = 'scratch checked out' if how == 'another branch' else how
        ok = (
            ran.code == 2
            and expect in ran.text
            and snapshot(w, w.worktree) == before
        )
        return verdict(ok, 'refused, nothing changed', ran)


@case("land.py: refuses a lane landed from another lane's worktree")
def _(w: World) -> tuple[bool, str]:
    w.start('T1.11g')
    other = w.tmp / 'lane-1c'
    started = w.lane(
        'start', '--lane', '1c', '--package', 'T1.11e',
        '--worktree', str(other),
    )  # fmt: skip
    sha = w.commit(other, 'T1.11e: the work')
    w.git('push', '--quiet', 'origin', f'HEAD:{PHASE}-lane-1c', cwd=other)
    w.set_runs(green_on=[BRANCH, f'{PHASE}-lane-1c', PHASE])
    before = snapshot(w, w.worktree, other)
    # The title's package is 1c's, so only the branch check stands
    # between this and lane 1b, whose worktree records T1.11g.
    ran = w.land(*land_args(w, sha, 'Land T1.11e: the work'), cwd=other)
    ok = (
        started.code == 0
        and ran.code == 2
        and f'{PHASE}-lane-1c checked out' in ran.text
        and snapshot(w, w.worktree, other) == before
    )
    return verdict(ok, 'refused, nothing changed', ran)


@case('land.py: a red landing run, a fix, a second landing, the next start')
def _(w: World) -> tuple[bool, str]:
    # The earlier landing commit a red --wait-for-landing-run left on the
    # lane, made here by hand: land.py refuses that flag as obsolete since
    # E131, but a lane may still hold such a commit, which the package owns.
    sha = candidate(w)
    earlier = landing_commit(w, 'Land T1.11e: the work', w.phase)
    w.git('push', '--quiet', 'origin', f'HEAD:{BRANCH}', cwd=w.worktree)
    stuck = w.tip(BRANCH) == earlier != sha and w.tip(PHASE) == w.phase
    fix = w.commit(w.worktree, 'T1.11e fix 1: what the red run found')
    checked = w.lane('check', cwd=w.worktree)
    w.git('push', '--quiet', 'origin', f'HEAD:{BRANCH}', cwd=w.worktree)
    state = w.gh_state()
    state['conclusion'] = 'success'
    w.gh_state(state)
    again = w.land(*land_args(w, fix, 'Land T1.11e: the work'))
    landing = w.git('rev-parse', 'HEAD', cwd=w.worktree).strip()
    parent = w.git('rev-parse', 'HEAD^', cwd=w.worktree).strip()
    base = w.git(
        'log', '-1', '--format=%(trailers:key=Base-SHA,valueonly)',
        cwd=w.worktree,
    ).strip()  # fmt: skip
    nxt = w.start('T1.11f')
    ok = (
        stuck
        and checked.code == 0
        and 'carries T1.11e: 3 commit(s)' in checked.text
        and again.code == 0
        and parent == fix
        and base == w.phase
        and w.tip(BRANCH) == w.tip(PHASE) == landing
        and nxt.code == 0
        and w.package(w.worktree) == 'T1.11f'
    )
    return verdict(
        ok,
        'stuck, fixed, landed again on the fix, the next package started',
        f'{checked.text}\n{again.text}\n{nxt.text}',
    )


def landing_commit(
    w: World, subject: str, base: str, change: bool = False
) -> str:
    if change:
        (w.worktree / 'changed.txt').write_text('a change\n')
        w.git('add', 'changed.txt', cwd=w.worktree)
    w.git(
        'commit', '--quiet', '--allow-empty', '-m',
        f'{subject}\n\nReviewed-by: r\nBase-SHA: {base}', cwd=w.worktree,
    )  # fmt: skip
    return w.git('rev-parse', 'HEAD', cwd=w.worktree).strip()


for label, why in (
    ('changes a file', 'changes no file'),
    ('has a Base-SHA outside the phase', 'not in the phase branch'),
    # Since T3.0 another package's landing commit below is a stack: both
    # refuse it, the check as a stack, land.py naming stack-land.
    ('lands another package', 'T1.11f'),
):

    @case(f'check and land.py: refuse a landing commit that {label}')
    def _(w: World, label: str = label, why: str = why) -> tuple[bool, str]:
        work = candidate(w)
        sha = landing_commit(
            w,
            'Land T1.11f: other' if 'another' in label else 'Land T1.11e: x',
            work if 'outside' in label else w.phase,
            change='changes' in label,
        )
        checked = w.lane('check', cwd=w.worktree)
        w.git('push', '--quiet', 'origin', f'HEAD:{BRANCH}', cwd=w.worktree)
        before = w.refs()
        ran = w.land(*land_args(w, sha, 'Land T1.11e: the work'))
        ok = (
            checked.code == 2
            and why in checked.text
            and ran.code == 2
            and why in ran.text
            and w.refs() == before
        )
        return verdict(ok, 'refused by both', f'{checked.text}\n{ran.text}')


@case('start: with core.hooksPath set, says the hook would not run')
def _(w: World) -> tuple[bool, str]:
    w.git('config', 'core.hooksPath', str(w.tmp / 'elsewhere'))
    ran = w.start()
    hooks = w.clone / '.git' / 'hooks'
    ok = (
        ran.code == 0
        and 'core.hooksPath is' in ran.text
        and 'pre-push hook was not installed' in ran.text
        and not (hooks / 'pre-push').exists()
        and not (hooks / 'prepush-worktrees').exists()
    )
    return verdict(ok, 'reported, nothing installed', ran)


# --- the phase is one setting (T2.0) ------------------------------------------

PHASE_2 = 'rust-core-phase-2'


def phase_2(w: World) -> str:
    """A Phase 2 branch on the remote, cut from the Phase 1 head."""
    w.git('push', '--quiet', 'origin', f'{w.phase}:refs/heads/{PHASE_2}', cwd=w.seed)  # fmt: skip
    return w.phase


@case('phase: --phase-branch takes effect over SDM_PHASE_BRANCH')
def _(w: World) -> tuple[bool, str]:
    head = phase_2(w)
    ran = w.lane('--phase-branch', PHASE_2, 'start', '--lane', '1b',
                 '--package', 'T2.0', '--worktree', str(w.worktree))  # fmt: skip
    lane = f'{PHASE_2}-lane-1b'
    ok = (
        ran.code == 0
        and w.tip(lane) == head
        and w.tip(BRANCH) is None
        and w.package(w.worktree) == 'T2.0'
        and f'STARTED T2.0 on {lane} at {head}' in ran.text
    )
    return verdict(ok, 'a Phase 2 lane started', ran)


@case('phase: SDM_PHASE_BRANCH takes effect, for start and status')
def _(w: World) -> tuple[bool, str]:
    phase_2(w)
    w.env['SDM_PHASE_BRANCH'] = PHASE_2
    ran = w.start('T2.0a')
    shown = w.lane('status')
    lane = f'{PHASE_2}-lane-1b'
    ok = (
        ran.code == 0
        and w.tip(lane) is not None
        and shown.code == 0
        and f'{PHASE_2} at ' in shown.text
        and lane in shown.text
    )
    return verdict(
        ok, 'start and status on Phase 2', f'{ran.text} | {shown.text}'
    )


@case('phase: with no flag and no SDM_PHASE_BRANCH, the phase in progress')
def _(w: World) -> tuple[bool, str]:
    del w.env['SDM_PHASE_BRANCH']
    before = w.refs()
    ran = w.start()
    # The clone has main checked out, so phases.CURRENT, which this
    # remote lacks.
    return verdict(
        refused_untouched(w, ran, f'origin has no {CURRENT}', before),
        'refused, naming the phase in progress',
        ran,
    )


PHASE_3 = 'rust-core-phase-3'


def phase_3(w: World, headings: tuple[str, ...] | None = None) -> Path:
    """A Phase 3 branch on the remote, cut from the Phase 1 head, the
    tools pointed at it, and a brief holding the template's headings (all
    of them, or ``headings``)."""
    w.git('push', '--quiet', 'origin', f'{w.phase}:refs/heads/{PHASE_3}', cwd=w.seed)  # fmt: skip
    w.env['SDM_PHASE_BRANCH'] = PHASE_3
    import brief_check

    wanted = brief_check.required(brief_check.TEMPLATE)
    brief = w.tmp / 'brief.md'
    brief.write_text(
        '# T3.4: a brief\n\n'
        + ''.join(
            f'{heading}\n\nnone: a seeded brief.\n\n'
            for heading in (wanted if headings is None else headings)
        )
    )
    return brief


@case('phase 3: a whole package ID and a full brief start the lane')
def _(w: World) -> tuple[bool, str]:
    brief = phase_3(w)
    ran = w.start('T3.4', '--brief', str(brief))
    ok = ran.code == 0 and w.package(w.worktree) == 'T3.4'
    return verdict(ok, 'started', ran)


@case('phase 3: a package ID with a letter is refused (T3.2)')
def _(w: World) -> tuple[bool, str]:
    brief = phase_3(w)
    before = w.refs()
    ran = w.start('T3.4a', '--brief', str(brief))
    return verdict(
        refused_untouched(w, ran, "the package 'T3.4a' has a letter", before),
        'refused, nothing changed',
        ran,
    )


@case('phase 3: a lane started with no brief still starts')
def _(w: World) -> tuple[bool, str]:
    phase_3(w)
    ran = w.start('T3.4')
    return verdict(ran.code == 0, 'started', ran)


@case('phase: a lane is never started on a finished phase (T3.2)')
def _(w: World) -> tuple[bool, str]:
    w.git('push', '--quiet', 'origin', f'{w.phase}:refs/heads/rust-core', cwd=w.seed)  # fmt: skip
    w.commit(w.seed, 'Merge the phase')
    w.git('push', '--quiet', 'origin', 'HEAD:refs/heads/rust-core', cwd=w.seed)  # fmt: skip
    before = w.refs()
    ran = w.start()
    return verdict(
        refused_untouched(w, ran, f'{PHASE} is finished', before),
        'refused, nothing changed',
        ran,
    )


@case('phase 3: a brief missing a heading is refused, naming it (T3.2)')
def _(w: World) -> tuple[bool, str]:
    import brief_check

    wanted = brief_check.required(brief_check.TEMPLATE)
    brief = phase_3(w, tuple(wanted[1:]))
    before = w.refs()
    ran = w.start('T3.4', '--brief', str(brief))
    return verdict(
        refused_untouched(w, ran, f'lacks the heading {wanted[0][3:]}', before),
        'refused, naming the heading',
        ran,
    )


@case('phase 1: a lettered ID and no brief still start, as before T3.2')
def _(w: World) -> tuple[bool, str]:
    ran = w.start('T1.11e')
    return verdict(ran.code == 0, 'started', ran)


@case('phase: a phase name that is no phase branch is refused, by name')
def _(w: World) -> tuple[bool, str]:
    before = w.refs()
    ran = w.lane('--phase-branch', 'rust-core-phase-02', 'status')
    ok = ran.code == 2 and "'rust-core-phase-02' is not a phase branch" in (
        ran.text
    )
    return verdict(ok and w.refs() == before, 'refused', ran)


@case("phase: check measures a lane against its own phase's head")
def _(w: World) -> tuple[bool, str]:
    head = phase_2(w)
    w.env['SDM_PHASE_BRANCH'] = PHASE_2
    started = w.start('T2.0')
    w.commit(w.worktree, 'T2.0: the package')
    # Under another phase's SDM_PHASE_BRANCH, the lane is still measured
    # from its own phase's head.
    w.env['SDM_PHASE_BRANCH'] = PHASE
    ran = w.lane('check', cwd=w.worktree)
    ok = (
        started.code == 0
        and ran.code == 0
        and f'{PHASE_2}-lane-1b carries T2.0: 1 commit(s)' in ran.text
        and head
    )
    return verdict(bool(ok), 'measured from the Phase 2 head', ran)


# --- status -------------------------------------------------------------------


@case(
    'start: past MAX_LANES a lane is refused, naming the active one, until it pauses'
)
def _(w: World) -> tuple[bool, str]:
    # The owner's rule (2026-10-22), T3.2 fix 4: at most MAX_LANES packages.
    conf = w.tmp / 'heavy.conf'
    conf.write_text('MAX_LANES=1\n')
    w.env['SDM_HEAVY_CONFIG'] = str(conf)
    second = w.tmp / 'lane-2c'

    def start_second() -> Ran:
        return w.lane(
            'start', '--lane', '2c', '--package', 'T1.11g',
            '--worktree', str(second),
        )  # fmt: skip

    first = w.start()
    refused = start_second()
    held = w.lane('pause', '--lane', '1b')
    again = start_second()
    ok = (
        first.code == 0
        and refused.code == 2
        and 'lane 1b (T1.11e)' in refused.text
        and 'lane.py pause --lane' in refused.text
        and held.code == 0
        and again.code == 0
        and w.package(second) == 'T1.11g'
    )
    return verdict(
        ok,
        'refused past the limit, started once the first paused',
        f'{refused.text} || {held.text} || {again.text}',
    )


@case('status: equal, behind, ahead with its package, and the latest run')
def _(w: World) -> tuple[bool, str]:
    w.start()
    equal = w.lane('status')
    w.advance_phase()
    behind = w.lane('status')
    work = w.commit(w.worktree, 'T1.11e: the work')
    w.git('push', '--quiet', 'origin', f'HEAD:{BRANCH}', cwd=w.worktree)
    w.set_runs(ci_run(5, work, 'in_progress'))
    ahead = w.lane('status', '--lane', '1b')
    ok = (
        'equals the phase head' in equal.text
        and 'package  T1.11e (worktree' in equal.text
        and 'behind the phase head by 1, nothing unlanded' in behind.text
        and 'AHEAD by 1 unlanded commit(s), naming T1.11e (1 of 1); behind by 1'
        in ahead.text
        and 'run 5: in_progress' in ahead.text
        and all(r.code == 0 for r in (equal, behind, ahead))
    )
    return verdict(
        ok, 'all three shown', f'{equal.text}\n{behind.text}\n{ahead.text}'
    )


@case('status: no lanes on the remote')
def _(w: World) -> tuple[bool, str]:
    ran = w.lane('status')
    return verdict(
        ran.code == 0 and 'no lane branches' in ran.text, 'reported', ran
    )


def main() -> int:
    results: list[Result] = []
    for name, check in CASES:
        with tempfile.TemporaryDirectory(prefix='sdm-lane-') as tmp:
            try:
                ok, detail = check(World(Path(tmp)))
            except Exception as error:  # a crash is a failed self-test
                ok, detail = False, f'raised {type(error).__name__}: {error}'
        results.append(Result(name, ok, detail.replace('\n', ' | ')[:600]))
    for result in results:
        print(
            f'{"ok  " if result.ok else "FAIL"} {result.name}: {result.detail}'
        )
    failed = [r for r in results if not r.ok]
    print(
        f'{len(results) - len(failed)} of {len(results)} self-tests as expected'
    )
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
