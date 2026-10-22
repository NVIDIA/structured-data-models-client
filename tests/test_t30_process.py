# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T3.0's process changes, each shown to block or to work: the handoff
stamp and what withholds it, the commit-msg ``Rule:`` check, the review
launches that need a stamp, the banned-command hook, the per-lane
environments, ``heavy``'s admission, the runners' refusal of the shared
``.venv``, the dry run's judging gaps and the slip report.

The stack, the train and the checkpoint per landing are land.py's
self-tests (``scripts/rust_core/land_selftest.py``, the ``T3.0`` cases);
the CI split, the timing guard and the comparison base are in
``tests/test_ci_workflow.py`` (the ``test_t30_*`` tests).

The interfaces tested, as T3.0's first implementer gives them:

- ``scripts/rust_core/handoff.py run --package ID [--previous SHA]
  [--case ID ...] [--python PY]`` writes
  ``<git-common-dir>/sdm-handoff/<sha>.json`` only when every step passes
  (reference, fails-before, affected, lint), else exits 1 with no stamp;
  ``handoff.py check SHA`` exits 0 or 1;
- ``scripts/rust_core/affected.py [--base REV] [FILES...]`` prints suite
  commands, and ``affected.py --unmapped`` exits 1 listing unmapped paths;
- ``scripts/rust_core/commit_msg.py MSGFILE``: a ``<ID> fix <n>:`` subject
  whose commit touches ``rust/crates/**`` or ``packages/**`` needs a
  ``Rule: ...`` line;
- ``codex_pane.py review`` exits 2 on a SHA with no stamp;
- ``scripts/rust_core/orchestration/guard_cmd.py``, the PreToolUse hook for
  Bash and Agent, reads the hook JSON on stdin; a deny prints the
  ``permissionDecision: deny`` JSON with ``<rule>; instead: <alternative>``
  and exits 0, an allow prints nothing; ``SDM_APPROVED=<slug> <cmd>`` passes
  and is logged to ``$SDM_ORCH_STATE/logs/approved-commands.log``;
  ``guard_cmd.py --tokens`` lists the slugs; an Agent prompt naming
  ``briefs/reviewer.md`` with an unstamped SHA is denied.

And, as named in :data:`ASSUMED` until the second implementer's names are
relayed: ``cells.sh --print-key [--rev REV]`` prints the lane environment's
key; ``scripts/rust_core/orchestration/heavy`` reads its thresholds from
the file ``SDM_HEAVY_CONFIG`` names (``SLOTS``, ``MIN_FREE_MEM_GB``,
``MIN_FREE_DISK_GB``, ``DISK``, one ``KEY=VALUE`` a line) and runs its
sibling ``clean_tmp.sh --apply`` first when the disk is short;
``scripts/rust_core/gate_dryrun.sh PLAN`` reads a YAML plan
(``kinds: [{name, case}]``, each case a stub case object) and lists every
kind the tools cannot judge; ``phase_stats.py --forecast FILE --now TIME``
prints the slip against a forecast (``start`` and ``packages: [{id,
hours}]``) and its main cause, and never fails on it.

Each test listed in ``tests/t30-pending.json`` failed when written, before
the implementation, and is held to failing (``xfail``, strict) until the
implementer takes it out of ``pending``; ``pending`` stays within
``authored``, the test author's list.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUST_CORE = ROOT / 'scripts/rust_core'
ORCH = RUST_CORE / 'orchestration'
HANDOFF = 'scripts/rust_core/handoff.py'
AFFECTED = RUST_CORE / 'affected.py'
COMMIT_MSG = RUST_CORE / 'commit_msg.py'
GUARD = ORCH / 'guard_cmd.py'
CODEX_PANE = ORCH / 'codex_pane.py'
SETTINGS = ROOT / '.claude/settings.json'
PHASE = 'rust-core-phase-3'
PACKAGE = 'T3.99'
TOPIC = f'{PHASE}-t3.99'
#: The second implementer's interfaces, assumed until relayed.
ASSUMED = {
    'cells_key': ('scripts/rust_core/cells.sh', '--print-key'),
    'heavy': 'scripts/rust_core/orchestration/heavy',
    'heavy_config': 'SDM_HEAVY_CONFIG',
    'dry_run': 'scripts/rust_core/gate_dryrun.sh',
    'slip': ('scripts/rust_core/orchestration/phase_stats.py', '--forecast'),
}
GIT_ENV = {
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_AUTHOR_NAME': 'Test Person',
    'GIT_AUTHOR_EMAIL': 'test@example.com',
    'GIT_COMMITTER_NAME': 'Test Person',
    'GIT_COMMITTER_EMAIL': 'test@example.com',
    'GIT_TERMINAL_PROMPT': '0',
}
#: An acceptance case on DuckDB, answered alike by the old client and the
#: core since Phase 1.
SEEDED_CASE = 's0.accept.binary-classification'
PENDING_FILE = ROOT / 'tests/t30-pending.json'
#: These tests copy the checkout into temporary repositories and run the
#: checking tools there: the judging-tool job's (T3.0, item 3).
pytestmark = pytest.mark.tooling


def _pending_names(key: str) -> set[str]:
    return set(json.loads(PENDING_FILE.read_text(encoding='utf-8'))[key])


def pending(test: Any) -> Any:
    """Holds ``test`` to failing while ``tests/t30-pending.json`` lists it."""
    if test.__name__ in _pending_names('pending'):
        return pytest.mark.xfail(
            strict=True,
            reason='T3.0: failing until implemented; once it passes, take '
            'it out of tests/t30-pending.json pending',
        )(test)
    return test


def test_t30_the_pending_list_only_shrinks() -> None:
    assert _pending_names('pending') <= _pending_names('authored')


# --- helpers ---------------------------------------------------------------


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """The environment without the caller's SDM_ or GIT_ settings, a state
    directory of the test's own, and git's identity."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('SDM_', 'GIT_'))
    }
    return {
        **env,
        **GIT_ENV,
        'SDM_ORCH_STATE': str(tmp_path / 'state'),
        **extra,
    }


def _git(where: Path, *args: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        ['git', '-C', str(where), *args],
        env=env or {**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )
    if done.returncode != 0:
        raise AssertionError(f'git {args}: {done.stderr.strip()}')
    return done.stdout.strip()


def _commit(
    work: Path, subject: str, files: dict[str, str | None] | None = None
) -> str:
    for name, text in (files or {}).items():
        path = work / name
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        _git(work, 'add', '-A', name)
    _git(work, 'commit', '-q', '--allow-empty', '-s', '-m', subject)
    return _git(work, 'rev-parse', 'HEAD')


def _overlay(work: Path) -> None:
    """The caller's uncommitted and untracked files under scripts/, tests/,
    contract/ and .claude/, copied into ``work``, so a clone runs the tools
    as they stand in this worktree."""
    changed = _git(ROOT, 'diff', '--name-only', 'HEAD').splitlines()
    added = _git(
        ROOT, 'ls-files', '--others', '--exclude-standard'
    ).splitlines()
    for name in {*changed, *added}:
        if not name.startswith(('scripts/', 'tests/', 'contract/', '.claude/')):
            continue
        source, target = ROOT / name, work / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif target.exists():
            target.unlink()


def _clone(tmp_path: Path) -> tuple[Path, Path, str]:
    """A bare origin and a clone of this repository, its HEAD and this
    worktree's changes as the phase head (and the epic branch, so it is
    the phase's fork point), and a topic branch checked out there."""
    origin, work = tmp_path / 'origin.git', tmp_path / 'work'
    _git(tmp_path, 'clone', '-q', '--bare', '--shared', str(ROOT), str(origin))
    _git(tmp_path, 'clone', '-q', '--shared', str(origin), str(work))
    _git(work, 'checkout', '-q', '--detach', _git(ROOT, 'rev-parse', 'HEAD'))
    _overlay(work)
    if _git(work, 'status', '--porcelain'):
        _git(work, 'add', '-A')
        _git(work, 'commit', '-q', '-s', '-m', 'T3.0: the tools under test')
    head = _git(work, 'rev-parse', 'HEAD')
    _git(
        work, 'push', '-q', 'origin',
        f'{head}:refs/heads/{PHASE}', f'{head}:refs/heads/rust-core',
    )  # fmt: skip
    _git(work, 'fetch', '-q', 'origin')
    _git(work, 'checkout', '-q', '-b', TOPIC, head)
    return origin, work, head


def _stamp(work: Path, sha: str) -> Path:
    common = Path(_git(work, 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = work / common
    return common / 'sdm-handoff' / f'{sha}.json'


def _handoff(
    work: Path, env: dict[str, str], *args: str, timeout: float = 3600
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(work / HANDOFF), *args],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )


# --- 3. the handoff stamp ----------------------------------------------------


def _core_python() -> str:
    """An interpreter holding the candidate's client (the native build), not
    the shared ``.venv``: ``SDM_TEST_CORE_PYTHON``. The reference step runs
    the package's cases on it; without one the test cannot run."""
    found = os.environ.get('SDM_TEST_CORE_PYTHON', '')
    if not found:
        pytest.skip(
            'the reference step needs SDM_TEST_CORE_PYTHON: an interpreter '
            "holding the candidate's client"
        )
    return found


#: ``--python`` for the handoff: the core interpreter, which, when told to,
#: makes the core's answer to the seeded case differ from the recorded
#: old-client answer (a float64 column read as float32) in every
#: python-core observation file it writes.
SEEDING_PYTHON = r"""#!{python}
import json, os, subprocess, sys

args = sys.argv[1:]
done = subprocess.run([os.environ['T30_CORE_PYTHON'], *args])
seed = os.environ.get('T30_SEED_CASE')
if (
    seed
    and any(a.endswith('run_python.py') for a in args)
    and 'python-core' in args
    and '--out' in args
):
    path = args[args.index('--out') + 1]

    def differ(value):
        if isinstance(value, dict):
            return {{k: differ(v) for k, v in value.items()}}
        if isinstance(value, list):
            return [differ(v) for v in value]
        return 'float32' if value == 'float64' else value

    with open(path) as handle:
        lines = handle.read().splitlines()
    out = []
    for line in lines:
        seen = json.loads(line)
        if seen.get('case') == seed:
            seen['public'] = differ(seen.get('public'))
            line = json.dumps(seen, sort_keys=True)
        out.append(line)
    with open(path, 'w') as handle:
        handle.write(''.join(line + '\n' for line in out))
sys.exit(done.returncode)
"""


def _seeding_python(tmp_path: Path) -> Path:
    path = tmp_path / 'seeding-python'
    path.write_text(SEEDING_PYTHON.format(python=sys.executable))
    path.chmod(0o755)
    return path


@pending
@pytest.mark.parametrize('seeded', [True, False])
def test_t30_no_stamp_when_a_seeded_case_differs_from_the_old_client(
    tmp_path: Path, seeded: bool
) -> None:
    core = _core_python()
    _, work, _ = _clone(tmp_path)
    sha = _commit(
        work,
        f'{PACKAGE}: a package that touches the binary-classification case',
        {'docs/t399-probe.md': 'A probe for the handoff tests.\n'},
    )
    env = _env(
        tmp_path,
        SDM_PHASE_BRANCH=PHASE,
        T30_CORE_PYTHON=core,
        **({'T30_SEED_CASE': SEEDED_CASE} if seeded else {}),
    )
    done = _handoff(
        work, env, 'run', '--package', PACKAGE, '--case', SEEDED_CASE,
        '--python', str(_seeding_python(tmp_path)),
    )  # fmt: skip
    said = done.stdout + done.stderr
    check = _handoff(work, env, 'check', sha, timeout=120)
    if seeded:
        assert done.returncode == 1, said[-3000:]
        assert SEEDED_CASE in said, said[-3000:]
        assert not _stamp(work, sha).exists()
        assert check.returncode == 1, check.stdout + check.stderr
    else:
        assert done.returncode == 0, said[-3000:]
        stamp = json.loads(_stamp(work, sha).read_text(encoding='utf-8'))
        assert stamp['sha'] == sha and stamp['ok'] is True, stamp
        assert check.returncode == 0, check.stdout + check.stderr


#: A fix round's new test: on the previous round's SHA it passes (the fix
#: changed nothing it reads), or fails (it reads what the fix changed).
FIX_TEST = '''\
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fix round's new test, for the handoff's fails-before step."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_probe_reads_the_fixed_answer() -> None:
    {body}
'''


@pending
@pytest.mark.parametrize('fails_before', [False, True])
def test_t30_no_stamp_for_a_fix_whose_new_tests_all_pass_on_the_previous_sha(
    tmp_path: Path, fails_before: bool
) -> None:
    _, work, _ = _clone(tmp_path)
    previous = _commit(
        work,
        f'{PACKAGE}: the first round',
        {'docs/t399-probe.md': 'The answer is 1.\n'},
    )
    body = (
        "text = (ROOT / 'docs/t399-probe.md').read_text()\n"
        "    assert text == 'The answer is 2.\\n'"
        if fails_before
        else 'assert ROOT.is_dir()'
    )
    sha = _commit(
        work,
        f'{PACKAGE} fix 1: the answer is 2',
        {
            'docs/t399-probe.md': 'The answer is 2.\n',
            'tests/test_t399_probe.py': FIX_TEST.format(body=body),
        },
    )
    env = _env(tmp_path, SDM_PHASE_BRANCH=PHASE)
    done = _handoff(
        work, env, 'run', '--package', PACKAGE, '--previous', previous
    )
    said = done.stdout + done.stderr
    if fails_before:
        assert done.returncode == 0, said[-3000:]
        assert _stamp(work, sha).is_file()
    else:
        assert done.returncode == 1, said[-3000:]
        assert 'test_t399_probe' in said, said[-3000:]
        assert not _stamp(work, sha).exists()
        check = _handoff(work, env, 'check', sha, timeout=120)
        assert check.returncode == 1


@pending
def test_t30_affected_maps_every_path_under_contract_scripts_and_crates() -> (
    None
):
    done = subprocess.run(
        [sys.executable, str(AFFECTED), '--unmapped'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for root in ('contract/', 'scripts/rust_core/', 'rust/crates/'):
        tracked = _git(ROOT, 'ls-files', root).splitlines()
        assert tracked, root
        # One path of each root, through the map, names a suite to run.
        sample = tracked[len(tracked) // 2]
        named = subprocess.run(
            [sys.executable, str(AFFECTED), sample],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert named.returncode == 0 and named.stdout.strip(), (
            f'{sample}: {named.stdout}{named.stderr}'
        )


def _msg_repo(tmp_path: Path, staged: str) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', TOPIC)
    judging = ROOT / 'scripts/rust_core/judging-paths.txt'
    _commit(
        repo,
        'Land T3.98: the phase so far',
        {
            'README.md': 'x\n',
            'scripts/rust_core/judging-paths.txt': (
                judging.read_text(encoding='utf-8')
                if judging.is_file()
                else 'scripts/rust_core/**\n'
            ),
        },
    )
    path = repo / staged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('changed\n', encoding='utf-8')
    _git(repo, 'add', staged)
    return repo


@pending
@pytest.mark.parametrize(
    ('staged', 'message', 'refused'),
    [
        (
            'rust/crates/sdm-core/src/lib.rs',
            'T3.99 fix 1: keep the order',
            True,
        ),
        (
            'packages/structured-data-models-client/src/x.py',
            'T3.99 fix 2: keep the order',
            True,
        ),
        (
            'rust/crates/sdm-core/src/lib.rs',
            'T3.99 fix 1: keep the order\n\nRule: the old client keeps the '
            "caller's column order (client.py, _columns).",
            False,
        ),
        ('rust/crates/sdm-core/src/lib.rs', 'T3.99: the first round', False),
        ('docs/notes.md', 'T3.99 fix 1: a wording fix', False),
        ('tests/test_order.py', 'T3.99 fix 1: the test reads it', False),
        (
            'rust/crates/sdm-core/src/lib.rs',
            'T3.99 fix 1: keep the order\n\nRule: keep order',
            True,
        ),
    ],
    ids=[
        'fix-crates',
        'fix-packages',
        'with-rule',
        'not-a-fix',
        'docs-fix',
        'tests-fix',
        'rule-too-short',
    ],
)
def test_t30_the_commit_msg_check_wants_a_rule_on_a_product_fix(
    tmp_path: Path, staged: str, message: str, refused: bool
) -> None:
    repo = _msg_repo(tmp_path, staged)
    msg = tmp_path / 'COMMIT_EDITMSG'
    msg.write_text(message + '\n\nSigned-off-by: T <t@example.com>\n')
    done = subprocess.run(
        [sys.executable, str(COMMIT_MSG), str(msg)],
        cwd=repo,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    said = done.stdout + done.stderr
    if refused:
        assert done.returncode != 0, said
        assert 'Rule:' in said, said
    else:
        assert done.returncode == 0, said


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pending
def test_t30_codex_pane_refuses_a_round_on_an_unstamped_sha(
    tmp_path: Path,
) -> None:
    fake = _module('codex_pane_selftest', ORCH / 'codex_pane_selftest.py').FAKE
    origin, repo = tmp_path / 'origin.git', tmp_path / 'repo'
    _git(tmp_path, 'init', '-q', '--bare', str(origin))
    _git(tmp_path, 'clone', '-q', str(origin), str(repo))
    _commit(repo, 'one', {'a.txt': 'one\n'})
    _git(repo, 'push', '-q', 'origin', 'HEAD:rust-core')
    sha = _commit(repo, 'T3.99: two', {'a.txt': 'two\n'})
    _git(repo, 'push', '-q', 'origin', f'HEAD:{TOPIC}')
    _git(repo, 'fetch', '-q', 'origin')
    herdr = tmp_path / 'herdr'
    herdr.write_text(fake)
    herdr.chmod(0o755)
    brief = tmp_path / 'brief.md'
    brief.write_text('Review it.\n')
    env = _env(
        tmp_path,
        SDM_REPO=str(repo),
        SDM_HERDR=str(herdr),
        FAKE_STATE=str(tmp_path / 'fake.json'),
        SDM_CODEX_POLL='0.05',
        SDM_CODEX_GRACE='0',
    )

    def tool(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CODEX_PANE), *args],
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )

    opened = tool('open', '--package', PACKAGE)
    assert opened.returncode == 0, opened.stdout + opened.stderr
    done = tool(
        'review', '--package', PACKAGE, '--sha', sha, '--brief', str(brief)
    )
    said = done.stdout + done.stderr
    assert done.returncode == 2, said
    assert 'SENT' not in said
    assert 'stamp' in said or 'handoff' in said, said
    world = json.loads((tmp_path / 'fake.json').read_text())
    assert not [c for c in world['log'] if c[:2] == ['agent', 'prompt']]
    assert not _stamp(repo, sha).exists()


# --- 6. the banned-command hook ----------------------------------------------

#: A session with no directory of its own, and where a Bash call runs:
#: neither under any scratch, wherever the tests run.
SESSION = '0f3c2a1e-7d6b-4c5a-9e8f-1a2b3c4d5e6f'
CWD = '/home/user/code/sdmc-rust-core-lane-a'


@pytest.fixture
def sessions() -> Any:
    """Two Claude sessions' directories, as Claude Code makes them
    (``/tmp/claude-*/<project>/<session id>/scratchpad``): this agent's and
    another's, as (session id, its scratchpad, the other's scratchpad)."""
    top = Path(tempfile.mkdtemp(prefix='claude-t30-', dir='/tmp'))
    own, other = str(uuid.uuid4()), str(uuid.uuid4())
    pads = []
    for session in (own, other):
        pad = top / '-home-user-code-repo' / session / 'scratchpad'
        (pad / 'probe' / 'build').mkdir(parents=True)
        pads.append(str(pad))
    yield own, pads[0], pads[1]
    shutil.rmtree(top)


def _hook(
    tmp_path: Path,
    command: str,
    tool: str = 'Bash',
    cwd: Path | None = None,
    tool_input: dict[str, Any] | None = None,
    session: str = SESSION,
) -> tuple[str | None, str, subprocess.CompletedProcess[str]]:
    """The hook's decision on one tool call: 'deny' or None, and why."""
    event = {
        'session_id': session,
        'transcript_path': f'/tmp/claude-1000/transcripts/{session}.jsonl',
        'cwd': str(cwd or CWD),
        'hook_event_name': 'PreToolUse',
        'tool_name': tool,
        'tool_input': tool_input or {'command': command},
    }
    done = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(event),
        env=_env(
            tmp_path,
            SDM_SCRATCH=str(tmp_path),
            SDM_OWNER_ANSWERS=str(_answers_file(tmp_path)),
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    if not done.stdout.strip():
        return None, '', done
    out = json.loads(done.stdout)['hookSpecificOutput']
    assert out['hookEventName'] == 'PreToolUse', out
    return (
        out['permissionDecision'],
        out.get('permissionDecisionReason', ''),
        done,
    )


@pending
def test_t30_the_guard_is_the_repository_s_pretooluse_hook_for_bash_and_agent() -> (
    None
):
    settings = json.loads(SETTINGS.read_text(encoding='utf-8'))
    matched: dict[str, str] = {}
    for entry in settings['hooks']['PreToolUse']:
        for hook in entry['hooks']:
            if 'guard_cmd.py' in hook.get('command', ''):
                for tool in ('Bash', 'Agent'):
                    # A matcher is a regular expression on the tool's name.
                    if re.fullmatch(entry.get('matcher', ''), tool):
                        matched[tool] = hook['command']
    assert set(matched) == {'Bash', 'Agent'}, matched


BANNED = [
    'rm -rf /home/user/code/sdmc-rust-core/rust/target',
    'rm -r ../another-worktree',
    'rm -r {other}/probe',
    'git push --force origin rust-core-phase-3-t3.99',
    'git push -f origin rust-core-phase-3-t3.99',
    'git push --force-with-lease origin rust-core-phase-3-t3.99',
    'git worktree remove --force ../old',
    'git worktree remove -f ../old',
    'docker rm -f probe',
    'git commit --no-verify -s -m "T3.99: x"',
    'git push --no-verify origin HEAD:rust-core-phase-3-t3.99',
    'sudo apt-get install -y jq',
    'git config user.email someone@example.com',
    'git push origin +HEAD:refs/heads/rust-core-phase-3-t3.99',
    'git stash',
    'git stash push -m wip',
    'git branch -D rust-core-phase-3-t3.98',
    'git push origin --delete rust-core-phase-3-t3.98',
    'gh run watch 12345',
    'git fetch --prune origin',
    'cd /tmp && git stash',
    'echo start; sudo true',
    'rm -rf /mnt/localssd/tmp',
    'rm -rf /mnt/localssd/target',
    'rm -rf /mnt/localssd/tmp/another-agent/build',
    'rm -r /mnt/localssd/target/another-agent',
    'rm -r "$HOME/old-probe"',
    'rm -r $(git rev-parse --show-toplevel)/rust/target',
    'ls -d build-* | xargs rm -r',
    "find . -name '*.pyc' -delete",
    'find . -name build -exec rm -rf {{}} +',
]

ALLOWED = [
    'git push origin HEAD:refs/heads/rust-core-phase-3-t3.99',
    'git worktree remove ../old',
    'git branch -d rust-core-phase-3-t3.98',
    'git fetch origin',
    'rm notes.txt',
    'rm -r {scratch}/probe',
    'rm -rf {scratch}/probe/build',
    'find {scratch}/probe -name build -exec rm -rf {{}} +',
    # T3.0 fix 1 (Codex r1): a shared build or temp directory is the
    # session's once it made it; one it did not make is another agent's.
    'mkdir -p /mnt/localssd/tmp/t30-claim-probe-{session} && rm -rf /mnt/localssd/tmp/t30-claim-probe-{session}/build',
    "git commit -s -m 'T3.99: document `git config` reads'",
    "rg '$(sudo true)' docs",
    'gh run list --limit 5',
    'git commit -s -m "T3.99: never git stash or --no-verify"',
    "grep -rn 'git config' docs",
    'git log --oneline -5',
]


@pending
@pytest.mark.parametrize('command', BANNED)
def test_t30_the_guard_denies_each_banned_command_with_its_alternative(
    tmp_path: Path, sessions: Any, command: str
) -> None:
    session, scratch, other = sessions
    command = command.format(scratch=scratch, other=other, session=session)
    decision, why, done = _hook(tmp_path, command, session=session)
    assert decision == 'deny', done.stdout + done.stderr
    rule, _, instead = why.partition('; instead: ')
    assert rule.strip() and instead.strip(), why


@pending
@pytest.mark.parametrize('command', ALLOWED)
def test_t30_the_guard_lets_the_allowed_forms_through(
    tmp_path: Path, sessions: Any, command: str
) -> None:
    session, scratch, other = sessions
    command = command.format(scratch=scratch, other=other, session=session)
    decision, why, done = _hook(tmp_path, command, session=session)
    assert decision is None, f'{decision}: {why}'
    assert done.stdout.strip() == ''


def _answers_file(tmp_path: Path) -> Path:
    return tmp_path / 'state' / 'phase-3' / 'owner-answers-2026-10-22.md'


@pending
def test_t30_the_guard_lets_rm_r_through_in_the_listed_scratch(
    tmp_path: Path,
) -> None:
    (tmp_path / 'probe' / 'build').mkdir(parents=True)
    for command in (
        f'rm -r {tmp_path}/probe',
        f'find {tmp_path}/probe -name build -exec rm -rf {{}} +',
    ):
        decision, why, _ = _hook(tmp_path, command)
        assert decision is None, f'{command}: {why}'


def _owner_answers(tmp_path: Path) -> None:
    _answers_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _answers_file(tmp_path).write_text(
        '# Owner answers\n\n'
        '## 2026-10-22: branch cleanup approval\n\n'
        'Owner, verbatim: "Yes, delete the superseded rounds."\n',
        encoding='utf-8',
    )


@pending
@pytest.mark.parametrize(
    'command',
    [
        'git branch -D rust-core-phase-3-t3.98',
        'git push origin --delete rust-core-phase-3-t3.98',
    ],
)
def test_t30_an_approval_token_lets_a_one_off_through_and_is_logged(
    tmp_path: Path, command: str
) -> None:
    _owner_answers(tmp_path)
    listed = subprocess.run(
        [sys.executable, str(GUARD), '--tokens'],
        env=_env(tmp_path, SDM_OWNER_ANSWERS=str(_answers_file(tmp_path))),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert listed.returncode == 0, listed.stdout + listed.stderr
    tokens = [
        word
        for word in listed.stdout.split()
        if 'branch-cleanup-approval' in word
    ]
    assert tokens, listed.stdout
    approved = f'SDM_APPROVED={tokens[0]} {command}'
    decision, why, _ = _hook(tmp_path, approved)
    assert decision is None, why
    log = tmp_path / 'state/logs/approved-commands.log'
    text = log.read_text(encoding='utf-8')
    assert tokens[0] in text and command in text, text
    # A token no owner-answers section has is no approval.
    decision, _, _ = _hook(tmp_path, f'SDM_APPROVED=not-a-section {command}')
    assert decision == 'deny'


@pending
def test_t30_a_reviewer_launch_on_an_unstamped_sha_is_denied(
    tmp_path: Path,
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', TOPIC)
    sha = _commit(repo, 'T3.99: the candidate', {'a.txt': 'a\n'})
    prompt = (
        'Review round 1 of T3.99 at commit '
        f'{sha}. Read $SDM_ORCH_STATE/phase-3/briefs/reviewer.md in full.'
    )
    decision, why, done = _hook(
        tmp_path,
        '',
        tool='Agent',
        cwd=repo,
        tool_input={'description': 'Review T3.99', 'prompt': prompt},
    )
    assert decision == 'deny', done.stdout + done.stderr
    assert 'stamp' in why or 'handoff' in why, why


# --- 5. environments within the machine's limits ----------------------------


@pending
def test_t30_the_lane_environment_is_reused_while_its_hash_holds(
    tmp_path: Path,
) -> None:
    _, work, base = _clone(tmp_path)
    script, flag = ASSUMED['cells_key']
    docs = _commit(work, f'{PACKAGE}: docs only', {'docs/t399.md': 'x\n'})
    lock = _commit(
        work,
        f'{PACKAGE}: the lockfile',
        {'uv.lock': (work / 'uv.lock').read_text() + '\n# t399\n'},
    )
    _git(work, 'checkout', '-q', '--detach', base)
    native = _commit(
        work,
        f'{PACKAGE}: the native sources',
        {'rust/crates/sdm-core/src/t399.rs': '// t399\n'},
    )

    def key(rev: str) -> str:
        done = subprocess.run(
            ['bash', str(work / script), flag, '--rev', rev],
            cwd=work,
            env=_env(tmp_path, SDM_CELLS_DIR=str(tmp_path / 'cells')),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        lines = done.stdout.split()
        assert lines, done.stderr
        return lines[-1]

    assert key(docs) == key(base)
    assert key(lock) != key(base)
    assert key(native) != key(base)
    assert not (tmp_path / 'cells').exists() or not any(
        (tmp_path / 'cells').glob('cell-*')
    ), 'printing the key built an environment'


def _heavy(
    tmp_path: Path, config: dict[str, str], *command: str, wait: float = 8
) -> tuple[int | None, str]:
    """``heavy`` with these thresholds, copied beside a stand-in
    ``clean_tmp.sh`` that records its run; None when it is still waiting
    after ``wait`` seconds (it queued)."""
    tools = tmp_path / 'tools'
    tools.mkdir(exist_ok=True)
    heavy = tools / 'heavy'
    shutil.copy2(ROOT / ASSUMED['heavy'], heavy)
    clean = tools / 'clean_tmp.sh'
    clean.write_text(
        f'#!/bin/sh\necho "$@" >> "{tmp_path / "cleaned"}"\n', encoding='utf-8'
    )
    clean.chmod(0o755)
    conf = tmp_path / 'heavy.conf'
    conf.write_text(''.join(f'{k}={v}\n' for k, v in config.items()))
    (tmp_path / 'state').mkdir(exist_ok=True)
    proc = subprocess.Popen(
        ['bash', str(heavy), *command],
        env=_env(
            tmp_path,
            **{ASSUMED['heavy_config']: str(conf)},
            SDM_REPO=str(tmp_path),
            SDM_CLEAN_TMP=str(clean),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=wait)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        out, _ = proc.communicate()
        return None, out


def _thresholds(tmp_path: Path, **over: str) -> dict[str, str]:
    return {
        'SLOTS': '2',
        'MIN_FREE_MEM_GB': '0',
        'MIN_FREE_DISK_GB': '0',
        'DISK': str(tmp_path),
        **over,
    }


@pending
def test_t30_heavy_runs_a_job_when_cpu_memory_and_disk_allow(
    tmp_path: Path,
) -> None:
    ran = tmp_path / 'ran'
    code, out = _heavy(
        tmp_path, _thresholds(tmp_path), 'sh', '-c', f'touch {ran}; exit 7'
    )
    assert code == 7 and ran.exists(), out
    assert not (tmp_path / 'cleaned').exists()


@pending
@pytest.mark.parametrize('short', ['memory', 'disk'])
def test_t30_heavy_refuses_or_queues_below_its_thresholds(
    tmp_path: Path, short: str
) -> None:
    ran = tmp_path / 'ran'
    over = (
        {'MIN_FREE_MEM_GB': '100000000'}
        if short == 'memory'
        else {'MIN_FREE_DISK_GB': '100000000'}
    )
    code, out = _heavy(
        tmp_path, _thresholds(tmp_path, **over), 'touch', str(ran)
    )
    assert not ran.exists(), out
    assert code is None or code != 0, out
    assert short in out.lower() or 'mem' in out.lower(), out
    if short == 'disk':
        cleaned = tmp_path / 'cleaned'
        assert cleaned.exists(), 'clean_tmp.sh did not run first'
        assert '--apply' in cleaned.read_text()


@pending
def test_t30_the_runner_refuses_the_shared_venv(tmp_path: Path) -> None:
    main = tmp_path / 'main'
    main.mkdir()
    _git(main, 'init', '-q', '-b', 'rust-core')
    runners = main / 'contract/runners'
    runners.mkdir(parents=True)
    for path in (ROOT / 'contract/runners').glob('*.py'):
        shutil.copy2(path, runners / path.name)
    (main / 'contract/cases').mkdir()
    _git(main, 'add', '-A')
    _git(main, 'commit', '-q', '-m', 'the runners')
    lane = tmp_path / 'lane-a'
    _git(main, 'worktree', 'add', '-q', '-b', f'{PHASE}-lane-a', str(lane))

    def venv(where: Path) -> Path:
        subprocess.run(
            [sys.executable, '-m', 'venv', '--without-pip',
             '--system-site-packages', str(where)],
            check=True,
            timeout=300,
        )  # fmt: skip
        return where / 'bin/python'

    shared = venv(main / '.venv')
    own = venv(tmp_path / 'cells' / 'lane-a')

    def run(python: Path) -> tuple[int, str, Path]:
        out = tmp_path / f'{python.parent.parent.name}.jsonl'
        done = subprocess.run(
            [str(python), '-I', str(lane / 'contract/runners/run_python.py'),
             '--mode', 'python-core', '--out', str(out),
             str(lane / 'contract/cases')],
            cwd=lane,
            # CI holds no runner to the machine's environments (it builds its
            # own per run), so this case runs as on the machine.
            env={
                k: v
                for k, v in _env(tmp_path).items()
                if k != 'GITHUB_ACTIONS'
            },
            capture_output=True,
            text=True,
            timeout=300,
        )  # fmt: skip
        return done.returncode, done.stdout + done.stderr, out

    code, said, out = run(shared)
    assert code != 0, said
    assert '.venv' in said, said
    assert 'cells.sh' in said or 'lane' in said, said
    assert not out.exists() or not out.read_text().strip()
    code, said, _ = run(own)
    assert '.venv' not in said, said


# --- 1. the dry run and the slip report ----------------------------------------


@pending
def test_t30_the_dry_run_lists_a_case_kind_the_tools_cannot_judge(
    tmp_path: Path,
) -> None:
    _, work, _ = _clone(tmp_path)
    template = json.loads(
        next((work / 'contract/cases').rglob(f'{SEEDED_CASE}.json')).read_text()
    )
    future = {
        **template,
        'id': 's3.stub.future-explain',
        'slice': 'S3',
        'paths': ['query.explain/accept/any'],
        'routes': ['query.explain/accept/any'],
        'until': 'explain',
    }
    plan = tmp_path / 'kinds.yaml'
    plan.write_text(
        yaml.safe_dump(
            {
                'kinds': [
                    {'name': 'binary-classification', 'case': template},
                    {'name': 'future-explain', 'case': future},
                ]
            }
        )
    )
    done = subprocess.run(
        ['bash', str(work / ASSUMED['dry_run']), str(plan)],
        cwd=work,
        env=_env(tmp_path, SDM_PHASE_BRANCH=PHASE),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=3600,
    )
    said = done.stdout + done.stderr
    gaps = [
        line
        for line in said.splitlines()
        if 'future-explain' in line and 'gap' in line.lower()
    ]
    assert gaps, said[-3000:]
    assert _git(work, 'rev-parse', '--abbrev-ref', 'HEAD') == TOPIC
    assert not _git(work, 'status', '--porcelain')


@pending
def test_t30_the_slip_report_computes_the_slip_from_a_forecast(
    tmp_path: Path,
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', PHASE)

    def at(when: str, subject: str, files: dict[str, str]) -> None:
        for name, text in files.items():
            (repo / name).write_text(text)
            _git(repo, 'add', name)
        env = {
            **os.environ,
            **GIT_ENV,
            'GIT_AUTHOR_DATE': when,
            'GIT_COMMITTER_DATE': when,
        }
        _git(repo, 'commit', '-q', '--allow-empty', '-m', subject, env=env)

    at('2026-10-22T00:00:00Z', 'Land T3.0: the process changes', {'a': '0'})
    at('2026-10-22T01:00:00Z', 'T3.1: the design', {'a': '1'})
    at('2026-10-22T03:00:00Z', 'Land T3.1: the design', {})
    at('2026-10-22T04:00:00Z', 'T3.2: the first product package', {'a': '2'})
    at('2026-10-22T10:00:00Z', 'Land T3.2: the first product package', {})
    forecast = tmp_path / 'forecast.json'
    forecast.write_text(
        json.dumps(
            {
                'start': '2026-10-22T00:00:00Z',
                'packages': [
                    {'id': 'T3.1', 'hours': 2},
                    {'id': 'T3.2', 'hours': 4},
                    {'id': 'T3.3', 'hours': 4},
                ],
            }
        )
    )
    script, flag = ASSUMED['slip']
    done = subprocess.run(
        [sys.executable, str(ROOT / script), '--phase-branch', PHASE,
         '--repo', str(repo), '--no-ci', flag, str(forecast),
         '--now', '2026-10-22T10:00:00Z', '--utc-offset', '0'],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )  # fmt: skip
    said = done.stdout + done.stderr
    # Never blocks; the slip is printed. T3.2 was forecast at 6:00 and
    # landed at 10:00: four hours late, its own four hours over, and
    # T3.1's one hour over the rest.
    assert done.returncode == 0, said
    slip = [line for line in said.splitlines() if 'slip' in line.lower()]
    assert slip, said
    assert any('4' in line for line in slip), slip
    assert 'T3.2' in said[said.lower().index('slip') :], said
