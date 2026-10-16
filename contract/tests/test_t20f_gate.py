# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0f's acceptance tests through the gate (the independent test
author's): the locked-set reader for Phase 2, the ``main`` step (the
``origin/main`` ancestry, run only by a half that lists it; check 7 maps to
``[ratchet, main]``), and ``compare.py``'s known-case check taught the
``<case>@<database>`` keys.

Through the real ``scripts/rust_core/gate.sh`` on the temporary repository
and stubs ``contract/tests/test_gate_modes.py`` builds (its fixtures,
reused, as ``contract/tests/test_t20_locked_set.py`` reuses them: the
checkpoint holding the locked steps' stubs, a ``cargo`` stub, the Python
runners' real observations). Whether a locked check passes as a whole
depends on the runtime cell the runners observed in (H1), so the locked
runs here assert the steps' own results in the gate summary, never the
whole run's pass. A real gate run needs the built native core, so this
file lives in ``contract/tests`` beside ``test_gate_modes.py`` and
``test_t20_locked_set.py``, which CI runs in the ``contract-python`` job
that installs the core, never in the plain ``test`` jobs.

The reader (exit 1): Phase 2's real locked set, the design's
``docs/design/rust-core/phase-2/locked-set.yaml`` (T2.0a lands it; until
then its copy from the design branch, ``tests/fixtures/t20f/``), landed at
the checkpoint, is read: ``--locked 2a`` is no usage error, prints the
table of its checks 1 to 8, 10, 11 and 12, and each step whose tool has
not landed (``census``, ``coverage-s1``, ``l3``, ``l7``, ``ci``), and
``main`` with no ``origin/main`` fetched, fails the run by name. Seeded faults, each a usage error naming what is wrong, on
that file with one change: an unknown step, an unknown key of the half, a
malformed limit (not a number, negative, a boolean, an unknown name, a
baseline that is no commit, ``statements`` other than ``equal``, limits
not a mapping), a malformed CI pattern (empty, not a string, a check's
patterns not a list, patterns for a check the half does not have, a check
holding the ``ci`` step with no patterns).

Check 7 (exit 2): the ``main`` step, run only by a half whose ``steps``
list it (half 2a, its check 7 holding ``[ratchet, main]``), requires the
fetched ``origin/main`` (``refs/remotes/origin/main``, never a local
``main``) to be an ancestor of the candidate and of the reference commit
``contract/reference-env.txt`` names. A landing at the checkpoint
re-keys the reference to a commit of the temporary repository, ``R``
(its root), every citation of the old SHA outside ``docs/`` moved with it;
``M`` is a later commit of ``main``, a child of ``R`` that the candidate
does not hold. ``origin/main`` at ``R`` passes the ``main`` step, a local
``main`` at ``M`` notwithstanding (gated, since today's reader refuses the
step). Seeded: ``origin/main`` at ``M`` (``main``
moved and not merged; a local ``main`` at ``R`` changes nothing), no
``origin/main`` at all, and ``M`` merged into the candidate but the
reference not rerun each fail the ``main`` step and check 7, by name; so
does an ambiguous ``reference-env.txt`` (a second ``reference:`` field,
malformed or naming another commit, beside the valid one).
Control: Phase 1's half ``1b`` (no ``main`` step) is judged as before with
``main`` moved, and runs no step of T2.0f's.

``compare.py`` (exit 3): with the reference and both Python runners holding
``X@sqlite`` and ``X@postgres`` lines (copies of ``X``'s), ``gate.sh --only
compare`` passes. Seeded: a runner's line for no case, an unknown
database, ``@duckdb``, an empty part, two suffixes, upper case, and a
reference line for no case each fail it, named by key or by the file and
line. Control: the bare IDs pass as before.

Gating: the tests that need the feature are marked :data:`PENDING`, a
strict xfail; the implementer deletes the marker (its one definition below)
in the commit that makes them pass. Unmarked tests are controls that pass
on the base and must keep passing.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

#: The marker the implementer deletes (see the module docstring).
PENDING = pytest.mark.xfail(
    strict=True,
    reason="T2.0f: the gate's Phase 2 reader and database keys are not implemented yet",
)

# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: Phase 2's locked set, where T2.0a lands it; until then the design
#: branch's copy (``origin/rust-core-phase-2-design`` at dce97ec9, the
#: approved design).
LOCKED_SET = 'docs/design/rust-core/phase-2/locked-set.yaml'
COPY = ROOT / 'tests/fixtures/t20f/phase-2-locked-set.yaml'
#: The steps T2.0f's reader adds, each failing the locked run by name in
#: the summary until its tool lands (``main``, T2.0f's own, until
#: ``origin/main`` is fetched and merged).
NEW_STEPS = ('census', 'coverage-s1', 'l3', 'l7', 'ci', 'main')
#: A malformed locked set is a usage error (exit 2) before any step, as
#: the reader's refusals are today, its message naming what is wrong.
USAGE = 2
#: The check that holds the ``main`` step (the ``origin/main`` ancestry, its
#: own gate step named ``main``, run only by a half whose ``steps`` list
#: it), beside ``ratchet``.
MAIN_CHECK = '7'
MAIN_STEP = 'main'
ORIGIN_MAIN = 'refs/remotes/origin/main'
LOCAL_MAIN = 'refs/heads/main'
# ---------------------------------------------------------------------------------


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GM = _module(ROOT / 'contract/tests/test_gate_modes.py', 't20f_gate_modes')
helpers = GM.helpers
git, commit, land = helpers.git, helpers.commit, helpers.land

# test_gate_modes' fixtures, used here as there.
origin = GM.origin
observed = GM.observed
stubbed = GM.stubbed
stubs = GM.stubs
repo = GM.repo

#: A fixture half's acceptance cases (test_t20_locked_set's).
CASES = [
    's0.accept.binary-classification',
    's0.accept.multiclass-classification',
    's0.accept.sum-window',
]
#: The case whose database lines compare.py is given.
X = 's0.accept.sum-window'


def phase_2_text() -> str:
    real = ROOT / LOCKED_SET
    return (real if real.is_file() else COPY).read_text(encoding='utf-8')


def with_locked_set(text: str) -> Callable[[Path], None]:
    def edit(tree: Path) -> None:
        path = tree / LOCKED_SET
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    return edit


def changed(change: Callable[[dict[str, Any]], None]) -> str:
    r"""Phase 2's locked set with one change to its half 2a."""
    document = yaml.safe_load(phase_2_text())
    change(document['halves']['2a'])
    return yaml.safe_dump(document, sort_keys=False)


def run_locked(
    tree: Path,
    stubs: dict[str, Any],
    observed: Any,
    tmp_path: Path,
    half: str = '2a',
) -> subprocess.CompletedProcess[str]:
    return GM.locked(
        tree, stubs, half, '--run', 'rust-host', *GM.files(observed, tmp_path)
    )


def step_log(done: subprocess.CompletedProcess[str], name: str) -> str:
    r"""The lines the gate printed under step ``name``."""
    lines = done.stdout.splitlines()
    found: list[str] = []
    inside = False
    for line in lines:
        head = re.match(r'^(\S+)\s+(ok|FAIL)\b', line)
        if head is not None:
            inside = head[1] == name
            if inside:
                found.append(line)
            continue
        if inside and line.startswith('    |'):
            found.append(line)
        elif inside:
            inside = False
    assert found, done.stdout + done.stderr
    return '\n'.join(found)


def named(text: str, token: str) -> bool:
    r"""``token`` in ``text`` as a word of its own, not inside a longer one."""
    return (
        re.search(
            rf'(?<![A-Za-z0-9_.@-]){re.escape(token)}(?![A-Za-z0-9_@-])', text
        )
        is not None
    )


# --- the reader: Phase 2's real locked set ----------------------------------------


def test_the_phase_2_copy_is_a_locked_set_of_half_2a() -> None:
    r"""The fixture's premise: the file holds half 2a with the steps and
    keys T2.0f's reader adds, so the test below reads them all."""
    half = yaml.safe_load(phase_2_text())['halves']['2a']
    assert set(NEW_STEPS) <= set(half['steps'])
    assert {'limits', 'ci'} <= set(half)
    assert half['checks'][MAIN_CHECK] == ['ratchet', MAIN_STEP]


@PENDING
def test_phase_2_s_locked_set_is_read_and_each_step_without_its_tool_fails_by_name(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
) -> None:
    r"""Exit 1, and the post-landing state judged against itself: the
    landed harness reads the real half 2a (no usage error), prints its
    checks, writes its evidence, and each step whose tool has not landed
    fails the locked run by name."""
    land(stubbed, 'T2.0a', with_locked_set(phase_2_text()), protected='docs/**')
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    steps = GM.summary(done)
    for name in NEW_STEPS:
        assert steps.get(name) == 'FAIL', (name, steps)
    checks = GM.table(done, '2a')
    assert set(checks) == {
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '10',
        '11',
        '12',
    }
    # Check 11 holds l7 alone and 6 l3 alone: each fails on its own step.
    assert checks['11'] == checks['6'] == 'FAIL', checks
    assert GM.evidence_of(stubbed, stubs)['halves']['2a']['passed'] is False


def _fault(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(half: dict[str, Any]) -> None:
        half[key] = value

    return change


def _limit(name: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(half: dict[str, Any]) -> None:
        half['limits'][name] = value

    return change


def _pattern(
    check: str, value: Any, replace: bool = False
) -> Callable[[dict[str, Any]], None]:
    def change(half: dict[str, Any]) -> None:
        if replace:
            half['ci'][check] = value
        else:
            half['ci'][check] = [*half['ci'][check], value]

    return change


def _without_patterns(check: str) -> Callable[[dict[str, Any]], None]:
    r"""Check ``check`` still holds the ci step, and its patterns are gone."""

    def change(half: dict[str, Any]) -> None:
        assert 'ci' in half['checks'][check]
        del half['ci'][check]

    return change


def _step(name: str) -> Callable[[dict[str, Any]], None]:
    def change(half: dict[str, Any]) -> None:
        half['steps'] = [*half['steps'], name]
        half['checks']['1'] = [*half['checks']['1'], name]

    return change


#: Each seeded fault: the change to half 2a, and the word the refusal
#: must name.
FAULTS = {
    'unknown-step': (_step('bogus-step'), 'bogus-step'),
    'unknown-key': (_fault('extra-key', 1), 'extra-key'),
    'limit-not-a-number': (
        _limit('cpu-median-ratio', 'fast'),
        'cpu-median-ratio',
    ),
    'limit-negative': (_limit('cpu-max-ratio', -1.1), 'cpu-max-ratio'),
    'limit-a-boolean': (
        _limit('boundary-overhead-floor-ms', True),
        'boundary-overhead-floor-ms',
    ),
    'limit-unknown': (_limit('made-up-ratio', 1.0), 'made-up-ratio'),
    'limit-baseline-no-commit': (
        _limit('native-size-baseline', 'not-a-sha'),
        'native-size-baseline',
    ),
    'limit-statements-not-equal': (_limit('statements', 'fewer'), 'statements'),
    'limits-not-a-mapping': (_fault('limits', [1.0, 1.1]), 'limits'),
    'pattern-empty': (_pattern('1', ''), 'ci'),
    'pattern-not-a-string': (_pattern('3', 42), 'ci'),
    'patterns-not-a-list': (
        _pattern('4', 'l5-conversion*', replace=True),
        'ci',
    ),
    'patterns-for-no-check': (_pattern('99', ['x*'], replace=True), '99'),
    'ci-check-without-patterns': (_without_patterns('8'), '8'),
}


@PENDING
@pytest.mark.parametrize('fault', sorted(FAULTS))
def test_a_malformed_phase_2_half_is_a_usage_error_naming_the_fault(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
    fault: str,
) -> None:
    r"""Seeded (exit 1): an unknown step or key, a malformed limit or CI
    pattern, each refused before any step, naming what is wrong. Its twin,
    the file unchanged, is read (the test above)."""
    change, word = FAULTS[fault]
    land(
        stubbed, 'T2.0a', with_locked_set(changed(change)), protected='docs/**'
    )
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path)
    assert done.returncode == USAGE, done.stdout + done.stderr
    assert 'gate summary' not in done.stdout
    assert named(done.stdout + done.stderr, word), done.stdout + done.stderr


# --- check 7: origin/main an ancestor of the candidate and the reference ---------


def _root(tree: Path) -> str:
    return git(tree, 'rev-list', '--max-parents=0', 'HEAD')


def _rekey_reference(sha: str) -> Callable[[Path], None]:
    r"""The reference re-keyed to ``sha``: the file's name and every
    citation of the old SHA outside ``docs/`` (``reference-env.txt``, the
    observations' identities, the vector files)."""

    def edit(tree: Path) -> None:
        env = (tree / 'contract/reference-env.txt').read_text()
        old = re.search(r'^reference: ([0-9a-f]{40})$', env, re.M)
        assert old is not None
        listed = git(tree, 'ls-files', '-z').split('\0')
        for name in filter(None, listed):
            if name.startswith('docs/'):
                continue
            path = tree / name
            data = path.read_bytes()
            if old[1].encode() in data:
                path.write_bytes(data.replace(old[1].encode(), sha.encode()))
        reference = tree / f'contract/reference/{old[1]}.jsonl'
        reference.rename(reference.with_name(f'{sha}.jsonl'))

    return edit


def _main_half() -> str:
    r"""A half 2a listing the main step, check 7 holding the ratchet and
    main steps, and its cases."""
    return yaml.safe_dump(
        {
            'format': 'sdm-locked-set/1',
            'halves': {
                '2a': {
                    'cases': CASES,
                    'steps': ['inventory', MAIN_STEP],
                    'checks': {
                        '1': [
                            'runners',
                            'observations',
                            'ratchet',
                            'compare',
                            'coverage',
                            'l1',
                            'inventory',
                        ],
                        MAIN_CHECK: ['ratchet', MAIN_STEP],
                    },
                }
            },
        },
        sort_keys=False,
    )


def _refs(tree: Path, origin_main: str | None, local_main: str | None) -> None:
    for ref, sha in ((ORIGIN_MAIN, origin_main), (LOCAL_MAIN, local_main)):
        if sha is None:
            subprocess.run(
                ['git', '-C', str(tree), 'update-ref', '-d', ref],
                capture_output=True,
                check=False,
            )
        else:
            git(tree, 'update-ref', ref, sha)


@pytest.fixture
def mainline(stubbed: Path) -> dict[str, str]:
    r"""The checkpoint: the reference re-keyed to R and half 2a mapping
    check 7; and M, a later commit of main the candidate does not hold."""
    root = _root(stubbed)
    land(
        stubbed,
        'T2.0c',
        lambda t: (_rekey_reference(root)(t), with_locked_set(_main_half())(t)),
        protected='contract/** docs/** rust/**',
    )
    moved = git(
        stubbed,
        'commit-tree',
        f'{root}^{{tree}}',
        '-p',
        root,
        '-m',
        'main: a later commit',
    )
    return {'R': root, 'M': moved}


@PENDING
def test_main_at_the_reference_commit_passes_the_main_step(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
    mainline: dict[str, str],
) -> None:
    r"""Exit 2, the seeds' twin: ``origin/main`` at R, which the candidate
    and the reference commit hold, passes the main step; a local ``main``
    at M is not read."""
    _refs(stubbed, origin_main=mainline['R'], local_main=mainline['M'])
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path)
    steps = GM.summary(done)
    assert steps.get(MAIN_STEP) == 'ok', done.stdout + done.stderr
    assert steps.get('ratchet') == 'ok', done.stdout + done.stderr


@PENDING
@pytest.mark.parametrize(
    'where',
    [
        pytest.param('moved', id='main-moved-not-merged'),
        pytest.param('missing', id='no-origin-main'),
    ],
)
def test_main_not_merged_fails_the_main_step_by_name(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
    mainline: dict[str, str],
    where: str,
) -> None:
    r"""Seeded (exit 2): ``origin/main`` moved to M, which the candidate
    does not hold (a local ``main`` at R changes nothing), or no fetched
    ``origin/main`` at all: the main step fails, naming origin/main, and
    check 7 with it."""
    _refs(
        stubbed,
        origin_main=mainline['M'] if where == 'moved' else None,
        local_main=mainline['R'],
    )
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path)
    assert GM.summary(done).get(MAIN_STEP) == 'FAIL', done.stdout + done.stderr
    assert 'origin/main' in step_log(done, MAIN_STEP), done.stdout
    assert GM.table(done, '2a')[MAIN_CHECK] == 'FAIL'


@PENDING
def test_main_merged_without_the_reference_rerun_fails_the_main_step(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
    mainline: dict[str, str],
) -> None:
    r"""Seeded (exit 2): the candidate merges M, but the reference commit
    is still R, which M is no ancestor of: the main step fails, naming
    origin/main and the reference, and check 7 with it."""
    _refs(stubbed, origin_main=mainline['M'], local_main=None)
    git(
        stubbed,
        'merge',
        '-q',
        '--no-ff',
        '-m',
        'T2.0c: merge main',
        mainline['M'],
    )
    assert git(stubbed, 'rev-parse', 'HEAD^{tree}') == git(
        stubbed, 'rev-parse', 'HEAD^1^{tree}'
    )
    done = run_locked(stubbed, stubs, observed, tmp_path)
    assert GM.summary(done).get(MAIN_STEP) == 'FAIL', done.stdout + done.stderr
    log = step_log(done, MAIN_STEP)
    assert 'origin/main' in log, log
    assert mainline['R'][:7] in log or 'reference-env.txt' in log, log
    assert GM.table(done, '2a')[MAIN_CHECK] == 'FAIL'


def _second_reference(where: str, line: str) -> Callable[[Path], None]:
    r"""``reference-env.txt`` with a second ``reference:`` field ``line``,
    ``before`` or ``after`` the valid one."""

    def edit(tree: Path) -> None:
        path = tree / 'contract/reference-env.txt'
        text = path.read_text()
        valid = re.search(r'(?m)^reference: [0-9a-f]{40}$', text)
        assert valid is not None
        both = (
            f'{line}\n{valid[0]}'
            if where == 'before'
            else f'{valid[0]}\n{line}'
        )
        path.write_text(text.replace(valid[0], both, 1))

    return edit


@PENDING
@pytest.mark.parametrize(
    ('where', 'extra'),
    [
        pytest.param('after', 'malformed', id='malformed-after'),
        pytest.param('before', 'malformed', id='malformed-before'),
        pytest.param('after', 'another-commit', id='another-commit'),
    ],
)
def test_an_ambiguous_reference_env_fails_the_main_step_by_name(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
    mainline: dict[str, str],
    where: str,
    extra: str,
) -> None:
    r"""Seeded (exit 2, review round 1): with ``origin/main`` at R, which
    passes (the twin above), a second ``reference:`` field beside the valid
    one, ``reference: not-a-commit`` or one naming M, makes the reference
    commit ambiguous: the main step fails, naming ``reference-env.txt``, and
    check 7 with it."""
    _refs(stubbed, origin_main=mainline['R'], local_main=None)
    line = (
        'reference: not-a-commit'
        if extra == 'malformed'
        else f'reference: {mainline["M"]}'
    )
    commit(stubbed, 'T2.0c: a change', _second_reference(where, line))
    done = run_locked(stubbed, stubs, observed, tmp_path)
    assert GM.summary(done).get(MAIN_STEP) == 'FAIL', done.stdout + done.stderr
    assert 'reference-env.txt' in step_log(done, MAIN_STEP), done.stdout
    assert GM.table(done, '2a')[MAIN_CHECK] == 'FAIL'


def test_phase_1_s_half_is_judged_as_before_with_main_moved(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
) -> None:
    r"""Control: half 1b lists no main step, so ``origin/main`` moved and
    not merged leaves its ratchet step as it was, no main step runs, and
    its steps are its own."""
    root = _root(stubbed)
    moved = git(
        stubbed,
        'commit-tree',
        f'{root}^{{tree}}',
        '-p',
        root,
        '-m',
        'main: moved',
    )
    _refs(stubbed, origin_main=moved, local_main=None)
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path, half='1b')
    steps = GM.summary(done)
    assert steps.get('ratchet') == 'ok', done.stdout + done.stderr
    assert not set(NEW_STEPS) & set(steps), steps
    assert set(GM.table(done, '1b')) == {'4', '5', '6'}


# --- compare.py's known-case check ------------------------------------------------


def _reference_file(tree: Path) -> Path:
    found = list((tree / 'contract/reference').glob('*.jsonl'))
    assert len(found) == 1, found
    return found[0]


def _keyed(line: dict[str, Any], key: str) -> dict[str, Any]:
    added = copy.deepcopy(line)
    added['case'] = key
    return added


def _with_reference_keys(*keys: str) -> Callable[[Path], None]:
    def edit(tree: Path) -> None:
        path = _reference_file(tree)
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        bare = next(x for x in lines if x['case'] == X)
        lines += [_keyed(bare, key) for key in keys]
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))

    return edit


def _with_keys(
    keys: tuple[str, ...],
    extra: tuple[str, ...] = (),
    at: dict[str, int] | None = None,
) -> Callable[[str, list[dict[str, Any]]], None]:
    r"""Each runner's X lines copied under ``keys``; python-old's also under
    ``extra``, whose line numbers go into ``at``."""

    def change(runner: str, lines: list[dict[str, Any]]) -> None:
        bare = next(x for x in lines if x['case'] == X)
        lines += [_keyed(bare, key) for key in keys]
        if runner == 'python-old':
            for key in extra:
                lines.append(_keyed(bare, key))
                if at is not None:
                    at[key] = len(lines)

    return change


def _only_compare(
    tree: Path, landed: str, files: list[str]
) -> subprocess.CompletedProcess[str]:
    return GM.gate(
        tree, '--no-l0', '--base', landed, '--only', 'compare', *files
    )


def test_bare_case_ids_compare_as_before(
    repo: Path, tmp_path: Path, observed: Any
) -> None:
    r"""Control: DuckDB's lines keep the bare case ID, compared as before."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    done = _only_compare(repo, landed, GM.files(observed, tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert GM.summary(done) == {'compare': 'ok'}


DATABASE_KEYS = (f'{X}@sqlite', f'{X}@postgres')
#: python-old's seeded lines: no case, an unknown database, DuckDB's
#: suffix, an empty part, two suffixes, upper case.
BAD_KEYS = (
    'nocase.at-all@sqlite',
    f'{X}@oracle',
    f'{X}@duckdb',
    f'{X}@',
    '@sqlite',
    f'{X}@sqlite@postgres',
    f'{X}@SQLITE',
)
#: The reference's seeded line: no case.
BAD_REFERENCE = 'nocase.reference-only@postgres'


@PENDING
def test_compare_takes_database_keys_and_refuses_malformed_ones_by_name(
    repo: Path, tmp_path: Path, observed: Any
) -> None:
    r"""Exit 3. Twin: the reference and both Python runners hold
    ``X@sqlite`` and ``X@postgres``: ``--only compare`` passes. Seeded:
    python-old's lines for no case, an unknown database, ``@duckdb``, an
    empty part, two suffixes and upper case, and the reference's line for
    no case, each fail it, named by key or by file and line."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, 'T2.1a: database lines', _with_reference_keys(*DATABASE_KEYS))
    done = _only_compare(
        repo,
        landed,
        GM.files(observed, tmp_path, change=_with_keys(DATABASE_KEYS)),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert GM.summary(done) == {'compare': 'ok'}

    commit(repo, 'T2.1a: a seeded line', _with_reference_keys(BAD_REFERENCE))
    at: dict[str, int] = {}
    done = _only_compare(
        repo,
        landed,
        GM.files(
            observed, tmp_path, change=_with_keys(DATABASE_KEYS, BAD_KEYS, at)
        ),
    )
    assert done.returncode == 1, done.stdout + done.stderr
    differs = [x for x in done.stdout.splitlines() if 'DIFFERS' in x]
    for key in BAD_KEYS:
        assert any(
            named(x, key) or f'python-old.jsonl:{at[key]}:' in x
            for x in differs
        ), (key, differs)
    reference = _reference_file(repo)
    number = len(reference.read_text().splitlines())
    assert any(
        named(x, BAD_REFERENCE) or f'{reference.name}:{number}:' in x
        for x in differs
    ), differs
    for key in DATABASE_KEYS:
        assert not any(named(x, key) for x in differs), (key, differs)
