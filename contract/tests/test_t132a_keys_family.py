# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32a's exits for the keys family on ``python-core``.

    python -m pytest contract/tests/test_t132a_keys_family.py -q

Written by the package's independent test author from the brief, the
design's T1.32a row and the reference, before the implementation. The
family is the nine ``key.*`` cases of ``contract/cases/s1``; each is run by
``contract/runners/run_python.py --mode python-core`` on the core route
(its own ``routes`` once admitted; until then every DuckDB route of
``Schema.from_source``, ``materialize`` and ``predict``, forced on a copy).

- Admission ("the ``key.*`` cases admitted"): each case is an acceptance
  case with routes, and carries its entities to the core
  (``core.materialize``), the decimal keys' included (their gap in
  ``contract/refusal-maps/keys.json``).
- "Each with its reference observation" and the cases' expectations: each
  case's ``expect`` is what ``python-ref`` observed
  (``contract/reference/<commit>.jsonl``), and ``python-core``'s outcome
  (result, stage, class and code) is ``python-ref``'s, or a register
  entry, proposed or approved, lists the case in its ``cases`` (P1B-1,
  the inventory's placeholder, does not count: its own notes give each
  changed case an entry of its own). ``sdm-testkit``'s
  ``tests/t132a_keys_sql.rs`` holds ``rust-host`` the same way.
- "Statement counts equal the reference's or are registered,
  cross-checked against DuckDB's query log": per public call, as
  ``compare.py`` counts them, with the capture complete against the log.
- "Capture on and off give the same public projection on the Python
  runners": ``capture_check.py`` over the family run with and without the
  capture, every case required; and the check fails on this family's own
  evidence when one public projection is seeded to differ.
- "The bridge's cast leaves the freeze as ``moved``": since the phase head
  T1.32a started from (``START``), ``_rust/materialize.py::_indices``
  changed or left the freeze's baseline, and ``bridge_changes.json``
  declared it ``moved`` naming T1.32a (at HEAD, or in a commit since, so
  the test survives a later pruning of spent declarations); the bridge
  holds no pandas cast of the entities (no ``ArrowDtype``,
  ``type_for_alias`` or ``_ARROW_TYPES``); and the design's ledger row
  "The entities cast to the key's type" is gone (E50(c)).

Gating (``contract/tests/t132a_gate.py``): a test whose id is in
``t132a-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list, which may only shrink.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = sorted((ROOT / 'contract/cases/s1').glob('key.*.json'))
IDS = [path.stem for path in CASES]
#: The phase head T1.32a started from (the brief: 1efd5882, "Land T1.20
#: part 2").
START = '1efd58827a36a1a999603e3e57d6766eaca813db'
BRIDGE = (
    'packages/structured-data-models-client/src/sdm_client/_rust/materialize.py'
)
CAST = f'{BRIDGE}::_indices'
#: The register's placeholder for the inventory's cases (see above).
PLACEHOLDER = 'P1B-1'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132a_gate', ROOT / 'contract/tests/t132a_gate.py')


def gated(request: pytest.FixtureRequest, check: Any) -> None:
    r"""Runs ``check`` held to the pending list, as ``py:<test id>``."""
    G.held('exits', f'py:{request.node.name}', check)


def per_case() -> list[Any]:
    return [pytest.param(case, id=case) for case in IDS]


def case_file(case: str) -> dict[str, Any]:
    path = ROOT / f'contract/cases/s1/{case}.json'
    return json.loads(path.read_text(encoding='utf-8'))


def reference() -> dict[str, dict[str, Any]]:
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        out[observation['case']] = observation
    return out


def registered_case(case: str) -> str | None:
    r"""The entry, proposed or approved, other than the placeholder, that
    lists ``case`` in its ``cases``."""
    path = ROOT / 'docs/design/rust-core/register.yaml'
    for entry in yaml.safe_load(path.read_text(encoding='utf-8'))['entries']:
        if entry['id'] == PLACEHOLDER:
            continue
        if entry.get('status') in ('proposed', 'approved') and case in (
            entry.get('cases') or []
        ):
            return str(entry['id'])
    return None


# --- the family's runs ---------------------------------------------------------


def _forced(tmp: Path) -> Path:
    r"""Copies of the family's cases, each on its own routes, or on every
    DuckDB route of the three public calls while it names none."""
    from sdm_client import _routing

    routes = list(
        _routing.FROM_SOURCE_DUCKDB
        + _routing.MATERIALIZE_DUCKDB
        + _routing.PREDICT_DUCKDB
    )
    out = tmp / 'cases'
    out.mkdir()
    for path in CASES:
        case = json.loads(path.read_text(encoding='utf-8'))
        case['routes'] = case['routes'] or routes
        (out / path.name).write_text(json.dumps(case), encoding='utf-8')
    return out


def _run(cases: Path, out: Path, *extra: str) -> list[dict[str, Any]]:
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            'python-core',
            *extra,
            '--out',
            str(out),
            str(cases),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return [
        json.loads(line)
        for line in out.read_text(encoding='utf-8').splitlines()
    ]


@pytest.fixture(scope='module')
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp('t132a')
    cases = _forced(tmp)
    on = _run(cases, tmp / 'on.jsonl')
    off = _run(cases, tmp / 'off.jsonl', '--no-capture')
    return {
        'on': {o['case']: o for o in on},
        'off': {o['case']: o for o in off},
        'on_path': tmp / 'on.jsonl',
        'off_path': tmp / 'off.jsonl',
        'tmp': tmp,
    }


# --- admission and the cases' expectations ------------------------------------


def test_the_family_is_the_nine_key_cases() -> None:
    assert sorted(IDS) == [
        'key.decimal-18-0',
        'key.decimal-38-10',
        'key.declared-int32',
        'key.hugeint',
        'key.uint64-above-int64-max',
        'key.uint64-small',
        'key.uint8',
        'key.varchar',
        'key.varchar-large-buffers',
    ]


@pytest.mark.parametrize('case', per_case())
def test_each_key_case_is_admitted(
    case: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        loaded = case_file(case)
        assert loaded['status'] == 'acceptance', loaded['status']
        assert loaded['routes'], 'an admitted case names its routes'
        materialize = (loaded.get('core') or {}).get('materialize')
        assert materialize, 'the case carries its entities to the core'
        assert materialize['indices'], 'and they are not empty'

    gated(request, check)


@pytest.mark.parametrize('case', per_case())
def test_each_key_case_expects_what_python_ref_observed(
    case: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        expect = case_file(case)['expect']
        observed = reference()[case]
        assert observed['identity']['runner'] == 'python-ref'
        outcome = observed['outcome']
        assert expect['outcome'] == outcome['result']
        assert expect['stage'] == outcome['stage']
        if outcome['result'] == 'refusal':
            assert expect['error'] == outcome['error']
            assert expect['code'] == outcome['code']

    gated(request, check)


@pytest.mark.parametrize('case', per_case())
def test_python_core_answers_each_key_case_as_python_ref(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        ours = runs['on'][case]['outcome']
        theirs = reference()[case]['outcome']
        assert ours.get('code') != 'UNSUPPORTED_REQUEST', ours
        fields = ('result', 'stage', 'error', 'code')
        differs = {
            k: (ours.get(k), theirs.get(k))
            for k in fields
            if ours.get(k) != theirs.get(k)
        }
        assert not differs or registered_case(case), differs

    gated(request, check)


# --- statement counts and the capture -----------------------------------------


@pytest.mark.parametrize(
    'case',
    per_case(),
)
def test_statement_counts_equal_the_reference_s_or_are_registered(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')
        check = _module(
            'capture_check', ROOT / 'contract/runners/capture_check.py'
        )
        ours = runs['on'][case]
        assert check.complete(ours) == [], 'the capture is DuckDB query log'
        mine = compare.counts(ours)
        theirs = compare.counts(reference()[case])
        assert mine is not None and theirs is not None
        assert mine == theirs or registered_case(case), (mine, theirs)

    gated(request, check)


def _capture_check(on: Path, off: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/capture_check.py'),
            '--on',
            str(on),
            '--off',
            str(off),
            *(arg for case in IDS for arg in ('--require', case)),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_capture_on_and_off_give_the_same_public_projection(
    runs: dict[str, Any],
) -> None:
    done = _capture_check(runs['on_path'], runs['off_path'])
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'complete and conserving' in done.stdout


def test_the_capture_check_fails_on_a_seeded_difference_in_the_family(
    runs: dict[str, Any],
) -> None:
    # Not vacuous on this family's evidence: one case's public outcome
    # changed in the capture-off run fails the check.
    seeded = runs['tmp'] / 'seeded-off.jsonl'
    lines = []
    for line in runs['off_path'].read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        if observation['case'] == 'key.uint8':
            observation['outcome'] = dict(
                observation['outcome'], result='refusal', code='SEEDED'
            )
        lines.append(json.dumps(observation))
    seeded.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    done = _capture_check(runs['on_path'], seeded)
    assert done.returncode != 0, done.stdout
    assert 'key.uint8' in done.stdout + done.stderr


# --- the bridge's cast leaves the freeze as moved -----------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _declared_moved(text: str) -> bool:
    declared = json.loads(text).get('changes', {}).get(CAST)
    return (
        isinstance(declared, dict)
        and set(declared) == {'moved'}
        and 'T1.32a' in str(declared['moved'])
    )


@pytest.mark.parametrize(
    'part',
    [
        pytest.param(part, id=part)
        for part in ('fingerprint', 'declared', 'no-pandas-cast', 'ledger')
    ],
)
def test_the_bridge_s_cast_left_the_freeze_as_moved(
    part: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        baseline = 'scripts/rust_core/bridge_freeze.json'
        changes = 'scripts/rust_core/bridge_changes.json'
        if part == 'fingerprint':
            # The cast changed or left the baseline since T1.32a's start.
            then = json.loads(_git('show', f'{START}:{baseline}'))
            now = json.loads((ROOT / baseline).read_text(encoding='utf-8'))
            before = then['files'][BRIDGE]['_indices']
            assert now['files'].get(BRIDGE, {}).get('_indices') != before
        elif part == 'declared':
            texts = [(ROOT / changes).read_text(encoding='utf-8')]
            for sha in _git(
                'log', '--format=%H', f'{START}..HEAD', '--', changes
            ).split():
                texts.append(_git('show', f'{sha}:{changes}'))
            assert any(_declared_moved(text) for text in texts), (
                f'{CAST} is declared moved, naming T1.32a, in {changes}'
            )
        elif part == 'no-pandas-cast':
            tree = ast.parse((ROOT / BRIDGE).read_text(encoding='utf-8'))
            names = {
                node.attr if isinstance(node, ast.Attribute) else node.id
                for node in ast.walk(tree)
                if isinstance(node, (ast.Attribute, ast.Name))
            }
            assert not names & {'ArrowDtype', 'type_for_alias', '_ARROW_TYPES'}
        else:
            design = (
                ROOT / 'docs/design/rust-core/phase-1/design.md'
            ).read_text(encoding='utf-8')
            assert "| The entities cast to the key's type" not in design

    gated(request, check)


def test_the_freeze_s_checks_hold_on_seeded_evidence() -> None:
    # The moved rule reads a declaration exactly: moved, naming T1.32a.
    good = {'changes': {CAST: {'moved': "T1.32a: the core's cast"}}}
    assert _declared_moved(json.dumps(good))
    for bad in (
        {'changes': {}},
        {'changes': {CAST: {'refactor': 'T1.32a'}}},
        {'changes': {CAST: {'moved': 'T1.32b: elsewhere'}}},
        {'changes': {CAST: {'moved': 'T1.32a', 'gone': 'x'}}},
        {'changes': {f'{BRIDGE}::other': {'moved': 'T1.32a'}}},
    ):
        assert not _declared_moved(json.dumps(bad)), bad
    # And the start's baseline holds the cast, so the fingerprint is real.
    then = json.loads(
        _git('show', f'{START}:scripts/rust_core/bridge_freeze.json')
    )
    assert then['files'][BRIDGE]['_indices']
