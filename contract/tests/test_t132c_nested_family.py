# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32c's exits for the nested family on ``python-core``.

    python -m pytest contract/tests/test_t132c_nested_family.py -q

Written by the package's independent test author from the brief, the
design's T1.32c row ("The ``nested.*`` cases, as T1.32a", whose exits apply
to this family) and the reference, before the implementation. The family is
the five ``nested.*`` cases of ``contract/cases/s1`` (``nested.union`` among
them, whose refusal T1.32c resolves when it admits it: register P1B-7);
each is run by ``contract/runners/run_python.py --mode python-core`` on the
core route (its own ``routes`` once admitted; until then every DuckDB route
of ``Schema.from_source``, ``materialize`` and ``predict``, forced on a
copy).

- Admission ("the ``nested.*`` cases admitted"): each case is an acceptance
  case with routes, carrying its inputs to the core (``describe_source``,
  ``materialize`` and ``predict``).
- "Each with its reference observation" and the cases' expectations: each
  case's ``expect`` is what ``python-ref`` observed
  (``contract/reference/<commit>.jsonl``), and ``python-core``'s outcome
  (result, stage, class and code) is ``python-ref``'s, or an approved
  register entry lists the case in its ``cases`` (a proposed one excuses
  nothing, so its case stays pending until the owner approves it; P1B-1,
  the inventory's placeholder, does not count: its own notes give each
  changed case an entry of its own). The core's ``UNSUPPORTED_REQUEST``
  is no exception: it passes where an approved entry lists the case
  (P1B-10's ``nested.map`` and ``nested.struct``), as any other
  difference does. ``sdm-testkit``'s
  ``tests/t132c_nested_sql.rs`` holds ``rust-host`` the same way, and
  ``test_t132c_nested_types.py`` both hosts over the family's types.
- "Statement counts equal the reference's or are registered,
  cross-checked against DuckDB's query log": per public call, as
  ``compare.py`` counts them, with the capture complete against the log;
  a difference passes where an approved entry lists the case.
- "Capture on and off give the same public projection on the Python
  runners": ``capture_check.py`` over the family's cases and the nested
  type vectors (``contract/nested-type-vectors.json``, as cases) run with
  and without the capture, every one required; and the check fails on
  this family's own evidence when one public projection is seeded to
  differ.

Gating (``contract/tests/t132c_gate.py``): a test whose id is in
``t132c-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list, which may only shrink.
"""

from __future__ import annotations

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
CASES = sorted((ROOT / 'contract/cases/s1').glob('nested.*.json'))
IDS = [path.stem for path in CASES]
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


G = _module('t132c_gate', ROOT / 'contract/tests/t132c_gate.py')
R = _module('nested_type_record', ROOT / 'contract/tools/nested_type_record.py')


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
    r"""The approved entry, other than the placeholder, that lists ``case``
    in its ``cases``."""
    path = ROOT / 'docs/design/rust-core/register.yaml'
    for entry in yaml.safe_load(path.read_text(encoding='utf-8'))['entries']:
        if entry['id'] == PLACEHOLDER:
            continue
        if entry.get('status') == 'approved' and case in (
            entry.get('cases') or []
        ):
            return str(entry['id'])
    return None


# --- the family's runs ---------------------------------------------------------


def _routes() -> list[str]:
    from sdm_client import _routing

    return list(
        _routing.FROM_SOURCE_DUCKDB
        + _routing.MATERIALIZE_DUCKDB
        + _routing.PREDICT_DUCKDB
    )


def _forced(tmp: Path) -> Path:
    r"""Copies of the family's cases, each on its own routes, or on every
    DuckDB route of the three public calls while it names none; and the
    nested type vectors as cases on those routes."""
    routes = _routes()
    out = tmp / 'cases'
    out.mkdir()
    for path in CASES:
        case = json.loads(path.read_text(encoding='utf-8'))
        case['routes'] = case['routes'] or routes
        (out / path.name).write_text(json.dumps(case), encoding='utf-8')
    for vector in R.TYPES:
        case = R.case_for(vector, routes)
        (out / f'{case["id"]}.json').write_text(
            json.dumps(case), encoding='utf-8'
        )
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
    tmp = tmp_path_factory.mktemp('t132c')
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


def test_the_family_is_the_five_nested_cases() -> None:
    assert IDS == [
        'nested.fixed-array',
        'nested.list-int',
        'nested.map',
        'nested.struct',
        'nested.union',
    ]


@pytest.mark.parametrize('case', per_case())
def test_each_nested_case_is_admitted(
    case: str, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        loaded = case_file(case)
        assert loaded['status'] == 'acceptance', loaded['status']
        assert loaded['routes'], 'an admitted case names its routes'
        core = loaded.get('core') or {}
        for operation in ('describe_source', 'materialize', 'predict'):
            assert core.get(operation), (
                f'the case carries {operation} to the core'
            )

    gated(request, check)


@pytest.mark.parametrize('case', per_case())
def test_each_nested_case_expects_what_python_ref_observed(
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
def test_python_core_answers_each_nested_case_as_python_ref(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        ours = runs['on'][case]['outcome']
        theirs = reference()[case]['outcome']
        fields = ('result', 'stage', 'error', 'code')
        differs = {
            k: (ours.get(k), theirs.get(k))
            for k in fields
            if ours.get(k) != theirs.get(k)
        }
        assert not differs or registered_case(case), differs

    gated(request, check)


# --- statement counts and the capture -----------------------------------------


@pytest.mark.parametrize('case', per_case())
def test_statement_counts_equal_the_reference_s_or_are_registered(
    case: str, runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')
        capture = _module(
            'capture_check', ROOT / 'contract/runners/capture_check.py'
        )
        ours = runs['on'][case]
        assert capture.complete(ours) == [], 'the capture is DuckDB query log'
        mine = compare.counts(ours)
        theirs = compare.counts(reference()[case])
        assert mine is not None and theirs is not None
        assert mine == theirs or registered_case(case), (mine, theirs)

    gated(request, check)


def _capture_check(on: Path, off: Path, required: list[str]) -> Any:
    return subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/capture_check.py'),
            '--on',
            str(on),
            '--off',
            str(off),
            *(arg for case in required for arg in ('--require', case)),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _required() -> list[str]:
    return IDS + [R.CASE_PREFIX + vector['id'] for vector in R.TYPES]


def test_capture_on_and_off_give_the_same_public_projection(
    runs: dict[str, Any],
) -> None:
    assert set(runs['on']) == set(_required())
    done = _capture_check(runs['on_path'], runs['off_path'], _required())
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'complete and conserving' in done.stdout


@pytest.mark.parametrize(
    'seeded_case', ['nested.list-int', R.CASE_PREFIX + 'list.integer']
)
def test_the_capture_check_fails_on_a_seeded_difference_in_the_family(
    seeded_case: str, runs: dict[str, Any]
) -> None:
    # Not vacuous on this family's evidence: one case's public outcome
    # changed in the capture-off run fails the check.
    seeded = runs['tmp'] / f'seeded-off-{seeded_case}.jsonl'
    lines = []
    for line in runs['off_path'].read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        if observation['case'] == seeded_case:
            observation['outcome'] = dict(
                observation['outcome'], result='refusal', code='SEEDED'
            )
        lines.append(json.dumps(observation))
    seeded.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    done = _capture_check(runs['on_path'], seeded, _required())
    assert done.returncode != 0, done.stdout
    assert seeded_case in done.stdout + done.stderr
