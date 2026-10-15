# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The corrected-reference path of the differential fails closed
(``contract/tools/differential.py``'s ``classify_corrected`` and
``compare --corrected``; T1.33 part 1's review, Codex P1).

    python -m pytest contract/tests/test_t133_corrected_controls.py -q

Written by the package's independent test author on records built here,
in the shape of Codex's reproduction (a successful rel-hm reference, the
core and the corrected reference both failing materialize with
``INTERNAL_ERROR``, their payload, predict and result absent as a failed
probe leaves them), never its data: no RelBench row is committed.

P1B-13 explains a source only when the corrected reference asks exactly
the core's request and the two then answer alike. Two crashes are no
request and no answer:

- A crash on the core or the corrected route, ``INTERNAL_ERROR`` or
  unnamed (an exception with no code), at ``from_source``,
  ``materialize`` or ``predict``, while the reference succeeded, is never
  ``approved:P1B-13`` (nor any other class): the source is UNCLASSIFIED,
  or the classifier fails. One side crashing and the other not is
  UNCLASSIFIED already, a control.
- Absent evidence is not an equal request: each stage the routes reached
  carries its evidence (a ``materialize`` that succeeded, its payload; a
  ``predict`` that succeeded, its result), or the source is UNCLASSIFIED.
- A malformed record explains nothing (the subagent's P2-3): the
  reference's record, or the core's and the corrected one alike, carries
  every request field with ``from_source`` a non-empty list, or the source
  is UNCLASSIFIED, on a source P1B-13 could explain and on one that agrees.
- The corrected record fails closed (the T1.32f plan's controls): missing,
  emptied, incomplete, asking the reference's request, differing from the
  core's, P1B-13 absent or rejected, each UNCLASSIFIED; and ``compare
  --corrected`` exits 1 for a corrected probe missing a source, empty or
  naming one twice, and for the crash seed.

Gating (``contract/tests/t133_gate.py``): a test whose id is in
``t133-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list, which may only shrink.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / 'contract/tools/differential.py'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t133_gate', ROOT / 'contract/tests/t133_gate.py')
differential = _module('t133_corrected_differential', TOOL)

#: A crash: the reference's internal error, and an exception with no code.
CRASHES = {
    'internal-error': ['StructuredDataError', 'INTERNAL_ERROR', None],
    'unnamed': ['RuntimeError', None, None],
}
STAGES = ('from_source', 'materialize', 'predict')


def gated(request: pytest.FixtureRequest, check: Callable[[], None]) -> None:
    r"""Runs ``check`` held to the pending list, as ``py:<test id>``."""
    G.held('exits', f'py:{request.node.name}', check)


def committed() -> dict[str, dict[str, Any]]:
    register: dict[str, dict[str, Any]] = differential._register(REGISTER)
    return register


def p1b_13(status: str = 'approved') -> dict[str, dict[str, Any]]:
    return {
        'P1B-13': {
            'id': 'P1B-13',
            'status': status,
            'new_behavior': 'a null foreign key',
        }
    }


def _payload(label: bool) -> dict[str, Any]:
    return {
        'context': {
            'instance_table': {
                'columns': ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
                'rows': [[{'number': '0'}, {'number': '2'}, label]],
            }
        }
    }


def succeeded() -> dict[str, Any]:
    r"""A source's record through predict's result: the reference's."""
    return {
        'id': 's00',
        'from_source': ['ok'],
        'edits': [],
        'schema': {'tables': [['e', 'id', None, []]], 'links': []},
        'materialize': ['ok'],
        'payload': _payload(True),
        'warnings': ['context examples'],
        'seeds': [[2, 1, 0]],
        'predict': ['ok'],
        'result': ['digest'],
    }


def crashed(stage: str, crash: list[Any]) -> dict[str, Any]:
    r"""The reference's record with ``stage`` failed by ``crash``, and what
    a failed probe leaves out after it."""
    record = succeeded()
    record[stage] = list(crash)
    after = {
        'from_source': (
            'edits',
            'schema',
            'materialize',
            'payload',
            'warnings',
            'seeds',
            'predict',
            'result',
        ),
        'materialize': ('payload', 'predict', 'result'),
        'predict': ('result',),
    }[stage]
    for key in after:
        record.pop(key, None)
    if stage == 'predict':
        # A request of its own, which the reference's is not: the corrected
        # reference asks it (P1B-13's shape), then both crash on it.
        record['payload'] = _payload(False)
    return record


def never_explained(
    old: dict[str, Any],
    core: dict[str, Any],
    corrected: Any,
    register: dict[str, dict[str, Any]],
) -> None:
    r"""The source is UNCLASSIFIED, or the classifier fails; never a
    class."""
    try:
        kind, why = differential.classify_corrected(
            old, core, corrected, [0], register, 0
        )
    except Exception:  # noqa: BLE001 - failing is fail-closed too
        return
    assert kind == 'UNCLASSIFIED', (kind, why)


# --- crashes ----------------------------------------------------------------------


def test_codex_s_seed_is_never_approved_p1b_13(
    request: pytest.FixtureRequest,
) -> None:
    # Codex P1: the reference succeeds; the core and the corrected
    # reference both fail materialize with INTERNAL_ERROR.
    old = succeeded()
    core = crashed('materialize', CRASHES['internal-error'])
    corrected = copy.deepcopy(core)

    def check() -> None:
        for register in (p1b_13(), committed()):
            never_explained(old, core, corrected, register)

    gated(request, check)


@pytest.mark.parametrize('crash', sorted(CRASHES))
@pytest.mark.parametrize('stage', STAGES)
def test_a_crash_on_both_core_and_corrected_explains_nothing(
    stage: str, crash: str, request: pytest.FixtureRequest
) -> None:
    old = succeeded()
    core = crashed(stage, CRASHES[crash])
    corrected = copy.deepcopy(core)

    def check() -> None:
        never_explained(old, core, corrected, p1b_13())

    gated(request, check)


@pytest.mark.parametrize('crash', sorted(CRASHES))
@pytest.mark.parametrize('stage', STAGES)
@pytest.mark.parametrize('side', ['core', 'corrected'])
def test_a_crash_on_one_side_explains_nothing(
    side: str, stage: str, crash: str
) -> None:
    # The control: one side crashing, the other answering the corrected
    # request (P1B-13's shape, a label of its own).
    old = succeeded()
    answered = succeeded()
    answered['payload'] = _payload(False)
    failed = crashed(stage, CRASHES[crash])
    core, corrected = (
        (failed, answered) if side == 'core' else (answered, failed)
    )
    never_explained(old, core, corrected, p1b_13())


# --- evidence of each stage reached --------------------------------------------


def _without(key: str, *, absent: bool) -> dict[str, Any]:
    r"""P1B-13's shape (a label of its own), every stage reached, ``key``
    absent or null."""
    record = succeeded()
    record['payload'] = _payload(False)
    if absent:
        del record[key]
    else:
        record[key] = None
    return record


@pytest.mark.parametrize('form', ['absent', 'null'])
@pytest.mark.parametrize('key', ['payload', 'result'])
def test_a_reached_stage_without_its_evidence_explains_nothing(
    key: str, form: str, request: pytest.FixtureRequest
) -> None:
    # materialize succeeded without its payload, or predict without its
    # result, on the core and the corrected reference alike: equal absence
    # is no equal request or answer.
    old = succeeded()
    core = _without(key, absent=form == 'absent')
    corrected = copy.deepcopy(core)

    def check() -> None:
        never_explained(old, core, corrected, p1b_13())

    gated(request, check)


def test_a_schema_absent_after_from_source_explains_nothing() -> None:
    old = succeeded()
    core = _without('schema', absent=True)
    never_explained(old, core, copy.deepcopy(core), p1b_13())


# --- the corrected record -------------------------------------------------------


def genuine() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r"""P1B-13's shape: the reference's request differs from the core's in
    a context label; the corrected reference asks exactly the core's and
    answers as it does."""
    old = succeeded()
    core = succeeded()
    core['payload'] = _payload(False)
    return old, core, copy.deepcopy(core)


def test_the_genuine_shape_is_approved_p1b_13() -> None:
    # The positive control the fail-closed checks below are read against.
    old, core, corrected = genuine()
    for register in (p1b_13(), committed()):
        kind, why = differential.classify_corrected(
            old, core, corrected, [0], register, 0
        )
        assert kind == 'approved:P1B-13', why


def _incomplete() -> dict[str, Any]:
    record = genuine()[2]
    del record['materialize']
    return record


def _asks_the_reference_s() -> dict[str, Any]:
    return genuine()[0]


def _differs_from_the_core_s() -> dict[str, Any]:
    record = genuine()[2]
    record['seeds'] = [[2, 1, 1]]
    return record


CORRECTED = {
    'missing': lambda: None,
    'emptied': lambda: {},
    'retyped': lambda: [genuine()[2]],
    'incomplete': _incomplete,
    "the reference's request": _asks_the_reference_s,
    "not the core's request": _differs_from_the_core_s,
}


@pytest.mark.parametrize('fault', sorted(CORRECTED))
def test_a_faulty_corrected_record_explains_nothing(fault: str) -> None:
    old, core, _ = genuine()
    never_explained(old, core, CORRECTED[fault](), p1b_13())


@pytest.mark.parametrize('register', ['absent', 'rejected', 'retyped'])
def test_p1b_13_absent_or_rejected_explains_nothing(register: str) -> None:
    old, core, corrected = genuine()
    entries = {
        'absent': {},
        'rejected': p1b_13('rejected'),
        'retyped': {'P1B-13': None},
    }[register]
    never_explained(old, core, corrected, entries)


# --- compare --corrected --------------------------------------------------------


def _compare(
    tmp: Path, old: Any, core: Any, corrected: Any
) -> subprocess.CompletedProcess[str]:
    paths = []
    for name, records in (
        ('old', old),
        ('core', core),
        ('corrected', corrected),
    ):
        path = tmp / f'{name}.json'
        path.write_text(json.dumps(records), encoding='utf-8')
        paths.append(path)
    return subprocess.run(
        [
            sys.executable,
            '-I',
            str(TOOL),
            'compare',
            str(paths[0]),
            str(paths[1]),
            '--corrected',
            str(paths[2]),
            '--register',
            str(REGISTER),
        ],
        capture_output=True,
        text=True,
        cwd=tmp,
    )


def test_compare_names_the_genuine_shape_p1b_13(tmp_path: Path) -> None:
    old, core, corrected = genuine()
    done = _compare(tmp_path, [old], [core], [corrected])
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'approved:P1B-13' in done.stdout


@pytest.mark.parametrize('fault', ['missing a source', 'empty', 'doubled'])
def test_compare_fails_a_faulty_corrected_probe(
    fault: str, tmp_path: Path
) -> None:
    old, core, corrected = genuine()
    other = {**copy.deepcopy(old), 'id': 's01'}
    probes = {
        'missing a source': [corrected],
        'empty': [],
        'doubled': [corrected, corrected, {**corrected, 'id': 's01'}],
    }[fault]
    done = _compare(tmp_path, [old, other], [core, other], probes)
    assert done.returncode == 1, done.stdout + done.stderr


def test_compare_fails_codex_s_seed(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    old = succeeded()
    core = crashed('materialize', CRASHES['internal-error'])

    def check() -> None:
        done = _compare(tmp_path, [old], [core], [copy.deepcopy(core)])
        assert done.returncode != 0, done.stdout
        assert ':P1B-13' not in done.stdout, done.stdout

    gated(request, check)


# --- malformed records (T1.33 part 1's subagent review, P2-3) ------------------
#
# Both the reference's record and the corrected one carry every request
# field, from_source a non-empty list, or the source is UNCLASSIFIED: a
# malformed reference record always "differs", so it must explain nothing,
# on a source P1B-13 could explain and on one that agrees outright.

REQUEST_KEYS = (
    'from_source',
    'edits',
    'schema',
    'materialize',
    'payload',
    'seeds',
    'warnings',
)


def _only_id(record: dict[str, Any]) -> dict[str, Any]:
    return {'id': record['id']}


def _dropped(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def drop(record: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in record.items() if k != key}

    return drop


def _from_source(value: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def set_it(record: dict[str, Any]) -> dict[str, Any]:
        return {**record, 'from_source': value}

    return set_it


MALFORMED = {
    'only its id': _only_id,
    'from_source retyped': _from_source('ok'),
    'from_source empty': _from_source([]),
    **{f'no {key}': _dropped(key) for key in REQUEST_KEYS},
}


def agreeing() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r"""A source the routes answer alike, nothing about null keys in it."""
    return succeeded(), succeeded(), succeeded()


SHAPES = {'genuine': genuine, 'agreeing': agreeing}


@pytest.mark.parametrize('shape', sorted(SHAPES))
@pytest.mark.parametrize('fault', sorted(MALFORMED))
def test_a_malformed_reference_record_explains_nothing(
    fault: str, shape: str, request: pytest.FixtureRequest
) -> None:
    old, core, corrected = SHAPES[shape]()
    old = MALFORMED[fault](old)

    def check() -> None:
        for register in (p1b_13(), committed()):
            never_explained(old, core, corrected, register)

    gated(request, check)


@pytest.mark.parametrize('fault', sorted(MALFORMED))
def test_a_malformed_corrected_and_core_record_explains_nothing(
    fault: str, request: pytest.FixtureRequest
) -> None:
    # The same fault on the core's record and the corrected one, which are
    # then equal: equal malformation is no request.
    old, core, _ = genuine()
    core = MALFORMED[fault](core)

    def check() -> None:
        never_explained(old, core, copy.deepcopy(core), p1b_13())

    gated(request, check)
