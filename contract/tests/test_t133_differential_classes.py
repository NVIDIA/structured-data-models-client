# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.33's changes to the differential's classifier, seeded
(``contract/tools/differential.py``; T1.32f's real-data part).

    python -m pytest contract/tests/test_t133_differential_classes.py -q

Written by the package's independent test author on records built here,
after the implementer's hand checks, so they are committed seeds:

- ``DESCRIBED_RULES``: the core refusing ``text_time`` at
  ``Schema.from_source`` where the reference describes is DT-15's, approved
  (T1.33 part 3, the owner's item 24: "text"; nothing of it is pending the
  owner, gated by ``t133p3_gate.py``); it is the only rule left. ``multicategorical`` (ported) and
  ``time_whole_table`` (fixed), both by T1.32e (E92), are differences again,
  UNCLASSIFIED even under an entry that names them.
- A rejected entry classifies nothing: P1B-11 approved gives
  ``approved:P1B-11`` for an empty table the core refuses by name; the
  same with P1B-11 rejected is UNCLASSIFIED, "is rejected".
- ``classify_corrected``: a source the corrected reference explains is
  ``approved:P1B-13`` while P1B-13 is approved, and UNCLASSIFIED once it
  is rejected.
- ``CODE_CHANGES``: a refusal by ``wire_dtype`` under another code where
  the reference's is ``INVALID_REQUEST`` reaches P1B-19, which the register
  rejects, so it is UNCLASSIFIED; under a register where P1B-19 is
  proposed, the same records are ``proposed:P1B-19``, so the path is
  reached.

Each class is also read against the committed register
(``docs/design/rust-core/register.yaml``), so a change of an entry's
status shows here.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    't133_differential', ROOT / 'contract/tools/differential.py'
)
assert _spec is not None and _spec.loader is not None
differential = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(differential)

REFUSED = 'UnsupportedRequestError'


def committed() -> dict[str, dict[str, Any]]:
    register: dict[str, dict[str, Any]] = differential._register(
        ROOT / 'docs/design/rust-core/register.yaml'
    )
    return register


def entry(name: str, status: str, names: str) -> dict[str, dict[str, Any]]:
    r"""A register holding ``name`` alone, with ``status``, naming the
    rules in ``names``."""
    return {name: {'id': name, 'status': status, 'new_behavior': names}}


def with_status(
    register: dict[str, dict[str, Any]], name: str, status: str
) -> dict[str, dict[str, Any]]:
    changed = copy.deepcopy(register)
    changed[name]['status'] = status
    return changed


def _payload(label: bool) -> dict[str, Any]:
    return {
        'context': {
            'instance_table': {
                'columns': ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
                'rows': [[{'number': '0'}, {'number': '2'}, label]],
            }
        }
    }


def _record(**changes: Any) -> dict[str, Any]:
    r"""A source's record through predict, both routes alike unless
    ``changes`` says otherwise."""
    record = {
        'id': 's00',
        'from_source': ['ok'],
        'edits': [['ok']],
        'schema': {'tables': [['e', 'id', None, []]], 'links': []},
        'materialize': ['ok'],
        'payload': _payload(True),
        'warnings': [],
        'seeds': [[2, 1, 0]],
        'predict': ['ok'],
        'result': ['digest'],
    }
    record.update(changes)
    return record


# --- DESCRIBED_RULES ------------------------------------------------------------


def _described(rule: str) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""The reference describes the source; the core refuses it by
    ``rule``."""
    old = {'id': 's00', 'from_source': ['ok']}
    core = {'id': 's00', 'from_source': [REFUSED, 'UNSUPPORTED_REQUEST', rule]}
    return old, core


def _p3() -> Any:
    spec = importlib.util.spec_from_file_location(
        't133p3_gate', ROOT / 'contract/tests/t133p3_gate.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _held(name: str, check: Any) -> None:
    _p3().held('exits', f'py:test_t133_differential_classes::{name}', check)


def test_text_time_is_the_only_described_rule() -> None:
    # T1.33 part 3 (the owner's item 24): text_time is DT-15's, and nothing
    # of it is pending the owner.
    def check() -> None:
        assert differential.DESCRIBED_RULES == {'text_time': 'DT-15'}
        assert ('P1B-2', 'text_time') not in differential.PENDING_OWNER

    _held('test_text_time_is_the_only_described_rule', check)


def test_a_core_refusing_text_time_is_dt_15_s() -> None:
    old, core = _described('text_time')

    def check() -> None:
        for register in (committed(), entry('DT-15', 'approved', 'text_time')):
            kind, why = differential.classify(old, core, [0], register, 0)
            assert kind == 'approved:DT-15', (kind, why)
        # Under DT-15 proposed, a proposed entry the review sees.
        kind, _ = differential.classify(
            old, core, [0], entry('DT-15', 'proposed', 'text_time'), 0
        )
        assert kind == 'proposed:DT-15'

    _held('test_a_core_refusing_text_time_is_dt_15_s', check)


@pytest.mark.parametrize('rule', ['multicategorical', 'time_whole_table'])
def test_multicategorical_and_time_whole_table_are_unclassified(
    rule: str,
) -> None:
    # Even under an approved P1B-2 that names the rule: no longer a
    # described rule, the refusal is a difference.
    old, core = _described(rule)
    register = entry(
        'P1B-2', 'approved', 'text_time multicategorical time_whole_table'
    )
    for under in (register, committed()):
        kind, why = differential.classify(old, core, [0], under, 0)
        assert kind == 'UNCLASSIFIED', (rule, why)


def test_a_text_time_refusal_fails_closed() -> None:
    old, core = _described('text_time')

    def check() -> None:
        # DT-15 rejected, not naming the rule, or not in the register (an
        # umbrella naming it classifies nothing).
        unclassified = [
            entry('DT-15', 'rejected', 'text_time'),
            entry('DT-15', 'approved', 'multicategorical'),
            entry('P1B-2', 'approved', 'text_time'),
        ]
        for register in unclassified:
            kind, _ = differential.classify(old, core, [0], register, 0)
            assert kind == 'UNCLASSIFIED', register
        kind, why = differential.classify(
            old, core, [0], entry('DT-15', 'rejected', 'text_time'), 0
        )
        assert 'DT-15 is rejected' in why
        # Under another code, it is no described refusal.
        other = {
            **core,
            'from_source': [REFUSED, 'INVALID_REQUEST', 'text_time'],
        }
        kind, _ = differential.classify(
            old, other, [0], entry('DT-15', 'approved', 'text_time'), 0
        )
        assert kind == 'UNCLASSIFIED'

    _held('test_a_text_time_refusal_fails_closed', check)


# --- a rejected entry -----------------------------------------------------------


def _empty_table() -> tuple[dict[str, Any], dict[str, Any]]:
    r"""An empty table: the reference fails unnamed, the core refuses it
    by name (E66(b)'s class), INVALID_REQUEST since the owner's switch of
    P1B-11 (E97, T1.32g)."""
    old = {'id': 's00', 'from_source': ['RuntimeError', None, None]}
    core = {
        'id': 's00',
        'from_source': [
            'StructuredDataError',
            'INVALID_REQUEST',
            'empty_table',
        ],
    }
    return old, core


def test_p1b_11_approved_classes_an_empty_table_and_rejected_does_not() -> None:
    old, core = _empty_table()
    approved = entry('P1B-11', 'approved', 'empty_table')
    kind, _ = differential.classify(old, core, [0], approved, 0)
    assert kind == 'approved:P1B-11'
    rejected = with_status(approved, 'P1B-11', 'rejected')
    kind, why = differential.classify(old, core, [0], rejected, 0)
    assert kind == 'UNCLASSIFIED'
    assert 'P1B-11 is rejected' in why
    # The committed register: P1B-11 as it is there, then rejected.
    register = committed()
    assert register['P1B-11']['status'] == 'approved'
    assert differential.classify(old, core, [0], register, 0)[0] == (
        'approved:P1B-11'
    )
    kind, why = differential.classify(
        old, core, [0], with_status(register, 'P1B-11', 'rejected'), 0
    )
    assert kind == 'UNCLASSIFIED' and 'is rejected' in why


# --- classify_corrected ---------------------------------------------------------


def _corrected() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r"""The reference's request differs from the core's in a context label
    (P1B-13's shape: a null key taken for the last one looked up); the
    corrected reference asks exactly the core's, and answers as it does."""
    old = _record(payload=_payload(True))
    core = _record(payload=_payload(False))
    corrected = copy.deepcopy(core)
    return old, core, corrected


def test_classify_leaves_the_seed_unclassified() -> None:
    old, core, _ = _corrected()
    kind, why = differential.classify(old, core, [0], committed(), 1)
    assert kind == 'UNCLASSIFIED' and 'payload' in why


def test_p1b_13_approved_classes_the_corrected_source() -> None:
    old, core, corrected = _corrected()
    register = entry('P1B-13', 'approved', 'null foreign key')
    kind, _ = differential.classify_corrected(
        old, core, corrected, [0], register, 1
    )
    assert kind == 'approved:P1B-13'
    assert committed()['P1B-13']['status'] == 'approved'
    kind, _ = differential.classify_corrected(
        old, core, corrected, [0], committed(), 1
    )
    assert kind == 'approved:P1B-13'


def test_p1b_13_rejected_leaves_the_corrected_source_unclassified() -> None:
    old, core, corrected = _corrected()
    for register in (
        entry('P1B-13', 'rejected', 'null foreign key'),
        with_status(committed(), 'P1B-13', 'rejected'),
    ):
        kind, why = differential.classify_corrected(
            old, core, corrected, [0], register, 1
        )
        assert kind == 'UNCLASSIFIED'
        assert 'P1B-13 is rejected' in why


# --- CODE_CHANGES ---------------------------------------------------------------


def _wire_dtype() -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Both routes refuse materialize: the reference with
    ``INVALID_REQUEST``, the core by ``wire_dtype`` under another code."""
    old = _record(materialize=['StructuredDataError', 'INVALID_REQUEST', None])
    core = _record(materialize=[REFUSED, 'UNSUPPORTED_REQUEST', 'wire_dtype'])
    for record in (old, core):
        for key in ('payload', 'predict', 'result'):
            del record[key]
    return old, core


def test_wire_dtype_under_another_code_is_code_changes_p1b_19() -> None:
    assert differential.CODE_CHANGES == {
        ('wire_dtype', 'INVALID_REQUEST'): 'P1B-19'
    }
    old, core = _wire_dtype()
    # The path is reached: under a P1B-19 that is proposed, these records
    # are its.
    proposed = entry('P1B-19', 'proposed', 'wire_dtype')
    kind, why = differential.classify(old, core, [0], proposed, 1)
    assert kind == 'proposed:P1B-19', why
    assert 'another code' in why


def test_p1b_19_rejected_leaves_wire_dtype_unclassified() -> None:
    old, core = _wire_dtype()
    assert committed()['P1B-19']['status'] == 'rejected'
    for register in (
        entry('P1B-19', 'rejected', 'wire_dtype'),
        committed(),
    ):
        kind, why = differential.classify(old, core, [0], register, 1)
        assert kind == 'UNCLASSIFIED'
        assert 'P1B-19 is rejected' in why
    # Another pair of rule and code is no code change.
    other = copy.deepcopy(core)
    other['materialize'][2] = 'another_rule'
    kind, why = differential.classify(
        old, other, [0], entry('P1B-19', 'proposed', 'wire_dtype'), 1
    )
    assert kind == 'UNCLASSIFIED', why
