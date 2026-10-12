# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The old-against-core differential's classification (T1.27,
``contract/tools/differential.py``), on records written here: the tool is
a local pre-push probe, not a gate, and this holds only that it fails
closed. A difference no register entry covers is UNCLASSIFIED; so is
evidence missing, empty or retyped (labels absent where the core refuses
after them, a label of another type, an entry that does not name the rule,
probes of other sources), each seeded below.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    'differential', ROOT / 'contract/tools/differential.py'
)
assert _spec is not None and _spec.loader is not None
differential = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(differential)

REGISTER = {
    'P1B-3': {
        'id': 'P1B-3',
        'status': 'proposed',
        'new_behavior': 'classification_task',
    },
    'P1B-18': {
        'id': 'P1B-18',
        'status': 'approved',
        'new_behavior': 'target_type',
    },
}


def _payload() -> dict[str, Any]:
    return {
        'context': {
            'instance_table': {
                'columns': [
                    'instance_id',
                    '__kumo_entity_ref_0',
                    '__kumo_anchor_time',
                    'TARGET',
                ],
                'rows': [
                    [
                        {'number': '0'},
                        {'number': '2'},
                        '2024-01-31T00:00:00.000000Z',
                        True,
                    ],
                    [
                        {'number': '1'},
                        {'number': '7'},
                        '2024-01-31T00:00:00.000000Z',
                        False,
                    ],
                ],
            }
        },
        'metadata': {'num_context_examples': {'number': '3'}},
    }


def _old() -> dict[str, Any]:
    return {
        'id': 'g00',
        'from_source': ['ok'],
        'materialize': ['ok'],
        'payload': _payload(),
        'warnings': [],
        'seeds': [[2, 1, 0]],
        'predict': ['ok'],
        'result': ['digest'],
    }


def _core() -> dict[str, Any]:
    return {
        'id': 'g00',
        'from_source': ['ok'],
        'materialize': [
            'UnsupportedRequestError',
            'UNSUPPORTED_REQUEST',
            'classification_task',
        ],
        'warnings': [],
        'seeds': [[2, 1, 0]],
        # The third example is one predicted for, which the reference's
        # static context leaves out.
        'labels': [
            [2, '2024-01-31T00:00:00.000000Z', True],
            [7, '2024-01-31T00:00:00.000000Z', False],
            [0, '2024-01-31T00:00:00.000000Z', True],
        ],
    }


def test_a_refusal_after_equal_labels_is_its_entry_s() -> None:
    kind, _ = differential.classify(_old(), _core(), [0, 1], REGISTER)
    assert kind == 'proposed:P1B-3'


def test_equal_requests_agree_and_any_difference_is_unclassified() -> None:
    old = _old()
    same = copy.deepcopy(old)
    assert differential.classify(old, same, [0, 1], REGISTER)[0] == 'agree'
    for field, value in [
        ('payload', {}),
        ('seeds', []),
        ('warnings', ['Missing']),
        ('result', ['other']),
    ]:
        changed = {**copy.deepcopy(old), field: value}
        assert (
            differential.classify(old, changed, [0, 1], REGISTER)[0]
            == 'UNCLASSIFIED'
        )


def test_the_classification_fails_closed() -> None:
    old = _old()
    cases = []
    flipped = _core()
    flipped['labels'][0][2] = False
    cases.append(flipped)
    retyped = _core()
    retyped['labels'][1][2] = 0
    cases.append(retyped)
    missing = _core()
    del missing['labels']
    cases.append(missing)
    empty = _core()
    empty['labels'] = []
    cases.append(empty)
    failed = _core()
    failed['labels'] = ['StructuredDataError', 'INTERNAL_ERROR', None]
    cases.append(failed)
    not_predicted = _core()
    not_predicted['labels'][2][0] = 9
    cases.append(not_predicted)
    unknown_rule = _core()
    unknown_rule['materialize'][2] = 'a_rule_no_entry_names'
    cases.append(unknown_rule)
    for core in cases:
        assert (
            differential.classify(old, core, [0, 1], REGISTER)[0]
            == 'UNCLASSIFIED'
        ), core
    # An entry that does not name the rule, or is not in the register.
    unnamed = {**REGISTER, 'P1B-3': {'id': 'P1B-3', 'status': 'proposed'}}
    assert (
        differential.classify(old, _core(), [0, 1], unnamed)[0]
        == 'UNCLASSIFIED'
    )
    gone = {k: v for k, v in REGISTER.items() if k != 'P1B-3'}
    assert (
        differential.classify(old, _core(), [0, 1], gone)[0] == 'UNCLASSIFIED'
    )
    # The core taking what the reference refuses.
    refused = {
        **old,
        'materialize': ['StructuredDataError', 'INVALID_REQUEST', None],
    }
    taken = {**_core(), 'materialize': ['ok']}
    assert (
        differential.classify(refused, taken, [0, 1], REGISTER)[0]
        == 'UNCLASSIFIED'
    )


def test_an_internal_error_named_by_the_core_is_e66_b_s_class() -> None:
    old = {
        **_old(),
        'materialize': ['StructuredDataError', 'INTERNAL_ERROR', None],
    }
    core = {
        **_core(),
        'materialize': [
            'UnsupportedRequestError',
            'UNSUPPORTED_REQUEST',
            'target_type',
        ],
    }
    assert (
        differential.classify(old, core, [0, 1], REGISTER)[0]
        == 'approved:P1B-18'
    )
    other = {
        **core,
        'materialize': ['StructuredDataError', 'INVALID_REQUEST', None],
    }
    assert (
        differential.classify(old, other, [0, 1], REGISTER)[0] == 'UNCLASSIFIED'
    )


def test_probes_of_other_sources_fail(tmp_path: Path) -> None:
    old, core = tmp_path / 'old.json', tmp_path / 'core.json'
    old.write_text(json.dumps([_old()]))
    core.write_text(json.dumps([{**_core(), 'id': 'g01'}]))
    register = tmp_path / 'register.yaml'
    register.write_text('entries: []\n')
    assert differential.compare(old, core, None, register) == 1
    core.write_text('[]')
    old.write_text('[]')
    assert differential.compare(old, core, None, register) == 1
