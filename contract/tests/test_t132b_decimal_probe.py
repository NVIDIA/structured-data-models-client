# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decimal probe's P1B-71 note: never an explanation of a difference.

    python -m pytest contract/tests/test_t132b_decimal_probe.py -q

Written by the bundle's independent test author after Codex's reviews of
T1.32g part 2b. Round 1 (P2): ``contract/tools/decimal_probe.py``'s
``EXPLAINED`` took every difference of a static float condition at a scale
of 28 or more as P1B-71's, an emptied schema among them. Round 2 (P2): its
proximity allowance still explained a label histogram swapped where no
value's label differs under either conversion. The orchestrator's decision
(conservative): the probe never explains a label or request difference; it
reports it unexplained, with a hint naming P1B-71 as a possible, unverified
cause. P1B-71 (the owner's decision of 2026-10-13: a float constant
compared with a static DECIMAL(p, s) column read from its exact binary
value, rounded half to even, where the reference's pyarrow approximates
from scale 28) stays the core's behavior; the probe only does not take it
as proof.

- Negative controls, each reported unexplained (``compare`` exits 1, the
  vector counted unexplained, no ``under P1B-71``): the schema emptied
  (round 1's repro); a payload table's rows, the feature column's
  description or the result changed; a warning added; a label flipped
  where no value of the column lies near the constant; the request changed
  with its labels unchanged; two labels flipped where one value lies at
  the constant; a histogram swapped where every value's label is the same
  under both conversions (round 2's repro).
- The former positive control, the exact conversion's own one-label
  consequence, is unexplained too, its line naming P1B-71 as a hint.
- Controls of the controls: the recorded answer agrees with itself and is
  the probe's own vector; an outcome change is never explained.

The reference's answer is ``static.gt-float-scale-30.d38-30``'s, recorded
by ``decimal_probe.py observe python-ref`` in the reference environment of
``contract/reference-env.txt`` (b413e9ad) on 2026-10-14, as ``REFERENCE``
below; recorded again with the column holding round 2's values
``[0.7, -1, 0, 0.25, 1, 2, 3.75, null]``, it is the same answer. The
constant's exact value at scale 30, 699999999999999955591079014994, was
computed apart, in Python:
``int((Decimal(0.7) * 10**30).quantize(1, ROUND_HALF_EVEN))``.

A vector names its column's values as ``values`` (decimal text, or None)
and its condition in its ``pql``
(``test_each_static_vector_names_its_values``).

Gating (``contract/tests/t132b_gate.py``): a test whose id is in
``t132b-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132b_gate', ROOT / 'contract/tests/t132b_gate.py')
P = _module('decimal_probe', ROOT / 'contract/tools/decimal_probe.py')
R = _module('decimal_record', ROOT / 'contract/tools/decimal_record.py')

VECTOR_ID = 'static.gt-float-scale-30.d38-30'
#: The reference's projected answer of VECTOR_ID (see above).
REFERENCE: dict[str, Any] = {
    'outcome': {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    },
    'schema': {
        'links': [['orders', 'user_id', 'users', 'user_id']],
        'tables': [
            {
                'columns': [
                    {'dtype': 'int64', 'name': 'order_id', 'stype': 'ID'},
                    {'dtype': 'int64', 'name': 'user_id', 'stype': 'ID'},
                    {'dtype': 'float64', 'name': 'price', 'stype': 'numerical'},
                    {
                        'dtype': 'timestamp[us]',
                        'name': 'ts',
                        'stype': 'timestamp',
                    },
                ],
                'duplicate_keys': 0,
                'name': 'orders',
                'primary_key': 'order_id',
                'time_column': 'ts',
            },
            {
                'columns': [
                    {'dtype': 'int64', 'name': 'user_id', 'stype': 'ID'},
                    {'dtype': 'int64', 'name': 'age', 'stype': 'numerical'},
                    {
                        'dtype': 'string',
                        'name': 'country',
                        'stype': 'categorical',
                    },
                    {'dtype': 'float64', 'name': 'f', 'stype': 'categorical'},
                ],
                'duplicate_keys': 0,
                'name': 'users',
                'primary_key': 'user_id',
                'time_column': None,
            },
        ],
    },
    'warnings': ['UserWarning', 'SDMWarning'],
    'payload': [
        {
            'method': 'POST',
            'path': '/v1/predictions',
            'digest': 'a93397602c500f99a18ec4edc8b49922272b4c5ebc876c8f777c3eed79202e28',
            'tables': {
                'context.instance_table': {'rows': 5, 'nulls': 0},
                'context.related_tables.orders': {'rows': 50, 'nulls': 0},
                'context.related_tables.users': {'rows': 5, 'nulls': 0},
                'predict.instance_table': {'rows': 2, 'nulls': 0},
                'predict.related_tables.orders': {'rows': 20, 'nulls': 0},
                'predict.related_tables.users': {'rows': 2, 'nulls': 0},
            },
        }
    ],
    'result': {
        'columns': [
            {'dtype': 'int64', 'name': 'row_index'},
            {'dtype': 'int64', 'name': 'id'},
            {'dtype': 'bool', 'name': 'prediction'},
        ],
        'digest': 'f0e9391d7fd6e8401c5816a7630af17c59454bf8ed01e7c54d4c0fa9c3aa054a',
        'rows': 2,
    },
    'columns': {
        'instance_table.TARGET': {
            'described': None,
            'request': [
                {
                    'context': {
                        'dtype': 'bool',
                        'stype': 'categorical',
                        'cells': {'false': 2, 'true': 3},
                    },
                    'predict': None,
                }
            ],
        },
        'users.f': {
            'described': {'dtype': 'float64', 'stype': 'categorical'},
            'request': [{'context': None, 'predict': None}],
        },
    },
}
#: The double nearest 0.7, exactly, at scale 30, rounded half to even.
EXACT_0_7_AT_30 = Decimal('0.699999999999999955591079014994')
#: The targeted vector's column values (decimal_probe.targeted's spread at
#: DECIMAL(38, 30)): none lies near 0.7.
SPREAD = ['-2.5', '-1', '0', '0.25', '1', '2', '3.75', None]


def gated(request: pytest.FixtureRequest, check: Callable[[], None]) -> None:
    G.held('exits', f'py:{request.node.name}', check)


def vector(values: list[str | None], op: str = '>') -> dict[str, Any]:
    r"""VECTOR_ID's vector, its column holding values at
    DECIMAL(38, 30), compared with 0.7 by op."""
    decimals = [None if v is None else Decimal(v) for v in values]
    return {
        'id': VECTOR_ID,
        'columns': [['instance_table', 'TARGET'], ['users', 'f']],
        'setup': [R.add_f('users', decimals, 38, 30)],
        'pql': f'PREDICT users.f {op} 0.7 FOR EACH users.user_id',
        'anchor_time': None,
        'static_condition': True,
        'scale': 30,
        'float_constant': True,
        'values': values,
    }


def judged(
    capsys: pytest.CaptureFixture[str],
    core: dict[str, Any],
    the_vector: dict[str, Any] | None = None,
) -> tuple[int, str]:
    the_vector = the_vector or vector(SPREAD)
    status = P.compare([the_vector], {VECTOR_ID: REFERENCE}, {VECTOR_ID: core})
    return status, capsys.readouterr().out


def assert_unexplained(status: int, out: str) -> None:
    assert status == 1, out
    assert '1 unexplained' in out, out
    assert 'under P1B-71' not in out, out


def changed(edit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    core = copy.deepcopy(REFERENCE)
    edit(core)
    return core


def flipped(answer: dict[str, Any]) -> None:
    r"""One context label from false to true, and the request's digest
    with it: what one row labelled otherwise gives."""
    cells = answer['columns']['instance_table.TARGET']['request'][0]
    cells = cells['context']['cells']
    cells['false'] -= 1
    cells['true'] += 1
    answer['payload'][0]['digest'] = 'f' * 64


# --- controls of the controls ------------------------------------------------


def test_the_recorded_answer_agrees_with_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert REFERENCE['outcome']['result'] == 'success'
    status, out = judged(capsys, copy.deepcopy(REFERENCE))
    assert status == 0, out
    assert '1 agree' in out, out
    # The probe's own vector is the one recorded.
    [mine] = [v for v in P.targeted() if v['id'] == VECTOR_ID]
    assert mine['pql'] == vector(SPREAD)['pql']
    assert mine['setup'] == vector(SPREAD)['setup']


def test_an_outcome_change_is_never_explained(
    capsys: pytest.CaptureFixture[str],
) -> None:
    core = changed(
        lambda a: a['outcome'].update(
            result='refusal', code='INVALID_REQUEST', stage='materialize'
        )
    )
    assert_unexplained(*judged(capsys, core))


# --- negative controls ---------------------------------------------------------


def test_an_emptied_schema_is_unexplained(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        core = changed(lambda a: a['schema'].update(tables=[]))
        assert_unexplained(*judged(capsys, core))

    gated(request, check)


@pytest.mark.parametrize(
    'part',
    ['payload-rows', 'feature-cells', 'result'],
)
def test_an_unrelated_payload_or_result_change_is_unexplained(
    part: str,
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> None:
    def edit(answer: dict[str, Any]) -> None:
        if part == 'payload-rows':
            tables = answer['payload'][0]['tables']
            tables['context.related_tables.users']['rows'] -= 1
            answer['payload'][0]['digest'] = 'e' * 64
        elif part == 'feature-cells':
            column = answer['columns']['users.f']
            column['described'] = {'dtype': 'string', 'stype': 'categorical'}
            answer['payload'][0]['digest'] = 'e' * 64
        else:
            answer['result']['rows'] += 1

    def check() -> None:
        assert_unexplained(*judged(capsys, changed(edit)))

    gated(request, check)


def test_a_warnings_change_is_unexplained(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        core = changed(lambda a: a['warnings'].append('DeprecationWarning'))
        assert_unexplained(*judged(capsys, core))

    gated(request, check)


def test_a_label_flip_with_no_value_near_the_constant_is_unexplained(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> None:
    # SPREAD holds no value within the approximation of 0.7 at scale 30,
    # so the exact reading and the reference's must label every row alike.
    def check() -> None:
        assert_unexplained(*judged(capsys, changed(flipped)))

    gated(request, check)


def test_a_request_change_without_a_label_change_is_unexplained(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        core = changed(lambda a: a['payload'][0].update(digest='d' * 64))
        assert_unexplained(*judged(capsys, core))

    gated(request, check)


def test_more_labels_flipped_than_values_near_the_constant_is_unexplained(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> None:
    # One value of the column lies at the constant's exact value, so at
    # most the rows holding it may be labelled otherwise: one user here.
    near = [str(EXACT_0_7_AT_30), '-1', '0', '0.25', '1', '2', '3.75', None]

    def twice(answer: dict[str, Any]) -> None:
        flipped(answer)
        flipped(answer)

    def check() -> None:
        assert_unexplained(
            *judged(capsys, changed(twice), vector(near, op='>='))
        )

    gated(request, check)


# --- a label difference, never explained ------------------------------------


def assert_hinted(out: str) -> None:
    r"""The vector's line names P1B-71 as a possible cause, not as an
    explanation."""
    [line] = [each for each in out.splitlines() if each.startswith(VECTOR_ID)]
    assert 'P1B-71' in line, out
    assert 'under P1B-71' not in line, out


def test_the_exact_conversion_s_own_consequence_is_unexplained_with_a_hint(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> None:
    # The column holds the constant's exact value: compared by >=, the core
    # labels that row true from the exact reading, where an approximation
    # past it labels it false. One label and the request's digest differ,
    # nothing else: still unexplained, P1B-71 named as a possible cause.
    near = [str(EXACT_0_7_AT_30), '-1', '0', '0.25', '1', '2', '3.75', None]

    def check() -> None:
        status, out = judged(capsys, changed(flipped), vector(near, op='>='))
        assert_unexplained(status, out)
        assert_hinted(out)

    gated(request, check)


def test_a_swapped_histogram_no_conversion_explains_is_unexplained(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> None:
    # Round 2's repro: users.f > 0.7 at DECIMAL(38, 30) over 0.7, -1, 0,
    # 0.25, 1, 2, 3.75 and a null. The column's 0.7 is above the exact
    # constant and the reference's alike, so no value's label differs under
    # either conversion; the context histogram swapped from false 2, true 3
    # to false 3, true 2, and the request's digest with it, is unexplained.
    values = ['0.7', '-1', '0', '0.25', '1', '2', '3.75', None]
    assert Decimal('0.7') > EXACT_0_7_AT_30

    def swapped(answer: dict[str, Any]) -> None:
        cells = answer['columns']['instance_table.TARGET']['request'][0]
        cells = cells['context']['cells']
        assert cells == {'false': 2, 'true': 3}
        cells.update(false=3, true=2)
        answer['payload'][0]['digest'] = 'c' * 64

    def check() -> None:
        status, out = judged(capsys, changed(swapped), vector(values))
        assert_unexplained(status, out)
        assert_hinted(out)

    gated(request, check)


def test_each_static_vector_names_its_values(
    request: pytest.FixtureRequest,
) -> None:
    # What the explanation reads to know which rows lie near the constant.
    def check() -> None:
        statics = [v for v in P.targeted() if v.get('static_condition')]
        statics += [v for v in P.generated(32, 50) if v.get('static_condition')]
        assert statics
        for each in statics:
            assert 'values' in each, each['id']
            assert len(each['values']) == 8, each['id']

    gated(request, check)
