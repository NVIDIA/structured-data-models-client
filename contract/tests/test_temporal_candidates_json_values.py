# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public regressions for temporal candidates, semantic types and JSON.

Expected values follow authored candidate and source-data rules.
PostgreSQL scenarios require SDM_POSTGRES_TEST_DSN, and CI can require them
with SDM_REQUIRE_POSTGRES.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / 'contract/support/temporal_candidate_scenarios.py'


def scenarios() -> ModuleType:
    name = 'temporal_candidate_scenarios'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCENARIOS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _dsn() -> str | None:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        return None
    return dsn


@functools.cache
def core() -> dict[str, Any]:
    """Observe each scenario through the current public client."""
    return scenarios().observe(dsn=_dsn())


# Selected input positions express candidate identity and absent-key behavior.
CANDIDATE_CASES = {
    'dates-repeated': (2, [0, 2]),
    'dates-distinct': (2, [0, 1]),
    'aware-one-instant': (2, [0, 2]),
    'aware-distinct': (2, [0, 1]),
    'zone-fold': (2, [0, 2]),
    'zone-against-utc': (3, [0, 1, 2]),
    'same-text-two-zones': (3, [0, 1, 2]),
    'times-one-instant': (2, [0, 2]),
    'timestamp-beside-datetime64': (2, [2]),
    'naive-beside-aware': (3, [1, 2]),
}


def test_authored_candidate_cases_are_all_exercised():
    assert set(CANDIDATE_CASES) == set(scenarios().CANDIDATES)


@pytest.mark.parametrize('name', CANDIDATE_CASES)
def test_temporal_candidates_preserve_identity_and_order(name: str) -> None:
    parts, positions = CANDIDATE_CASES[name]
    inputs = scenarios().CANDIDATES[name][1]()
    candidates = [inputs[i] for i in positions]
    got = core()['candidates'][name]
    assert got['refusal'] is None
    assert got['parts'] == parts
    assert got['asked'] == [
        f'PREDICT COUNT(buys.* WHERE buys.item_id = {json.dumps(str(candidate))}, 0, 30, days) > 0 FOR EACH users.user_id'
        for candidate in candidates
    ]
    assert len(got['rankings']) == 2
    for row, ranking in enumerate(got['rankings']):
        scores = [
            (3 * call + 5 * row) % 9 + 1 for call in range(len(candidates))
        ]
        winners = sorted(
            range(len(candidates)), key=lambda call: -scores[call]
        )[:2]
        assert len(ranking) == len(winners)
        for item, call in zip(ranking, winners, strict=True):
            candidate = candidates[call]
            assert item['candidate'] == {
                'type': type(candidate).__name__,
                'text': str(candidate),
            }
            assert type(item['score']) is float
            assert item['score'] == scores[call] / 10


def test_a_numeric_semantic_type_has_a_useful_refusal() -> None:
    refusal = core()['stype']['refusal']
    assert refusal['error'] == 'StructuredDataError'
    assert refusal['code'] == 'INVALID_REQUEST'
    assert refusal['message'].startswith(
        '[INVALID_REQUEST] 5 is not a semantic type; use one of '
    )
    for name in (
        'ID',
        'categorical',
        'multicategorical',
        'numerical',
        'sequence',
        'text',
        'timestamp',
    ):
        assert repr(name) in refusal['message']


def test_a_filter_reading_a_double_past_float32_warns() -> None:
    assert core()['where']['refusal'] is None
    assert core()['where']['runtime_warnings'] == [
        'overflow encountered in cast'
    ]


def test_json_documents_preserve_escaped_lone_surrogates() -> None:
    if _dsn() is None:
        pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')
    got = core()['json']
    assert got['refusal'] is None
    assert len(got['bodies']) == 1
    body = got['bodies'][0]
    table_name = scenarios().JSON_TABLE
    assert body['task'] == {
        'kind': 'regression',
        'target': {'column_name': 'TARGET', 'dtype': 'int64'},
        'entity_table_names': [table_name],
    }
    assert body['model'] == 'kumo-relational'
    assert body['output'] == {
        'fields': ['prediction'],
        'prediction_statistic': 'median',
    }
    assert body['inference'] == {
        'run_mode': 'fast',
        'use_prediction_time': False,
    }
    assert body['metadata'] == {
        'source_query': f'PREDICT {table_name}.age FOR EACH {table_name}.id',
        'num_context_examples': 8,
        'num_prediction_examples': 2,
    }
    expected_docs = [
        '{"s":"\\ud800"}',
        '{"\\udc00":"x"}',
        '["\\ud800","\\ud83d\\ude00"]',
        '{"s":"ok"}',
    ]
    for half in ('context', 'predict'):
        instance = body[half]['instance_table']
        assert instance['format'] == 'arrays'
        assert instance['columns'] == ['instance_id', '__kumo_entity_ref_0'] + (
            ['TARGET'] if half == 'context' else []
        )
        rows = instance['rows']
        assert len(rows) == (6 if half == 'context' else 2)
        assert {r[1] for r in rows} == (
            set(range(2, 8)) if half == 'context' else {0, 1}
        )
        assert len({r[0] for r in rows}) == len(rows)
        for row in rows:
            assert all(type(v) is int for v in row)
            if half == 'context':
                assert row[2] == 20 + row[1]
        if half == 'predict':
            assert [r[1] for r in rows] == [0, 1]
        related = body[half]['related_tables']
        assert set(related) == {table_name}
        assert related[table_name]['columns'] == ['instance_id', 'id', 'doc']
        assert related[table_name]['rows'] == [
            [r[0], r[1], expected_docs[r[1] % 4]] for r in rows
        ]
    schema = body['schema']
    assert set(schema) == {'instance_table', 'related_tables', 'relationships'}
    assert schema['instance_table']['columns'] == {
        'instance_id': {'dtype': 'int64', 'stype': 'ID', 'nullable': False},
        '__kumo_entity_ref_0': {'dtype': 'int64', 'stype': 'ID'},
        'TARGET': {'dtype': 'int64', 'stype': 'numerical'},
    }
    assert set(schema['related_tables']) == {table_name}
    assert schema['related_tables'][table_name]['columns'] == {
        'instance_id': {'dtype': 'int64', 'stype': 'ID', 'nullable': False},
        'id': {'dtype': 'int64', 'nullable': False},
        'doc': {'dtype': 'string', 'stype': 'categorical'},
    }
    assert schema['instance_table']['primary_key'] == 'instance_id'
    assert schema['related_tables'][table_name]['primary_key'] == [
        'instance_id',
        'id',
    ]
    assert schema['related_tables'][table_name]['columns']['doc'] == {
        'dtype': 'string',
        'stype': 'categorical',
    }
    assert schema['relationships'] == [
        {
            'source_columns': ['__kumo_entity_ref_0'],
            'target_table': table_name,
            'target_columns': ['id'],
        }
    ]


# --- FLOAT32_OVERFLOW and the host NumPy version ---------------------------


def _codes_reported(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    r"""Each warning code the core reports, as the host takes its
    warnings at ``sdm_client._native``'s boundary (the one function through
    which the host imports the native module)."""
    from sdm_client._rust import codec, run

    imported = run.native
    codes: list[str] = []

    class Handle:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def take_warnings(self) -> Any:
            taken = self._handle.take_warnings()
            if taken is not None:
                control, sent = taken
                read = codec.read(control, sent)
                codes.extend(w['code'] for w in read.get('warnings', []))
            return taken

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    class Run:
        def __init__(self, native: Any) -> None:
            self._native = native

        def start(self, message: bytes, attachments: Any) -> Handle:
            return Handle(self._native.Run.start(message, attachments))

    class Native:
        def __init__(self, native: Any) -> None:
            self._native = native
            self.Run = Run(native)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._native, name)

    monkeypatch.setattr(run, 'native', lambda: Native(imported()))
    return codes


@pytest.mark.parametrize(
    ('numpy', 'shown'),
    [('1.23.5', False), ('1.24.0', True), ('2.3.4', True)],
)
def test_float32_overflow_is_shown_where_the_runtime_s_numpy_warns(
    numpy: str, shown: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""The core reports a double past
    float32's range (``FLOAT32_OVERFLOW``) on every runtime, and the Python
    host shows it as ``RuntimeWarning`` ("overflow encountered in cast")
    only where the installed NumPy is 1.24 or later, which first warns on
    such a cast; NumPy 1.23 casts
    silently. The version the host's function reads is set either side of
    1.24; the core's fact is there either way."""
    import numpy as np
    from sdm_client._rust import errors

    monkeypatch.setattr(np, '__version__', numpy)
    assert errors.numpy_warns_on_cast_overflow() is shown
    codes = _codes_reported(monkeypatch)
    observed = scenarios().where()
    assert observed['refusal'] is None, observed
    assert 'FLOAT32_OVERFLOW' in codes, codes
    assert observed['runtime_warnings'] == (
        ['overflow encountered in cast'] if shown else []
    ), (numpy, observed['runtime_warnings'])
