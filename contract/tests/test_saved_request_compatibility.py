# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Older serialized requests and schemas remain usable by the current client.

Frozen pickle bytes exercise saved-format compatibility, including forecasts,
rankings, direct frames and schema conversion notes.
Loads, round trips and predictions retain independent expected values.
These fixtures must not be regenerated using the current implementation.
"""

from __future__ import annotations

import copy
import importlib.util
import pickle
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PICKLES = ROOT / 'contract/fixtures/request-pickles'
NAMES = {
    'request-one-call',
    'request-forecast',
    'request-ranking',
    'request-frames',
    'schema-notes',
}


def writer() -> ModuleType:
    name = 'test_saved_request_compatibility_write_request_pickles'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'tests/support/pickle_observations.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _load(name: str) -> tuple[Any, list[Any]]:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        loaded = pickle.loads((PICKLES / f'{name}.pkl').read_bytes())
    return loaded, [
        w for w in shown if not issubclass(w.category, ResourceWarning)
    ]


def test_saved_pickle_formats_and_cases_are_covered() -> None:
    """Keep every saved-format fixture and the scenarios it exercises."""
    written = {path.stem for path in PICKLES.glob('*.pkl')}
    assert written == NAMES
    assert all(
        p.read_bytes().startswith(b'\x80\x04') for p in PICKLES.glob('*.pkl')
    )


def _assert_request(request, name):
    from sdm_client import Request

    assert isinstance(request, Request)
    assert request.task.kind == (
        'binary_classification' if name == 'request-ranking' else 'regression'
    )
    if name == 'request-frames':
        assert (
            request.query is None
            and request.payloads is None
            and request.parts is None
        )
        assert getattr(request, '_windowed', None) is None
        assert not getattr(request, 'indices', None)
        assert request.entity_table == 'instance'
        assert request.task.entity_table is None
        assert (
            request.task.target.column_name,
            request.task.target.dtype,
            request.task.target.table_name,
        ) == ('y', 'float64', 'instance')
        assert sorted(request.schema.tables) == ['instance']
        pd.testing.assert_frame_equal(
            request.context.instance_table,
            pd.DataFrame({'x': [1, 2, 3], 'y': [0.5, 1.5, 2.5]}),
        )
        pd.testing.assert_frame_equal(
            request.predict.instance_table, pd.DataFrame({'x': [4, 5]})
        )
        return
    ranking = name == 'request-ranking'
    assert request.entity_table == request.task.entity_table == 'users'
    assert list(request.indices) == [0, 1]
    assert request._windowed is True
    assert sorted(request.schema.tables) == ['buys', 'items', 'orders', 'users']
    assert (
        request.task.target.column_name,
        request.task.target.dtype,
        request.task.target.table_name,
    ) == ('TARGET', 'bool' if ranking else 'float32', None)
    forecast = name == 'request-forecast'
    parts = request.parts if (forecast or ranking) else [request]
    assert len(parts) == (3 if ranking else 2 if forecast else 1)
    assert len(request.payloads) == len(parts)
    if forecast or ranking:
        decomposition = request._decomposition
        assert decomposition.kind == ('ranking' if ranking else 'forecast')
        assert decomposition.top == (2 if ranking else None)
        assert [about for _, about in decomposition.parts] == (
            [2, 1, 0] if ranking else [1, 2]
        )
    else:
        assert request.parts is None
    for i, part in enumerate(parts):
        assert isinstance(part, Request)
        assert part.task == request.task
        assert part.entity_table == 'users'
        assert list(part.indices) == [0, 1]
        assert part._windowed is True
        assert sorted(part.schema.tables) == [
            'buys',
            'items',
            'orders',
            'users',
        ]
        assert part.parts is None
        assert list(part.payloads) == [request.payloads[i]]
        query = (
            f'PREDICT COUNT(buys.* WHERE buys.item_id = {2 - i}, 0, 30, days) > 0 FOR EACH users.user_id'
            if ranking
            else f'PREDICT SUM(orders.price, {7 * i}, {7 * (i + 1)}, days) FOR EACH users.user_id'
        )
        assert part.query == query
        if forecast or ranking:
            assert decomposition.parts[i][0] == (
                query.replace(' = ', '=') if ranking else query
            )
        assert len(part.context.instance_table) == (
            16 if ranking else [64, 56][i]
        )
        assert len(part.predict.instance_table) == 2
        assert part.predict.instance_table['__kumo_entity_ref_0'].tolist() == [
            0,
            1,
        ]
        assert part.predict.instance_table['TARGET'].isna().all()
    assert request.query == parts[0].query


def _assert_wire_views(request: Any) -> None:
    for part in getattr(request, 'parts', None) or ():
        _assert_wire_views(part)
    payloads = getattr(request, 'payloads', None)
    if not payloads:
        return
    for half in ('context', 'predict'):
        frame = getattr(request, half).instance_table
        assert len(frame) == sum(
            len(p[half]['instance_table']['rows']) for p in payloads
        )
        if len(payloads) == 1:
            table = payloads[0][half]['instance_table']
            expected = pd.DataFrame(table['rows'], columns=table['columns'])
            target = request.task.target.column_name
            if half == 'predict' and target not in expected:
                assert frame[target].isna().all()
                frame = frame.drop(columns=[target])
            if '__kumo_anchor_time' in expected:
                expected['__kumo_anchor_time'] = pd.to_datetime(
                    expected['__kumo_anchor_time'], utc=True
                ).dt.tz_localize(None)
            pd.testing.assert_frame_equal(
                frame.reset_index(drop=True), expected, check_dtype=False
            )


@pytest.mark.parametrize('name', sorted(NAMES))
def test_historical_pickle_loads_and_round_trips_without_warnings(name):
    loaded, shown = _load(name)
    assert shown == [], [str(w.message) for w in shown]
    if name != 'schema-notes':
        _assert_request(loaded, name)
    _assert_wire_views(loaded)
    again = pickle.loads(pickle.dumps(loaded, protocol=4))
    # Round-trip identity supplements the independent semantic checks above.
    assert writer().projection(again) == writer().projection(loaded)
    _assert_wire_views(again)
    if name != 'schema-notes':
        _assert_request(again, name)


@pytest.mark.parametrize(
    'name',
    ['request-one-call', 'request-forecast', 'request-ranking'],
)
def test_saved_request_sends_saved_data_and_assembles_scripted_answers(name):
    loaded, _ = _load(name)
    saved = copy.deepcopy(list(loaded.payloads))
    observed = writer().predicted(loaded)
    calls = {
        'request-one-call': 1,
        'request-forecast': 2,
        'request-ranking': 3,
    }[name]
    assert len(observed['bodies']) == len(saved) == calls
    for actual, original in zip(observed['bodies'], saved, strict=True):
        expected = copy.deepcopy(original)
        expected.pop('client_analysis', None)
        expected.pop('client_analysis_delta', None)
        expected['model'] = 'kumo-relational'
        expected['output'] = {
            'fields': (
                ['prediction', 'probabilities']
                if name == 'request-ranking'
                else ['prediction']
            )
        }
        if name != 'request-ranking':
            expected['output']['prediction_statistic'] = 'median'
        expected.get('inference', {}).pop('inference_config', None)
        expected.get('metadata', {}).pop('request_id', None)
        assert actual == expected
    assert list(loaded.payloads) == saved

    # The endpoint emits regression=1 and positive probability
    # ((3*call + 5*row) % 9 + 1)/10. No client assembler computes this oracle.
    rows = []
    for row in range(2):
        if name == 'request-ranking':
            scores = [
                (candidate, ((3 * call + 5 * row) % 9 + 1) / 10)
                for call, candidate in enumerate([2, 1, 0])
            ]
            top = sorted(scores, key=lambda pair: -pair[1])[:2]
            rows.append(
                {
                    'row_index': row,
                    'id': row,
                    'prediction': [candidate for candidate, _ in top],
                    'rankings': [
                        {'candidate': candidate, 'score': score}
                        for candidate, score in top
                    ],
                }
            )
        else:
            for step in [1, 2] if name == 'request-forecast' else [None]:
                item = {'row_index': row, 'id': row}
                if step is not None:
                    item['forecast_step'] = step
                item['prediction'] = 1.0
                rows.append(item)
    pd.testing.assert_frame_equal(observed['result'], pd.DataFrame(rows))


def test_the_schema_s_conversion_notes_and_links_survive_the_pickle() -> None:
    r"""The ``main``-era schema: its conversion notes load as the
    reference wrote them, it holds no source, and its relationships are
    still checked when edited."""
    from sdm_client import Relationship, Schema, StructuredDataError

    loaded, _ = _load('schema-notes')
    assert isinstance(loaded, Schema)
    assert list(loaded.conversion_notes) == [
        "'total_quantity' is SUM(order_lines.quantity, 0, 30, days) over the next 30 days"
    ]
    assert list(loaded.tables) == ['order_lines', 'customers', 'products']
    columns = {
        'order_lines': [
            'line_id',
            'ordered_at',
            'quantity',
            'discounted',
            'customer_id',
            'product_id',
        ],
        'customers': ['segment', 'customer_id'],
        'products': ['color', 'product_id'],
    }
    for name, table in loaded.tables.items():
        assert type(table).__name__ == 'SourceTable'
        assert list(table.columns) == columns[name]
        assert table.primary_key == {
            'customers': 'customer_id',
            'products': 'product_id',
        }.get(name)
        assert table.time_column == (
            'ordered_at' if name == 'order_lines' else None
        )
        for column, info in table.columns.items():
            dtype, stype = (
                ('int64', 'ID')
                if column.endswith('_id')
                else ('timestamp[us]', 'timestamp')
                if column == 'ordered_at'
                else ('int64', 'numerical')
                if column == 'quantity'
                else ('bool', 'categorical')
                if column == 'discounted'
                else ('string', 'categorical')
            )
            assert (info.dtype, info.stype, info.stype_declared) == (
                dtype,
                stype,
                column == 'ordered_at',
            )
    assert [
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in loaded.relationships
    ] == [
        ('order_lines', f'{entity}_id', plural, f'{entity}_id')
        for entity, plural in [
            ('customer', 'customers'),
            ('product', 'products'),
        ]
    ]

    assert loaded.source is None
    with pytest.raises(StructuredDataError):
        loaded.relationships.append(
            Relationship('order_lines', 'nothing', 'customers', 'customer_id')
        )
