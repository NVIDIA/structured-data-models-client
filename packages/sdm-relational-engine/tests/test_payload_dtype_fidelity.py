# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Every column a request declares must be serialized the way it is declared.

The checks here mirror the NIM's own ``_validate_cell`` and dtype whitelist, so
a payload that passes them is one the server accepts, and a column that
declares one encoding while sending another fails here rather than as an
HTTP 422 or 500 the caller cannot map back to their frame.
"""

import datetime
import math
import re
import uuid
from decimal import Decimal
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
import pytest
from predictive_query.schema.task import TaskType
from sdm_relational_engine.rfm import Graph, KumoRelational, TaskTable

# A fixed offset rather than an IANA zone: the CI image has no tzdata,
# and these cases only need a non-UTC offset.
_TZ = datetime.timezone(datetime.timedelta(hours=-8))

# `tfm_serving.data.dtypes.DTYPE_WHITELIST`.
DTYPE_WHITELIST = frozenset(
    {
        'float32',
        'float64',
        'int32',
        'int64',
        'bool',
        'string',
        'stringlist',
        'timestamp[us]',
    }
)
NUMERIC_DTYPES = frozenset({'float32', 'float64', 'int32', 'int64'})
JSON_SAFE_INT = 9007199254740991
RFC3339 = re.compile(
    r'\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(\.\d{1,6})?'
    r'([Zz]|[+-]\d{2}:\d{2})'
)

NUM_USERS = 24


def _check_cell(value: Any, dtype: str, path: str, problems: list[str]) -> None:
    if value is None:
        return
    if dtype == 'string':
        if not isinstance(value, str):
            problems.append(f'{path}: expected str, got {type(value).__name__}')
    elif dtype == 'stringlist':
        if not isinstance(value, list):
            problems.append(
                f'{path}: expected list, got {type(value).__name__}'
            )
            return
        for index, item in enumerate(value):
            if not isinstance(item, str):
                problems.append(
                    f'{path}[{index}]: expected str in stringlist, '
                    f'got {type(item).__name__}'
                )
    elif dtype == 'bool':
        if not isinstance(value, bool):
            problems.append(
                f'{path}: expected bool, got {type(value).__name__}'
            )
    elif dtype in ('float32', 'float64'):
        if isinstance(value, bool) or not isinstance(value, Real):
            problems.append(
                f'{path}: expected number, got {type(value).__name__}'
            )
        elif isinstance(value, float) and not math.isfinite(value):
            problems.append(f'{path}: non-finite number')
    elif dtype in ('int32', 'int64'):
        if isinstance(value, bool):
            problems.append(f'{path}: expected int, got bool')
        elif isinstance(value, Integral):
            if dtype == 'int64' and abs(int(value)) > JSON_SAFE_INT:
                problems.append(
                    f'{path}: int64 outside the JSON safe range '
                    f'must travel as a base-10 string'
                )
        elif isinstance(value, str):
            if re.fullmatch(r'[+-]?[0-9]+', value) is None:
                problems.append(
                    f'{path}: expected a base-10 integer string, got {value!r}'
                )
        else:
            problems.append(f'{path}: expected int, got {type(value).__name__}')
    elif dtype == 'timestamp[us]':
        if not isinstance(value, str) or not RFC3339.fullmatch(value):
            problems.append(
                f'{path}: expected an RFC 3339 string, got {value!r}'
            )


def payload_fidelity_problems(payload: dict[str, Any]) -> list[str]:
    r"""Every way ``payload`` describes itself differently from what it sends."""
    problems: list[str] = []
    for section in ('context', 'predict'):
        tables = [
            (
                'instance_table',
                payload[section]['instance_table'],
                payload['schema']['instance_table'],
            )
        ]
        tables.extend(
            (name, table, payload['schema']['related_tables'][name])
            for name, table in payload[section]['related_tables'].items()
        )
        for name, table, schema in tables:
            declared = schema['columns']
            for index, column in enumerate(table['columns']):
                spec = declared.get(column)
                if spec is None:
                    problems.append(
                        f'{section}.{name}.{column}: not declared in schema'
                    )
                    continue
                dtype = spec['dtype']
                if dtype not in DTYPE_WHITELIST:
                    problems.append(
                        f'{section}.{name}.{column}: dtype '
                        f'{dtype!r} is not one the NIM accepts'
                    )
                    continue
                if (
                    spec.get('stype') == 'numerical'
                    and dtype not in NUMERIC_DTYPES
                ):
                    problems.append(
                        f'{section}.{name}.{column}: stype numerical declared '
                        f'with non-numeric dtype {dtype!r}'
                    )
                for row_index, row in enumerate(table['rows']):
                    _check_cell(
                        row[index],
                        dtype,
                        f'{section}.{name}.{column}[{row_index}]',
                        problems,
                    )
    target = payload.get('task', {}).get('target', {})
    for index, label in enumerate(target.get('classes', [])):
        if not isinstance(label, str):
            problems.append(f'task.target.classes[{index}]: not a string')
        elif target.get('dtype') == 'bool' and label not in ('true', 'false'):
            problems.append(
                f'task.target.classes[{index}]: {label!r} is not '
                f'valid for a bool target dtype'
            )
    return problems


def _orders() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            'ORDER_ID': np.arange(6 * NUM_USERS, dtype='int64'),
            'USER_ID': np.tile(np.arange(NUM_USERS, dtype='int64'), 6),
            'AMOUNT': rng.uniform(5, 100, 6 * NUM_USERS).round(2),
            'TIME': pd.to_datetime('2024-01-01')
            + pd.to_timedelta(
                rng.integers(0, 420, 6 * NUM_USERS).astype('int64'), unit='D'
            ),
        }
    )


def _regression_task() -> TaskTable:
    return TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': np.arange(NUM_USERS - 2, dtype='int64'),
                'TARGET': np.arange(NUM_USERS - 2, dtype='float64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(
                    ['2025-02-20'] * (NUM_USERS - 2)
                ),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': np.arange(NUM_USERS - 2, NUM_USERS, dtype='int64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-02-20'] * 2),
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )


def feature_payload(feature: Any) -> tuple[Graph, dict[str, Any]]:
    r"""A request whose ``USERS`` table carries one extra ``FEATURE`` column."""
    users = pd.DataFrame({'USER_ID': np.arange(NUM_USERS, dtype='int64')})
    users['FEATURE'] = feature
    graph = Graph.from_data(
        {'USERS': users, 'ORDERS': _orders()}, verbose=False
    )
    payload = (
        KumoRelational(graph, verbose=False)
        .materialize_task(_regression_task(), verbose=False)[0]
        .payload
    )
    return graph, payload


def feature_schema(payload: dict[str, Any]) -> dict[str, Any]:
    return payload['schema']['related_tables']['USERS']['columns']['FEATURE']


def feature_cells(payload: dict[str, Any]) -> list[Any]:
    table = payload['context']['related_tables']['USERS']
    index = table['columns'].index('FEATURE')
    return [row[index] for row in table['rows']]


# The systematic dtype x serialization x cell-union sweep. Every spelling a
# pandas frame can arrive in, checked against the rules the NIM enforces.
FEATURE_CASES = {
    'int64': np.arange(NUM_USERS, dtype='int64'),
    'int32': np.arange(NUM_USERS, dtype='int32'),
    'uint8': np.arange(NUM_USERS, dtype='uint8'),
    'float64': np.arange(NUM_USERS, dtype='float64') / 2,
    'float32': (np.arange(NUM_USERS) / 2).astype('float32'),
    'bool': np.array([True, False] * (NUM_USERS // 2)),
    'str': [f's{index}' for index in range(NUM_USERS)],
    'str_low_cardinality': ['a', 'b'] * (NUM_USERS // 2),
    'datetime': pd.date_range('2024-01-01', periods=NUM_USERS),
    'datetime_tz': pd.date_range('2024-01-01', periods=NUM_USERS, tz=_TZ),
    'timedelta': pd.to_timedelta(np.arange(NUM_USERS), unit='D'),
    'timedelta_beyond_js_safe': pd.to_timedelta(
        np.arange(NUM_USERS) + 4000, unit='D'
    ),
    'bytes': pd.Series(
        [bytes([index % 3, 7]) for index in range(NUM_USERS)], dtype=object
    ),
    'decimal': pd.Series(
        [Decimal(f'{index}.1234') for index in range(NUM_USERS)], dtype=object
    ),
    'decimal_low_cardinality': pd.Series(
        [Decimal('1.25'), Decimal('2.50')] * (NUM_USERS // 2), dtype=object
    ),
    'object_int': pd.Series([1, 2, 3] * (NUM_USERS // 3), dtype=object),
    'object_int_with_null': pd.Series(
        [1, 2, None] * (NUM_USERS // 3), dtype=object
    ),
    'object_float': pd.Series([1.5, 2.5, 3.5] * (NUM_USERS // 3), dtype=object),
    'object_bool': pd.Series([True, False] * (NUM_USERS // 2), dtype=object),
    'object_mixed': pd.Series(
        [1, 'a', 2.5, None] * (NUM_USERS // 4), dtype=object
    ),
    'object_all_null': pd.Series([None] * NUM_USERS, dtype=object),
    'object_uuid': pd.Series(
        [uuid.uuid4() for _ in range(NUM_USERS)], dtype=object
    ),
    'object_date': pd.Series(
        [datetime.date(2024, 1, 1 + index % 28) for index in range(NUM_USERS)],
        dtype=object,
    ),
    'object_datetime': pd.Series(
        [
            datetime.datetime(2024, 1, 1 + index % 28, tzinfo=datetime.UTC)
            for index in range(NUM_USERS)
        ],
        dtype=object,
    ),
    'object_timedelta': pd.Series(
        [pd.Timedelta(days=index) for index in range(NUM_USERS)], dtype=object
    ),
    'object_int_beyond_js_safe': pd.Series(
        [9007199254740993 + 2 * index for index in range(NUM_USERS)],
        dtype=object,
    ),
    'int64_beyond_js_safe': np.array(
        [9007199254740993 + 2 * index for index in range(NUM_USERS)],
        dtype='int64',
    ),
    'category_int': pd.Categorical([1, 2, 3] * (NUM_USERS // 3)),
    'category_str': pd.Categorical(['a', 'b'] * (NUM_USERS // 2)),
    'nullable_int64': pd.array([1, 2, None] * (NUM_USERS // 3), dtype='Int64'),
    'nullable_float64': pd.array(
        [1.5, 2.5, None] * (NUM_USERS // 3), dtype='Float64'
    ),
    'nullable_boolean': pd.array(
        [True, False, None] * (NUM_USERS // 3), dtype='boolean'
    ),
    'string_extension': pd.array(
        [f's{index}' for index in range(NUM_USERS)], dtype='string'
    ),
    'stringlist': pd.Series([['a', 'b']] * NUM_USERS, dtype=object),
    'intlist': pd.Series([[1, 2, 3]] * NUM_USERS, dtype=object),
    'floatlist': pd.Series([[1.5, 2.5]] * NUM_USERS, dtype=object),
    'ndarray_float': pd.Series(
        [np.array([1.0, 2.0])] * NUM_USERS, dtype=object
    ),
    'all_nan_float': pd.Series([np.nan] * NUM_USERS),
    'float_with_nan': pd.Series([1.5, np.nan] * (NUM_USERS // 2)),
}


@pytest.mark.parametrize('case', sorted(FEATURE_CASES))
def test_every_column_spelling_is_sent_the_way_it_is_declared(case) -> None:
    # Regression: the whole dtype/payload-fidelity cluster. `_dtype_name` and
    # the cell writer used to decide independently, so `object`, `category`,
    # `Decimal`, `timedelta`, `bytes` and list columns each declared one
    # encoding and sent another.
    _, payload = feature_payload(FEATURE_CASES[case])
    assert payload_fidelity_problems(payload) == []


def test_decimal_column_is_a_number_not_a_string() -> None:
    # Regression: `dtype: 'string'` beside `stype: 'numerical'` is what the NIM
    # answers with an HTTP 500 the client then reports as GPU capacity pressure.
    graph, payload = feature_payload(
        [Decimal(f'{index}.1234') for index in range(NUM_USERS)]
    )
    assert str(graph['USERS']['FEATURE'].stype) == 'numerical'
    assert feature_schema(payload) == {'dtype': 'float64', 'stype': 'numerical'}
    assert feature_cells(payload)[:3] == pytest.approx([0.1234, 1.1234, 2.1234])


def test_decimal_column_is_not_inferred_as_free_text() -> None:
    # Regression: a high-cardinality Decimal column became `Stype.text`, whose
    # normalizer strips the decimal point, so `18.1234` travelled as `['18',
    # '1234']`.
    graph, payload = feature_payload(
        [Decimal(f'{index}.1234') for index in range(NUM_USERS)]
    )
    column = graph['USERS']['FEATURE']
    assert (str(column.dtype), str(column.stype)) == ('float', 'numerical')
    assert all(isinstance(cell, float) for cell in feature_cells(payload))


@pytest.mark.parametrize(
    'feature',
    [
        pd.Categorical([1, 2, 3] * (NUM_USERS // 3)),
        pd.Series(np.arange(NUM_USERS) % 4).astype('category'),
        pd.Series([1, 2, 3] * (NUM_USERS // 3), dtype=object),
        pd.Series([1.5, 2.5, 3.5] * (NUM_USERS // 3), dtype=object),
        pd.Series([True, False] * (NUM_USERS // 2), dtype=object),
        pd.Series([1, 'a', 2.5, None] * (NUM_USERS // 4), dtype=object),
    ],
)
def test_object_and_category_columns_send_what_they_declare(feature) -> None:
    # Regression: these declared `string` and sent JSON numbers, and the NIM
    # refused the whole prediction with a 422 naming a row index.
    _, payload = feature_payload(feature)
    assert payload_fidelity_problems(payload) == []


@pytest.mark.parametrize(
    'feature',
    [
        pd.Series([[1, 2, 3]] * NUM_USERS, dtype=object),
        pd.Series([[1.5, 2.5]] * NUM_USERS, dtype=object),
        pd.Series([np.array([1.0, 2.0])] * NUM_USERS, dtype=object),
    ],
)
def test_numeric_list_columns_travel_as_lists_of_strings(feature) -> None:
    # Regression: `stringlist` was declared for every list column while the
    # items went out as JSON numbers, which the NIM's cell union has no branch
    # for.
    _, payload = feature_payload(feature)
    assert feature_schema(payload)['dtype'] == 'stringlist'
    assert feature_cells(payload)[0] == [str(item) for item in feature.iloc[0]]
    assert payload_fidelity_problems(payload) == []


def test_timedelta_column_is_serializable_as_the_quantity_it_is() -> None:
    # Regression: the graph typed the column `timedelta`/`numerical`,
    # `validate()` passed, and the first `predict()` died inside `json.dumps`
    # naming neither table nor column.
    _, payload = feature_payload(
        pd.to_timedelta(np.arange(NUM_USERS), unit='D')
    )
    assert feature_schema(payload) == {'dtype': 'int64', 'stype': 'numerical'}
    assert feature_cells(payload)[:2] == [0, 86_400_000_000_000]
    assert payload_fidelity_problems(payload) == []


def test_bytes_column_is_serializable_as_the_string_it_declares() -> None:
    # Regression.
    _, payload = feature_payload(
        pd.Series(
            [bytes([index % 3, 7]) for index in range(NUM_USERS)], dtype=object
        )
    )
    assert feature_schema(payload)['dtype'] == 'string'
    assert feature_cells(payload)[0] == 'AAc='
    assert payload_fidelity_problems(payload) == []


def test_boolean_multiclass_target_labels_match_its_declared_dtype() -> None:
    # Regression: the target was declared `dtype: 'bool'` while its classes
    # went out as `str(np.True_)`, which the NIM refuses; and a boolean
    # *multiclass* label cannot be mapped back to a declared class at all, so
    # the column travels as its two labels.
    users = pd.DataFrame(
        {
            'USER_ID': np.arange(NUM_USERS, dtype='int64'),
            'FLAG': np.array([True, False] * (NUM_USERS // 2)),
        }
    )
    graph = Graph.from_data(
        {'USERS': users, 'ORDERS': _orders()}, verbose=False
    )
    model = KumoRelational(graph, verbose=False)
    query = model._parse_query('PREDICT USERS.FLAG FOR EACH USERS.USER_ID')
    task = model._get_task_table(query, indices=[1, 2], random_seed=0)
    payload = model.materialize_task(task, random_seed=0, verbose=False)[
        0
    ].payload

    target = payload['task']['target']
    assert payload['task']['kind'] == 'multiclass_classification'
    assert target['dtype'] == 'string'
    assert sorted(target['classes']) == ['False', 'True']
    assert payload_fidelity_problems(payload) == []


def test_binary_target_still_declares_the_contract_bool_spelling() -> None:
    users = pd.DataFrame({'USER_ID': np.arange(NUM_USERS, dtype='int64')})
    graph = Graph.from_data(
        {'USERS': users, 'ORDERS': _orders()}, verbose=False
    )
    model = KumoRelational(graph, verbose=False)
    query = model._parse_query(
        'PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH USERS.USER_ID'
    )
    task = model._get_task_table(query, indices=[1, 2], random_seed=0)
    payload = model.materialize_task(task, random_seed=0, verbose=False)[
        0
    ].payload

    target = payload['task']['target']
    assert target['dtype'] == 'bool'
    assert target['classes'] == ['false', 'true']
    assert payload_fidelity_problems(payload) == []


def test_tz_aware_anchor_timestamps_are_converted_not_refused() -> None:
    # Regression: `predict` returns `ANCHOR_TIMESTAMP` tz-aware and
    # `predict_task`'s docstring points callers back at it, but
    # `astype('datetime64[ns]')` raises on a tz-aware series rather than
    # converting it.
    users = pd.DataFrame({'USER_ID': np.arange(NUM_USERS, dtype='int64')})
    graph = Graph.from_data(
        {'USERS': users, 'ORDERS': _orders()}, verbose=False
    )
    task = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': np.arange(NUM_USERS - 2, dtype='int64'),
                'TARGET': np.arange(NUM_USERS - 2, dtype='float64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(
                    ['2025-02-20T00:00:00Z'] * (NUM_USERS - 2)
                ),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': np.arange(NUM_USERS - 2, NUM_USERS, dtype='int64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(
                    ['2025-02-19T16:00:00-08:00'] * 2
                ),
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )
    payload = (
        KumoRelational(graph, verbose=False)
        .materialize_task(task, verbose=False)[0]
        .payload
    )

    anchor = payload['task']['anchor_time_column']
    table = payload['context']['instance_table']
    index = table['columns'].index(anchor)
    assert table['rows'][0][index] == '2025-02-20T00:00:00.000000Z'
    assert payload_fidelity_problems(payload) == []


def test_tz_aware_scalar_anchor_time_is_accepted() -> None:
    # Regression: the scalar path failed separately, comparing a tz-aware
    # Timestamp against the graph's tz-naive range.
    users = pd.DataFrame({'USER_ID': np.arange(NUM_USERS, dtype='int64')})
    graph = Graph.from_data(
        {'USERS': users, 'ORDERS': _orders()}, verbose=False
    )
    model = KumoRelational(graph, verbose=False)
    query = model._parse_query(
        'PREDICT SUM(ORDERS.AMOUNT, 0, 30, days) FOR EACH USERS.USER_ID'
    )
    task = model._get_task_table(
        query,
        indices=[1, 2],
        anchor_time=pd.Timestamp('2025-02-20', tz=_TZ),
        random_seed=0,
    )
    payload = model.materialize_task(task, random_seed=0, verbose=False)[
        0
    ].payload
    assert payload_fidelity_problems(payload) == []
