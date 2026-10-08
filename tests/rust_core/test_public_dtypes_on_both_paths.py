# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Frames of every common pandas dtype, through the public calls, on
today's Python and on the core: the two must agree.

Two frames, ``users`` and ``orders``, go through ``Schema.from_data`` and a
``PredictiveQuery(...).materialize``; one column of them at a time takes a
dtype under test, as a feature of ``users``, as the time column of
``orders`` or as the key ``users.user_id`` and its foreign key. Compared:
each column's dtype and semantic type, the time column, the payloads
(request id aside), the error raised (class, code, message) and the
warnings (category and text). pyarrow-backed columns are what
``read_parquet(dtype_backend='pyarrow')`` and Arrow-native warehouses give,
and the parity corpus held none of them.

Beyond those 35 cases of the probe, the other Arrow scalars and the
unsigned widths are held the same way, as features, keys, time columns and
the summed target.

Some answers differ on purpose, each a fix of today's bugs: a column of a
type the engine has no reading of (``large_string``, an Arrow duration or
dictionary) is refused by name where today's ended in the engine's
failure; a time of day is text, not a date; and a ``uint64`` key of 2**63
or more is sent as the text the schema calls it, where today's refused the
values it sent.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest

pytest.importorskip('sdm_relational_engine')
pytest.importorskip('sdm_core._native')
np = pytest.importorskip('numpy')
pd = pytest.importorskip('pandas')
pa = pytest.importorskip('pyarrow')

QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ORDERS = 80
ANCHOR = '2024-03-01'


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('view', ['metric', 'semantic'])
def test_view_conversion_has_a_shared_filterable_category(
    monkeypatch, old, view
):
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2]
            / 'packages/structured-data-models-client/tests'
        )
    )
    from sdm_client import Schema, _impl
    from sdm_client.core import warnings as categories

    if view == 'metric':
        from test_a_metric_view_filter_holds import (
            _VIEW,
            DatabricksSource,
            _Warehouse,
        )

        source = DatabricksSource(
            _Warehouse(
                _VIEW.replace(
                    'expr: source.quantity', 'expr: mystery(source.quantity)'
                )
            ),
            catalog='cat',
            schema='sch',
        ).semantic_model('sales_mv')
    else:
        import test_semantic_model_snowflake as semantic
        from test_semantic_model_snowflake import (
            SnowflakeSource,
            _FakeConnection,
        )

        monkeypatch.setattr(
            semantic,
            '_SEMANTIC_VIEW_YAML',
            semantic._SEMANTIC_VIEW_YAML.replace(
                'expr: racks.HALL', 'expr: mystery(racks.HALL)'
            ),
        )

        source = SnowflakeSource(
            _FakeConnection(), database='DB', schema='SCH'
        ).semantic_model('GPU_FLEET_SV')
    monkeypatch.setattr(_impl, 'OLD', old)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Schema.from_source(source)
    converted = [
        warning for warning in caught if 'Failed to add' in str(warning.message)
    ]
    assert converted
    category = categories.ViewConversionWarning
    assert issubclass(category, UserWarning)
    assert all(warning.category is category for warning in converted)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.filterwarnings('ignore', category=category)
        Schema.from_source(source)
    assert not [
        warning for warning in caught if 'Failed to add' in str(warning.message)
    ]


def _features() -> dict[str, Any]:
    a = pd.ArrowDtype
    stamp = pd.Timestamp
    return {
        'Int64': pd.array([1, None, 3, 4], dtype='Int64'),
        'UInt64': pd.array([1, None, 3, 4], dtype='UInt64'),
        'Float32': pd.array([1.5, None, 3, 4], dtype='Float32'),
        'boolean': pd.array([True, None, False, True], dtype='boolean'),
        'string_python': pd.array(
            ['a', None, 'b', 'c'], dtype='string[python]'
        ),
        'string_pyarrow': pd.array(
            ['a', None, 'b', 'c'], dtype='string[pyarrow]'
        ),
        'category': pd.Categorical(['a', 'b', 'a', None]),
        'arrow_int64': pd.array([1, None, 3, 4], dtype=a(pa.int64())),
        'arrow_double': pd.array([1.5, None, 3, 4], dtype=a(pa.float64())),
        'arrow_bool': pd.array([True, None, False, True], dtype=a(pa.bool_())),
        'arrow_string': pd.array(['a', None, 'b', 'c'], dtype=a(pa.string())),
        'arrow_large_string': pd.array(
            ['a', None, 'b', 'c'], dtype=a(pa.large_string())
        ),
        'arrow_decimal': pd.array(
            [
                decimal.Decimal('1.5'),
                None,
                decimal.Decimal('2'),
                decimal.Decimal('3'),
            ],
            dtype=a(pa.decimal128(10, 2)),
        ),
        'arrow_date32': pd.array(
            [
                dt.date(2020, 1, 1),
                None,
                dt.date(2020, 1, 3),
                dt.date(2020, 1, 4),
            ],
            dtype=a(pa.date32()),
        ),
        'arrow_timestamp_us': pd.array(
            [
                stamp('2020-01-01'),
                None,
                stamp('2020-01-03'),
                stamp('2020-01-04'),
            ],
            dtype=a(pa.timestamp('us')),
        ),
        'datetime64_ms': pd.Series(
            pd.to_datetime(['2020-01-01', None, '2020-01-03', '2020-01-04'])
        )
        .astype('datetime64[ms]')
        .array,
        'datetime64_tz': pd.Series(
            pd.to_datetime(['2020-01-01', None, '2020-01-03', '2020-01-04'])
        )
        .dt.tz_localize('Europe/Berlin')
        .array,
        'date_objects': np.array(
            [
                dt.date(2020, 1, 1),
                None,
                dt.date(2020, 1, 3),
                dt.date(2020, 1, 4),
            ],
            dtype=object,
        ),
        'arrow_list_int': pd.array(
            [[1, 2], None, [3], []], dtype=a(pa.list_(pa.int64()))
        ),
    }


def _times() -> dict[str, Any]:
    a = pd.ArrowDtype
    ts = _weekly()
    return {
        'datetime64_ns': ts,
        'datetime64_us': ts.astype('datetime64[us]'),
        'datetime64_s': ts.astype('datetime64[s]'),
        'datetime64_utc': ts.dt.tz_localize('UTC'),
        'arrow_timestamp_us': ts.astype(a(pa.timestamp('us'))),
        'arrow_timestamp_ns': ts.astype(a(pa.timestamp('ns'))),
        'arrow_date32': ts.dt.date.astype(a(pa.date32())),
        'text': ts.dt.strftime('%Y-%m-%d'),
        'arrow_text': ts.dt.strftime('%Y-%m-%d').astype(a(pa.string())),
    }


def _keys() -> dict[str, Any]:
    a = pd.ArrowDtype
    return {
        'int64': np.arange(4),
        'Int64': pd.array(np.arange(4), dtype='Int64'),
        'arrow_int64': pd.array(np.arange(4), dtype=a(pa.int64())),
        'text': np.array(['u0', 'u1', 'u2', 'u3'], dtype=object),
        'arrow_string': pd.array(
            ['u0', 'u1', 'u2', 'u3'], dtype=a(pa.string())
        ),
        'string_pyarrow': pd.array(
            ['u0', 'u1', 'u2', 'u3'], dtype='string[pyarrow]'
        ),
        'uint64_past_int64': np.array(
            [2**63 + 1, 2**63 + 2, 2**63 + 3, 5], dtype='uint64'
        ),
    }


def _more_features() -> dict[str, Any]:
    r"""Beyond the probe: the other Arrow scalars, and plain widths."""
    a = pd.ArrowDtype
    stamp = pd.Timestamp
    out = {
        'arrow_float32': pd.array([1.5, None, 3, 4], dtype=a(pa.float32())),
        'arrow_date64': pd.array(
            [
                dt.date(2020, 1, 1),
                None,
                dt.date(2020, 1, 3),
                dt.date(2020, 1, 4),
            ],
            dtype=a(pa.date64()),
        ),
        'arrow_timestamp_utc': pd.array(
            [stamp('2020-01-01', tz='UTC'), None, stamp('2020-01-03', tz='UTC')]
            + [stamp('2020-01-04', tz='UTC')],
            dtype=a(pa.timestamp('us', tz='UTC')),
        ),
        'arrow_binary': pd.array(
            [b'a', None, b'b', b'c'], dtype=a(pa.binary())
        ),
        'arrow_duration': pd.array(
            [pd.Timedelta('1D'), None, pd.Timedelta('2h'), pd.Timedelta('3s')],
            dtype=a(pa.duration('us')),
        ),
        'arrow_time64': pd.array(
            [dt.time(1, 2), None, dt.time(3, 4), dt.time(5, 6)],
            dtype=a(pa.time64('us')),
        ),
        'arrow_time32': pd.array(
            [dt.time(1, 2), None, dt.time(3, 4), dt.time(5, 6)],
            dtype=a(pa.time32('s')),
        ),
        'arrow_dictionary': pd.Series(
            pa.chunked_array(
                [pa.array(['a', None, 'b', 'a']).dictionary_encode()]
            ).to_pandas(types_mapper=pd.ArrowDtype)
        ).array,
    }
    for width in ('uint8', 'uint32', 'uint64'):
        out[f'arrow_{width}'] = pd.array(
            [1, None, 3, 4], dtype=a(getattr(pa, width)())
        )
    return out


def _more_keys() -> dict[str, Any]:
    a = pd.ArrowDtype
    out: dict[str, Any] = {
        'arrow_large_string': pd.array(
            ['u0', 'u1', 'u2', 'u3'], dtype=a(pa.large_string())
        ),
    }
    for width in ('uint8', 'uint16', 'uint32', 'uint64'):
        out[width] = np.arange(4).astype(width)
        nullable = width.replace('uint', 'UInt')
        out[nullable] = pd.array(np.arange(4), dtype=nullable)
        out[f'arrow_{width}'] = pd.array(
            list(range(4)), dtype=a(getattr(pa, width)())
        )
    return out


def _more_times() -> dict[str, Any]:
    a = pd.ArrowDtype
    ts = _weekly()
    return {
        'arrow_date64': ts.dt.date.astype(a(pa.date64())),
        'arrow_timestamp_utc': ts.dt.tz_localize('UTC').astype(
            a(pa.timestamp('us', tz='UTC'))
        ),
        'arrow_timestamp_s': ts.astype(a(pa.timestamp('s'))),
    }


def _targets() -> dict[str, Any]:
    r"""The summed column, ``orders.price``, of other dtypes."""
    a = pd.ArrowDtype
    return {
        'uint64': np.arange(ORDERS).astype('uint64'),
        'arrow_float32': pd.array(
            [i * 1.5 for i in range(ORDERS)], dtype=a(pa.float32())
        ),
        'arrow_uint8': pd.array(
            [i % 200 for i in range(ORDERS)], dtype=a(pa.uint8())
        ),
    }


def _weekly() -> Any:
    return pd.Series(
        pd.Timestamp('2023-01-01')
        + pd.to_timedelta(np.arange(ORDERS) * 7, unit='D')
    )


def _orders(user_ids: Any, ts: Any, price: Any = None) -> Any:
    keys = pd.Series(user_ids)
    return pd.DataFrame(
        {
            'order_id': np.arange(ORDERS),
            'user_id': keys.iloc[np.arange(ORDERS) % 4].reset_index(drop=True),
            'price': np.arange(ORDERS, dtype='float64')
            if price is None
            else price,
            'ts': ts,
        }
    )


def frames(case: str) -> dict[str, Any]:
    r"""The two frames of ``case``, ``KIND:NAME``."""
    kind, name = case.split(':')
    ids = np.arange(4)
    if kind == 'feature':
        given = {**_features(), **_more_features()}[name]
        users = pd.DataFrame({'user_id': ids, 'f': given})
        return {'users': users, 'orders': _orders(ids, _weekly())}
    if kind == 'time':
        users = pd.DataFrame({'user_id': ids, 'age': [1, 2, 3, 4]})
        given = {**_times(), **_more_times()}[name]
        return {'users': users, 'orders': _orders(ids, given)}
    if kind == 'target':
        users = pd.DataFrame({'user_id': ids, 'age': [1, 2, 3, 4]})
        price = _targets()[name]
        return {'users': users, 'orders': _orders(ids, _weekly(), price)}
    keys = pd.Series({**_keys(), **_more_keys()}[name])
    users = pd.DataFrame({'user_id': keys, 'age': [1, 2, 3, 4]})
    return {'users': users, 'orders': _orders(keys, _weekly())}


CASES = (
    [f'feature:{name}' for name in _features()]
    + [f'time:{name}' for name in _times()]
    + [f'key:{name}' for name in _keys()]
)

#: Beyond the probe: the other Arrow scalars and unsigned widths, as
#: features, keys, time columns and the summed target.
MORE = (
    [f'feature:{name}' for name in _more_features()]
    + [f'time:{name}' for name in _more_times()]
    + [f'key:{name}' for name in _more_keys()]
    + [f'target:{name}' for name in _targets()]
)

STRUCTURED = 'sdm_client.errors.StructuredDataError'

#: Columns of a type the engine has no reading of: today's ended in its
#: failure, ``INTERNAL_ERROR``, and the core refuses each by name, as it
#: does a complex, period or interval column.
UNREAD = {
    'feature:arrow_large_string': 'users.f is large_string[pyarrow]',
    'feature:arrow_duration': 'users.f is duration[us][pyarrow]',
    'feature:arrow_dictionary': 'users.f is dictionary<values=string',
    'key:arrow_large_string': 'users.user_id is large_string[pyarrow]',
}

#: A time of day: today's took it for a date and refused the semantic type
#: the schema gave it; the core reads it as text, which no model reads.
TIME_OF_DAY = {
    'feature:arrow_time64': 'users.f is time64[us][pyarrow]',
    'feature:arrow_time32': 'users.f is time32[s][pyarrow]',
}

#: Cases whose answer a fix in the core changes.
FIXED = {*UNREAD, *TIME_OF_DAY, 'key:uint64_past_int64'}


def _plain(value: Any) -> Any:
    text = json.dumps(value, default=str, sort_keys=True)
    return json.loads(text.replace('"request_id": ', '"request_id_": '))


def _without_request_ids(payloads: Any) -> Any:
    found = _plain(payloads)
    for payload in found if isinstance(found, list) else []:
        if isinstance(payload, dict):
            payload.pop('request_id_', None)
            inference = payload.get('inference')
            if isinstance(inference, dict):
                inference.pop('request_id_', None)
    return found


def outcome(case: str) -> dict[str, Any]:
    r"""What the public calls over ``case``'s frames give."""
    from sdm_client import PredictiveQuery, Schema

    out: dict[str, Any] = {}
    given = frames(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            schema = Schema.from_data(given)
            out['columns'] = {
                table: {
                    name: [str(column.dtype), str(column.stype)]
                    for name, column in schema[table].columns.items()
                }
                for table in ('users', 'orders')
            }
            out['time_column'] = schema['orders'].time_column
            request = PredictiveQuery(QUESTION, schema).materialize(
                indices=list(given['users']['user_id'][:2]),
                anchor_time=pd.Timestamp(ANCHOR),
            )
            out['payloads'] = _without_request_ids(request.payloads)
        except Exception as error:  # noqa: BLE001 - compared
            cls = type(error)
            out['error'] = [
                f'{cls.__module__}.{cls.__qualname__}',
                getattr(error, 'code', None),
                str(error),
            ]
    out['warnings'] = [
        [f'{w.category.__module__}.{w.category.__qualname__}', str(w.message)]
        for w in caught
    ]
    return out


@pytest.fixture
def on_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], tuple[dict[str, Any], dict[str, Any]]]]:
    r"""Runs a case on today's Python, then on the core."""
    from sdm_client import _impl

    def run(case: str) -> tuple[dict[str, Any], dict[str, Any]]:
        monkeypatch.setattr(_impl, 'OLD', True)
        try:
            old = outcome(case)
        finally:
            monkeypatch.setattr(_impl, 'OLD', False)
        return old, outcome(case)

    yield run


def _first_difference(a: Any, b: Any, path: str = '$') -> str | None:
    if type(a) is not type(b):
        return f'{path}: {a!r:.300} != {b!r:.300}'
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f'{path}.{key}: on one side only'
            found = _first_difference(a[key], b[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            found = _first_difference(x, y, f'{path}[{i}]')
            if found:
                return found
        return None if len(a) == len(b) else f'{path}: {len(a)} != {len(b)}'
    return None if a == b else f'{path}: {a!r:.300} != {b!r:.300}'


def test_the_cases_are_the_dtype_probe() -> None:
    r"""19 features, 9 time columns, 6 keys and the ``uint64`` key."""
    assert len(CASES) == 35, len(CASES)


@pytest.mark.parametrize('case', [c for c in CASES + MORE if c not in FIXED])
def test_a_dtype_gives_what_it_gave(case, on_both_paths) -> None:
    old, core = on_both_paths(case)
    found = _first_difference(old, core)
    assert found is None, f'{case}: {found}'


@pytest.mark.parametrize('case', sorted(UNREAD))
def test_a_type_the_engine_cannot_read_is_refused_by_name(
    case, on_both_paths
) -> None:
    old, core = on_both_paths(case)
    assert old['error'][:2] == [STRUCTURED, 'INTERNAL_ERROR'], old
    assert 'the relational engine failed' in old['error'][2], old
    assert core['error'][:2] == [STRUCTURED, 'UNSUPPORTED_REQUEST'], core
    assert UNREAD[case] in core['error'][2], core
    assert 'a type no model reads' in core['error'][2], core
    assert old['columns'] == core['columns']
    assert old['warnings'] == core['warnings']


@pytest.mark.parametrize('case', sorted(TIME_OF_DAY))
def test_a_time_of_day_feature_is_text_no_model_reads(
    case, on_both_paths
) -> None:
    old, core = on_both_paths(case)
    assert old['error'][:2] == [STRUCTURED, 'INVALID_REQUEST'], old
    assert "got dtype='date'" in old['error'][2], old
    assert core['error'][:2] == [STRUCTURED, 'UNSUPPORTED_REQUEST'], core
    assert TIME_OF_DAY[case] in core['error'][2], core
    assert old['columns'] == core['columns']


def test_a_uint64_key_past_int64_is_sent_as_text(on_both_paths) -> None:
    r"""The schema calls such a key ``string`` on both paths; today's then
    sent the numbers and refused them, the core sends the text."""
    old, core = on_both_paths('key:uint64_past_int64')
    assert old['columns'] == core['columns']
    assert old['columns']['users']['user_id'] == ['string', 'ID']
    assert old['error'][:2] == [STRUCTURED, 'INVALID_REQUEST'], old
    assert 'outside the range int64 represents' in old['error'][2]
    assert 'error' not in core, core
    predict = core['payloads'][0]['predict']['instance_table']
    assert [row[1] for row in predict['rows']] == [
        str(2**63 + 1),
        str(2**63 + 2),
    ]
