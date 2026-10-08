# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The binding's own behaviour, beyond the smoke test, tested on the
installed wheel. cibuildwheel runs this in each build container, after
``tests/rust_core/smoke.py``.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import pyarrow as pa
import pytest
from sdm_core import _native


def test_low_level_host_operations_do_not_import_pandas():
    child = subprocess.run(
        [
            sys.executable,
            '-c',
            """
import builtins
import json
from sdm_core import _native

original_import = builtins.__import__
attempts = []
def guarded(name, *args, **kwargs):
    if name in ('pandas', 'numpy') or name.startswith(('pandas.', 'numpy.')):
        attempts.append(name)
        raise AssertionError('low-level host operation imported pandas or NumPy')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
inputs = json.dumps({
    'backend': 'local', 'tables': [], 'edges': [], 'state': {}, 'arguments': {},
})
for function, arguments in [
    (_native.data_schema, ({}, '{}', '{}')),
    (_native.query_text, ('PREDICT COUNT(items.*)',)),
    (_native.materialize_run, ('{}', {})),
    (_native.task_table, (inputs, {})),
]:
    try:
        result = function(*arguments)
        if isinstance(result, _native.Run):
            result.next()
    except (ValueError, _native.CoreError):
        pass
assert not attempts, 'low-level host operation attempted pandas or NumPy import'
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize(
    'facts',
    [
        None,
        {},
        [],
        ['ns'] * 3,
        ['ns'] * 5,
        ['ns', 'ns', 1, 'ns'],
        ['s', 'ms', 'object', 'ns'],
        ['s', 'ms', 'us', 'us'],
    ],
)
@pytest.mark.parametrize('operation', ['materialize', 'task'])
def test_timestamp_series_fact_descriptors_are_validated(facts, operation):
    error_class = (
        _native.CoreError if operation == 'materialize' else ValueError
    )
    with pytest.raises(error_class, match='timestamp Series facts') as caught:
        if operation == 'materialize':
            _native.materialize_run(
                json.dumps({'timestamp_series_units': facts}), {}
            )
        else:
            inputs = {
                'backend': 'local',
                'tables': [],
                'edges': [],
                'state': {},
                'arguments': {'timestamp_series_units': facts},
            }
            _native.task_table(json.dumps(inputs), {}).next()
    if operation == 'materialize':
        assert caught.value.error_class == 'builtins.ValueError'
        assert caught.value.code is None


@pytest.mark.parametrize('capability', [None, {}, [], 0, 1, 'prefix'])
def test_numpy_parser_capability_is_a_strict_boolean(capability):
    with pytest.raises(
        _native.CoreError, match='invalid NumPy text policy'
    ) as caught:
        _native.materialize_run(
            json.dumps({'numpy_text_prefix': capability}), {}
        )
    assert caught.value.error_class == 'builtins.ValueError'
    assert caught.value.code is None


REQUEST = {
    'model': 'kumo-tabular',
    'task': {
        'kind': 'regression',
        'target': {'column_name': 'y', 'dtype': 'float64'},
    },
    'schema': {
        'instance_table': {
            'columns': {
                'id': {'dtype': 'int64', 'stype': 'ID'},
                'y': {'dtype': 'float64'},
            }
        }
    },
    'context': {
        'instance_table': {
            'format': 'arrays',
            'columns': ['id', 'y'],
            'rows': [[1, 2.0], [2, None]],
        }
    },
    'predict': {
        'instance_table': {
            'format': 'arrow_ipc',
            'transport': 'base64',
            'payload_base64': 'QUJD',
        }
    },
    'output': {'fields': ['prediction']},
}


def _core_error() -> _native.CoreError:
    run = _native.count_rows('t')
    assert run.next().kind == 'run_queries'
    with pytest.raises(_native.CoreError) as raised:
        run.submit(99, _native.Outcome.error(None, None, 'unused'))
    return raised.value


def test_core_error_is_named_for_the_module_it_is_in():
    assert _native.CoreError.__module__ == 'sdm_core._native'
    assert _native.CoreError.__qualname__ == 'CoreError'
    assert issubclass(_native.CoreError, Exception)


def test_a_core_error_pickles_with_its_message_and_code():
    error = _core_error()
    assert error.code == 'UNKNOWN_QUERY'
    back = pickle.loads(pickle.dumps(error))
    assert type(back) is _native.CoreError
    assert str(back) == str(error)
    assert back.args == error.args
    assert back.code == 'UNKNOWN_QUERY'


def test_a_payload_round_trips_with_its_json_types():
    back = json.loads(_native.roundtrip_payload(json.dumps(REQUEST)))
    assert back == REQUEST
    assert isinstance(back['context']['instance_table']['rows'][0][1], float)


@pytest.mark.parametrize(
    'number', ['18446744073709551616', '-9223372036854775809']
)
def test_an_integer_wider_than_64_bits_is_refused(number):
    text = json.dumps(REQUEST).replace(
        '"kind": "regression"', f'"kind": "regression", "big": {number}'
    )
    with pytest.raises(ValueError, match='does not fit in 64 bits'):
        _native.roundtrip_payload(text)


def test_the_widest_64_bit_integers_come_back_unchanged():
    payload = json.loads(json.dumps(REQUEST))
    payload['task']['big'] = 2**64 - 1
    payload['task']['small'] = -(2**63)
    back = json.loads(_native.roundtrip_payload(json.dumps(payload)))
    assert back == payload


def test_a_table_comes_back_cut_wherever_any_column_s_chunk_ends():
    a = pa.chunked_array([[1, 2], [3]])
    b = pa.chunked_array([['x'], ['y', 'z']])
    table = pa.Table.from_arrays([a, b], names=['a', 'b'])
    back = _native.roundtrip_arrow(table)
    assert back.equals(table)
    for name in ('a', 'b'):
        lengths = [len(chunk) for chunk in back.column(name).chunks]
        assert lengths == [1, 1, 1]


def test_a_table_chunked_alike_keeps_its_chunks():
    a = pa.chunked_array([[1, 2], [3]])
    b = pa.chunked_array([['x', 'y'], ['z']])
    table = pa.Table.from_arrays([a, b], names=['a', 'b'])
    back = _native.roundtrip_arrow(table)
    assert back.equals(table)
    assert [len(c) for c in back.column('a').chunks] == [2, 1]


PQL_SCHEMA = {
    'tables': {
        'users': {
            'cols': [
                {'name': 'user_id', 'stype': 'ID', 'dtype': 'int'},
                {'name': 'age', 'stype': 'numerical', 'dtype': 'int'},
            ],
            'pkey': 'user_id',
        },
    },
    'col_groups': [],
}


def test_pql_validate_answers_with_a_tree_or_errors():
    schema = json.dumps(PQL_SCHEMA)
    tree = json.loads(
        _native.pql_validate(
            'PREDICT users.age FOR EACH users.user_id', schema, 'FINE_TUNING'
        )
    )
    assert tree['tree']['target_ast']['fqn'] == 'users.age'
    assert tree['warnings'] == []
    refused = json.loads(
        _native.pql_validate('PREDICT', schema, 'RELATIONAL', None)
    )
    assert [e['code'] for e in refused['errors']] == ['PQL_PARSE_SYNTAX']


@pytest.mark.parametrize(
    ('args', 'error'),
    [
        (('PREDICT', '{', 'RELATIONAL', None), ValueError),
        (
            ('PREDICT', '{"tables": {}, "col_groups": []}', 'NONE', None),
            ValueError,
        ),
        (
            ('PREDICT', '{"tables": {}, "col_groups": []}', 'TABULAR', 'X'),
            ValueError,
        ),
        (
            (None, '{"tables": {}, "col_groups": []}', 'TABULAR', None),
            TypeError,
        ),
    ],
)
def test_pql_validate_refuses_what_it_cannot_read(args, error):
    with pytest.raises(error):
        _native.pql_validate(*args)


def test_pql_parse_date_reads_as_pandas_does():
    assert json.loads(_native.pql_parse_date('29.02.2024')) == {
        'value': '2024-02-29T00:00:00',
        'warnings': [
            {
                'category': 'builtins.UserWarning',
                'message': 'Parsing dates in %d.%m.%Y format when '
                'dayfirst=False (the default) was specified. Pass '
                '`dayfirst=True` or specify a format to silence this '
                'warning.',
            }
        ],
    }
    assert json.loads(_native.pql_parse_date('2023-02-29'))['value'] is None


def test_pql_parse_date_warns_on_a_year_below_1000_as_this_python_would():
    """pandas guesses a day-first format for ``13/01/0999``, and so warns,
    only where Python's ``strftime`` writes the year as ``0999``: on macOS,
    and on Linux from Python 3.14."""
    import datetime

    pads = datetime.datetime(999, 1, 1).strftime('%Y') == '0999'
    reading = json.loads(_native.pql_parse_date('13/01/0999'))
    assert reading['value'] == '0999-01-13T00:00:00'
    assert len(reading['warnings']) == (1 if pads else 0)


def test_dialect_render_answers_a_statement_and_a_value():
    sql, params = _native.dialect_render(
        'sqlite',
        'holds_any',
        json.dumps({'table': 't', 'identity': 'k', 'values': ['a', 'é']}),
    )
    assert sql == (
        'SELECT 1 FROM t\nWHERE (k) IN (SELECT value FROM json_each(?))\n'
        'LIMIT 1'
    )
    assert json.loads(params) == ['["a","\\u00e9"]']
    sql, result = _native.dialect_render(
        'postgres', 'sql.quote_ident', json.dumps({'ident': 'a"b', 'char': '"'})
    )
    assert sql is None
    assert json.loads(result) == '"a""b"'


@pytest.mark.parametrize(
    ('backend', 'function', 'args'),
    [
        ('mysql', 'list_tables', '{}'),
        ('sqlite', 'no_such_function', '{}'),
        ('sqlite', 'table_info', '{"table": 5}'),
        ('sqlite', 'table_info', '{"table": "t", "extra": 1}'),
        ('sqlite', 'list_tables', 'not json'),
    ],
)
def test_dialect_render_refuses_what_it_cannot_read(backend, function, args):
    with pytest.raises(ValueError):
        _native.dialect_render(backend, function, args)


@pytest.mark.parametrize('name', ['id', 'id\x85', 'id '])
def test_a_missing_column_is_a_key_error_of_its_name(name):
    # As `df[name]` raises it: the name is the argument, so the text is its
    # repr once, with what `str.isprintable` rejects escaped.
    graph = {
        'tables': [
            {
                'name': 'users',
                'primary_key': name,
                'columns': [{'name': name, 'dtype': 'int'}],
            }
        ],
        'edges': [],
    }
    with pytest.raises(KeyError) as raised:
        _native.graph_store(
            {'users': pa.table({'x': [1, 2]})}, json.dumps(graph)
        )
    assert raised.value.args == (name,)
    assert str(raised.value) == repr(name)


def test_an_edge_to_a_table_without_a_primary_key_is_key_error_none():
    # LocalGraphStore reaches `df[None]`, whose KeyError has None as its
    # argument. Graph.validate keeps such a graph from the public API.
    graph = {
        'tables': [
            {
                'name': 'users',
                'primary_key': None,
                'columns': [{'name': 'id', 'dtype': 'int'}],
            },
            {
                'name': 'orders',
                'primary_key': 'oid',
                'columns': [
                    {'name': 'oid', 'dtype': 'int'},
                    {'name': 'uid', 'dtype': 'int'},
                ],
            },
        ],
        'edges': [{'src_table': 'orders', 'fkey': 'uid', 'dst_table': 'users'}],
    }
    tables = {
        'users': pa.table({'id': [1, 2]}),
        'orders': pa.table({'oid': [1, 2], 'uid': [1, 2]}),
    }
    with pytest.raises(KeyError) as raised:
        _native.graph_store(tables, json.dumps(graph))
    assert raised.value.args == (None,)
    assert str(raised.value) == 'None'


def test_a_container_met_again_inside_itself_is_written_as_a_reference_back():
    # Plain JSON cannot hold a cycle; the core refuses the reference back as
    # json_ready refuses the cycle, rather than the encoder recursing until
    # Python gives up.
    from sdm_core.answers import encode

    holding: dict = {}
    holding['self'] = holding
    nested: list = [1]
    nested.append({'again': nested})
    shared = [1]
    assert encode({'options': holding}) == {
        'options': {'self': {'$refers_back': 'dict'}}
    }
    assert encode(nested) == [1, {'again': {'$refers_back': 'list'}}]
    # A value held twice side by side is no cycle.
    assert encode([shared, shared]) == [[1], [1]]
