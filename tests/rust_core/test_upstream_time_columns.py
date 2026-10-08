# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sdm_core import _native

_ROOT = Path(__file__).resolve().parents[2]
_ENCODE = runpy.run_path(str(_ROOT / 'scripts/parity/_parity/values.py'))[
    'encode'
]

_TESTS = runpy.run_path(
    str(
        _ROOT / 'packages/structured-data-models-pql/tests/test_time_columns.py'
    )
)
_NAMESPACE = _TESTS['_choose'].__globals__
_ORACLE = _NAMESPACE['choose_time_column']


def _encoded(value):
    return {'$nat': True} if value is pd.NaT else _ENCODE(value)


def _choice(frame, candidates, *, full_range=None, linked=False):
    observed = []

    def observe(names):
        value = full_range(names)
        observed.append(value)
        return value

    expected = _ORACLE(
        frame,
        candidates,
        full_range=None if full_range is None else observe,
        linked=linked,
    )
    whole = observed[0] if observed else None
    arguments = {
        'frame': {
            'columns': {
                name: {'values': [_encoded(value) for value in frame[name]]}
                for name in frame
            }
        },
        'rows': len(frame),
        'candidates': list(candidates),
        'linked': linked,
        'attempted': full_range is not None,
        'whole': None
        if whole is None
        else {
            'rows': whole.rows,
            'columns': {
                name: [_encoded(value) for value in column]
                for name, column in whole.columns.items()
            },
        },
    }
    actual = json.loads(
        _native.dates_op('time_column.choose', 'null', json.dumps(arguments))
    )['output']
    wanted = dataclasses.asdict(expected)
    for key in ('refused', 'unconfirmed'):
        wanted[key] = list(wanted[key])
        for rejection in wanted[key]:
            if rejection['value'] is not None:
                rejection['value'] = str(rejection['value'])
        wanted[f'{key}_why'] = [
            rejection.why for rejection in getattr(expected, key)
        ]
    wanted['among'] = list(wanted['among'])
    assert actual == wanted
    return expected


def _name_rule(name, reason):
    frame = pd.DataFrame({name: _TESTS['DAYS'], 'ordinary': _TESTS['LATER']})
    choice = _choice(frame, list(frame))
    return any(
        rejected.column == name and rejected.reason == reason
        for rejected in choice.refused
    )


_NAMESPACE['choose_time_column'] = _choice
_NAMESPACE['is_load_stamp'] = lambda name: _name_rule(name, 'load_stamp')
_NAMESPACE['is_edit_stamp'] = lambda name: _name_rule(name, 'edit_stamp')
_NAMESPACE['is_birth_date'] = lambda name: _name_rule(name, 'birth_date')
globals().update(
    (name, value) for name, value in _TESTS.items() if name.startswith('test_')
)


def test_core_public_time_inference_does_not_import_the_oracle():
    environment = dict(os.environ, SDM_CLIENT_IMPLEMENTATION='rust')
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import pandas as pd; from sdm_client import Schema; '
            'frame=pd.DataFrame({"created_at":pd.date_range("2024-01-01",periods=2),'
            '"updated_at":pd.date_range("2025-01-01",periods=2)}); '
            'assert Schema.from_data({"events":frame})["events"].time_column=="created_at"; '
            'assert not any(name=="predictive_query" or name.startswith("predictive_query.") '
            'or name=="sdm_relational_engine" or name.startswith("sdm_relational_engine.") '
            'for name in sys.modules)',
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_public_frame_reads_no_source_and_source_batches_all_time_ranges(
    monkeypatch,
):
    import duckdb
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource
    from sdm_connectors.backends import duckdb as backend

    frame = pd.DataFrame(
        {
            'id': range(1200),
            'created_at': pd.date_range('2024-01-01', periods=1200, freq='h'),
            'updated_at': pd.date_range('2025-01-01', periods=1200, freq='h'),
        }
    )
    connection = duckdb.connect()
    try:
        connection.register('input_frame', frame)
        connection.execute('CREATE TABLE events AS SELECT * FROM input_frame')
        statements = []
        original = backend.run

        def recorded(connection, sql, params=None):
            statements.append(sql)
            return original(connection, sql, params)

        monkeypatch.setattr(backend, 'run', recorded)
        assert (
            Schema.from_data({'events': frame})['events'].time_column
            == 'created_at'
        )
        assert statements == []
        schema = Schema.from_source(
            DuckDBSource(connection), tables={'events': 'events'}
        )
        assert schema['events'].time_column == 'created_at'
        assert [
            sql for sql in statements if sql.startswith('SELECT COUNT(*), MIN(')
        ] == [
            'SELECT COUNT(*), MIN("created_at"), MAX("created_at"), COUNT("created_at"), '
            'MIN("updated_at"), MAX("updated_at"), COUNT("updated_at")\nFROM "events"'
        ]
    finally:
        connection.close()
