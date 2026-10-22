# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The capture writer's own tests (``contract/runners/captures.py``, T3.3;
design, choice 2): placeholders written for every private name, in each
case form, in the capture's text and its tables' cells, names and
metadata; the write failing, with no file written, on any private name
the environment holds, named by its variable, never its value; the
converter's chain; a table read back only when its content is its name. The acceptance tests are the test
author's.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'contract/tests/t33-capture-v1'


def _module(name: str) -> ModuleType:
    key = f'capture_tests_{name}'
    if key in sys.modules:
        return sys.modules[key]
    path = ROOT / 'contract/runners' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


captures = _module('captures')

ENVIRON = {
    'SNOWFLAKE_USERNAME': 'jdoe_private',
    'SNOWFLAKE_URL': 'https://acme-xy123.example.invalid',
    'SNOWFLAKE_ACCESS_TOKEN': 'tok-SECRET-0001',
    'SDM_CAPTURE_CATALOG': 'Real_Cat',
    'SDM_CAPTURE_SCHEMA': 'Hidden_Schema',
    'SDM_LIVE_SOMETHING_ELSE': 'other-resource',
    'HOME': '/home/someone',
}


def _capture(**overrides):
    found = {
        'case': 'c1',
        'leg': 'snowflake',
        'recording_commit': 'a' * 40,
        'reference_commit': 'b' * 40,
        'versions': {'python': '3.12'},
        'input_hash': '0' * 64,
        'session': {'time_zone': 'UTC', 'null_order': 'nulls_last'},
        'reference': {'outcome': 'ok'},
        'tables': {'users': pa.table({'id': [1, 2]})},
    }
    found.update(overrides)
    return found


def test_private_values_are_read_from_the_environment() -> None:
    found = captures.private_values(ENVIRON)
    assert found['SNOWFLAKE_USERNAME'] == ['jdoe_private']
    assert found['SNOWFLAKE_URL'][1:] == [
        'acme-xy123.example.invalid',
        'acme-xy123',
    ]
    assert found['SDM_LIVE_SOMETHING_ELSE'] == ['other-resource']
    assert 'HOME' not in found


def test_every_text_has_its_placeholder_in_each_case_form(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            'TABLE_CATALOG': ['REAL_CAT', 'real_cat', 'Real_Cat', None],
            'nested': [
                {'name': 'Real_Cat.Hidden_Schema.users'},
                None,
                {'name': 'x'},
                {'name': None},
            ],
            'listed': [['HIDDEN_SCHEMA'], [], None, ['acme-xy123']],
        }
    ).replace_schema_metadata({'origin': 'Real_Cat'})
    found, files = captures.document(
        _capture(
            reference={'message': 'no table Real_Cat.Hidden_Schema.users'},
            tables={'users': table},
            statements=[
                {
                    'text': 'DESCRIBE TABLE "REAL_CAT"."HIDDEN_SCHEMA"."users"',
                    'parameters': ['hidden_schema'],
                    'result': table,
                }
            ],
        ),
        ENVIRON,
    )
    assert found['reference'] == {
        'message': 'no table {catalog}.{schema}.users'
    }
    statement = found['statements'][0]
    assert statement['text'] == 'DESCRIBE TABLE "{CATALOG}"."{SCHEMA}"."users"'
    assert statement['parameters'] == ['{schema}']
    stored = captures.read_table(files[found['tables'][0]['data']])
    assert stored.column('TABLE_CATALOG').to_pylist() == [
        '{CATALOG}',
        '{catalog}',
        '{catalog}',
        None,
    ]
    assert stored.column('nested').to_pylist()[0] == {
        'name': '{catalog}.{schema}.users'
    }
    assert stored.column('listed').to_pylist() == [
        ['{SCHEMA}'],
        [],
        None,
        ['{host}'],
    ]
    assert stored.schema.metadata == {b'origin': b'{catalog}'}


@pytest.mark.parametrize(
    'where',
    ['reference', 'cell', 'field', 'statement', 'binary', 'token', 'prefix'],
)
def test_a_private_name_left_fails_the_write_and_writes_nothing(
    tmp_path: Path, where: str
) -> None:
    found = _capture()
    if where == 'reference':
        found['reference'] = {'message': 'grant TOK-SECRET-0001'}
    elif where == 'cell':
        found['tables'] = {'users': pa.table({'who': ['tok-secret-0001']})}
    elif where == 'field':
        found['tables'] = {'users': pa.table({'by_tok-secret-0001': [1]})}
    elif where == 'statement':
        found['statements'] = [{'text': "SELECT 'tok-secret-0001'"}]
    elif where == 'binary':
        found['tables'] = {'users': pa.table({'b': [b'xx tok-secret-0001']})}
    elif where == 'token':
        found['draws'] = [
            {
                'table': 'users',
                'seed': 1,
                'fraction': '1.0',
                'rows': pa.table({'t': ['TOK-secret-0001']}),
            }
        ]
    else:
        found['reference'] = {'message': 'reached OTHER-RESOURCE'}
    with pytest.raises(captures.PrivateName) as raised:
        captures.write(tmp_path / 'capture', found, environ=ENVIRON)
    message = str(raised.value)
    assert 'the value of S' in message
    for value in ENVIRON.values():
        assert value.lower() not in message.lower()
    assert not any(tmp_path.iterdir())


def test_a_written_capture_reads_back(tmp_path: Path) -> None:
    draw = {
        'table': 'users',
        'seed': 2**40 + 3,
        'fraction': '1e-05',
        'rows': pa.table({'id': [2]}),
    }
    path = captures.write(tmp_path / 'c', _capture(draws=[draw]), environ={})
    back = captures.read(path)
    assert back['format'] == captures.FORMAT
    assert back['draws'][0]['seed'] == 2**40 + 3
    assert back['draws'][0]['fraction'] == '1e-05'
    assert back['draws'][0]['rows'].to_pylist() == [{'id': 2}]
    assert back['tables']['users'].column('id').to_pylist() == [1, 2]
    with pytest.raises(captures.CaptureError, match='written once'):
        captures.write(path, _capture(), environ={})


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        (
            {
                'draws': [
                    {'table': 't', 'seed': 1.5, 'fraction': '1', 'rows': None}
                ]
            },
            'integer',
        ),
        (
            {
                'draws': [
                    {'table': 't', 'seed': 1, 'fraction': 10.0, 'rows': None}
                ]
            },
            'as written',
        ),
        ({'recording_commit': 'abc1234'}, 'full SHA'),
        ({'session': {'time_zone': 'UTC'}}, 'null_order'),
        ({'leg': 'duckdb'}, 'no leg'),
        ({'extra': 1}, 'unknown'),
    ],
)
def test_a_capture_is_refused_by_name_where_it_breaks_the_format(
    tmp_path: Path, change: dict, message: str
) -> None:
    with pytest.raises(captures.CaptureError, match=message):
        captures.document(_capture(**change), {})


def test_a_value_nested_past_recursion_is_refused_by_name(
    tmp_path: Path,
) -> None:
    deep: dict = {}
    cursor = deep
    for _ in range(5000):
        cursor['a'] = {}
        cursor = cursor['a']
    with pytest.raises(captures.CaptureError, match='too deep'):
        captures.document(_capture(reference=deep), {})
    (tmp_path / 'capture.json').write_text('[' * 100000 + ']' * 100000)
    with pytest.raises(captures.CaptureError, match='RecursionError'):
        captures.read(tmp_path)


def test_a_converter_chain_is_read_and_a_directory_converted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = json.loads((FIXTURE / 'capture.json').read_text())
    monkeypatch.setattr(captures, 'FORMAT', 'sdm-capture/2')
    monkeypatch.setattr(
        captures, 'VERSIONS', ('sdm-capture/1', 'sdm-capture/2')
    )
    monkeypatch.setitem(
        captures.CONVERTERS,
        'sdm-capture/1',
        lambda old: {**old, 'format': 'sdm-capture/2', 'added': True},
    )
    upgraded = captures.upgrade(dict(found))
    assert upgraded['format'] == 'sdm-capture/2' and upgraded['added']
    (tmp_path / 'leg/case').mkdir(parents=True)
    (tmp_path / 'leg/case/capture.json').write_text(json.dumps(found))
    assert captures.convert(tmp_path) == [tmp_path / 'leg/case/capture.json']
    assert captures.read_document(tmp_path / 'leg/case')['added'] is True
    assert captures.convert(tmp_path) == []


def test_a_table_whose_content_is_not_its_name_is_refused(
    tmp_path: Path,
) -> None:
    path = captures.write(tmp_path / 'c', _capture(), environ={})
    name = captures.read_document(path)['tables'][0]['data']
    (path / name).write_bytes(captures.table_bytes(pa.table({'id': [9]})))
    with pytest.raises(captures.CaptureError, match='not its name'):
        captures.read(path)


def test_every_sdm_capture_variable_is_a_placeholder() -> None:
    environ = {
        'SDM_CAPTURE_CATALOG': 'Real_Cat',
        'SDM_CAPTURE_SHOP_METRIC_VIEW': 'secret_view_x',
        'SDM_CAPTURE_SOMETHING_NEW': 'other_name_y',
    }
    found, _ = captures.document(
        _capture(
            reference={
                'message': 'Real_Cat secret_view_x OTHER_NAME_Y',
            }
        ),
        environ,
    )
    assert found['reference'] == {
        'message': '{catalog} {shop_metric_view} {SOMETHING_NEW}'
    }


def test_a_leg_s_manifest_lists_its_captures(tmp_path: Path) -> None:
    leg = tmp_path / 'snowflake'
    for case in ('c1', 'c2'):
        captures.write(leg / case, _capture(case=case), environ={})
    path = captures.write_manifest(leg)
    manifest = json.loads(path.read_text())
    assert manifest['format'] == captures.FORMAT
    assert manifest['recording_commit'] == 'a' * 40
    assert manifest['legs'] == {'snowflake': {'python': '3.12'}}
    assert manifest['cases'] == {
        'c1@snowflake': {'input_hash': '0' * 64, 'capture': 'c1'},
        'c2@snowflake': {'input_hash': '0' * 64, 'capture': 'c2'},
    }
    captures.write(
        leg / 'c3', _capture(case='c3', recording_commit='c' * 40), environ={}
    )
    with pytest.raises(captures.CaptureError, match='2 recordings'):
        captures.write_manifest(leg)


def test_a_recording_mode_line_is_written_as_a_capture(tmp_path: Path) -> None:
    import base64
    import io

    def text(table: pa.Table) -> str:
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return base64.b64encode(sink.getvalue()).decode()

    sql = 'SELECT * FROM `{catalog}`.`{schema}`.`users` TABLESAMPLE (12.5 PERCENT) REPEATABLE (7)'
    line = {
        'format': 'sdm-record/1',
        'case': 'c1@databricks-sql',
        'leg': 'databricks-sql',
        'input': '0' * 64,
        'recording': {'commit': 'a' * 40, 'reference': 'b' * 40},
        'versions': {'python': '3.12'},
        'zone': 'Etc/UTC',
        'observation': {'case': 'c1@databricks-sql'},
        'statements': [
            {
                'sql': sql,
                'parameters': [{'type': 'builtins.int', 'value': '3'}],
                'error': None,
                'results': [{'arrow': text(pa.table({'id': [2]}))}],
            }
        ],
        'tables': [{'name': 'users', 'arrow': text(pa.table({'id': [1, 2]}))}],
        'relations': [],
        'draws': [
            {
                'statement': 0,
                'kind': 'tablesample',
                'table': '`{catalog}`.`{schema}`.`users`',
                'fraction': '12.5',
                'seed': '7',
            }
        ],
        'incomplete': [],
    }
    recording = tmp_path / 'recording.jsonl'
    recording.write_text(json.dumps(line) + '\n')
    (manifest,) = captures.import_recording(
        recording, tmp_path / 'out', environ={}
    )
    back = captures.read(tmp_path / 'out/databricks-sql/c1')
    assert (
        back['case'] == 'c1' and back['session']['null_order'] == 'nulls_first'
    )
    assert back['statements'][0]['parameters'] == [3]
    assert (
        back['draws'][0]['seed'] == 7 and back['draws'][0]['fraction'] == '12.5'
    )
    assert back['draws'][0]['rows'].to_pylist() == [{'id': 2}]
    assert json.loads(manifest.read_text())['cases'] == {
        'c1@databricks-sql': {'input_hash': '0' * 64, 'capture': 'c1'}
    }


def test_a_recording_line_of_another_format_is_refused_by_name() -> None:
    with pytest.raises(captures.CaptureError, match='sdm-record/9'):
        captures.from_recording({'format': 'sdm-record/9', 'leg': 'snowflake'})


def test_a_nested_field_s_metadata_holding_a_private_name_is_refused(
    tmp_path: Path,
) -> None:
    # Review r1 P1-1: a struct child's metadata is searched too.
    child = pa.field(
        'child', pa.int64(), metadata={b'origin': b'tok-secret-0001'}
    )
    table = pa.table(
        {'parent': pa.array([{'child': 1}], type=pa.struct([child]))}
    )
    with pytest.raises(captures.PrivateName, match='SNOWFLAKE_ACCESS_TOKEN'):
        captures.write(
            tmp_path / 'c', _capture(tables={'t': table}), environ=ENVIRON
        )
    assert not any(tmp_path.iterdir())
    # A placeholder's value in a nested name or metadata is replaced.
    named = pa.field(
        'Hidden_Schema', pa.int64(), metadata={b'from': b'Real_Cat'}
    )
    held = pa.table(
        {'p': pa.array([{'Hidden_Schema': 1}], type=pa.struct([named]))}
    )
    path = captures.write(
        tmp_path / 'd', _capture(tables={'t': held}), environ=ENVIRON
    )
    back = captures.read(path)['tables']['t'].schema.field('p').type.field(0)
    assert back.name == '{schema}' and back.metadata == {b'from': b'{catalog}'}


def test_nested_types_keep_their_fields_through_write_and_read(
    tmp_path: Path,
) -> None:
    # Review r1 P2-3: a child's name and nullability are the driver's.
    element = pa.field('element', pa.int64(), nullable=False)
    member = pa.field('x', pa.string(), nullable=False)
    table = pa.table(
        {
            'l': pa.array([[1, 2], []], type=pa.list_(element)),
            's': pa.array([{'x': 'a'}, {'x': 'b'}], type=pa.struct([member])),
            'm': pa.array(
                [[('k', 1)], []],
                type=pa.map_(
                    pa.string(), pa.field('v', pa.int32(), nullable=False)
                ),
            ),
        }
    )
    for environ in ({}, {'SDM_CAPTURE_CATALOG': 'b'}):
        path = captures.write(
            tmp_path / f'c{len(environ)}',
            _capture(tables={'t': table}),
            environ=environ,
        )
        back = captures.read(path)['tables']['t']
        for name in ('l', 's', 'm'):
            got, want = (
                back.schema.field(name).type,
                table.schema.field(name).type,
            )
            assert got == want, (environ, name, got, want)
            if name != 'm':
                # Arrow's IPC names a map's entries key and value whatever
                # they were (they are no part of its type's equality).
                assert str(got) == str(want)


def test_every_read_of_a_recorded_statement_is_kept(tmp_path: Path) -> None:
    # Review r1 P2-4 and P3-5: a fetchmany then a fetchall are one result,
    # every row kept; a named binding keeps its name.
    import base64

    def text(table: pa.Table) -> str:
        return base64.b64encode(captures.table_bytes(table)).decode()

    line = {
        'format': 'sdm-record/1',
        'case': 'c@databricks-sql',
        'leg': 'databricks-sql',
        'recording': {'commit': 'a' * 40, 'reference': 'b' * 40},
        'versions': {},
        'input': '0' * 64,
        'zone': 'UTC',
        'observation': {},
        'tables': [],
        'relations': [],
        'draws': [],
        'statements': [
            {
                'sql': 'SELECT x FROM t WHERE x = %(x)s',
                'parameters': [
                    {'name': 'x', 'type': 'builtins.int', 'value': '3'}
                ],
                'error': None,
                'results': [
                    {'arrow': text(pa.table({'x': [1]}))},
                    {'arrow': text(pa.table({'x': [2]}))},
                ],
            }
        ],
    }
    found = captures.from_recording(line)
    assert found['statements'][0]['result'].to_pylist() == [{'x': 1}, {'x': 2}]
    assert found['statements'][0]['parameters'] == [{'name': 'x', 'value': 3}]
    line['statements'][0]['results'].append({'arrow': None})
    with pytest.raises(captures.CaptureError, match='kept no table'):
        captures.from_recording(line)
