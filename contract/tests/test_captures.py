# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The capture writer's own tests (``contract/runners/captures.py``, T3.3;
design, choice 2): placeholders written for every private name, in each
case form, in the capture's text and its tables' cells, names and
metadata; the write failing, with no file written, on any private name
the environment holds, named by its variable, never its value; the
version-1 fixture (``contract/fixtures/captures/v1``) read; an unknown
version refused; the converter's chain. The acceptance tests are the test
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
FIXTURE = ROOT / 'contract/fixtures/captures/v1'


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
    'SDM_LIVE_TEST_SCHEMA': 'Hidden_Schema',
    'HOME': '/home/someone',
}


def _writer(root: Path, **overrides):
    arguments = {
        'recording_commit': 'a' * 40,
        'reference_commit': 'b' * 40,
        'runtime': {'python': '3.12'},
        'session': {'time_zone': 'UTC'},
        'placeholders': {'catalog': 'Real_Cat', 'schema': 'Hidden_Schema'},
        'environ': ENVIRON,
    }
    arguments.update(overrides)
    return captures.CaptureWriter(root, 'snowflake', **arguments)


def _parts(**overrides):
    parts = {
        'input_hash': 'sha256:' + '0' * 64,
        'reference': {'outcome': 'ok'},
        'tables': {'users': pa.table({'id': [1, 2]})},
    }
    parts.update(overrides)
    return parts


def test_private_names_are_read_from_the_environment() -> None:
    found = captures.private_names(ENVIRON)
    assert found['SNOWFLAKE_USERNAME'] == 'jdoe_private'
    assert 'acme-xy123.example.invalid' in found.values()
    assert 'acme-xy123' in found.values()
    assert found['SDM_LIVE_TEST_SCHEMA'] == 'Hidden_Schema'
    assert 'HOME' not in found


def test_every_text_has_its_placeholder_in_each_case_form(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    table = pa.table(
        {
            'TABLE_CATALOG': ['REAL_CAT', 'real_cat', 'Real_Cat', None],
            'nested': [
                {'name': 'Real_Cat.Hidden_Schema.users'},
                None,
                {'name': 'x'},
                {'name': None},
            ],
            'listed': [['HIDDEN_SCHEMA'], [], None, ['a']],
        }
    ).replace_schema_metadata({'origin': 'Real_Cat'})
    document, files = writer.document(
        'c1',
        **_parts(
            reference={'message': 'no table Real_Cat.Hidden_Schema.users'},
            tables={'users': table},
            statements=[
                {
                    'sql': 'DESCRIBE TABLE "REAL_CAT"."HIDDEN_SCHEMA"."users"',
                    'parameters': ['hidden_schema'],
                    'result': table,
                }
            ],
        ),
    )
    assert document['reference'] == {
        'message': 'no table {catalog}.{schema}.users'
    }
    statement = document['statements'][0]
    assert statement['sql'] == 'DESCRIBE TABLE "{CATALOG}"."{SCHEMA}"."users"'
    assert statement['parameters'] == ['{schema}']
    stored = captures.read_table(files[document['tables'][0]['data']])
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
        ['a'],
    ]
    assert stored.schema.metadata == {b'origin': b'{catalog}'}


@pytest.mark.parametrize(
    'where',
    ['reference', 'cell', 'field', 'statement', 'binary', 'host', 'token'],
)
def test_a_private_name_left_fails_the_write_and_writes_nothing(
    tmp_path: Path, where: str
) -> None:
    writer = _writer(tmp_path)
    parts = _parts()
    if where == 'reference':
        parts['reference'] = {'message': 'user JDOE_PRIVATE has no grant'}
    elif where == 'cell':
        parts['tables'] = {'users': pa.table({'who': ['jdoe_private']})}
    elif where == 'field':
        parts['tables'] = {'users': pa.table({'by_jdoe_private': [1]})}
    elif where == 'statement':
        parts['statements'] = [{'sql': "SELECT 'jdoe_private'"}]
    elif where == 'binary':
        parts['tables'] = {'users': pa.table({'b': [b'xx jdoe_private']})}
    elif where == 'host':
        parts['reference'] = {'message': 'reached acme-xy123'}
    else:
        parts['reference'] = {'message': 'tok-secret-0001'}
    with pytest.raises(captures.PrivateName) as raised:
        writer.write('c1', **parts)
    message = str(raised.value)
    assert 'the value of SNOWFLAKE_' in message
    for value in ENVIRON.values():
        assert value.lower() not in message.lower()
    assert not any(tmp_path.rglob('*'))


def test_a_written_capture_reads_back(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    draw = {
        'table': 'users',
        'seed': 2**40 + 3,
        'fraction': '1e-05',
        'statement': 'SELECT 1',
        'picked': pa.table({'id': [2]}),
    }
    path = writer.write('case.one', **_parts(draws=[draw]))
    document = captures.read(path)
    assert document['format'] == captures.FORMAT
    assert document['session'] == {
        'null_order': 'nulls_last',
        'time_zone': 'UTC',
    }
    assert document['draws'][0]['seed'] == 2**40 + 3
    assert document['draws'][0]['fraction'] == '1e-05'
    picked = captures.table(path.parent, document['draws'][0]['data'])
    assert picked.to_pylist() == [{'id': 2}]
    users = captures.table(path.parent, document['tables'][0]['data'])
    assert users.column('id').to_pylist() == [1, 2]


@pytest.mark.parametrize(
    ('draw', 'message'),
    [
        ({'seed': 1.5}, 'seed'),
        ({'seed': True}, 'seed'),
        ({'fraction': 10.0}, 'as written'),
        ({'rows': '5'}, 'row count'),
    ],
)
def test_a_draw_keeps_its_seed_and_fraction_as_given(
    tmp_path: Path, draw: dict, message: str
) -> None:
    entry = {'table': 't', 'picked': pa.table({'id': [1]}), **draw}
    with pytest.raises(captures.CaptureError, match=message):
        _writer(tmp_path).document('c1', **_parts(draws=[entry]))


def test_a_commit_is_a_full_sha(tmp_path: Path) -> None:
    with pytest.raises(captures.CaptureError, match='full SHA'):
        _writer(tmp_path, recording_commit='abc1234')
    with pytest.raises(captures.CaptureError, match='time zone'):
        _writer(tmp_path, session={})


def test_the_version_1_fixture_reads() -> None:
    for leg in ('databricks-sql', 'snowflake'):
        directory = FIXTURE / leg
        document = captures.read(directory / 'fixture.standin.json')
        assert document['format'] == 'sdm-capture/1'
        assert document['leg'] == leg
        names = [entry['name'] for entry in document['tables']]
        assert names == ['users', 'orders']
        for entry in document['tables'] + document['statements']:
            if entry['data'] is not None:
                captures.table(directory, entry['data'])
        assert document['draws'][0]['seed'] == 2**33 + 1
        assert document['views'][1]['refused'] == ['odd']


def test_an_unknown_version_is_refused_and_a_converter_chain_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = json.loads(
        (FIXTURE / 'snowflake/fixture.standin.json').read_text()
    )
    with pytest.raises(captures.CaptureError, match='unknown'):
        captures.upgrade({**document, 'format': 'sdm-capture/9'})
    # A later version: version 1 read through its converter, and the
    # directory rewritten by `convert`.
    monkeypatch.setattr(captures, 'FORMAT', 'sdm-capture/2')
    monkeypatch.setattr(
        captures, 'VERSIONS', ('sdm-capture/1', 'sdm-capture/2')
    )
    monkeypatch.setitem(
        captures.CONVERTERS,
        'sdm-capture/1',
        lambda old: {**old, 'format': 'sdm-capture/2', 'added': True},
    )
    upgraded = captures.upgrade(dict(document))
    assert upgraded['format'] == 'sdm-capture/2' and upgraded['added']
    (tmp_path / 'snowflake').mkdir()
    (tmp_path / 'snowflake/c.json').write_text(json.dumps(document))
    assert captures.convert(tmp_path) == [tmp_path / 'snowflake/c.json']
    assert captures.read(tmp_path / 'snowflake/c.json')['added'] is True
    assert captures.convert(tmp_path) == []


def test_a_table_whose_content_is_not_its_name_is_refused(
    tmp_path: Path,
) -> None:
    path = _writer(tmp_path).write('c1', **_parts())
    name = captures.read(path)['tables'][0]['data']
    (path.parent / name).write_bytes(
        captures.table_bytes(pa.table({'id': [9]}))
    )
    with pytest.raises(captures.CaptureError, match='not its name'):
        captures.table(path.parent, name)
