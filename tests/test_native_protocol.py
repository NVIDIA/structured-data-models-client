# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current relation schema, native version handshake and public SQL behavior.

These assert supported shapes and results, not migration-version history.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / 'rust/crates/sdm-contract/schema'
LOCK = SCHEMA_DIR / 'contract.lock'
VECTORS = ROOT / 'rust/crates/sdm-contract/tests/contract-26-vectors.json'

SOURCE_TABLES = ('NamedTable', 'DescribedTable', 'HeldTable')


# --- helpers ---------------------------------------------------------------------


def _schema(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding='utf-8'))
    return document


def _lock_module() -> ModuleType:
    path = ROOT / 'scripts/rust_core/contract_lock.py'
    spec = importlib.util.spec_from_file_location('contract_lock_t21b', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip(node: Any) -> Any:
    r"""``node`` without its doc comments, as the lock hashes it."""
    if isinstance(node, dict):
        return {
            key: _strip(value)
            for key, value in node.items()
            if key not in ('description', 'title', 'examples')
        }
    if isinstance(node, list):
        return [_strip(item) for item in node]
    return node


def _variants(definition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    r"""A union's members by their tag's constant, or the definition itself
    under ``''``; an untagged member's fields count as the definition's."""
    if 'oneOf' not in definition:
        return {'': definition}
    found: dict[str, dict[str, Any]] = {'': {'properties': {}}}
    for member in definition['oneOf']:
        properties = member.get('properties', {})
        tags = [p['const'] for p in properties.values() if 'const' in p]
        if len(tags) == 1:
            found[tags[0]] = member
        else:
            found['']['properties'].update(properties)
    return found


def _relation_ref(node: dict[str, Any]) -> bool:
    return _strip(node) == {'$ref': '#/$defs/Relation'}


def _table_requests(schema: dict[str, Any]) -> list[str]:
    r"""The data requests whose ``table`` is a string in ``schema``."""
    return sorted(
        kind
        for kind, member in _variants(schema['$defs']['DataRequest']).items()
        if _strip(member['properties'].get('table')) == {'type': 'string'}
    )


def _validator(schema: dict[str, Any], definition: str) -> Any:
    import jsonschema

    wrapped = {
        '$schema': schema['$schema'],
        '$ref': f'#/$defs/{definition}',
        '$defs': schema['$defs'],
    }
    return jsonschema.Draft202012Validator(wrapped)


def _vectors() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
    assert document['format'] == 'sdm-contract-vectors/1'
    return document


# --- the schema and its lock ------------------------------------------------------


def _current_schema() -> dict[str, Any]:
    entries = _lock_module().parse_lock(LOCK.read_text())
    assert len(entries) == 1
    return _schema(SCHEMA_DIR / next(iter(entries.values()))[1])


def test_the_lock_matches_the_installed_native_version() -> None:
    from sdm_client._rust.run import native

    entries = _lock_module().parse_lock(LOCK.read_text())
    current = native().CONTRACT_VERSION
    assert list(entries) == [current]
    assert entries[current][1] == f'sdm-contract-{current}.json'
    assert _current_schema()['$id'] == f'sdm-contract/{current}'
    assert _lock_module().main(['--no-base']) == 0


def test_the_core_speaks_the_locked_version_and_refuses_a_mismatched_host() -> (
    None
):
    # Reject a mismatched host before interpreting its operation or attachments.
    from sdm_client._rust import codec, errors, facts
    from sdm_client._rust.run import native

    core = native()
    assert core.CONTRACT_VERSION == max(
        _lock_module().parse_lock(LOCK.read_text())
    )
    start = codec.encode(
        {
            'message': 'start',
            'contract': core.CONTRACT_VERSION - 1,
            'facts': facts.host_facts(core),
            'operation': {
                'op': 'describe_source',
                'provider': {'provider': 'sql', 'dialect': 'duckdb'},
                'source': 1,
                'check_keys': True,
            },
        }
    )
    spec = {
        'tables': [{'name': 'users', 'source': {'source': ['users']}}],
        'primary_keys': {},
        'time_columns': {},
    }
    with pytest.raises(core.CoreError) as raised:
        core.Run.start(start, {1: codec.attachment(spec)})
    error = errors.error_from_core(
        raised.value.control, raised.value.attachments
    )
    assert error.code == 'INVALID_MESSAGE'
    assert error.details['contract'] == core.CONTRACT_VERSION - 1


def test_relation_sources_are_explicit_parts_and_derived_fields_are_optional() -> (
    None
):
    relation = _strip(_current_schema()['$defs']['Relation'])
    assert relation['type'] == 'object'
    assert relation['required'] == ['source']
    assert relation['additionalProperties'] is False
    assert set(relation['properties']) == {
        'source',
        'columns',
        'filter',
        'found',
    }
    assert relation['properties']['source'] == {
        'type': 'array',
        'items': {'type': 'string'},
    }


def test_every_table_source_is_a_relation() -> None:
    schema = _current_schema()
    requests = _variants(schema['$defs']['DataRequest'])
    kinds = [
        'column_ranges',
        'count_rows',
        'lookup',
        'neighbors',
        'preview',
        'repeated_key',
        'sample_entities',
        'scan',
    ]
    for kind in kinds:
        assert _relation_ref(requests[kind]['properties']['table']), kind
        assert 'table' in requests[kind]['required'], kind
    assert _table_requests(schema) == []
    for name in SOURCE_TABLES:
        definition = schema['$defs'][name]
        assert _relation_ref(definition['properties']['source']), name
        assert 'source' in definition['required'], name


def test_neighbors_and_lookup_take_a_key_of_several_columns() -> None:
    requests = _variants(_current_schema()['$defs']['DataRequest'])
    for kind in ('neighbors', 'lookup', 'repeated_key'):
        assert _strip(requests[kind]['properties']['key']) == {
            'type': 'array',
            'items': {'type': 'string'},
        }
        assert 'key' in requests[kind]['required']


def test_every_provider_declares_its_largest_seed() -> None:
    provider = _current_schema()['$defs']['ProviderCapabilities']
    assert 'max_seed' in provider['required']
    declared = _strip(provider['properties']['max_seed'])
    assert declared.get('type') == 'integer'
    assert 'anyOf' not in declared and 'oneOf' not in declared


def test_python_schema_validates_supported_shapes_and_rejects_malformed_ones() -> (
    None
):
    document, schema = _vectors(), _current_schema()
    current = int(schema['$id'].split('/')[-1])
    for vector in document['accept'] + document['refuse']:
        expected = vector in document['accept'] or current > vector.get(
            'refused_through', current
        )
        assert (
            _validator(schema, vector['definition']).is_valid(vector['value'])
            == expected
        ), vector['name']


# --- the public route ------------------------------------------------------------


def _source_scenarios() -> ModuleType:
    path = ROOT / 'tests/support/sql_source_scenarios.py'
    spec = importlib.util.spec_from_file_location(
        'test_native_protocol_source_scenarios', path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_dotted_source_name_still_reads_as_one_table() -> None:
    # A dotted source name still reads as one table: `orders` held in a
    # table named `a.b` materializes the plain base's payload, though a
    # schema `a` holds a table `b` whose prices differ.
    record = _source_scenarios()
    plain, _ = record.run('orders', ())
    dotted, statements = record.run('a.b', record.DOTTED)
    assert dotted == plain
    # Not vacuous: reading the decoy's prices changes the payload.
    decoy = (
        'CREATE TABLE priced AS SELECT order_id, user_id, price * 100 AS price,'
        ' ts FROM orders',
    )
    other, _ = record.run('priced', decoy)
    assert other != plain
    # Each statement names the table as one quoted identifier or text.
    texts = [text for fetch in statements for text in fetch]
    assert any('"a.b"' in text for text in texts)
    for text in texts:
        assert not re.search(r'"a"\s*\.\s*"b"', text), text
        assert not re.search(
            r'\ba\s*\.\s*b\b', text.replace('"a.b"', '').replace("'a.b'", '')
        ), text


def test_the_dotted_statements_are_the_plain_ones_on_another_name() -> None:
    # Compare executed plans after a physical table rename, including the
    # catalog lookup needed when a single identifier contains a dot.
    record = _source_scenarios()
    _, plain = record.run('orders', ())
    _, dotted = record.run('a.b', record.DOTTED)
    assert len(plain) == len(dotted)
    renamed = 0
    for one, other in zip(plain, dotted, strict=True):
        if one == ['PRAGMA table_info("orders")']:
            assert len(other) == 1 and "c.table_name = 'a.b'" in other[0]
            continue
        expected = [
            text.replace('"orders"', '"a.b"').replace("'orders'", "'a.b'")
            for text in one
        ]
        assert other == expected
        renamed += expected != one
    assert renamed >= 5


def test_executed_sql_preserves_labels_history_and_prediction_order() -> None:
    import pandas as pd

    record = _source_scenarios()
    payload, _ = record.run('orders', ())
    origin = pd.Timestamp('2023-01-01', tz='UTC')
    anchor = pd.Timestamp(record.ANCHOR, tz='UTC')
    expected_days = {
        (anchor - pd.Timedelta(days=30 * step) - origin).days
        for step in range(1, 15)
    }
    seen = set()
    for half in ('context', 'predict'):
        rows = payload[half]['instance_table']['rows']
        assert len(rows) == (112 if half == 'context' else 2)
        instances = {}
        for instance, user, stamp, *label in rows:
            day = (pd.Timestamp(stamp) - origin).days
            assert instance not in instances
            instances[instance] = (user, day)
            if half == 'context':
                assert (user, day) not in seen
                seen.add((user, day))
                expected = sum(
                    10 + order * 1.5
                    for order in range(80)
                    if order % 8 == user and day < order * 7 <= day + 30
                )
                assert label == [float(expected)]
                assert type(label[0]) is float
            else:
                assert not label
                assert day == (anchor - origin).days
        related = payload[half]['related_tables']
        people = related['users']
        assert people['columns'] == ['instance_id', 'user_id', 'age', 'country']
        assert people['rows'] == [
            [instance, user, 20 + user, ['US', 'DE', 'IN'][user % 3]]
            for instance, (user, _) in instances.items()
        ]
        history = related['orders']
        assert history['columns'] == [
            'instance_id',
            '__node_id',
            'price',
            'ts',
            'user_id',
        ]
        expected_history = sorted(
            (
                instance,
                10 + order * 1.5,
                (origin + pd.Timedelta(days=order * 7)).isoformat(),
                user,
            )
            for instance, (user, day) in instances.items()
            for order in range(80)
            if order % 8 == user and order * 7 <= day
        )
        actual = sorted(
            (instance, price, pd.Timestamp(stamp).isoformat(), user)
            for instance, _, price, stamp, user in history['rows']
        )
        assert actual == expected_history
        for instance, (user, day) in instances.items():
            actual_orders = [
                (price - 10) / 1.5
                for row_instance, _, price, _, _ in history['rows']
                if row_instance == instance
            ]
            expected_orders = [
                order
                for order in reversed(range(80))
                if order % 8 == user and order * 7 <= day
            ]
            assert actual_orders == expected_orders
        assert len({(r[0], r[1]) for r in history['rows']}) == len(
            history['rows']
        )
    assert seen == {(user, day) for user in range(8) for day in expected_days}
    context = payload['context']['instance_table']['rows']
    assert [r[2] for r in context] == sorted(
        (r[2] for r in context), reverse=True
    )
    assert [r[1] for r in payload['predict']['instance_table']['rows']] == [
        0,
        1,
    ]


def _describe(connection: Any, parts: list[str]) -> Any:
    r"""``describe_source`` over ``connection`` with ``orders`` read from the
    relation of ``parts``, through the Python host's own run."""
    from sdm_client._rust import codec
    from sdm_client._rust.providers.duckdb import DuckDBProvider
    from sdm_client._rust.run import run

    spec = {
        'tables': [{'name': 'orders', 'source': {'source': parts}}],
        'primary_keys': {},
        'time_columns': {},
    }
    operation = {
        'op': 'describe_source',
        'provider': {'provider': 'sql', 'dialect': 'duckdb'},
        'source': 1,
        'check_keys': True,
    }
    finished = run(
        operation,
        {1: codec.attachment(copy.deepcopy(spec))},
        provider=DuckDBProvider(connection),
    )
    return json.loads(finished.take(finished.result['description']))


def test_a_relation_duckdb_cannot_read_is_refused_by_name() -> None:
    # Choice 12: a relation with no part, or more parts than DuckDB takes,
    # is refused by name (rule `relation`); the same run over a one-part
    # dotted relation describes the table, so the refusal is the parts'.
    import duckdb
    from sdm_client import StructuredDataError

    record = _source_scenarios()
    connection = duckdb.connect()
    for statement in [*record.shop(), *record.DOTTED]:
        connection.execute(statement)
    described = _describe(connection, ['a.b'])
    (table,) = described['tables']
    assert table['source'] == {'source': ['a.b']}
    assert [c['name'] for c in table['columns']] == [
        'order_id',
        'user_id',
        'price',
        'ts',
    ]
    for parts in ([], ['memory', 'a', 'b', 'c']):
        with pytest.raises(StructuredDataError) as raised:
            _describe(connection, parts)
        assert raised.value.details.get('rule') == 'relation', (
            parts,
            raised.value.details,
        )


# --- a relation's parts follow DuckDB's identifier case (review round 1, P2) --


def _keyed_tables() -> Any:
    r"""A DuckDB database holding ``u(code VARCHAR UNIQUE, n DOUBLE)`` in
    ``main``, in a schema ``s``, and as ``s."q""t"``, a table name holding a
    double quote, each with the same two rows."""
    import duckdb

    connection = duckdb.connect()
    connection.execute('CREATE SCHEMA s')
    for table in ('u', 's.u', 's."q""t"'):
        connection.execute(
            f'CREATE TABLE {table} (code VARCHAR UNIQUE, n DOUBLE)'
        )
        connection.execute(f"INSERT INTO {table} VALUES ('a', 1), ('b', 2)")
    return connection


def _keys(described: Any) -> tuple[list[str], dict[str, bool]]:
    r"""A one-table description's primary key and each column's unique
    flag."""
    (table,) = described['tables']
    return table['primary_key'], {
        column['name']: column['unique'] for column in table['columns']
    }


# The declared unique code is the inferred primary key under every spelling.
EXPECTED_KEYS = (['code'], {'code': True, 'n': False})


def test_the_main_table_retains_its_declared_unique_key() -> None:
    # The control: one part, the table's own spelling.
    assert _keys(_describe(_keyed_tables(), ['u'])) == EXPECTED_KEYS


@pytest.mark.parametrize(
    'parts',
    [
        ['s', 'u'],
        ['S', 'u'],
        ['s', 'U'],
        ['S', 'U'],
        ['memory', 's', 'u'],
        ['MEMORY', 'S', 'u'],
        ['Memory', 's', 'U'],
    ],
    ids='.'.join,
)
def test_a_schema_qualified_table_keeps_its_keys_in_any_case(
    parts: list[str],
) -> None:
    # DuckDB resolves an identifier whatever its case, quoted or not, so
    # each spelling names s.u (shown by DuckDB itself below), and the
    # description's unique column and primary key are the same for each.
    connection = _keyed_tables()
    quoted = '.'.join('"' + part.replace('"', '""') + '"' for part in parts)
    assert connection.execute(f'SELECT count(*) FROM {quoted}').fetchone() == (
        2,
    )
    assert _keys(_describe(connection, parts)) == EXPECTED_KEYS, parts


@pytest.mark.parametrize(
    'parts',
    [
        ['s', 'q"t'],
        ['S', 'q"t'],
        ['s', 'Q"T'],
        ['MEMORY', 'S', 'q"t'],
    ],
    ids='.'.join,
)
def test_a_quoted_table_name_keeps_its_keys_when_its_qualifier_changes_case(
    parts: list[str],
) -> None:
    # A table name holding a double quote is read by name from the catalog;
    # a schema or database qualifier of another case still names it.
    connection = _keyed_tables()
    quoted = '.'.join('"' + part.replace('"', '""') + '"' for part in parts)
    assert connection.execute(f'SELECT count(*) FROM {quoted}').fetchone() == (
        2,
    )
    assert _keys(_describe(connection, parts)) == EXPECTED_KEYS, parts
