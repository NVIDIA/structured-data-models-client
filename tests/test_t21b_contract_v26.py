# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract v26 (T2.1b): the design's choice 1 vocabulary with choice 12's
table source, held at the schema, the lock, the loaders and the public route.

The design's row for T2.1b, each exit with the tests below that prove it:

- **Schema lock**: v26 is locked to its own file, and v25's merged entry
  and file are untouched (``test_v25_*``, ``test_the_lock_*``,
  ``test_the_core_speaks_*``). v26 was merged into rust-core mid-phase, so
  T2.12's key label is v27, the phase's working entry beside it.
- **Both loaders**: ``rust/crates/sdm-contract/tests/contract-26-vectors.json``
  holds accept and refuse vectors; the Python loader here validates each
  against ``sdm-contract-26.json`` (``test_the_python_loader_*``), and the
  Rust loader (written beside the implementation, in ``sdm-contract``'s
  tests) deserializes each into the type it names. Every v25 vector among
  the refusals is shown valid under v25, so its refusal is the bump's.
- **The binding jobs green on the bump**: the native module speaks 27, and
  a host speaking 25 is refused by name (``test_the_core_speaks_*``).
- **SQL goldens unchanged**: the statements the core renders on the public
  route over the ``shop`` base, recorded at v25 by
  ``tests/golden/t21b_statements_record.py``, are rendered the same at v26
  (``test_the_core_renders_the_v25_statements``).
- **A dotted source name still reads as one table**: ``orders`` held in a
  DuckDB table named ``a.b``, beside a schema ``a`` holding a table ``b`` of
  other prices, materializes the plain base's payload, and every statement
  names ``"a.b"`` as one identifier (``test_a_dotted_source_*``).
- **More parts than a dialect takes, or none, refused by name** (choice 12,
  rule ``relation``): ``test_a_relation_duckdb_cannot_read_*``.

The interface assumed (choice 12, verbatim where the design spells it):
``$defs/Relation`` is ``{"source": [<part>, ...]}``, an object of that one
required field, a list of strings, nothing else (KO2-46: Phase 3 adds its
fields later); it replaces, keeping each field's name, every data request's
``table`` string and the ``source`` string of ``NamedTable``,
``DescribedTable`` and ``HeldTable``. ``neighbors`` and ``lookup`` take
``key`` as a list of column names, as ``repeated_key`` and
``sample_entities`` already do (choice 1, multi-column keys).
``ProviderCapabilities.max_seed`` is required and never null (choice 1, no
"any seed" default). ``StatePayload``'s ``x-request-state-version`` is
raised (choice 1). No v25 definition or field name is dropped.

Gating: the tests that need v26 are marked :data:`PENDING`, a strict xfail,
so CI stays green until the bump lands and turns red if one passes before
it. The implementer deletes the ``PENDING`` marker (its one definition
below) in the commit that makes them pass. The unmarked tests are controls
that pass on the base and must keep passing.
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
V25 = SCHEMA_DIR / 'sdm-contract-25.json'
V26 = SCHEMA_DIR / 'sdm-contract-26.json'
VECTORS = ROOT / 'rust/crates/sdm-contract/tests/contract-26-vectors.json'
GOLDEN = ROOT / 'tests/golden/t21b-statements.json'

#: v25's entry as merged into rust-core: never rewritten.
V25_HASH = '0c478d9bd05d6125d5daf85d3af2bd0de2c5dd17d1e8f444af9ce5e63068de90'

#: The schema tables whose source name becomes a relation (choice 12:
#: ``sdm-contract/src/ops.rs:380-415,533-534``).
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


def _fields(schema: dict[str, Any]) -> set[str]:
    r"""Every ``<definition>[/<tag>].<field>`` of ``schema``'s ``$defs``."""
    names = set()
    for name, definition in schema['$defs'].items():
        for tag, member in _variants(definition).items():
            for field in member.get('properties', {}):
                names.add(f'{name}/{tag}.{field}' if tag else f'{name}.{field}')
    return names


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


def test_v25_stays_locked_as_merged() -> None:
    # Schema lock: the merged entry and its file are never rewritten.
    lock = _lock_module()
    entries = lock.parse_lock(LOCK.read_text(encoding='utf-8'))
    assert entries[25] == (V25_HASH, 'sdm-contract-25.json')
    assert lock.digest(V25.read_bytes()) == V25_HASH
    assert _schema(V25)['$id'] == 'sdm-contract/25'


def test_the_lock_holds_v26_locked_to_its_own_file() -> None:
    # Schema lock: v26, merged into rust-core mid-phase, is locked to its
    # own file; v27 is the phase's working entry beside it (T2.12).
    lock = _lock_module()
    entries = lock.parse_lock(LOCK.read_text(encoding='utf-8'))
    assert max(entries) == 27
    digest, file = entries[26]
    assert file == 'sdm-contract-26.json'
    assert lock.digest(V26.read_bytes()) == digest
    assert _schema(V26)['$id'] == 'sdm-contract/26'
    assert lock.main(['--no-base']) == 0


def test_the_core_speaks_v27_and_refuses_a_v25_host_by_name() -> None:
    # The binding on the bump: the native module is v27 (v26 with T2.12's
    # key label), and a start written for v25 is refused by name, never
    # read (decision H25).
    from sdm_client._rust import codec, errors, facts
    from sdm_client._rust.run import native

    core = native()
    assert core.CONTRACT_VERSION == 27
    start = codec.encode(
        {
            'message': 'start',
            'contract': 25,
            'facts': facts.host_facts(core.is_posix_zone),
            'operation': {
                'op': 'describe_source',
                'provider': {'provider': 'sql', 'dialect': 'duckdb'},
                'source': 1,
                'check_keys': True,
            },
        }
    )
    spec = {
        'tables': [{'name': 'users', 'source': 'users'}],
        'primary_keys': {},
        'time_columns': {},
    }
    with pytest.raises(core.CoreError) as raised:
        core.Run.start(start, {1: codec.attachment(spec)})
    error = errors.error_from_core(
        raised.value.control, raised.value.attachments
    )
    assert error.code == 'INVALID_MESSAGE'
    assert error.details['contract'] == 25


def test_a_relation_is_a_list_of_name_parts_and_nothing_more() -> None:
    # Choice 12 and KO2-46: `Relation { source: Vec<String> }`, plain data.
    relation = _strip(_schema(V26)['$defs']['Relation'])
    assert relation['type'] == 'object'
    assert relation['required'] == ['source']
    assert relation['additionalProperties'] is False
    assert set(relation['properties']) == {'source'}
    source = relation['properties']['source']
    assert source['type'] == 'array'
    assert source['items'] == {'type': 'string'}


def test_every_table_source_is_a_relation() -> None:
    # Choice 12: the relation replaces each data request's `table` string
    # and each schema table's `source` string, the field names kept.
    v25, v26 = _schema(V25), _schema(V26)
    kinds = _table_requests(v25)
    # Not vacuous: v25 names these eight by a string.
    assert kinds == [
        'column_ranges',
        'count_rows',
        'lookup',
        'neighbors',
        'preview',
        'repeated_key',
        'sample_entities',
        'scan',
    ]
    requests = _variants(v26['$defs']['DataRequest'])
    for kind in kinds:
        assert _relation_ref(requests[kind]['properties']['table']), kind
        assert 'table' in requests[kind]['required'], kind
    assert _table_requests(v26) == []
    for name in SOURCE_TABLES:
        assert _strip(v25['$defs'][name]['properties']['source']) == {
            'type': 'string'
        }
        definition = v26['$defs'][name]
        assert _relation_ref(definition['properties']['source']), name
        assert 'source' in definition['required'], name


def test_neighbors_and_lookup_take_a_key_of_several_columns() -> None:
    # Choice 1: multi-column keys in `Neighbors` and `Lookup`, replacing the
    # single `key: String`, spelled as `repeated_key`'s key already is.
    v25 = _variants(_schema(V25)['$defs']['DataRequest'])
    v26 = _variants(_schema(V26)['$defs']['DataRequest'])
    many = _strip(v25['repeated_key']['properties']['key'])
    assert many == {'type': 'array', 'items': {'type': 'string'}}
    for kind in ('neighbors', 'lookup'):
        assert _strip(v25[kind]['properties']['key']) == {'type': 'string'}
        assert _strip(v26[kind]['properties']['key']) == many, kind
        assert 'key' in v26[kind]['required'], kind


def test_every_provider_declares_its_largest_seed() -> None:
    # Choice 1: `max_seed` required of every provider, with no "any seed"
    # default (`P1 d:142`): neither left out nor null.
    v25 = _schema(V25)['$defs']['ProviderCapabilities']
    v26 = _schema(V26)['$defs']['ProviderCapabilities']
    assert 'max_seed' not in v25['required']
    assert 'max_seed' in v26['required']
    declared = _strip(v26['properties']['max_seed'])
    assert declared.get('type') == 'integer', declared
    assert 'anyOf' not in declared and 'oneOf' not in declared


def test_the_request_state_version_is_raised() -> None:
    # Choice 1: the `RequestState` version raised for the new state.
    def version(path: Path) -> int:
        value = _schema(path)['$defs']['StatePayload'][
            'x-request-state-version'
        ]
        assert isinstance(value, int)
        return value

    assert version(V26) > version(V25)


def test_no_v25_definition_or_field_is_dropped() -> None:
    # The bump reshapes the fields the design names and adds the rest:
    # every v25 definition and field name is still in v26 (hosts in Python
    # and Java keep their spelling).
    v25, v26 = _schema(V25), _schema(V26)
    assert set(v25['$defs']) - set(v26['$defs']) == set()
    assert _fields(v25) - _fields(v26) == set()


# --- both loaders -----------------------------------------------------------------


def test_the_vectors_cover_every_table_source_and_are_valid_v25_where_marked() -> (
    None
):
    # Both loaders, not vacuous: the vectors hold each request v25 names by
    # a string and each schema table, accepted as a relation and refused as
    # v25's string; each vector marked v25 is a valid v25 message, so its
    # refusal under v26 is the bump's.
    document, v25 = _vectors(), _schema(V25)
    kinds = _table_requests(v25)
    accepted = {
        v['value'].get('request', v['definition']) for v in document['accept']
    }
    marked = [v for v in document['refuse'] if v.get('v25')]
    refused = {v['value'].get('request', v['definition']) for v in marked}
    for name in [*kinds, *SOURCE_TABLES]:
        assert name in accepted, name
        assert name in refused, name
    for vector in marked:
        errors = list(
            _validator(v25, vector['definition']).iter_errors(vector['value'])
        )
        assert errors == [], (vector['name'], errors)
    names = [v['name'] for v in document['accept'] + document['refuse']]
    assert len(names) == len(set(names))


def test_the_python_loader_reads_and_refuses_the_v26_vectors() -> None:
    # Both loaders: the Python side, against the committed v26 schema.
    document, v26 = _vectors(), _schema(V26)
    for vector in document['accept']:
        errors = list(
            _validator(v26, vector['definition']).iter_errors(vector['value'])
        )
        assert errors == [], (vector['name'], errors)
    for vector in document['refuse']:
        validator = _validator(v26, vector['definition'])
        assert not validator.is_valid(vector['value']), vector['name']


# --- the public route ------------------------------------------------------------


def _recorder() -> ModuleType:
    path = ROOT / 'tests/golden/t21b_statements_record.py'
    spec = importlib.util.spec_from_file_location('t21b_record', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_dotted_source_name_still_reads_as_one_table() -> None:
    # A dotted source name still reads as one table: `orders` held in a
    # table named `a.b` materializes the plain base's payload, though a
    # schema `a` holds a table `b` whose prices differ.
    record = _recorder()
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
    # A golden shows it (choice 12): fetch for fetch, the dotted run's
    # statements are the plain run's with `orders` renamed `a.b`, as an
    # identifier and as text, save the column read, which reads a name
    # holding a dot from `duckdb_columns()` (register entry P1B-5).
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))['statements']
    plain, dotted = golden['plain'], golden['dotted']
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


def test_the_core_renders_the_v25_statements() -> None:
    # SQL goldens unchanged: the statements recorded at v25 are the ones the
    # core renders now. A later package that changes a rendering on purpose
    # re-records them with tests/golden/t21b_statements_record.py.
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    record = _recorder()
    for name, (orders, setup) in record.RUNS.items():
        _, statements = record.run(orders, setup)
        assert statements == golden['statements'][name], name


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

    record = _recorder()
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


#: What the reference (python-ref, 6298913c) reports of ``u`` read under
#: its own spelling, ``Schema.from_source(DuckDBSource(c), {'u': 'u'})``:
#: ``code`` its primary key, the unique column (an ``ID``), and ``n`` not.
#: The reference reads no schema-qualified name at all (``'s.u'`` is a
#: catalog error there), so the core's description of the same table under
#: any spelling DuckDB resolves to it is held to this.
REFERENCE_KEYS = (['code'], {'code': True, 'n': False})


def test_the_main_table_is_described_as_the_reference_describes_it() -> None:
    # The control: one part, the table's own spelling.
    assert _keys(_describe(_keyed_tables(), ['u'])) == REFERENCE_KEYS


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
    assert _keys(_describe(connection, parts)) == REFERENCE_KEYS, parts


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
    assert _keys(_describe(connection, parts)) == REFERENCE_KEYS, parts
