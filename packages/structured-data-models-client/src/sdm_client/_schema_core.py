# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a schema's objects ask the core when they are built, edited or read.

`Schema`, `Table`, `SourceTable`, `Column` and `Relationship` stay the objects
a caller holds, edits, copies and pickles. What an edit means (which columns a
key types, what a time column orders, which links are kept and what is said
about them, what is refused) is the core's: each edit hands the core what the
objects it reaches say (one column, a table held on its own, or the whole
schema), and the objects are given back what the core says they say
afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any

from sdm_core import _native

from sdm_client import _core

if TYPE_CHECKING:
    from sdm_client.schema import Column, Schema, _TableShape


# --- What the objects say, as the core reads it -----------------------------------


def _key_json(key: Any) -> Any:
    if key is None or isinstance(key, str):
        return key
    try:
        return [str(name) for name in key]
    except TypeError:
        return None


def column_json(name: Any, column: Column) -> dict[str, Any]:
    return {
        'name': name,
        'dtype': getattr(column, 'dtype', None),
        'stype': getattr(column, 'stype', None),
        'untyped': bool(getattr(column, 'untyped', False)),
        'stype_declared': bool(getattr(column, 'stype_declared', False)),
        'before_key': getattr(column, '_stype_before_key', None),
    }


def table_json(name: Any, table: _TableShape) -> dict[str, Any]:
    source_name = getattr(table, 'source_name', None)
    return {
        'name': name,
        'source_name': source_name,
        'columns': [
            column_json(key, column)
            for key, column in getattr(table, 'columns', {}).items()
        ],
        'primary_key': _key_json(getattr(table, 'primary_key', None)),
        'time_column': getattr(table, 'time_column', None),
        'duplicate_keys': getattr(table, 'duplicate_keys', None),
    }


def link_json(link: Any) -> Any:
    from sdm_client.schema import Relationship

    if isinstance(link, Relationship):
        return [
            link.source_table,
            link.source_column,
            link.target_table,
            link.target_column,
        ]
    return (
        _core.opaque(link)
        if not isinstance(link, (list, tuple))
        else {'$object': {'type': type(link).__name__, 'repr': repr(link)}}
    )


def schema_json(
    tables: Mapping[str, _TableShape], links: Iterable[Any] = ()
) -> dict[str, Any]:
    return {
        'tables': [table_json(name, table) for name, table in tables.items()],
        'relationships': [link_json(link) for link in links],
    }


# --- What the core says, put back on the objects ----------------------------------


def _key_of(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def apply_table(table: _TableShape, said: Mapping[str, Any]) -> None:
    r"""Put what the core says a table says back on it."""
    set_ = object.__setattr__
    set_(table, 'primary_key', _key_of(said.get('primary_key')))
    set_(table, 'time_column', said.get('time_column'))
    set_(table, 'duplicate_keys', said.get('duplicate_keys'))
    by_name = {c['name']: c for c in said.get('columns') or ()}
    for name in list(table.columns):
        column = table.columns[name]
        found = by_name.get(name)
        if found is None:
            continue
        apply_column(column, found)


def apply_column(column: Column, said: Mapping[str, Any]) -> None:
    set_ = object.__setattr__
    set_(column, 'stype', said.get('stype'))
    set_(column, 'untyped', bool(said.get('untyped')))
    set_(column, 'stype_declared', bool(said.get('stype_declared')))
    set_(column, '_stype_before_key', said.get('before_key'))


def _apply_schema(
    tables: Mapping[str, _TableShape], said: Mapping[str, Any]
) -> None:
    for name, found in zip(
        list(tables), said.get('tables') or (), strict=False
    ):
        apply_table(tables[name], found)


def links_of(said: Mapping[str, Any]) -> list[Any]:
    from sdm_client.schema import Relationship

    return [Relationship(*link) for link in said.get('relationships') or ()]


# --- Asking the core ----------------------------------------------------------------


def _frames_of(tables: Mapping[str, Any]) -> _core.Frames:
    r"""The frames of the tables that hold rows, named ``schema/NAME``."""
    import pandas as pd

    held = {
        f'schema/{name}': table.data
        for name, table in tables.items()
        if isinstance(getattr(table, 'data', None), pd.DataFrame)
    }
    return _core.Frames.described(held)


def edit(
    scope: str,
    tables: Mapping[str, Any],
    op: Mapping[str, Any],
    *,
    links: Iterable[Any] = (),
    rows: bool = False,
) -> dict[str, Any]:
    r"""The edit ``op`` made by the core on what ``tables`` (and ``links``)
    say: what they say afterwards. Its warnings are given; its refusal is
    raised as today's error. ``rows`` hands over the tables' frames, which
    an edit that counts a key reads."""
    frames = _frames_of(tables) if rows else _core.Frames()
    state = {
        'scope': scope,
        'schema': schema_json(tables, links),
        'op': dict(op),
        'tags': frames.tags,
    }
    found, said = _core.call(
        _native.schema_edit, json.dumps(state, default=repr), frames.tables
    )
    _core.give_json(said)
    return json.loads(found)


# --- The objects' edits -------------------------------------------------------------


def _scope_of_column(column: Column) -> tuple[str, Any, Any, Any]:
    r"""What an edit of ``column`` reaches: the column alone, its table, or
    its table's schema; and the names it is known by there."""
    table = getattr(column, '_table', None)
    if table is not None:
        key = next((k for k, c in table.columns.items() if c is column), None)
        if key is not None:
            schema = getattr(table, '_schema', None)
            if schema is not None:
                name = next(
                    (n for n, t in schema._tables.items() if t is table), None
                )
                if name is not None:
                    return 'schema', schema, name, key
            return 'table', table, table.name, key
    return 'column', None, '', getattr(column, 'name', '')


def column_init(column: Column) -> None:
    r"""``Column(name, dtype, stype)``: its meaning, checked."""
    edit(
        'column',
        {},
        {
            'op': 'column_init',
            'name': _core.plain(getattr(column, 'name', None)),
            'dtype': _core.plain(getattr(column, 'dtype', None)),
            'stype': _core.plain(getattr(column, 'stype', None)),
        },
    )


def set_column(column: Column, key: str, value: Any) -> bool:
    r"""``setattr(column, key, value)`` for the attributes the core decides
    about; whether it did."""
    if key not in ('stype', 'untyped'):
        return False
    if key == 'stype' and not hasattr(column, 'stype'):
        # Being built: the constructor's check, then the value.
        edit(
            'column',
            {},
            {
                'op': 'column_init',
                'name': _core.plain(getattr(column, 'name', None)),
                'dtype': _core.plain(getattr(column, 'dtype', None)),
                'stype': _core.plain(value),
            },
        )
        object.__setattr__(column, 'stype', value)
        if value == 'ID':
            object.__setattr__(column, '_stype_before_key', None)
        return True
    scope, holder, table_name, name = _scope_of_column(column)
    op = {
        'op': 'set_column',
        'table': table_name,
        'column': name,
        'attribute': key,
        'value': _core.plain(value),
    }
    if scope == 'schema':
        if key == 'stype' and value != 'ID':
            holder._own_tables()
        found = edit('schema', holder._tables, op, links=holder.relationships)
        _apply_schema(holder._tables, found)
        return True
    if scope == 'table':
        found = edit('table', {table_name: holder}, op)
        apply_table(holder, found['tables'][0])
        return True
    lone = _LoneTable(name, column)
    found = edit('column', {'': lone}, {**op, 'table': ''})
    apply_column(column, found['tables'][0]['columns'][0])
    return True


class _LoneTable:
    r"""A column held on its own, as the one column of a nameless table."""

    def __init__(self, name: Any, column: Column) -> None:
        self.name = ''
        self.columns = {name: column}
        self.primary_key = None
        self.time_column = None
        self.duplicate_keys = None


def column_check_consistent(column: Column) -> None:
    r"""``column.check_consistent()``."""
    name = getattr(column, 'name', '')
    lone = _LoneTable(name, column)
    edit(
        'column',
        {'': lone},
        {
            'op': 'check_consistent',
            'table': '',
            'column': name,
            'dtype_none': getattr(column, 'dtype', None) is None,
            'stype_none': getattr(column, 'stype', None) is None,
        },
    )


def key_value(value: Any) -> Any:
    r"""A primary key as the core reads one: a name, the names of an
    ordered iterable of them, read once, or what it was where it is
    neither."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bytes, Mapping, AbstractSet)) or not isinstance(
        value, Iterable
    ):
        return {'$object': {'type': type(value).__name__, 'repr': repr(value)}}
    shown = repr(value)
    names = list(value)
    if not all(isinstance(name, str) for name in names):
        return {'$object': {'type': type(value).__name__, 'repr': shown}}
    return names


def set_table(table: _TableShape, key: str, value: Any) -> None:
    r"""``setattr(table, key, value)`` for ``primary_key`` and
    ``time_column``, which the rest of the table follows."""
    schema = getattr(table, '_schema', None)
    name = None
    if schema is not None:
        name = next((n for n, t in schema._tables.items() if t is table), None)
    op = {
        'op': 'set_table',
        'attribute': key,
        'value': key_value(value)
        if key == 'primary_key'
        else _core.plain(value),
    }
    if schema is not None and name is not None:
        schema._own_tables()
        found = edit(
            'schema',
            schema._tables,
            {**op, 'table': name},
            links=schema.relationships,
            rows=key == 'primary_key',
        )
        _apply_schema(schema._tables, found)
        return
    found = edit(
        'table',
        {table.name: table},
        {**op, 'table': table.name},
        rows=key == 'primary_key',
    )
    apply_table(table, found['tables'][0])


def table_init(table: _TableShape) -> None:
    r"""A table's constructor: its names, and its frame's cells."""
    import pandas as pd

    frame = getattr(table, 'data', None)
    op: dict[str, Any] = {
        'op': 'table_init',
        'name': _core.plain(table.name),
    }
    if hasattr(table, 'source_name'):
        op['source_name'] = _core.plain(table.source_name)
        op['columns'] = [_core.plain(c) for c in getattr(table, 'columns', {})]
        edit('table', {}, op)
        return
    held: dict[str, Any] = {}
    if isinstance(frame, pd.DataFrame):
        op['frame_columns'] = [_core.plain(c) for c in frame.columns]
        held = {'': _FrameOnly(frame)}
    edit('table', held, op, rows=bool(held))


class _FrameOnly:
    r"""A frame, as the one table an edit reads the rows of."""

    def __init__(self, frame: Any) -> None:
        self.data = frame
        self.columns: dict[str, Any] = {}
        self.primary_key = None
        self.time_column = None
        self.duplicate_keys = None


def delete_column(table: _TableShape, column: Any) -> None:
    r"""``del table[column]`` of a table in its source."""
    name = table.name
    edit(
        'table',
        {name: table},
        {'op': 'delete_column', 'table': name, 'column': _core.plain(column)},
    )
    del table.columns[column]


def settle(schema: Schema, links: Sequence[Any]) -> list[Any]:
    r"""``schema.relationships = links``: checked, each kept once, what is
    said about them said, and the key columns derived again. Returns the
    links as the schema holds them."""
    given = list(links)
    schema._own_tables()
    found = edit(
        'schema',
        schema._tables,
        {'op': 'set_links', 'links': [link_json(link) for link in given]},
        links=(),
    )
    _apply_schema(schema._tables, found)
    kept = links_of(found)
    # The links the caller passed, each once, rather than copies of them.
    by_value: dict[Any, Any] = {}
    for link in given:
        by_value.setdefault(_link_key(link), link)
    return [by_value.get(_link_key(link), link) for link in kept]


def _link_key(link: Any) -> Any:
    return (
        getattr(link, 'source_table', None),
        getattr(link, 'source_column', None),
        getattr(link, 'target_table', None),
        getattr(link, 'target_column', None),
    )


def mark_keys(schema: Schema) -> None:
    r"""Which columns ``schema`` joins on, derived again."""
    schema._own_tables()
    found = edit(
        'schema',
        schema._tables,
        {'op': 'mark_keys'},
        links=schema.relationships,
    )
    _apply_schema(schema._tables, found)
    for table in schema._tables.values():
        for column in table.columns.values():
            object.__setattr__(column, '_table', table)


def check_consistent(schema: Schema) -> None:
    r"""``schema.check_consistent()``."""
    schema._own_tables()
    found = edit(
        'schema',
        schema._tables,
        {'op': 'check_consistent'},
        links=schema.relationships,
    )
    _apply_schema(schema._tables, found)


# --- Schemas the core described -------------------------------------------------------


def _relationship_json(link: Any) -> Any:
    return link_json(link)


def _keys_json(given: Any) -> Any:
    if given is None:
        return None
    return {str(table): key_value(key) for table, key in dict(given).items()}


def _column_from(said: Mapping[str, Any]) -> Column:
    from sdm_client.schema import Column

    made = Column.__new__(Column)
    set_ = object.__setattr__
    set_(made, 'name', said['name'])
    set_(made, 'dtype', said.get('dtype'))
    apply_column(made, said)
    return made


def _tables_from(
    said: Mapping[str, Any], frames: Mapping[str, Any] | None
) -> dict[str, _TableShape]:
    r"""The tables a description names: of frames, holding the caller's
    frames themselves, or in their source."""
    from sdm_client.schema import SourceTable, Table

    built: dict[str, _TableShape] = {}
    for found in said.get('tables') or ():
        name = found['name']
        cls = Table if frames is not None else SourceTable
        table = cls.__new__(cls)
        set_ = object.__setattr__
        set_(table, 'name', name)
        if frames is not None:
            set_(table, 'data', frames[name])
        else:
            set_(table, 'source_name', found.get('source_name') or name)
        set_(
            table,
            'columns',
            {c['name']: _column_from(c) for c in found.get('columns') or ()},
        )
        apply_table(table, found)
        for column in table.columns.values():
            set_(column, '_table', table)
        built[name] = table
    return built


def schema_from(
    cls: Any,
    said: Mapping[str, Any],
    frames: Mapping[str, Any] | None,
    *,
    source: Any = None,
    engine: Any = None,
) -> Schema:
    r"""A ``Schema`` of ``cls`` saying what the core described."""
    from sdm_client.schema import _Links

    schema = cls.__new__(cls)
    set_ = object.__setattr__
    set_(schema, '_source', source)
    tables = _tables_from(said, frames)
    set_(schema, '_tables', tables)
    set_(schema, 'relationships', _Links(links_of(said), schema))
    set_(schema, '_engine_tables', engine)
    for table in tables.values():
        set_(table, '_schema', schema)
    return schema


def _refused(
    found: Mapping[str, Any],
    caused: Mapping[str, BaseException] | None = None,
) -> None:
    r"""Raise a schema refusal the core gave as a value, as today's error,
    from what today's is raised from."""
    refused = found.get('error')
    if refused is None:
        return
    qualified = refused.get('class') or 'sdm_client.errors.StructuredDataError'
    code = refused.get('code')
    made = _core.error_of(
        _core._Refused(
            qualified,
            code if qualified.startswith('sdm_client.') else None,
            refused['message'],
            details=refused.get('details'),
            chain=refused.get('chain'),
        ),
        caused,
    )
    raise made from made.__cause__


def from_data(
    cls: type,
    tables: Mapping[str, Any],
    *,
    primary_keys: Any,
    time_columns: Any,
    relationships: Any,
) -> Schema:
    r"""``Schema.from_data``: the core infers; the schema holds the caller's
    frames."""
    import pandas as pd

    from sdm_client.errors import StructuredDataError

    if not tables:
        raise StructuredDataError(
            'a schema needs at least one table', code='INVALID_REQUEST'
        )
    given = dict(tables.items())
    for name, frame in given.items():
        if not isinstance(name, str) or not name:
            raise StructuredDataError(
                f'table names must be non-empty strings; got {name!r}',
                code='INVALID_REQUEST',
            )
        if not isinstance(frame, pd.DataFrame):
            raise StructuredDataError(
                f'table {name!r} must be a pandas DataFrame, got '
                f'{type(frame).__name__}',
                code='INVALID_REQUEST',
            )
    described = _core.Frames.described(given)
    options = {
        'primary_keys': _keys_json(primary_keys),
        'time_columns': None
        if time_columns is None
        else {str(t): _core.plain(c) for t, c in dict(time_columns).items()},
        'relationships': None
        if relationships is None
        else [_relationship_json(link) for link in relationships],
        'edits': [],
    }
    found, said = _core.call(
        _native.data_schema,
        described.tables,
        json.dumps(described.tags),
        json.dumps(options),
    )
    _core.give_json(said)
    handle = json.loads(found)
    _refused(handle)
    return schema_from(cls, handle['schema'], given)


#: The engine's name for each source's backend.
_BACKENDS = {
    'databricks': 'databricks',
    'duckdb': 'duckdb',
    'postgres': 'postgres',
    'snowflake': 'snow',
    'sqlite': 'sqlite',
}


def source_json(source: Any) -> dict[str, Any]:
    r"""What the core is told about a source: its backend, namespace and,
    for a semantic model, its name and whether its expressions are
    trusted."""
    model = None
    trust = False
    if hasattr(source, 'trust_expressions') and hasattr(source, 'source'):
        model = source.name
        trust = bool(source.trust_expressions)
    return {
        'backend': _BACKENDS.get(source.backend, source.backend),
        'namespace': dict(source.namespace),
        'semantic_model': model,
        'trust_expressions': trust,
    }


def from_source(
    cls: type,
    source: Any,
    tables: Any,
    *,
    primary_keys: Any,
    time_columns: Any,
    relationships: Any,
    check_keys: bool,
) -> Schema:
    r"""``Schema.from_source``: the core reads the source, a statement at a
    time through its ``_run``, and describes it."""
    from sdm_client.schema import _require_a_source

    _require_a_source(source)
    described = source_json(source)
    given = {
        'tables': None
        if tables is None
        else [
            [_core.plain(k), _core.plain(v)] for k, v in dict(tables).items()
        ],
        'primary_keys': _keys_json(primary_keys),
        'time_columns': None
        if time_columns is None
        else {str(t): _core.plain(c) for t, c in dict(time_columns).items()},
        'relationships': None
        if relationships is None
        else [_relationship_json(link) for link in relationships],
        'check_keys': bool(check_keys),
    }
    run = _core.call(
        _native.schema_source_run,
        described['backend'],
        json.dumps(described),
        json.dumps(given),
    )
    driver = _core.Driver(source)
    found, said = driver.drive(run)
    _core.give_json(said)
    handle = json.loads(found)
    _refused(handle, driver.caused)
    return schema_from(
        cls,
        handle['schema'],
        None,
        source=source,
        engine=handle.get('engine') or [],
    )


def schema_state(schema: Schema) -> tuple[dict[str, Any], _core.Frames]:
    r"""What a schema says, as the core holds one between calls, and its
    frames, as they stand, named ``schema/NAME``."""
    said = schema_json(schema._tables, schema.relationships)
    source = schema._source
    if source is not None:
        return (
            {
                'kind': 'source',
                'source': source_json(source),
                'schema': said,
                'engine': schema._engine_tables or [],
            },
            _core.Frames(),
        )
    return (
        {
            'kind': 'data',
            'tables': list(schema._tables),
            'options': {},
            'schema': said,
        },
        _frames_of(schema._tables),
    )
