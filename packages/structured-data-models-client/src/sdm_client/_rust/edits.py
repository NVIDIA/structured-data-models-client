# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A core-built schema's edits, checked and made on the core (T1.24b).

An edit to a schema ``Schema.from_source`` described on the core, whose
columns hold what the source said of them (``Column._source``, T3.16), (a key or
time column set or cleared, a column's meaning, a column left out, the
relationships replaced or edited as a list) reaches here from the setter
the caller's assignment ran (``Column.__setattr__``,
``_TableShape.__setattr__`` and ``Schema.__setattr__``, under the owner's
allowance of 2026-10-11, decision E66(g); ``SourceTable.__delitem__`` and
the relationships' list, ``_Links``), which passes it on and returns. The
core's ``edit_schema`` checks and makes it, as the reference's setters do,
over the schema as this client holds it, as plain data (decision E67,
T3.16), and answers the schema with the edit made,
or refuses it at the assignment; this module stores the answer on the
objects and decides nothing of it.

What it does is the host's own: finding the schema an object belongs to,
writing each value the caller gave as the contract writes it, or as
``unreadable`` where no contract type holds it (a key named by a number,
a set of names, a relationship that is not one), which the core refuses;
storing the answer on the objects the caller holds, keeping the caller's
own value where the answer agrees with it (a key given as a tuple, a
meaning given as a ``Stype``); and the objects' ownership, as the
reference's setters take it. Nothing here imports the core until an edit
is made, and nothing imports the relational engine.

Raises (from each edit):
    StructuredDataError: The core's refusal, as :mod:`.errors` maps it,
        ``INVALID_REQUEST`` where the reference refuses the edit.
"""

from __future__ import annotations

import json
from typing import Any

from sdm_client._rust.codec import given as _given
from sdm_client._rust.codec import given_key as _key
from sdm_client._rust.codec import given_name_or_none as _name_or_none
from sdm_client._rust.codec import given_text as _text
from sdm_client._rust.codec import spelled as _spelled


def _schema_holding(table: Any) -> tuple[Any, str] | None:
    r"""The core-built schema holding ``table``, and the name it holds it
    under; ``None`` for a table no core-built schema holds."""
    from sdm_client.schema import _described_on_the_core

    schema = getattr(table, '_schema', None)
    if not _described_on_the_core(schema):
        return None
    for name, held in vars(schema).get('_tables', {}).items():
        if held is table:
            return schema, name
    return None


def _link(link: Any) -> Any:
    from sdm_client.schema import Relationship

    if not isinstance(link, Relationship):
        return _spelled(link)
    names = (
        link.source_table,
        link.source_column,
        link.target_table,
        link.target_column,
    )
    if not all(isinstance(name, str) for name in names):
        return _spelled(link)
    return _given(
        [
            {'table': str.__str__(names[0]), 'column': str.__str__(names[1])},
            {'table': str.__str__(names[2]), 'column': str.__str__(names[3])},
        ]
    )


def _made(schema: Any, edit: dict[str, Any]) -> dict[str, Any]:
    r"""``edit`` made by the core to ``schema``, its answer stored: the
    schema the core answered."""
    from sdm_client._rust import codec, results
    from sdm_client._rust.run import run
    from sdm_client._rust.spec import held_schema

    # The schema as it stands, as plain data, as materialize sends it
    # (T3.16).
    operation = {'op': 'edit_schema', 'schema': 1, 'edit': 2}
    finished = run(
        operation,
        {1: codec.attachment(held_schema(schema)), 2: codec.attachment(edit)},
    )
    edited = results.expect(finished.result, results.Edited)
    answer: dict[str, Any] = json.loads(finished.take(edited.schema))
    _store(schema, answer, edited.declared)
    return answer


def _store(
    schema: Any, answer: dict[str, Any], declared: tuple[str, str] | None
) -> None:
    r"""The core's answer, the schema as the host holds it from now on, on
    the objects of ``schema``: each table's key, time column and repeated
    keys, its columns left out, each column's meaning and what it meant
    before the schema joined on it, what the source said of it, and its
    data type where the source's reading of it changed (a column of exact
    numbers is read by its role, T2.21); the column whose meaning the
    caller chose. A value the object holds that the answer agrees with is
    kept as the caller gave it."""
    from sdm_client._rust import codec
    from sdm_client._rust.results import SourceFacts
    from sdm_client.schema import _as_key_tuple

    # The tables and columns this schema marks are its own, as the
    # reference's setters take them before marking.
    schema._own_tables()
    answered = {table['name']: table for table in answer['tables']}
    for name, table in schema.tables.items():
        given = answered[name]
        keys = tuple(given['primary_key'])
        if _as_key_tuple(table.primary_key) != keys:
            key = None if not keys else keys[0] if len(keys) == 1 else keys
            object.__setattr__(table, 'primary_key', key)
        object.__setattr__(table, 'time_column', given['time_column'])
        repeated = given.get('duplicate_keys')
        object.__setattr__(
            table,
            'duplicate_keys',
            None if repeated is None else int(codec.decode_scalar(repeated)),
        )
        columns = {column['name']: column for column in given['columns']}
        for gone in [name for name in table.columns if name not in columns]:
            del table.columns[gone]
        for column_name, column in table.columns.items():
            said = columns[column_name]
            facts = (
                None
                if said.get('source') is None
                else SourceFacts.from_contract(said['source'])
            )
            if (
                facts is not None
                and facts != column._source
                and column.dtype != said['dtype']
            ):
                # The source's reading changed, and the type with it.
                object.__setattr__(column, 'dtype', said['dtype'])
            object.__setattr__(column, '_source', facts)
            # A meaning the column holds already, as its meaning or as what
            # it meant before the schema joined on it, is kept as the
            # object it is, as the reference's marks move it between the
            # two.
            held = (column.stype, column._stype_before_key)
            for field, value in (
                ('stype', said['stype']),
                ('_stype_before_key', said['stype_before_key']),
            ):
                kept = next(
                    (
                        each
                        for each in held
                        if each is not None
                        and value is not None
                        and each == value
                    ),
                    value,
                )
                object.__setattr__(column, field, kept)
    if declared is not None:
        chosen = schema.tables[declared[0]].columns[declared[1]]
        object.__setattr__(chosen, 'stype_declared', True)


def column(column: Any, stype: Any) -> bool:
    r"""``column.stype = stype``, made on the core when ``column`` is a
    core-built schema's; whether it was."""
    table: Any = getattr(column, '_table', None)
    holding = None if table is None else _schema_holding(table)
    if holding is None:
        return False
    schema, name = holding
    column_name = next(
        (key for key, held in table.columns.items() if held is column), None
    )
    if column_name is None:
        return False
    _made(
        schema,
        {
            'edit': 'stype',
            'table': name,
            'column': column_name,
            'stype': _text(stype),
        },
    )
    # The meaning as the caller gave it, where the answer agrees.
    if column.stype == stype:
        object.__setattr__(column, 'stype', stype)
    if column._stype_before_key == stype:
        object.__setattr__(column, '_stype_before_key', stype)
    return True


def table(table: Any, key: str, value: Any) -> bool:
    r"""``table.primary_key = value`` or ``table.time_column = value``, made
    on the core when ``table`` is a core-built schema's; whether it was."""
    holding = _schema_holding(table)
    if holding is None:
        return False
    schema, name = holding
    if key == 'primary_key':
        written, form = _key(value)
        _made(schema, {'edit': 'primary_key', 'table': name, 'key': written})
        # The key as the caller gave it, which the answer names.
        object.__setattr__(table, 'primary_key', form)
        return True
    _made(
        schema,
        {'edit': 'time_column', 'table': name, 'column': _name_or_none(value)},
    )
    # The time column as the caller gave it, which the answer names: a
    # ``str`` subclass (``np.str_``) kept as given, as the reference keeps
    # it, and as the key above.
    if value is not None:
        object.__setattr__(table, 'time_column', value)
    return True


def leave_out(table: Any, column_name: Any) -> bool:
    r"""``del table[column_name]``, made on the core when ``table`` is a
    core-built schema's; whether it was."""
    holding = _schema_holding(table)
    if holding is None:
        return False
    schema, name = holding
    _made(
        schema,
        {'edit': 'leave_out', 'table': name, 'column': _text(column_name)},
    )
    return True


def relationships(schema: Any, links: Any, into: Any = None) -> bool:
    r"""``schema.relationships = links``, or its list edited to ``links``
    (``into``, the list), made on the core when ``schema`` is core-built;
    whether it was. Each relationship the answer holds is the caller's own
    object where the caller gave one equal to it."""
    from sdm_client.schema import Relationship, _described_on_the_core, _Links

    if not _described_on_the_core(schema):
        return False
    given = list(links)
    answer = _made(
        schema,
        {'edit': 'relationships', 'links': [_link(link) for link in given]},
    )
    settled = []
    for foreign, referenced in answer['links']:
        made = Relationship(
            foreign['table'],
            foreign['column'],
            referenced['table'],
            referenced['column'],
        )
        settled.append(next((link for link in given if link == made), made))
    if into is None:
        object.__setattr__(schema, 'relationships', _Links(settled, schema))
    else:
        list.clear(into)
        list.extend(into, settled)
    return True
