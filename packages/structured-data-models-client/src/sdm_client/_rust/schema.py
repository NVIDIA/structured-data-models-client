# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt source declarations and Rust schema descriptions to Python objects.

Rust owns schema inference, relationship policy, validation and warnings.
This module encodes Python arguments and binds the returned metadata to public
Schema, SourceTable and Column objects. Providers perform source I/O.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec, errors, providers, results
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client.schema import Relationship, Schema


def from_source(
    cls: type[Schema],
    source: Any,
    tables: Mapping[str, str] | None,
    *,
    primary_keys: Mapping[str, str | Sequence[str]] | None,
    time_columns: Mapping[str, str | None] | None,
    relationships: Sequence[Relationship] | None,
    check_keys: bool,
    columns: Mapping[str, Sequence[str]] | None = None,
) -> Schema:
    r"""The schema of the tables named, or of every table, of the DuckDB,
    SQLite or Postgres ``source``, described on the core; of a table
    ``columns`` names, only the columns it keeps (contract v28), as the
    reference reads only the columns a schema describes.

    Each key and time column is written as the contract's ``Given``: a
    value the contract's type cannot hold (a number, a set, a mapping) as
    ``unreadable``, which the core refuses (decision E101); an iterable key
    is read once, here, before the run starts. A table named by anything
    but a ``str`` is refused here, before the run (P1B-74).

    Raises:
        StructuredDataError: The core's refusal, as :mod:`.errors` maps it,
            ``UNSUPPORTED_REQUEST`` naming the rule for an input it does not
            describe yet; ``INVALID_REQUEST`` for a relationship that is not
            a ``Relationship``, which the contract cannot carry; a source
            holding no table, named as the reference names it.
    """
    from sdm_client.schema import Relationship

    # A table is named by a str (P1B-74, the owner's): JSON would write
    # another key as its text, so the host refuses it before the run.
    for field, by_table in (
        ('primary_keys', primary_keys),
        ('time_columns', time_columns),
    ):
        if any(not isinstance(table, str) for table in by_table or {}):
            raise StructuredDataError(
                'a table is named by a string',
                code='INVALID_REQUEST',
                details={'field': field, 'rule': 'table_not_a_name'},
            )
    # Each table's source a relation of one part, its name never split on
    # a dot (contract v26, design choice 12).
    spec = {
        'tables': [
            {
                'name': name,
                'source': codec.relation(named),
                **(
                    {'columns': [str(c) for c in columns[name]]}
                    if columns is not None and name in columns
                    else {}
                ),
            }
            for name, named in (tables or {}).items()
        ],
        'primary_keys': {
            table: codec.given_key(key)[0]
            for table, key in (primary_keys or {}).items()
        },
        'time_columns': {
            table: codec.given_name_or_none(named)
            for table, named in (time_columns or {}).items()
        },
    }
    if relationships is not None:
        links = list(relationships)
        for link in links:
            if not isinstance(link, Relationship):
                raise StructuredDataError(
                    f'a relationship must be a Relationship, got '
                    f'{type(link).__name__}: {link!r}. Build one with '
                    f'Relationship(source_table, source_column, '
                    f'target_table, target_column)',
                    code='INVALID_REQUEST',
                )
        given: Any = [
            [
                {'table': link.source_table, 'column': link.source_column},
                {'table': link.target_table, 'column': link.target_column},
            ]
            for link in links
        ]
        spec['relationships'] = given
    choice, provider = providers.for_source(source)
    operation = {
        'op': 'describe_source',
        'provider': choice,
        'source': 1,
        'check_keys': bool(check_keys),
    }
    # The source, which only this host holds, as the core's sentence of a
    # source with no table names it (`engine_graph`'s refusal, T3.17), and
    # a semantic model's as its sentences name it (T3.5).
    with errors.holding(**errors.source_values(source)):
        finished = run(
            operation,
            {1: codec.attachment(spec)},
            provider=provider,
        )
    described = results.expect(finished.result, results.Described)
    schema = build(
        cls, source, json.loads(finished.take(described.description))
    )
    schema._conversion_notes = conversion_notes(described.notes, schema)
    return schema


def conversion_notes(
    notes: Sequence[results.ViewNote], schema: Schema
) -> tuple[str, ...]:
    r"""A semantic model's conversion notes, as the core gave them at
    describe, in words: each rendered from its sentence and fields, the
    core naming each table as the schema does. A note of a table the
    schema does not hold, or of a time column its table no longer has,
    says nothing the schema holds, so it is left out, as the old client
    leaves it out (``CL/schema.py:1401-1433``)."""
    said = []
    for note in notes:
        if note.table is not None:
            table = schema.tables.get(note.table)
            if table is None:
                continue
            if (
                note.time_column is not None
                and table.time_column != note.time_column
            ):
                continue
        said.append(errors.render(note.sentence, errors.decoded(note.fields)))
    return tuple(said)


def build(cls: type[Schema], source: Any, described: dict[str, Any]) -> Schema:
    r"""Today's ``Schema`` of a ``SourceDescription``, as the core assembled
    it: each column's meaning as the description gives it, with what it
    meant before the schema joined on it (``Column._stype_before_key``,
    which the schema gives back if it stops joining on it) and what the
    source said of it (``Column._source``, T3.16), each table's key and
    time column, and the links."""
    from sdm_client._rust.results import SourceFacts
    from sdm_client.core.warnings import SDMWarning
    from sdm_client.materializers.sources import wire_dtype
    from sdm_client.schema import Column, Relationship, SourceTable

    built: dict[str, SourceTable] = {}
    for table in described['tables']:
        name = table['name']
        # The columns a time column was set on, which the old route's setter
        # records as a meaning assigned (`stype_declared`): the column
        # inference took, then the time column the schema holds, as the
        # reference's schema shows them.
        timed = {table['time_column_inferred'], table['time_column']} - {None}
        columns = {}
        for column in table['columns']:
            dtype = wire_dtype(column['dtype'])
            if dtype is None:
                raise StructuredDataError(
                    'the core described a column with no wire dtype',
                    code='INTERNAL_ERROR',
                )
            # Written as the core answered it, field by field: the core
            # checked and marked it (decision E55), and the old route's
            # checks, which ask the old PQL package, are not run again
            # (T3.16, the structure design's rule 11). What the source said
            # of it is kept with it and handed back whole to the core.
            made = Column.__new__(Column)
            for field, value in (
                ('name', column['name']),
                ('dtype', dtype),
                ('stype', column['stype']),
                ('untyped', False),
                ('stype_declared', column['name'] in timed),
                ('_stype_before_key', column['stype_before_key']),
                ('_source', SourceFacts.of_described(column, table['source'])),
            ):
                object.__setattr__(made, field, value)
            columns[column['name']] = made
        # Bind the already checked result mechanically, without rerunning
        # standalone table policy or issuing one Rust operation per field.
        table_shape = SourceTable.__new__(SourceTable)
        object.__setattr__(table_shape, 'name', name)
        object.__setattr__(
            table_shape, 'source_name', codec.relation_name(table['source'])
        )
        object.__setattr__(table_shape, 'columns', columns)
        table_shape._own_columns()
        keys = tuple(table['primary_key'])
        # The key and time column as the core answered them, which marked
        # the columns already.
        object.__setattr__(
            table_shape,
            'primary_key',
            None if not keys else keys[0] if len(keys) == 1 else keys,
        )
        object.__setattr__(table_shape, 'time_column', table['time_column'])
        duplicates = table['duplicate_keys']
        object.__setattr__(
            table_shape,
            'duplicate_keys',
            None
            if duplicates is None
            else int(codec.decode_scalar(duplicates)),
        )
        built[name] = table_shape
    links = [
        Relationship(
            source_table=foreign['table'],
            source_column=foreign['column'],
            target_table=referenced['table'],
            target_column=referenced['column'],
        )
        for foreign, referenced in described['links']
    ]
    # The core gave the schema's warnings as it assembled it; building the
    # object repeats none of them.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SDMWarning)
        schema = cls(built, links, source=source)
    return schema
