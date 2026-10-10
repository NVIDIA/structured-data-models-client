# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``Schema.from_source`` on the core: the schema bridge (T1.11b).

``Schema.from_source`` comes here on the core route, for a DuckDB source
and no tables named. The core's ``describe_source`` reads the source through
the DuckDB provider (:mod:`.providers.duckdb`) and infers each table, its
columns' data and semantic types, its key and its time column, by the
reference's rules, and the source's declared foreign keys; this module
builds today's :class:`~sdm_client.schema.Schema` from that description,
with the client's own assembly as ``Schema.from_source`` does it after the
engine has read the source: the relationships inferred by name, the edit
stamps settled by them, the key columns marked, and the same warnings.

Nothing here imports the relational engine. The proof is the tests':
an import hook that refuses the engine for the length of the route is a
test's own, installed in a process of its own, since a hook on
``sys.meta_path`` is the whole process's and would refuse another
thread's call on the old route (design, "Routing seam"). The old
materializer, which a query still takes
until T1.11c, reads the source itself later; the schema hands it an empty
store of engine tables (``EngineTables``), which it fills then.

As built at T1.11b the core describes every table of the source with no
overrides, and refuses with ``UNSUPPORTED_REQUEST`` an input whose
description takes a rule it does not hold yet (register entry P1B-2);
relationships passed in are refused here, before the core runs, for the
same reason.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec
from sdm_client._rust.providers.duckdb import DuckDBProvider
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client.schema import Relationship, Schema


def from_source(
    cls: type[Schema],
    source: Any,
    *,
    primary_keys: Mapping[str, str | Sequence[str]] | None,
    time_columns: Mapping[str, str | None] | None,
    relationships: Sequence[Relationship] | None,
    check_keys: bool,
) -> Schema:
    r"""The schema of every table of the DuckDB ``source``, described on the
    core.

    Raises:
        StructuredDataError: The core's refusal, as :mod:`.errors` maps it,
            ``UNSUPPORTED_REQUEST`` naming the rule for an input it does not
            describe yet.
    """
    if relationships is not None:
        raise StructuredDataError(
            'Schema.from_source on the core does not take relationships yet',
            code='UNSUPPORTED_REQUEST',
            details={'rule': 'relationships', 'field': 'relationships'},
        )
    if any(column is None for column in (time_columns or {}).values()):
        raise StructuredDataError(
            'Schema.from_source on the core does not take a time column '
            'cleared yet',
            code='UNSUPPORTED_REQUEST',
            details={'rule': 'time_column_cleared', 'field': 'time_columns'},
        )
    spec = {
        'tables': [],
        'primary_keys': {
            table: [key] if isinstance(key, str) else list(key)
            for table, key in (primary_keys or {}).items()
        },
        'time_columns': dict(time_columns or {}),
    }
    operation = {
        'op': 'describe_source',
        'provider': {'provider': 'sql', 'dialect': 'duckdb'},
        'source': 1,
        'check_keys': bool(check_keys),
    }
    finished = run(
        operation,
        {1: codec.attachment(spec)},
        provider=DuckDBProvider(source.connection),
    )
    described = json.loads(finished.take(finished.result['description']))
    return build(cls, source, described)


def build(cls: type[Schema], source: Any, described: dict[str, Any]) -> Schema:
    r"""Today's ``Schema`` of a ``SourceDescription``, assembled as
    ``Schema.from_source`` assembles the engine's reading of a source."""
    from predictive_query.time_columns import TimeColumnChoice

    from sdm_client.core.warnings import SDMWarning
    from sdm_client.materializers.sources import EngineTables, wire_dtype
    from sdm_client.schema import (
        Column,
        Relationship,
        SourceTable,
        _infer_relationships,
        _joinable_inferred_links,
        _key_by_the_referenced_column,
        _settle_edit_stamps,
        _warn_how_time_columns_were_inferred,
        _warn_two_ways_to_reach_one_table,
    )

    built: dict[str, SourceTable] = {}
    chose: dict[str, TimeColumnChoice | None] = {}
    for table in described['tables']:
        name = table['name']
        columns = {}
        for column in table['columns']:
            dtype = wire_dtype(column['dtype'])
            if dtype is None:
                raise StructuredDataError(
                    'the core described a column with no wire dtype',
                    code='INTERNAL_ERROR',
                )
            columns[column['name']] = Column(
                column['name'], dtype, column['stype']
            )
        shaped = SourceTable(name=name, source_name=name, columns=columns)
        keys = tuple(table['primary_key'])
        if keys and all(key in columns for key in keys):
            shaped.primary_key = keys[0] if len(keys) == 1 else keys
        time = table['time_column']
        if time is not None and time in columns and time not in keys:
            shaped.time_column = time
        duplicates = table['duplicate_keys']
        object.__setattr__(
            shaped,
            'duplicate_keys',
            None
            if duplicates is None
            else int(codec.decode_scalar(duplicates)),
        )
        built[name] = shaped
        # The core takes a time column only as the one timestamp left, and
        # refuses every other choice, so the choice is that or none.
        chose[name] = (
            TimeColumnChoice(time, 'only', (time,), ())
            if time is not None
            else TimeColumnChoice(None, None, (), ())
        )
    declared = [
        Relationship(
            source_table=foreign['table'],
            source_column=foreign['column'],
            target_table=referenced['table'],
            target_column=referenced['column'],
        )
        for foreign, referenced in described['links']
    ]
    # The core refuses a keyless table the declared keys would key, so this
    # keys nothing; it runs where the reference runs it.
    _key_by_the_referenced_column(built, declared)
    covered = {(link.source_table, link.source_column) for link in declared}
    by_name = [
        link
        for link in _infer_relationships(built)
        if (link.source_table, link.source_column) not in covered
    ]
    links = _joinable_inferred_links(built, declared + by_name)
    _settle_edit_stamps(built, chose, None, links)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SDMWarning)
        schema = cls(built, links, source=source)
    schema._engine_tables = EngineTables()
    # Kept for materialize on the core (T1.11c), which needs what the
    # catalog said of each column and the schema does not hold: the type
    # received, whether it may be null, whether the source keys it.
    schema._core_description = described
    schema._mark_key_columns()
    _warn_how_time_columns_were_inferred(schema, chose, None)
    _warn_two_ways_to_reach_one_table(built, schema.relationships)
    return schema
