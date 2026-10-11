# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: the host-held rules here (design, "Host-held semantics"),
# the option refusals (T1.24), are deleted in Phase 1 by that task; building
# today's Schema from the description stays the host's. Frozen meanwhile
# (E50).

r"""``Schema.from_source`` on the core: the schema bridge (T1.11b).

``Schema.from_source`` comes here on the core route, for a DuckDB source
and no tables named. The core's ``describe_source`` reads the source through
the DuckDB provider (:mod:`.providers.duckdb`) and infers each table, its
columns' data and semantic types, its key and its time column, by the
reference's rules, with the source's declared foreign keys; from T1.21d it
assembles the schema as the client's ``Schema.from_source`` does after the
engine has read the source, and says so in the same warnings: the
relationships inferred by name, the edit stamps settled by them, the key
columns marked (decision E55). This module builds today's
:class:`~sdm_client.schema.Schema` from that description and decides
nothing of it; the run shows the core's warnings as it gives them.

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
    r"""Today's ``Schema`` of a ``SourceDescription``, as the core assembled
    it: each column's meaning as the description gives it, with what it
    meant before the schema joined on it (``Column._stype_before_key``,
    which the schema gives back if it stops joining on it), each table's
    key and time column, and the links."""
    from sdm_client.core.warnings import SDMWarning
    from sdm_client.materializers.sources import EngineTables, wire_dtype
    from sdm_client.schema import Column, Relationship, SourceTable

    built: dict[str, SourceTable] = {}
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
        table_shape = SourceTable(
            name=name, source_name=table['source'], columns=columns
        )
        keys = tuple(table['primary_key'])
        if keys:
            table_shape.primary_key = keys[0] if len(keys) == 1 else keys
        # The column inference took ordered the table before the links or
        # the caller settled another, as the reference's schema shows it.
        table_shape.time_column = table['time_column_inferred']
        table_shape.time_column = table['time_column']
        # Once the key and time column are set, whose setters re-derive
        # the marks of a table held on its own.
        for column in table['columns']:
            object.__setattr__(
                table_shape.columns[column['name']],
                '_stype_before_key',
                column['stype_before_key'],
            )
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
    schema._engine_tables = EngineTables()
    # Kept for materialize on the core (T1.11c), which needs what the
    # catalog said of each column and the schema does not hold: the type
    # received, whether it may be null, whether the source keys it.
    schema._core_description = described
    return schema
