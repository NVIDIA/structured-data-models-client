# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame schema composition using shared Rust schema policy."""

from __future__ import annotations

from typing import Any

from sdm_client._rust import (
    codec,
    direct,
    direct_admission,
    direct_schema,
    spec,
)


def infer_tables(tables: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Observe native columns once and retain full time-choice input facts."""
    built = {}
    times: list[dict[str, Any]] = []
    for name, frame in tables.items():
        direct_admission.table_entry_name(name, direct._native_calls)
        direct_admission.names(frame, name, direct._native_calls)
        single = direct_schema.infer_frame_schema(
            frame,
            name,
            direct._native_calls,
            direct._schema,
            time_observations=times,
            advice=False,
        )
        built[name] = single[name]
    return built, times


def assemble(
    cls: Any,
    built: dict[str, Any],
    times: list[dict[str, Any]],
    *,
    primary_keys: Any = None,
    time_columns: Any = None,
    relationships: Any = None,
) -> Any:
    """Let Rust settle relationships and time roles, then bind native owners.

    Public input validation and key-quality observations precede this step.
    No table data or public column owner is copied during result binding.
    """
    from sdm_client.schema import Relationship, _key_quality

    held = {'tables': [], 'links': []}
    for table in built.values():
        held['tables'].extend(spec.held_schema(table._schema)['tables'])
    links = (
        None
        if relationships is None
        else [
            [
                {'table': link.source_table, 'column': link.source_column},
                {'table': link.target_table, 'column': link.target_column},
            ]
            for link in relationships
        ]
    )
    counts = {}
    for name, key in (primary_keys or {}).items():
        names = [key] if isinstance(key, str) else key
        if (
            name not in built
            or not isinstance(names, (list, tuple))
            or not names
        ):
            continue  # Rust diagnoses the invalid override before using counts.
        frame = built[name].data
        if any(
            not isinstance(column, str) or column not in frame
            for column in names
        ):
            continue
        quality = _key_quality(frame, names)
        counts[name] = {
            'rows': len(frame),
            'missing': quality.nulls,
            'missing_columns': [
                column for column in names if bool(frame[column].isna().any())
            ],
            'distinct': len(frame) - quality.nulls - quality.duplicates,
            'example': None,
        }
    description = direct._call(
        'assemble_frame_schema',
        {
            'schema': held,
            'times': times,
            'key_counts': counts,
            'primary_keys': {
                name: codec.given_key(key)[0]
                for name, key in (primary_keys or {}).items()
            },
            'time_columns': {
                name: codec.given_name_or_none(value)
                for name, value in (time_columns or {}).items()
            },
            'relationships': links,
        },
    )
    result = cls.__new__(cls)
    rebound = {}
    for definition in description['tables']:
        name = definition['name']
        table = built[name]
        single = direct._schema(
            {'tables': [definition], 'links': []},
            table.data,
            native_columns=table.columns,
        )
        rebound[name] = single[name]
        # These are frame observations, not warehouse catalog facts.
        # Catalog provenance would incorrectly route edits through source-only checks.
        for column in rebound[name].columns.values():
            object.__setattr__(column, '_source', None)
        object.__setattr__(rebound[name], '_schema', result)
    for name, value in (
        ('_tables', rebound),
        ('_source', None),
        ('_engine_tables', None),
        ('_conversion_notes', ()),
        (
            'relationships',
            [
                Relationship(
                    source_table=left['table'],
                    source_column=left['column'],
                    target_table=right['table'],
                    target_column=right['column'],
                )
                for left, right in description['links']
            ],
        ),
    ):
        object.__setattr__(result, name, value)
    return result


def from_data(
    cls: Any,
    tables: Any,
    *,
    primary_keys: Any = None,
    time_columns: Any = None,
    relationships: Any = None,
) -> Any:
    """Public frame admission followed by core inference and assembly."""
    import pandas as pd

    from sdm_client.errors import StructuredDataError
    from sdm_client.schema import Relationship

    if not tables:
        raise StructuredDataError(
            'a schema needs at least one table', code='INVALID_REQUEST'
        )
    for name, frame in tables.items():
        if not isinstance(name, str) or not name:
            raise StructuredDataError(
                f'table names must be non-empty strings; got {name!r}',
                code='INVALID_REQUEST',
            )
        if not isinstance(frame, pd.DataFrame):
            raise StructuredDataError(
                f'table {name!r} must be a pandas DataFrame, got {type(frame).__name__}',
                code='INVALID_REQUEST',
            )
    if relationships is not None:
        relationships = list(relationships)
        if any(not isinstance(link, Relationship) for link in relationships):
            raise StructuredDataError(
                'relationships must contain Relationship objects',
                code='INVALID_REQUEST',
            )
    built, times = infer_tables(tables)
    result = assemble(
        cls,
        built,
        times,
        primary_keys=primary_keys,
        time_columns=time_columns,
        relationships=relationships,
    )
    # Advice sees the final schema, after explicit key/time/link overrides.
    # Native date parsing and user-facing remedies stay in the host.
    from sdm_client.schema import (
        _warn_dates_stored_as_text,
        _warn_one_key_name_keys_several_tables,
        _warn_repeats_cost_the_table_its_key,
        _warn_table_cannot_be_linked_to,
    )

    dates = {
        name: named
        for name, table in result.tables.items()
        if (named := _warn_dates_stored_as_text(table)) is not None
    }
    if relationships is None:
        spoken_for = _warn_repeats_cost_the_table_its_key(result.tables)
        _warn_table_cannot_be_linked_to(result.tables, dates, spoken_for)
        _warn_one_key_name_keys_several_tables(result.tables)
    direct_schema.categorical_advice(result, direct._native_calls)
    return result
