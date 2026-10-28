# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: hands native frame values and opaque checkpoint owners to
# the core; this routing seam is deleted in Phase 5.

"""Native input and opaque ownership for the first frame materializer."""

from __future__ import annotations

import secrets
from typing import Any

import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust import codec
from sdm_client._rust.frame_decimals import native_decimal_type
from sdm_client._rust.frame_facts import native_source_facts
from sdm_client._rust.frames import _convert


def check_caller(schema: Any, state: dict[str, Any]) -> None:
    """Adopt reached cold graph key marks before presenting caller refusal.

    Cached native models observe no fresh schema here. The current owner is
    tested after overlap handlers and any reentrant materialization return.
    """
    checkpoint = state.get('held', {}).get('frame_checkpoint')
    if checkpoint is not None and 'owner' in checkpoint:
        return
    from sdm_client._rust import edits, errors
    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare
    from sdm_client._rust.spec import held_schema

    bindings = _Bindings()
    report = _adopt(
        _prepare(
            {
                'stage': 'caller_meaning',
                'schema': bindings.json(held_schema(schema)),
            },
            bindings,
            state,
        ),
        'caller_meaning',
    )
    try:
        edits._store(schema, report.data['schema'], None)
        diagnostic = report.data['diagnostic']
        if diagnostic is not None:
            errors.raise_from_core(
                codec.encode({'message': 'error', 'error': diagnostic}), {}
            )
    finally:
        report.owners.clear()


def native_inputs(
    schema: Any, state: dict[str, Any]
) -> tuple[
    list[tuple[dict[str, Any], Any]], dict[str, Any], Any, list[dict[str, Any]]
]:
    """Export one fresh native source or reuse its actual retained owners.

    String arrays use their explicit lossless native representation. Other
    families retain the constructor converter's existing closed admission.
    No cached graph observes caller values or dtype metadata again.
    """
    import pyarrow as pa

    from sdm_client._rust.frame_identity import convert as convert_identity
    from sdm_client._rust.frame_native import NativeColumn, NativeFrameColumns
    from sdm_client._rust.frame_source import (
        source_meaning,
        table_meaning,
        table_key,
        table_metadata,
        time_feature,
        typed_table,
    )
    from sdm_client._rust.frames import _check_header
    from sdm_client._rust.spec import _links, held_table
    from sdm_client.schema import _as_key_tuple

    checkpoint = state.get('held', {}).get('frame_checkpoint')
    if checkpoint is not None and 'owner' in checkpoint:
        state['frame_row_identities'] = []
        return (
            [],
            state['frame_schema'],
            state['frame_columns'],
            state['frame_native_tables'],
        )
    exported = []
    native_tables = []
    native_types = {}
    categorical = {}
    held_tables = []
    prepared_tables = []
    for name, table in schema.tables.items():
        frame = table.data
        if type(frame) is not pd.DataFrame:
            raise RouteError(
                'frame materialization requires built-in DataFrames'
            )
        frame = typed_table(table, state)
        prepared_tables.append((name, frame, held_table(name, table)))
    # Graph.from_data constructs its physical tables only after ALL declared
    # casts. Keep their dtype declarations, but delay physical admission to
    # this boundary so an earlier table cannot skip a later cast's effects.
    physical_tables = []
    for name, frame, held in prepared_tables:
        empty = bool(frame.empty)
        if not empty:
            _check_header(name, frame)
            constructed = frame.copy(deep=False)
        else:
            constructed = frame
        held_tables.append(table_meaning(held, constructed, state, empty=empty))
        physical_tables.append((name, constructed, held))
    # Each physical owner exists before the first caller key assignment.
    # Read later metadata only after that table's actual native fold callbacks.
    constructed_tables = {
        name: (frame, meaning)
        for (name, frame, _), meaning in zip(
            physical_tables, held_tables, strict=True
        )
    }
    completed_tables = {}
    identities = []
    for name, table in schema.tables.items():
        frame, constructed = constructed_tables[name]
        primary = table.primary_key
        key_table = {
            **constructed,
            'table': {
                **constructed['table'],
                'frame_key_form': None
                if primary is None
                else 'scalar'
                if isinstance(primary, str)
                else 'sequence',
            },
        }
        key = table_key(
            key_table,
            [] if primary is None else list(_as_key_tuple(primary)),
            state,
        )
        definition = key['composite']
        derived = None
        if definition is not None:
            frame = convert_identity(frame, definition, state)
            constructed_tables[name] = (frame, constructed)
            if not any(
                column['name'] == definition['derived']
                for column in constructed['constructed']
            ):
                from sdm_client._rust.frame_constructor import (
                    column as construct_column,
                )
                from sdm_client._rust.frame_facts import native_declaration
                from sdm_client._rust.frame_source import constructor_values

                # Assignment releases the original constructor sample. A new
                # native column reads one new cached sample at this position.
                sample = frame.reset_index(drop=True)
                values = sample[definition['derived']]
                derived = construct_column(
                    name,
                    definition['derived'],
                    sample,
                    constructor_values(values),
                    native_declaration(values.dtype),
                )
            identities.append((definition, frame))
        live = held_table(name, table)
        metadata = table_metadata(key, live, state, derived)
        completed_tables[name] = metadata
    # Graph.from_data's physical table order is independent of the later
    # metadata assignment order. Keep the captured values in that order.
    typed_tables = [completed_tables[name] for name, _, _ in physical_tables]
    graph_schema = source_meaning(
        {
            'tables': [table['table'] for table in typed_tables],
            'links': _links(schema),
        },
        typed_tables,
        state,
    )
    for (name, _, _), held in zip(
        physical_tables, graph_schema['tables'], strict=True
    ):
        frame = constructed_tables[name][0]
        constructed_types = {
            column['name']: column['dtype']
            for column in completed_tables[name]['constructed']
        }
        frame = frame.copy(deep=False).reset_index(drop=True)
        arrays = []
        origins = []
        types = {}
        category_arrays = {}
        for ordinal, column in enumerate(frame.columns):
            series = time_feature(held, column, frame.iloc[:, ordinal], state)
            dtype = series.dtype
            encoding = None
            object_text = pd.api.types.is_object_dtype(
                dtype
            ) and pd.api.types.infer_dtype(series, skipna=True) in (
                'string',
                'empty',
            )
            if type(dtype) is pd.CategoricalDtype:
                values, encoding = NativeColumn(series).arrow()
                array = values.column(0)
                if not (
                    pa.types.is_string(array.type)
                    or pa.types.is_large_string(array.type)
                    or pa.types.is_binary(array.type)
                    or pa.types.is_large_binary(array.type)
                ):
                    encoding = None
                category_arrays[column] = series.array
            elif (
                object_text
                or type(dtype) is pd.StringDtype
                and dtype.storage
                in (
                    'python',
                    'pyarrow',
                )
            ):
                values, encoding = NativeColumn(series).arrow()
                array = values.column(0)
            else:
                values, converted_types = _convert(
                    name,
                    pd.DataFrame({column: series}),
                    check_header=False,
                    constructor_objects=False,
                )
                array = values.column(0)
                types.update(converted_types)
                if pa.types.is_string(array.type) or pa.types.is_large_string(
                    array.type
                ):
                    encoding = 'utf8'
            arrays.append(array)
            types[column] = dtype
            origins.append(
                {
                    'name': column,
                    'constructed': constructed_types.get(column),
                    **native_source_facts(dtype),
                    'strings': encoding,
                }
            )
        source = held['source']
        source_name = codec.relation_name(source)
        exported.append(
            (source, pa.Table.from_arrays(arrays, names=list(frame.columns)))
        )
        native_tables.append({'source': source, 'columns': origins})
        native_types[source_name] = types
        categorical[source_name] = category_arrays
    state['frame_row_identities'] = [
        (definition, *NativeColumn(frame[definition['derived']]).arrow())
        for definition, frame in identities
    ]
    return (
        exported,
        graph_schema,
        NativeFrameColumns(native_types, categorical),
        native_tables,
    )


def inputs(schema: Any, state: dict[str, Any]) -> dict[int, Any]:
    """Convert live inputs once; resume from opaque checkpoint owners."""
    import pyarrow as pa

    tables = []
    attached: dict[int, Any] = {}
    checkpoint = state.get('held', {}).get('frame_checkpoint')
    if checkpoint is None:
        for number, (name, table) in enumerate(schema.tables.items(), 8):
            frame = table.data
            if type(frame) is not pd.DataFrame:
                raise RouteError(
                    'the partial frame materializer requires built-in DataFrames'
                )
            rows, native_types = _convert(
                name, frame, constructor_objects=False
            )
            # Constructor mantissas require logical Decimal consumption in
            # the materializer before this development seam can take them.
            # This whole native family was already outside this unfinished
            # route; never let scale-zero storage become a model value.
            if any(
                native_decimal_type(dtype) is not None
                for dtype in native_types.values()
            ):
                raise RouteError(
                    'native Arrow Decimal columns remain unfinished in the partial frame materializer'
                )
            # Exact Arrow LargeUTF8 has a frozen native-engine failure;
            # StringDtype exports can use the same physical offset width.
            # Keep this native family unfinished after generic conversion
            # validation, and never inspect caller frames on checkpoint reuse.
            if any(
                type(dtype) is pd.ArrowDtype
                and pa.types.is_large_string(dtype.pyarrow_dtype)
                for dtype in native_types.values()
            ):
                raise RouteError(
                    'exact Arrow large_string columns remain unfinished in the partial frame materializer'
                )
            attached[number] = rows
            tables.append({'source': codec.relation(name), 'rows': number})
    attached[7] = codec.attachment(
        {'tables': tables, 'entropy': secrets.randbits(64)}
    )
    return attached


def rebind(
    held: dict[str, Any], attached: dict[int, Any]
) -> dict[str, Any] | None:
    """Bind retained owners under this call's ids, without reading metadata."""
    checkpoint = held.get('frame_checkpoint')
    if checkpoint is None:
        return None
    next_id = max(attached, default=0) + 1
    metadata = next_id
    attached[metadata] = checkpoint['metadata']
    tables = []
    for table in checkpoint['tables']:
        next_id += 1
        attached[next_id] = table['rows']
        tables.append({'source': table['source'], 'rows': next_id})
    row_identities = []
    for identity in checkpoint.get('row_identities', []):
        next_id += 1
        attached[next_id] = identity['rows']
        row_identities.append(
            {
                'definition': identity['definition'],
                'values': {'values': next_id, 'encoding': identity['encoding']},
            }
        )
    next_id += 1
    attached[next_id] = codec.attachment(
        {
            'metadata': metadata,
            'tables': tables,
            'row_identities': row_identities,
        }
    )
    return {'version': checkpoint['version'], 'manifest': next_id}


def take(handle: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Take every exported owner before dropping the completed native run."""
    import json

    manifest = json.loads(handle.take_attachment(checkpoint['manifest']))
    metadata = handle.take_attachment(manifest['metadata'])
    if manifest.get('retained') is not None:
        return {
            'version': checkpoint['version'],
            'metadata': metadata,
            'owner': handle.take_attachment(manifest['retained']),
            'tables': [],
        }
    import pyarrow as pa

    tables = []
    for table in manifest['tables']:
        owner = handle.take_attachment(table['rows'])
        tables.append({'source': table['source'], 'rows': pa.table(owner)})
    row_identities = []
    for identity in manifest.get('row_identities', []):
        owner = handle.take_attachment(identity['values']['values'])
        row_identities.append(
            {
                'definition': identity['definition'],
                'encoding': identity['values']['encoding'],
                'rows': pa.table(owner),
            }
        )
    return {
        'version': checkpoint['version'],
        'metadata': metadata,
        'tables': tables,
        'row_identities': row_identities,
    }


def publish(state: dict[str, Any], previous: dict[str, Any] | None) -> None:
    """Publish model ownership without replacing a later cold construction.

    A frame facade keeps a call-local ownership dictionary and a separate
    materializer cache. Only the first capture of a freshly constructed
    graph publishes a different owner. Later captures update that cache
    only while it still names the same actual core resource; concurrent
    calls keep their own local graph/task data irrespective of the cache.
    """
    cache = state.get('frame_cache')
    if cache is None:
        return
    checkpoint = state.get('held', {}).get('frame_checkpoint')
    if checkpoint is None or 'owner' not in checkpoint:
        return
    owner = checkpoint['owner']
    cached = cache.get('held', {}).get('frame_checkpoint', {}).get('owner')
    prior = (previous or {}).get('frame_checkpoint', {}).get('owner')
    if cached is None or cached.same_owner(owner) or prior is None:
        cache['held'] = state['held']
        # These are native ownership companions from the same graph
        # construction, not caller Schema reads or source-hash interning.
        for field in ('frame_columns', 'frame_schema', 'frame_native_tables'):
            if field in state:
                cache[field] = state[field]
