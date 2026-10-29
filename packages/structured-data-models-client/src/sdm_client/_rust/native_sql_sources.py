# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete existing Reader-owned temporal data before native source capture."""

from __future__ import annotations

import json
from typing import NoReturn, Protocol

import pandas as pd
import pyarrow as pa

from sdm_client._rust.codec import CodecError
from sdm_client._rust.frame_native import NativeFrameFailure
from sdm_client._rust.native_products import Product
from sdm_client._rust.native_source import finish_timestamp
from sdm_client._rust.native_source_completion import (
    SourceCompletion,
    _timestamp_declaration,
)


class SourcePreparation(Protocol):
    def native_failure(self, failure: NativeFrameFailure) -> NoReturn: ...


def complete_source_times(
    inputs: Product,
    entity: SourceCompletion,
    windowed: SourceCompletion,
    preparation: SourcePreparation,
) -> None:
    """Use complete same-read typed values; never parse the original text again.

    All source/extent/position bindings are checked before assignments.
    Already typed native storage is retained, including its mutable backing.
    """
    sources = {'entity': entity.tables, 'windowed': windowed.tables}
    expected = {
        (role, read) for role, tables in sources.items() for read in tables
    }
    seen = set()
    ready = []
    observations = table = data = stream = values = frame = None
    try:
        observations = inputs.owners[inputs.data['owner']].temporal_sources()
        for descriptor, stream in observations:
            if type(descriptor) is not bytes:
                raise CodecError(
                    'complete temporal source has no descriptor bytes'
                )
            try:
                value = json.loads(descriptor)
            except ValueError:
                raise CodecError(
                    'complete temporal source descriptor is not JSON'
                ) from None
            if type(value) is not dict or set(value) != {
                'source_role',
                'source_read',
                'source',
                'columns',
                'rows',
            }:
                raise CodecError(
                    'complete temporal source descriptor has unknown fields'
                )
            role, read, columns, count = (
                value['source_role'],
                value['source_read'],
                value['columns'],
                value['rows'],
            )
            if (
                type(role) is not str
                or role not in sources
                or type(read) is not int
                or not 0 <= read < 2**32
                or type(count) is not int
                or count < 0
                or type(columns) is not list
                or any(type(name) is not str for name in columns)
                or len(set(columns)) != len(columns)
            ):
                raise CodecError(
                    'complete temporal source binding is malformed'
                )
            key = role, read
            if key not in expected or key in seen:
                raise CodecError(
                    'complete temporal source repeats or names another issued read'
                )
            table = sources[role][read]
            if value['source'] != table.source:
                raise CodecError(
                    'complete temporal source names another logical source'
                )
            data = pa.table(stream)
            if (
                data.column_names != ['source_row', *columns]
                or data.num_rows != count
                or data.column(0).type != pa.uint64()
                or data.column(0).to_pylist() != list(range(count))
                or any(
                    not pa.types.is_timestamp(field.type)
                    or field.type.tz is not None
                    for field in list(data.schema)[1:]
                )
            ):
                raise CodecError(
                    'complete temporal source does not cover its ordered full rows'
                )
            try:
                names = list(table.frame.columns)
                rows = len(table.frame)
            except Exception as error:
                preparation.native_failure(NativeFrameFailure.capture(error))
            if rows != count or any(name not in names for name in columns):
                raise CodecError(
                    'complete temporal source differs from its retained native read'
                )
            seen.add(key)
            ready.extend(
                (table.frame, names.index(name), name, data.column(at + 1))
                for at, name in enumerate(columns)
            )
        if seen != expected:
            raise CodecError(
                'complete temporal source omitted an actual retained read'
            )
        for frame, ordinal, name, values in ready:
            try:
                # Only a representation completed by the existing Reader is
                # supplied here. Existing native timestamp owners stay intact.
                if (
                    _timestamp_declaration(frame.dtypes.iloc[ordinal])
                    is not None
                ):
                    continue
                converted = pd.Series(
                    values.combine_chunks()
                    .to_numpy(zero_copy_only=False)
                    .copy(),
                    index=frame.index,
                    name=name,
                )
                frame[name] = finish_timestamp(converted)
            except Exception as error:
                preparation.native_failure(NativeFrameFailure.capture(error))
    finally:
        # A retained native exception keeps this helper's traceback. Drop only
        # these redundant read views, leaving caller-owned products untouched.
        ready.clear()
        sources.clear()
        observations = table = data = stream = values = frame = None
        del inputs, entity, windowed, preparation
