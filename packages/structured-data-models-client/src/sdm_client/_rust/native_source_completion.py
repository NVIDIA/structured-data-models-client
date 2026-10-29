# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete actual source data around fixed core-owned construction meanings.

Each instance belongs to one producing Run.
Its retained native tables use that Run's actual logical-read IDs only.
Lookup admission and scalar/fallback policy remain in the domain service.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa

from sdm_client._rust.codec import CodecError
from sdm_client._rust.frame_decimals import native_decimal_type
from sdm_client._rust.frame_facts import native_declaration
from sdm_client._rust.frame_native import NativeFrameFailure
from sdm_client._rust.native_products import Product
from sdm_client._rust.native_source import scalar_column
from sdm_client._rust.native_source_rows import (
    SourceReadFailure,
    SourceRowsProvider,
    read_source_rows,
)
from sdm_client._rust.providers.result_columns import result_columns
from sdm_client._rust.run import FetchRefused

_BATCH = '__kumo_batch__'
_MATCHES = '__kumo_matches__'
_HELPERS = (_BATCH, _MATCHES, '__kumo_row__', '__kumo_id__')


@dataclass(frozen=True)
class NativeSourceTable:
    source: dict[str, Any]
    frame: pd.DataFrame
    time_column: str | None = None


class SourceConstruction(Protocol):
    def lookup(
        self, lookup: dict[str, Any], helpers: pa.Table, owner: object
    ) -> pa.Table: ...
    def column(
        self, binding: dict[str, Any], facts: dict[str, Any]
    ) -> dict[str, Any]: ...
    def fallback(
        self, meaning: dict[str, Any], failure: NativeFrameFailure
    ) -> dict[str, Any]: ...
    def read_failure(self, failure: SourceReadFailure) -> NoReturn: ...


def _helpers(columns: Sequence[str]) -> dict[str, str]:
    names = {}
    for canonical in _HELPERS:
        found = [
            name for name in (canonical, canonical.upper()) if name in columns
        ]
        if len(found) > 1:
            raise CodecError('source data repeats a transport helper spelling')
        if found:
            names[canonical] = found[0]
    return names


class SourceCompletion:
    def __init__(
        self,
        provider: SourceRowsProvider,
        source: Product,
        service: SourceConstruction,
    ) -> None:
        self.provider = provider
        self.source = source
        self.service = service
        self.tables: dict[int, NativeSourceTable] = {}
        self.native_error: Exception | None = None

    def __call__(
        self,
        steps: Sequence[dict[str, Any]],
        keys: Mapping[int, Mapping[int, object]],
    ) -> dict[int, pa.Table] | FetchRefused:
        self.native_error = None
        try:
            return self._complete(steps, keys)
        except Exception as error:
            # Service failures remain their selected public/Core owners.
            # Only the narrow reached native sections set native_error.
            return FetchRefused(error)

    def _binding(self, steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not steps or self.source.kind != 'source':
            raise CodecError('native source completion lacks its source read')
        read = steps[0]['request']['table']
        matches = [
            table
            for table in self.source.data['tables']
            if table['read'] == read
        ]
        if len(matches) != 1 or any(
            step['request']['table'] != read for step in steps
        ):
            raise CodecError(
                'native source read does not match its captured relation'
            )
        return matches[0]

    def _complete(
        self,
        steps: Sequence[dict[str, Any]],
        keys: Mapping[int, Mapping[int, object]],
    ) -> dict[int, pa.Table]:
        binding = self._binding(steps)
        first = steps[0]['partition']['first']
        if first in self.tables:
            raise CodecError(
                'native source read identity was already completed in this Run'
            )
        raw = read_source_rows(self.provider, steps, keys)
        if isinstance(raw, FetchRefused):
            self.native_error = raw.raised
            raise raw.raised
        if isinstance(raw, SourceReadFailure):
            self.service.read_failure(raw)
        rows = raw.rows
        columns = (
            rows.column_names
            if isinstance(rows, pa.Table)
            else list(rows.columns)
        )
        helpers = _helpers(columns)
        matches: np.ndarray | None = None
        try:
            if not isinstance(rows, pa.Table) and _MATCHES in helpers:
                # The scalar port reaches this pop/cast before counted lookup.
                matches = rows.pop(helpers[_MATCHES]).to_numpy(dtype=np.int64)
            helper_columns: dict[str, Any] = {}
            if _BATCH in helpers:
                helper_columns[_BATCH] = (
                    rows[helpers[_BATCH]].cast(pa.int64())
                    if isinstance(rows, pa.Table)
                    else pa.array(rows[helpers[_BATCH]], type=pa.int64())
                )
            if matches is not None:
                helper_columns[_MATCHES] = pa.array(matches, type=pa.int64())
            elif isinstance(rows, pa.Table) and _MATCHES in helpers:
                helper_columns[_MATCHES] = rows[helpers[_MATCHES]].cast(
                    pa.int64()
                )
            helper_rows = pa.table(helper_columns)
            if isinstance(rows, pa.Table):
                for canonical in ('__kumo_row__', '__kumo_id__'):
                    if canonical in helpers:
                        helper_columns[canonical] = rows[helpers[canonical]]
        except Exception as error:
            self.native_error = error
            raise
        lookup = steps[0].get('lookup')
        positions = None
        if lookup is not None:
            # This is outside all native handlers. The retained task owner
            # selects duplicate-scored-key admission before business conversion.
            validated = self.service.lookup(
                lookup, helper_rows, keys[steps[0]['id']][lookup['owner']]
            )
            if validated.column_names != [
                'position'
            ] or validated.num_rows != len(rows):
                raise CodecError(
                    'counted lookup returned no complete row positions'
                )
            positions = validated.column('position')
        if isinstance(rows, pa.Table):
            conversions = steps[0]['plan'].get('result_conversions', [])
            if any(
                step['plan'].get('result_conversions', []) != conversions
                for step in steps
            ):
                raise CodecError(
                    'source partition changes its result transport meaning'
                )
            try:
                rows = result_columns(rows, conversions)
            except (pa.ArrowException, MemoryError) as error:
                self.native_error = error
                raise
            try:
                if positions is None and _BATCH in helpers:
                    positions = rows[helpers[_BATCH]].cast(pa.int64())
                business = rows.select(
                    [
                        index
                        for index, name in enumerate(rows.column_names)
                        if name not in helpers.values()
                    ]
                )
                frame = business.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception as error:
                self.native_error = error
                raise
        else:
            frame = rows
            try:
                if _BATCH in helpers:
                    observed = frame.pop(helpers[_BATCH]).to_numpy(
                        dtype=np.int64
                    )
                    if positions is None:
                        positions = pa.array(observed, type=pa.int64())
                for canonical in ('__kumo_row__', '__kumo_id__'):
                    if canonical in helpers:
                        observed_row = frame.pop(helpers[canonical])
                        helper_columns[canonical] = pa.array(
                            observed_row, type=pa.int64()
                        )
            except Exception as error:
                self.native_error = error
                raise
            # Declaration order is distinct from the fetched business order.
            for name in binding['columns']:
                if name not in frame:
                    continue
                meaning = self.service.column(
                    {'source': binding['source'], 'column': name},
                    {'storage': 'cell_rows'},
                )
                if meaning['scalar']['meaning'] != 'original':
                    self._scalar(frame, name, meaning, decimal=False)
        # Only actual native Decimal storage reaches the common Decimal pass.
        for name in frame.columns:
            try:
                dtype = frame[name].dtype
                decimal = native_decimal_type(dtype)
                declaration = (
                    None if decimal is None else native_declaration(dtype)
                )
            except Exception as error:
                self.native_error = error
                raise
            if decimal is not None:
                meaning = self.service.column(
                    {'source': binding['source'], 'column': name},
                    {'storage': 'typed_columns', 'declaration': declaration},
                )
                self._scalar(frame, name, meaning, decimal=True)
        try:
            exported = pa.Table.from_pandas(frame, preserve_index=False)
            if positions is not None:
                # Core positions are UInt64 row data. The issued SQL batch
                # helper remains the established signed int64 transport.
                helper_columns[_BATCH] = positions.cast(pa.int64())
            # The completed native frame is helper-free; the replied source
            # still supplies every actual transport helper in its original
            # column slot for the plan and later shared Core readers.
            helper_names = {
                actual: canonical for canonical, actual in helpers.items()
            }
            arrays, fields = [], []
            business_index = 0
            for original_index, name in enumerate(columns):
                helper_name = helper_names.get(name)
                if helper_name is not None:
                    array = helper_columns[helper_name]
                    field = (
                        rows.schema.field(original_index)
                        .with_name(helper_name)
                        .with_type(array.type)
                        if isinstance(rows, pa.Table)
                        else pa.field(helper_name, array.type)
                    )
                else:
                    array = exported.column(business_index)
                    field = exported.schema.field(business_index)
                    business_index += 1
                arrays.append(array)
                fields.append(field)
            exported = pa.Table.from_arrays(
                arrays,
                schema=pa.schema(fields, metadata=exported.schema.metadata),
            )
            replies = {}
            offset = 0
            for step, count in zip(steps, raw.row_counts, strict=True):
                replies[step['id']] = exported.slice(offset, count)
                offset += count
        except Exception as error:
            self.native_error = error
            raise
        if offset != exported.num_rows:
            raise CodecError(
                'source completion changed its complete issued row count'
            )
        self.tables[first] = NativeSourceTable(
            binding['source'], frame, binding.get('time_column')
        )
        return replies

    def _scalar(
        self,
        frame: pd.DataFrame,
        name: str,
        meaning: dict[str, Any],
        *,
        decimal: bool,
    ) -> None:
        recipe = meaning['scalar']
        failure = None
        try:
            converted = scalar_column(frame[name], recipe)
        except CodecError:
            raise
        except (TypeError, ValueError, ArithmeticError) as error:
            if not decimal:
                self.native_error = error
                raise
            failure = NativeFrameFailure.capture(error)
        except Exception as error:
            self.native_error = error
            raise
        if failure is not None:
            meaning = self.service.fallback(meaning, failure)
            recipe = meaning['scalar']
            try:
                converted = scalar_column(frame[name], recipe)
            except CodecError:
                raise
            except Exception as error:
                self.native_error = error
                raise
        try:
            frame[name] = converted
        except Exception as error:
            self.native_error = error
            raise
