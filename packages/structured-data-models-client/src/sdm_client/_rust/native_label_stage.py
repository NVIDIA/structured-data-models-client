# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One fixed native target handover used by local and provider-backed data.

The preparation port consumes completed domain values; it supplies neither
an action program nor native callbacks. Every condition/window decision
remains in Rust, between the actual reached native observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa

from sdm_client._rust import codec
from sdm_client._rust.frame_native import NativeColumn, NativeFrameFailure
from sdm_client._rust.native_products import Bindings, Product
from sdm_client._rust.native_target import NativeTargetColumn


@dataclass(frozen=True)
class ColumnExport:
    rows: pa.Table
    presence: pa.Table | None
    encoding: str | None

    def bind(self, bindings: Bindings) -> dict[str, Any]:
        return {
            'rows': bindings.add(self.rows),
            'presence': None
            if self.presence is None
            else bindings.add(self.presence),
            'encoding': self.encoding,
        }


def export_column(
    column: NativeTargetColumn, *, presence: object | None
) -> ColumnExport:
    values, encoding = NativeColumn(column.values).arrow()
    target = values.column(0)
    if not (
        pa.types.is_string(target.type)
        or pa.types.is_large_string(target.type)
        or pa.types.is_binary(target.type)
        or pa.types.is_large_binary(target.type)
    ):
        encoding = None
    return ColumnExport(
        pa.table(
            {
                'position': pa.array(column.positions, type=pa.uint64()),
                'target': target,
            }
        ),
        None
        if presence is None
        else pa.table({'present': pa.array(presence, type=pa.bool_())}),
        encoding,
    )


class SourceColumns(Protocol):
    def target_series(
        self, binding: dict[str, Any], rows: list[int]
    ) -> pd.Series: ...


class ConditionSources(Protocol):
    def columns(self, view: Product) -> SourceColumns: ...


class TargetPreparation(Protocol):
    def target_column(
        self, inputs: Product, converted: ColumnExport
    ) -> Product: ...
    def condition_column(
        self,
        column: Product,
        occurrence: dict[str, Any],
        converted: ColumnExport,
    ) -> Product: ...
    def filtered_target(
        self, column: Product, converted: ColumnExport
    ) -> Product: ...
    def windowed_target(
        self, column: Product, converted: ColumnExport
    ) -> Product: ...
    def native_failure(self, failure: NativeFrameFailure) -> NoReturn: ...
    def eligibility_column(
        self, inputs: Product, column: Product
    ) -> Product: ...
    def what_if_column(self, groups: Product, column: Product) -> Product: ...


def complete_eligibility(
    inputs: Product, sources: ConditionSources, preparation: TargetPreparation
) -> Product:
    """Complete each actual entity column view before the target is read."""
    history = inputs.data.get('eligibility')
    if history is None:
        raise codec.CodecError(
            'native label input lacks its entity-column history'
        )
    current = inputs
    try:
        for view in history['columns']:
            column_input = Product(
                'label_inputs',
                view,
                {view['owner']: inputs.owners[view['owner']]},
            )
            column = None
            try:
                column, _ = complete_target(
                    column_input, sources.columns(column_input), preparation
                )
                following = preparation.eligibility_column(current, column)
            finally:
                column_input.owners.clear()
                if column is not None:
                    column.owners.clear()
            if current is not inputs:
                current.owners.clear()
            current = following
        return current
    except BaseException:
        if current is not inputs:
            current.owners.clear()
        raise


def complete_what_if(
    groups: Product, sources: ConditionSources, preparation: TargetPreparation
) -> Product:
    """Observe late condition inputs only after actual group restoration."""
    if groups.data.get('restored') is None:
        raise codec.CodecError('what-if inputs precede restored group data')
    history = groups.data.get('what_if')
    if history is None:
        return groups
    current = groups
    try:
        for view in history['columns']:
            column_input = Product(
                'label_inputs',
                view,
                {view['owner']: groups.owners[view['owner']]},
            )
            column = None
            try:
                column, _ = complete_target(
                    column_input, sources.columns(column_input), preparation
                )
                following = preparation.what_if_column(current, column)
            finally:
                column_input.owners.clear()
                if column is not None:
                    column.owners.clear()
            if current is not groups:
                current.owners.clear()
            current = following
        return current
    except BaseException:
        if current is not groups:
            current.owners.clear()
        raise


def complete_target(
    inputs: Product,
    source: SourceColumns,
    preparation: TargetPreparation,
) -> tuple[Product, NativeTargetColumn]:
    rows = (
        pa.table(inputs.owners[inputs.data['owner']].rows())
        .column('source_row')
        .to_pylist()
    )
    try:
        values = source.target_series(inputs.data['target'], rows)
        native = NativeTargetColumn.column(
            values, drop_missing=inputs.data['form'] == 'plain'
        )
        converted = export_column(native, presence=native.present)
    except Exception as error:
        preparation.native_failure(NativeFrameFailure.capture(error))
    column = preparation.target_column(inputs, converted)
    if inputs.data['form'] == 'condition':
        return column, native
    try:
        if inputs.data['form'] == 'filtered':
            for occurrence in inputs.data['condition_columns']:
                try:
                    # Repeated occurrences read their actual native source
                    # separately, after the preceding Rust comparison.
                    operand = NativeTargetColumn.column(
                        source.target_series(occurrence, rows),
                        drop_missing=False,
                    )
                    converted = export_column(operand, presence=operand.present)
                except Exception as error:
                    preparation.native_failure(
                        NativeFrameFailure.capture(error)
                    )
                next_column = preparation.condition_column(
                    column, occurrence, converted
                )
                column.owners.clear()
                column = next_column
                operand = None
            selected = (
                pa.table(column.owners[column.data['owner']].selection())
                .column(0)
                .to_numpy()
            )
            try:
                native = native.filtered(np.asarray(selected, dtype=bool))
                converted = export_column(native, presence=None)
            except Exception as error:
                preparation.native_failure(NativeFrameFailure.capture(error))
            next_column = preparation.filtered_target(column, converted)
            column.owners.clear()
            column = next_column
        selected = (
            pa.table(column.owners[column.data['owner']].selection())
            .column(0)
            .to_numpy()
        )
        try:
            native = native.windowed(np.asarray(selected, dtype=bool))
            converted = export_column(native, presence=native.window_presence)
        except Exception as error:
            preparation.native_failure(NativeFrameFailure.capture(error))
        next_column = preparation.windowed_target(column, converted)
    finally:
        column.owners.clear()
    return next_column, native
