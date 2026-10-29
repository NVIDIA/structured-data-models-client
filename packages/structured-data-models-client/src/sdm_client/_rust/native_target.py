# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual converted target values and native storage, without label policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

NativeGroupResult: TypeAlias = (
    npt.NDArray[np.object_] | pd.api.extensions.ExtensionArray
)


@dataclass(frozen=True)
class NativeTargetColumn:
    """A native derived column aligned to its actual fetched source rows.

    Core selects plain versus filtered query meaning and supplies subsequent
    condition and time selections. These methods perform only the original
    fixed vector conversions; no query or label arithmetic belongs here.
    """

    values: pd.Series
    positions: npt.NDArray[np.intp]
    present: npt.NDArray[np.bool_]

    @classmethod
    def column(
        cls, values: pd.Series, *, drop_missing: bool
    ) -> NativeTargetColumn:
        """Complete execute_column's native operations in their reached order."""
        out = values.reset_index(drop=True)
        if pd.api.types.is_float_dtype(out):
            out = out.astype('float32')
        out.name = None
        out.index.name = None
        mask = out.notna().to_numpy()
        positions = np.arange(len(mask), dtype=np.intp)
        if drop_missing:
            out = out[mask].reset_index(drop=True)
            positions = positions[mask]
            if pd.api.types.is_integer_dtype(out):
                out = out.astype('int64')
            elif pd.api.types.is_bool_dtype(out):
                out = out.astype('bool')
        return cls(out, positions, mask)

    def filtered(self, condition: npt.NDArray[np.bool_]) -> NativeTargetColumn:
        """Apply the core's actual condition to the filter-na-false column."""
        selected = condition & self.present
        return NativeTargetColumn(
            self.values[selected].reset_index(drop=True),
            self.positions[selected],
            selected,
        )

    def windowed(self, selected: npt.NDArray[np.bool_]) -> NativeTargetColumn:
        """Retain native time selection and the aggregation's second notna."""
        values = self.values[pd.Series(selected)]
        positions = self.positions[selected]
        present = values.notna()
        values = values[present]
        positions = positions[present.to_numpy()]
        mask = np.zeros(len(self.present), dtype=bool)
        mask[positions] = True
        return NativeTargetColumn(values, positions, mask)

    def group_result(self, groups: list[object]) -> NativeGroupResult:
        """Restore Rust's actual groups through this derived array's owner."""
        boxed = np.empty(len(groups), dtype=object)
        for ordinal, group in enumerate(groups):
            boxed[ordinal] = group
        native = self.values.array
        cast_result = getattr(native, '_cast_pointwise_result', None)
        if cast_result is not None:
            return cast_result(boxed)

        # Earlier native groupby first converts its actual object result and
        # attempts dtype restoration only for extension-backed input. A plain
        # integer cast has already changed the actual input to NumPy here.
        from pandas._libs import lib
        from pandas.core.dtypes.cast import maybe_cast_pointwise_result

        converted = lib.maybe_convert_objects(boxed, try_float=False)
        if not isinstance(self.values._values, np.ndarray):
            return maybe_cast_pointwise_result(
                converted, self.values.dtype, numeric_only=True
            )
        return converted
