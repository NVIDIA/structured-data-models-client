# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native equality/missingness observations for selected populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._rust import codec


@dataclass(frozen=True)
class Quality:
    owner: pd.DataFrame
    columns: tuple[str, ...]
    missing: int
    duplicates: int
    example: Any
    example_row: int | None

    def facts(self) -> bytes:
        return codec.attachment(
            {
                'missing': self.missing,
                'duplicates': self.duplicates,
                'example_row': self.example_row,
            }
        )


def quality(frame: pd.DataFrame, columns: list[str]) -> Quality:
    """Read one core-selected key population using actual pandas operations.

    The native example remains bound to this exact source owner/columns/row.
    No repr, scalar cast, uniqueness threshold or schema decision occurs here.
    """
    names = tuple(columns)
    if any(name not in frame.columns for name in names):
        return Quality(frame, names, 0, 0, None, None)
    key = frame[names[0]] if len(names) == 1 else frame[list(names)]
    missing = int(
        key.isna().any(axis=1).sum() if len(names) > 1 else key.isna().sum()
    )
    present_mask = ~key.isna().any(axis=1) if len(names) > 1 else key.notna()
    present = key[present_mask]
    repeated_mask = present.duplicated()
    duplicates = int(repeated_mask.sum())
    example = None
    row = None
    if duplicates:
        first = present[repeated_mask].iloc[0]
        example = first if len(names) == 1 else tuple(first)
        # These are already-created native Boolean masks. Reading positions
        # invokes no source-cell equality, hashing or conversion again.
        source_positions = np.flatnonzero(present_mask.to_numpy(copy=False))
        repeated_positions = np.flatnonzero(repeated_mask.to_numpy(copy=False))
        row = int(source_positions[repeated_positions[0]])
    return Quality(frame, names, missing, duplicates, example, row)
