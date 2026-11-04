# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What explaining a prediction gives back: gradients and column explanations."""

from __future__ import annotations

from sdm_client.explanations.column import ColumnExplanations
from sdm_client.explanations.gradient import GradientExplanations


class Explanations:
    r"""A prediction's explanations, in two parts.

    Attributes:
        gradients: Each explained row's gradient explanation, keyed as the row
            is in the frame. Empty when explanations were not asked for.
        columns: Per-column cohort statistics over the request's context rows.
            ``None`` when explanations were not asked for.
    """

    def __init__(
        self,
        gradients: GradientExplanations | None = None,
        columns: ColumnExplanations | None = None,
    ) -> None:
        self.gradients = (
            gradients if gradients is not None else GradientExplanations()
        )
        self.columns = columns

    def __repr__(self) -> str:
        explained = (
            f'{len(self.columns._columns)} columns explained'
            if self.columns is not None
            else 'no column explanations'
        )
        return (
            f'Explanations({len(self.gradients)} rows explained, {explained})'
        )
