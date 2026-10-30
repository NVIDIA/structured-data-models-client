# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What explaining a prediction gives back: gradients and column analysis."""

from __future__ import annotations

from sdm_client.explanations.column import ColumnAnalysis
from sdm_client.explanations.gradient import GradientExplanations


class Explanations:
    r"""A prediction's explanations, in two parts.

    Attributes:
        gradients: Each explained row's gradient explanation, keyed as the row
            is in the frame. Empty when explanations were not asked for.
        column_analysis: Per-column cohort statistics over the request's
            context rows. ``None`` when explanations were not asked for.
    """

    def __init__(
        self,
        gradients: GradientExplanations | None = None,
        column_analysis: ColumnAnalysis | None = None,
    ) -> None:
        self.gradients = (
            gradients if gradients is not None else GradientExplanations()
        )
        self.column_analysis = column_analysis

    def __repr__(self) -> str:
        analysed = (
            f'{len(self.column_analysis._columns)} columns analyzed'
            if self.column_analysis is not None
            else 'no column analysis'
        )
        return f'Explanations({len(self.gradients)} rows explained, {analysed})'
