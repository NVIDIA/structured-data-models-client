# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit task meaning for caller-supplied relational examples."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RelationalTask:
    """A relational task without PQL; Rust validates its schema and labels.

    Forecast steps are fixed nanoseconds. Link targets name another schema
    table. Context columns are ENTITY, ANCHOR_TIMESTAMP and TARGET; prediction
    columns omit TARGET. Static tasks may omit ANCHOR_TIMESTAMP.

    ``temporal_link_prediction`` is refused by ``materialize_examples`` before
    any work, with ``UnsupportedRequestError``, since no released Kumo
    Relational model serves it.
    """

    kind: Literal[
        'regression',
        'binary_classification',
        'multiclass_classification',
        'temporal_link_prediction',
        'forecasting',
    ]
    entity_table: str
    link_target_table: str | None = None
    step_size_ns: int | None = None
    num_forecasts: int | None = None
