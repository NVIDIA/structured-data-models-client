# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound traversal of host containers before the core validates them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class NestingTooDeep(TypeError):
    """A value exceeds the core's 100-level JSON value contract."""


class CyclicContainer(TypeError):
    """A column cell has no literal JSON form because it contains itself."""


def ancestors(
    value: Any, holding: frozenset[int], max_depth: int | None = 100
) -> frozenset[int]:
    """Track active ancestors, leaving repeated acyclic values intact."""
    try:
        import numpy as np
    except ImportError:
        array = False
    else:
        array = isinstance(value, np.ndarray)
    if not array and not isinstance(
        value, (Mapping, list, tuple, set, frozenset)
    ):
        return holding
    if max_depth is not None and len(holding) > max_depth:
        raise NestingTooDeep(
            'no plain form for a value nested more than 100 levels'
        )
    return holding | {id(value)}
