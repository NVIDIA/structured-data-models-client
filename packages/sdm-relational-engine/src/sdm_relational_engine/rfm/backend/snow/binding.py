# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sdm_relational_engine.rfm.backend.snow import Connection


@contextmanager
def paramstyle(
    connection: Connection,
    style: str = 'qmark',
) -> Iterator[None]:
    r"""Switches the driver's parameter style for the duration of the block.

    Server-side ``qmark`` binding is how a caller-supplied name or value
    reaches Snowflake as data rather than as SQL text. Snowflake honours
    backslash escapes inside single-quoted literals, so building a literal by
    doubling quotes is not sufficient escaping.

    This lives in its own module so both the table and the sampler layers can
    bind without importing each other.

    The connection is usually owned by the caller, so the previous style is
    restored no matter how the block exits.
    """
    previous = connection._paramstyle
    connection._paramstyle = style
    try:
        yield
    finally:
        connection._paramstyle = previous
