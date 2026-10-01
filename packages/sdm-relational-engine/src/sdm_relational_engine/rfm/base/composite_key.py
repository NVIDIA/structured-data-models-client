# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Identity spread across more than one column.

A warehouse often identifies a row by a combination of columns rather than by
one: a customer dimension partitioned by region holds one row per
``(customer, region)`` pair, and neither column alone is unique. A graph needs
a single node identity, and the serving contract carries a single
``instance_id`` per instance, so such a key is folded here into one derived
value.

The fold has to agree **exactly** on both sides of a join. Seed identities are
built in pandas while a SQL backend joins by evaluating an expression in the
warehouse, and a single character of disagreement produces no matched rows
rather than an error. Every value is therefore rendered by one documented rule,
and a part whose text a warehouse would render differently is refused outright
rather than trusted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from predictive_query.schema.typing import Dtype

DERIVED_PREFIX = '__kumo_key_'
SEPARATOR = '\x1f'
ESCAPE = '\\'

_REFUSED_DTYPES = (Dtype.float, Dtype.float32, Dtype.float64, Dtype.floatlist)


def escape_part(text: str) -> str:
    r"""Escapes the separator so that the fold stays injective.

    Without this, ``('a', 'b\x1fc')`` and ``('a\x1fb', 'c')`` would fold to the
    same value and two different rows would become one node.
    """
    return text.replace(ESCAPE, ESCAPE * 2).replace(SEPARATOR, ESCAPE + 'u')


def encode_values(values: Sequence[object]) -> str:
    r"""Folds one row's key parts into a single identity.

    Args:
        values: The key parts, in the order the key declares them.

    Returns:
        The derived identity.
    """
    return SEPARATOR.join(escape_part(_render(value)) for value in values)


def _render(value: object) -> str:
    r"""Renders one key part by the single rule both sides of a join use.

    A value's text is its identity here, so ``1`` and ``'1'`` fold alike; a
    key column holds one dtype, so this only matters across a badly matched
    pair of tables, which the link guard rejects.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        raise ValueError('a key column cannot hold a null value')
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, float) and math.isnan(value):
        raise ValueError('a key column cannot hold a null value')
    return str(value)


def encode_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    r"""Folds *columns* of *df* into one identity per row.

    Args:
        df: The frame holding the key columns.
        columns: The key columns, in the order the key declares them.

    Returns:
        One derived identity per row, aligned to ``df``.

    Raises:
        ValueError: If a key column is missing or holds a null.
    """
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(
            f'composite key columns {missing} are not present; a table '
            f'referencing this key must carry every one of {list(columns)}'
        )

    rendered = []
    for name in columns:
        column = df[name]
        if column.isna().any():
            raise ValueError(
                f"composite key column '{name}' holds a null value; every "
                f"part of a row's identity has to be present"
            )
        rendered.append(
            column.map(lambda value: escape_part(_render(value))).astype(str)
        )

    joined = rendered[0]
    for part in rendered[1:]:
        joined = joined + SEPARATOR + part
    return joined


def decode_value(encoded: str, arity: int) -> tuple[str, ...]:
    r"""Recovers the key parts from a derived identity.

    Used to report a prediction against the columns the caller named rather
    than against an identity this module invented.

    Args:
        encoded: A value produced by :func:`encode_values`.
        arity: How many parts the key declares.

    Returns:
        The parts, as text.

    Raises:
        ValueError: If *encoded* does not hold exactly *arity* parts.
    """
    parts: list[str] = []
    current: list[str] = []
    escaped = False
    for char in encoded:
        if escaped:
            current.append(SEPARATOR if char == 'u' else char)
            escaped = False
        elif char == ESCAPE:
            escaped = True
        elif char == SEPARATOR:
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    parts.append(''.join(current))
    if len(parts) != arity:
        raise ValueError(
            f'expected a key of {arity} parts, got {len(parts)}: {encoded!r}'
        )
    return tuple(parts)


def refuse_unfoldable_dtypes(
    dtypes: Sequence[tuple[str, Dtype | None]],
) -> None:
    r"""Rejects a key part a warehouse and pandas need not render alike.

    Raises:
        ValueError: If any part is approximate.
    """
    for name, dtype in dtypes:
        if dtype in _REFUSED_DTYPES:
            raise ValueError(
                f"composite key column '{name}' is {dtype}; a floating-point "
                f"column cannot be part of a row's identity because its text "
                f'is not reproducible. Use an exact column, or add a column '
                f'holding the identity you intend.'
            )


def sql_expression(
    columns: Sequence[str],
    *,
    quote: Callable[[str], str],
    text_type: str,
    chr_function: str = 'CHR',
) -> str:
    r"""Builds the fold as SQL, matching :func:`encode_frame` value for value.

    Character codes carry the separator and the escape rather than string
    literals: Snowflake and Databricks read a backslash inside a literal as an
    escape, so a literal would fold differently there than in pandas and the
    join would quietly match nothing.

    Args:
        columns: The key columns, in the order the key declares them.
        quote: Renders one column name as an identifier for this dialect.
        text_type: The dialect's text type, cast to before folding.
        chr_function: The dialect's code-point-to-character function.

    Returns:
        A SQL expression producing one row's identity.
    """
    backslash = f'{chr_function}(92)'
    separator = f'{chr_function}(31)'
    parts = []
    for name in columns:
        rendered = f'CAST({quote(name)} AS {text_type})'
        escaped = (
            f'REPLACE(REPLACE({rendered}, {backslash}, '
            f'{backslash} || {backslash}), {separator}, '
            f"{backslash} || 'u')"
        )
        parts.append(escaped)
    return f' || {separator} || '.join(parts)


#: A derived name has to survive a predictive query, a fully qualified name
#: being split on its dots, and a warehouse's identifier rules. Anything
#: longer is refused rather than truncated, since truncating would let two
#: identities collide.
MAX_DERIVED_NAME = 200


def encode_identity(columns: Sequence[str]) -> str:
    r"""Name the identity *columns* describe, recoverably and safely.

    Each name is written as hexadecimal and the parts are joined with
    underscores. Hexadecimal cannot contain an underscore, so the join is
    unambiguous. The result contains only ``[0-9a-f_]``, which matters because
    this name is written into predictive queries, split from a fully qualified
    name, and created as a warehouse column. Carrying the names directly let a
    dot break the split, a backtick break the quoting, and a space force every
    reference to be quoted.

    Args:
        columns: The key columns, in the order the key declares them.

    Returns:
        A name identifying that exact sequence.

    Raises:
        ValueError: If the result would exceed what a warehouse accepts.
    """
    encoded = '_'.join(name.encode('utf-8').hex() for name in columns)
    if len(DERIVED_PREFIX) + len(encoded) > MAX_DERIVED_NAME:
        raise ValueError(
            f'An identity over {list(columns)} needs a column name longer '
            f'than {MAX_DERIVED_NAME} characters, which a warehouse will not '
            f'accept. Add a column holding that identity and declare it '
            f'instead.'
        )
    return encoded


def decode_identity(derived: str) -> tuple[str, ...] | None:
    r"""Recovers the columns an identity was folded from.

    Args:
        derived: A column name produced with :data:`DERIVED_PREFIX` and
            :func:`encode_identity`.

    Returns:
        The key columns in declaration order, or :obj:`None` if *derived* does
        not name a folded identity.
    """
    if not derived.startswith(DERIVED_PREFIX):
        return None
    rest = derived[len(DERIVED_PREFIX) :]
    if not rest:
        return None
    columns: list[str] = []
    for part in rest.split('_'):
        try:
            columns.append(bytes.fromhex(part).decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return None
    return tuple(columns)
