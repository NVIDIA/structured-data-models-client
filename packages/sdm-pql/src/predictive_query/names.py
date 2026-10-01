# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""Spelling of table and column names in a predictive query.

A warehouse accepts names a predictive query's bare identifier cannot spell,
such as ``Customer ID``. Those are written backtick-quoted, and the quoting is
a spelling device only: once parsed, the AST holds the name the data itself
uses, so nothing downstream needs to know a name was ever quoted.
:func:`fqn` puts the quotes back when a query is rendered.

This module sits outside ``parser/`` and ``ast/`` so that the spelling rules
have one home that both import from, rather than a copy in each. The grammar's
side of it is the ``QUOTED_ID`` token in ``grammar/PQLGrammar.g4``.
"""

from __future__ import annotations

import re

QUOTE = '`'

# Mirrors the grammar's ID rule; a name matching it needs no quoting.
_BARE_NAME = re.compile(r'[_a-zA-Z][_a-zA-Z0-9]*')


def split_fqn(full_name: str) -> list[str]:
    r"""Splits a fully qualified name on its separating dots.

    A backtick-quoted part may itself hold a dot, so a plain :meth:`str.split`
    would cut such a name in half. Dots inside quotes are therefore kept, and
    the quotes are stripped from each part, leaving the name the warehouse
    knows.

    Args:
        full_name: Fully qualified column name in the format of
            `<table_name>`.`<col_name>`, either part optionally quoted.

    Returns:
        The parts, unquoted.
    """
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    for char in full_name:
        if char == QUOTE:
            quoted = not quoted
        elif char == '.' and not quoted:
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    parts.append(''.join(current))
    return parts


def canonical_fqn(full_name: str) -> str:
    r"""Render *full_name* with every part unquoted.

    Called once during parsing so the AST and its consumers, including
    dataframe lookups, schema lookups and the sampler, see the data's name
    instead of the query's spelling.

    A quoted part holding a dot is rejected. The parts are rejoined with dots,
    so such a name is indistinguishable from a table/column pair here and at
    every consumer that splits a fully qualified name on a dot.

    Args:
        full_name: Fully qualified column name, either part optionally quoted.

    Returns:
        The name, unquoted.

    Raises:
        ValueError: If a quoted part contains a dot.
    """
    parts = split_fqn(full_name)
    for part in parts:
        if '.' in part:
            raise ValueError(
                f"Column name '{part}' contains a dot, which is not supported "
                f'in a predictive query. Rename the column to reference it.'
            )
    return '.'.join(parts)


def quote_name(name: str) -> str:
    r"""Quotes *name* only when the grammar's bare identifier cannot spell it.

    Leaving spellable names bare keeps queries readable and keeps the output of
    this function stable for every name that parsed before quoting existed.

    Args:
        name: A table or column name.

    Returns:
        The name, backtick-quoted if it needs to be.

    Raises:
        ValueError: If the name holds a character no quoting can carry, so an
            unparseable query is never built silently.
    """
    if name == '*' or _BARE_NAME.fullmatch(name):
        return name
    for char, label in (
        (QUOTE, 'a backtick'),
        ('\r', 'a carriage return'),
        ('\n', 'a newline'),
    ):
        if char in name:
            raise ValueError(
                f"Name '{name}' contains {label}, which a predictive query "
                f'cannot express.'
            )
    return f'{QUOTE}{name}{QUOTE}'


def fqn(table_name: str, col_name: str) -> str:
    r"""Given a table name and column name, returns a fully qualified column
    name as `<table_name>`.`<col_name>`.

    Args:
        table_name: A table name.
        col_name: A column name.

    Returns:
        Fully qualified column name in the format of
            `<table_name>`.`<col_name>`.
    """
    return f'{quote_name(table_name)}.{quote_name(col_name)}'
