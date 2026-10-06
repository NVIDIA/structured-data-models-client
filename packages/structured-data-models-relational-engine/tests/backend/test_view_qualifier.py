# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Qualifier handling in imported semantic/metric view expressions.

Kept apart from ``test_view_expr_import.py``, which needs the Snowflake
connector installed to reach its stub warehouse; these are pure functions and
must run everywhere.
"""

from __future__ import annotations

import pytest
from sdm_relational_engine.rfm.graph import (
    _mask_literals,
    _qualifier_pattern,
    _sub_outside_literals,
)


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        ('USERS.USER_ID', 'USER_ID'),
        ('UPPER(USERS.NAME)', 'UPPER(NAME)'),
        ('"USERS".NAME', 'NAME'),
        ('users.name', 'name'),
        ("CONCAT('src: USERS.', NAME)", "CONCAT('src: USERS.', NAME)"),
        (
            "CASE WHEN NAME = 'USERS.WEB' THEN 1 ELSE 0 END",
            "CASE WHEN NAME = 'USERS.WEB' THEN 1 ELSE 0 END",
        ),
        (
            "CONCAT(USERS.NAME, 'USERS.B', USERS.USER_ID)",
            "CONCAT(NAME, 'USERS.B', USER_ID)",
        ),
        (
            "CASE WHEN NAME = 'it''s USERS.X' THEN USERS.NAME END",
            "CASE WHEN NAME = 'it''s USERS.X' THEN NAME END",
        ),
    ],
)
def test_own_table_qualifier_is_stripped_outside_string_literals(
    expr, expected
):
    """``ORDERS.`` inside a literal is data. Stripping it rewrote the expression
    to compare against a different value, still valid SQL, so nothing
    downstream could tell that what the client runs is not what the view declares.
    Double quotes stay in scope: ``"USERS".NAME`` is a quoted-identifier
    qualifier the pattern is built to match.
    """
    pattern = _qualifier_pattern('USERS')
    assert _sub_outside_literals(pattern, expr) == expected


@pytest.mark.parametrize(
    ('expr', 'is_cross_table'),
    [
        ('ORDERS.TOTAL', True),
        ('"ORDERS".TOTAL', True),
        ('SUM(ORDERS.TOTAL)', True),
        ("CONCAT('ORDERS.X', NAME)", False),
        ("CASE WHEN NAME = 'ORDERS.WEB' THEN 1 ELSE 0 END", False),
        ('NAME', False),
    ],
)
def test_cross_table_detection_ignores_string_literals(expr, is_cross_table):
    r"""The mirror-image defect: another table's name inside a literal is data,
    and dropping the column for it reported "references other tables"
    untruthfully.
    """
    pattern = _qualifier_pattern('ORDERS')
    assert bool(pattern.search(_mask_literals(expr))) is is_cross_table
