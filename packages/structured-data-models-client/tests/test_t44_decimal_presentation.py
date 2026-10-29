# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core-selected Decimal diagnostics retain sign, payload and native context."""

from decimal import InvalidOperation, localcontext

import pytest

from sdm_client._rust.frames import _frame_timestamp


@pytest.mark.parametrize(
    ('kind', 'negative', 'digits', 'exponent', 'expected'),
    [
        ('finite', False, '12345', -3, '12.345'),
        ('infinity', False, '123', 0, 'Infinity'),
        ('infinity', True, '123', 0, '-Infinity'),
        ('nan', False, '00123', 0, 'NaN123'),
        ('nan', True, '00123', 0, '-NaN123'),
        ('signaling_nan', False, '00123', 0, 'sNaN123'),
        ('signaling_nan', True, '00123', 0, '-sNaN123'),
    ],
)
def test_decimal_diagnostic_keeps_exact_specials_without_rounding_context(
    kind, negative, digits, exponent, expected
):
    fields = {
        'table': 'rows',
        'column': 'value',
        'value': digits,
        'frame_decimal_kind': kind,
        'frame_decimal_negative': negative,
        'frame_decimal_exponent': exponent,
    }
    with localcontext() as context:
        context.prec = 2
        context.traps[InvalidOperation] = True
        before = (
            context.prec,
            context.rounding,
            context.traps.copy(),
            context.flags.copy(),
        )
        assert _frame_timestamp(fields, {}, {}) == expected
        assert (
            context.prec,
            context.rounding,
            context.traps.copy(),
            context.flags.copy(),
        ) == before
