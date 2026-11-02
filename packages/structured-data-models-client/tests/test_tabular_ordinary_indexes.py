# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary object row-label ordering and Decimal regression contracts."""

from decimal import FloatOperation, InvalidOperation, localcontext

import pytest
from tabular_arrow_index_cases import digest
from tabular_ordinary_index_cases import CASES, frame_for, observe_case


def expected_case(name):
    """Pandas ordering and Decimal effects without invoking any client code."""
    family, entry, shape, mode = name.split('/')
    n, wanted = (81, 40) if entry == 'indices' else (10002, 10000)
    with localcontext() as context:
        context.prec = 2
        context.traps[FloatOperation] = False
        context.traps[InvalidOperation] = False
        frame = frame_for(family, n, shape)
        context.clear_flags()
        context.traps[FloatOperation] = mode == 'trapped'
        try:
            selected = (
                frame.iloc[:-1].sample(n=wanted, random_state=42).sort_index()
            )
        except FloatOperation as error:
            return {
                'error': {'type': 'FloatOperation', 'message': str(error)},
                'flags': sorted(
                    signal.__name__
                    for signal, set_ in context.flags.items()
                    if set_
                ),
            }
        ids = selected['id'].tolist()
        return {
            'context_count': wanted,
            'context_ids_sha256': digest(ids),
            'encoded_ids_sha256': digest(ids),
            'index_dtype': str(selected.index.dtype),
            'predictions': [{'row_index': 0, 'id': n - 1, 'prediction': 0.0}],
            'flags': sorted(
                signal.__name__
                for signal, set_ in context.flags.items()
                if set_
            ),
        }


@pytest.mark.parametrize('name', CASES)
def test_ordinary_labels_preserve_order_decimal_effects_and_prediction(name):
    expected = expected_case(name)
    actual = observe_case(name)
    assert actual == expected
    if 'error' in actual:
        assert name.startswith('decimal_float/') and name.endswith('/trapped')
        assert actual['error']['type'] == 'FloatOperation'
