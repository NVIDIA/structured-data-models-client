# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three things a review found after the fixes they belong to had landed."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import Schema
from sdm_client.materializers.relational import (
    _drop_context_that_answers_the_question,
)


def payload_with_a_shared_row() -> dict:
    """One entity scored at an anchor, and that same entity in the context at
    the same anchor -- the shape the drop exists to catch."""
    return {
        'task': {'anchor_time_column': '__kumo_anchor_time'},
        'context': {
            'instance_table': {
                'columns': [
                    'instance_id',
                    '__kumo_entity_ref_0',
                    '__kumo_anchor_time',
                    'v',
                ],
                'rows': [
                    [0, 'e1', '2024-01-01', 9.0],
                    [1, 'e2', '2024-01-01', 8.0],
                ],
            },
            'related_tables': {
                'e': {
                    'columns': ['instance_id', 'w'],
                    'rows': [[0, 1.0], [1, 2.0]],
                }
            },
        },
        'predict': {
            'instance_table': {
                'columns': ['__kumo_entity_ref_0', '__kumo_anchor_time'],
                'rows': [['e1', '2024-01-01']],
            }
        },
    }


def test_a_raised_warning_leaves_the_payload_whole():
    """The warning is the caller's only notice that rows went. Turning it into
    an error must not also be what strips them: a raise partway through left
    the payload short of rows nobody was told about."""
    payloads = [payload_with_a_shared_row()]
    before = [list(r) for r in payloads[0]['context']['instance_table']['rows']]
    related_before = [
        list(r) for r in payloads[0]['context']['related_tables']['e']['rows']
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UserWarning):
            _drop_context_that_answers_the_question(payloads, windowed=False)

    assert payloads[0]['context']['instance_table']['rows'] == before
    assert (
        payloads[0]['context']['related_tables']['e']['rows'] == related_before
    )


def test_the_rows_still_go_when_the_warning_is_only_a_warning():
    payloads = [payload_with_a_shared_row()]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _drop_context_that_answers_the_question(payloads, windowed=False)

    assert payloads[0]['context']['instance_table']['rows'] == [
        [1, 'e2', '2024-01-01', 8.0]
    ]
    assert payloads[0]['context']['related_tables']['e']['rows'] == [[1, 2.0]]
    assert any('left out' in str(w.message) for w in caught)


def test_a_unique_column_is_offered_as_a_key_and_as_a_foreign_key(recwarn):
    """A column unique in the rows read is either this table's key or a
    foreign key that happens not to repeat there. Naming only the first
    reading sends a caller to declare a key that invents a parent."""
    Schema.from_data(
        {
            'line_items': pd.DataFrame(
                {'ORDER_NO': ['O1', 'O2'], 'qty': [1, 2]}
            ),
            'shipments': pd.DataFrame(
                {'SHIPMENT_ID': [1, 2, 3], 'ORDER_NO': ['O1', 'O1', 'O2']}
            ),
        },
        primary_keys={'shipments': 'SHIPMENT_ID'},
    )

    said = ' '.join(str(w.message) for w in recwarn)
    assert 'nothing can link to a table without one' in said
    assert 'foreign key that happens not to repeat' in said
    assert 'that table is the one that needs the key' in said
