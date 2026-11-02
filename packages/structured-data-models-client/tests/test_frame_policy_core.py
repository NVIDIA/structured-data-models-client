# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public memory workflows do not depend on the legacy validator/edit rules."""

import pandas as pd
import pytest

from sdm_client import KumoRelational, KumoTabular, PredictiveQuery, Schema
from sdm_client._rust import edits, s0


def frame_schema():
    return Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(20), 'value': [i + 0.25 for i in range(20)]}
            )
        },
        primary_keys={'users': 'id'},
    )


@pytest.fixture
def resolved_queries(monkeypatch):
    calls = []
    validate = s0.validate

    def recorded(*args, **kwargs):
        calls.append(args[0])
        return validate(*args, **kwargs)

    monkeypatch.setattr(s0, 'validate', recorded)
    return calls


@pytest.mark.parametrize('model', [KumoRelational(), KumoTabular()])
def test_memory_pql_uses_rust_resolved_query(resolved_queries, model):
    schema = frame_schema()
    request = PredictiveQuery(
        'PREDICT users.value FOR EACH users.id', schema
    ).materialize(indices=[1, 2], model=model)
    assert request.query
    assert resolved_queries == [request.query]


def test_frame_semantic_edit_uses_rust_policy(monkeypatch):
    schema = frame_schema()
    column = schema['users']['value']
    calls = []
    edit = edits.column

    def recorded(owner, value):
        calls.append((owner, value))
        return edit(owner, value)

    monkeypatch.setattr(edits, 'column', recorded)
    column.stype = 'categorical'
    assert column.stype == 'categorical'
    assert (column, 'categorical') in calls


def test_tabular_quoted_identifiers_use_resolved_columns(resolved_queries):
    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {
                    'id': range(20),
                    'value with spaces': [i + 0.25 for i in range(20)],
                }
            )
        },
        primary_keys={'users': 'id'},
    )
    request = PredictiveQuery(
        'PREDICT users.`value with spaces` FOR EACH users.id', schema
    ).materialize(indices=[1, 2], model=KumoTabular())
    assert request.task.target.column_name == 'value with spaces'
    assert resolved_queries
