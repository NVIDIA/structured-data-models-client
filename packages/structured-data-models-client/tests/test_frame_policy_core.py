# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public memory workflows do not depend on the legacy validator/edit rules."""

import pandas as pd
import predictive_query
import pytest

from sdm_client import KumoRelational, KumoTabular, PredictiveQuery, Schema


def frame_schema():
    return Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(20), 'value': [i + 0.25 for i in range(20)]}
            )
        },
        primary_keys={'users': 'id'},
    )


def legacy_policy(*args, **kwargs):
    raise AssertionError('legacy computational policy executed')


@pytest.mark.parametrize('model', [KumoRelational(), KumoTabular()])
def test_memory_pql_uses_rust_resolved_query(monkeypatch, model):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    schema = frame_schema()
    monkeypatch.setattr(predictive_query, 'validate', legacy_policy)
    request = PredictiveQuery(
        'PREDICT users.value FOR EACH users.id', schema
    ).materialize(indices=[1, 2], model=model)
    assert request.query


def test_frame_semantic_edit_uses_rust_policy(monkeypatch):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    schema = frame_schema()
    column = schema['users']['value']
    monkeypatch.setattr(type(column), '_check_supports_dtype', legacy_policy)
    column.stype = 'categorical'
    assert column.stype == 'categorical'


def test_tabular_quoted_identifiers_use_resolved_columns(monkeypatch):
    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {
                    'id': range(20),
                    'value.with.dot': [i + 0.25 for i in range(20)],
                }
            )
        },
        primary_keys={'users': 'id'},
    )
    monkeypatch.setattr(predictive_query, 'validate', legacy_policy)
    request = PredictiveQuery(
        'PREDICT users.`value.with.dot` FOR EACH users.id', schema
    ).materialize(indices=[1, 2], model=KumoTabular())
    assert request.task.target.column_name == 'value.with.dot'
