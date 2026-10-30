# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native frame and label effects remain at their public prediction consumers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from test_tabular_direct_core import DIRECT, Endpoint, client

from sdm_client import KumoTabular, Request


def test_scoring_length_is_not_read_after_dtype_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScoringFrame(pd.DataFrame):
        def __getitem__(self, key: Any) -> Any:
            result = super().__getitem__(key)
            if self.__dict__.get('_arm_on_read', False) and key == 'feature':
                object.__setattr__(self, '_length_forbidden', True)
            return result

        def __len__(self) -> int:
            if self.__dict__.get('_length_forbidden', False):
                raise AssertionError('extra scoring length read after dtype resolution')
            return super().__len__()

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = pd.DataFrame({'feature': [1, 1, 2, 2], 'target': [0, 1, 0, 1]})
        scoring = ScoringFrame({'feature': [3]})
        request = Request(context, scoring, target='target', task='classification')
        assert request.schema[request.entity_table].primary_key is None
        object.__setattr__(scoring, '_arm_on_read', True)
        endpoint = Endpoint({
            'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': 1}],
        })
        result = client(endpoint).predict(request, model=KumoTabular())
        assert scoring.__dict__['_length_forbidden']
        payload = endpoint.calls[0]
        payload['metadata'].pop('request_id')
        assert payload['predict']['instance_table']['rows'] == [[3]]
        outcomes.append((payload, result.items))
    assert outcomes[0] == outcomes[1]


def test_label_native_target_comparison_selects_wire_semantic_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TargetLikeLabel(str):
        def __ne__(self, other: Any) -> bool:
            if self.__dict__.get('active', False) and other == 'target':
                self.comparisons += 1
                return False
            return str.__ne__(self, other)

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        label = TargetLikeLabel('feature')
        label.active = False
        label.comparisons = 0
        context = pd.DataFrame({'feature': [1.5, 1.5, 2.5, 2.5], 'target': [0, 1, 0, 1]})
        scoring = pd.DataFrame({'feature': [3.5]})
        context.columns = pd.Index([label, 'target'], dtype=object)
        scoring.columns = pd.Index([label], dtype=object)
        assert context.columns[0] is label
        request = Request(context, scoring, target='target', task='classification')
        assert request.schema[request.entity_table][label].stype == 'numerical'
        label.active = True
        endpoint = Endpoint({
            'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': 1}],
        })
        result = client(endpoint).predict(request, model=KumoTabular())
        payload = endpoint.calls[0]
        payload['metadata'].pop('request_id')
        assert label.comparisons > 0
        assert payload['schema']['instance_table']['columns']['feature']['stype'] == 'categorical'
        outcomes.append((payload, result.items))
    assert outcomes[0] == outcomes[1]
