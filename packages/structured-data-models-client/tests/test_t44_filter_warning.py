# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The entity-filter notice precedes every reached native child graph."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from test_t44_complete_task_references import question
from test_t44_decomposition_owners import question as decomposed_question
from test_t44_text_frames import p3e2_warning_codes, shop, text
from test_t44_text_frames import core as core

from sdm_client import PredictiveQuery, StructuredDataError


def test_filter_notice_keeps_frozen_order_before_complete_reference_validation(
    core,
):
    record = json.loads(
        Path(__file__)
        .with_name('t44_complete_task_references_reference.json')
        .read_text()
    )['records'][0]
    asked = question(shop('python'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(StructuredDataError) as raised:
            asked.materialize(
                indices=record['indices'],
                num_context_samples=1,
                num_neighbors=[3],
                num_hops=1,
                rows_per_call=1,
                random_seed=record['seed'],
            )
    assert {
        'type': type(raised.value).__name__,
        'code': raised.value.code,
        'message': str(raised.value),
    } == record['error']
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == (
        p3e2_warning_codes(record['warnings']) if core else record['warnings']
    )


@pytest.mark.parametrize(
    'kind',
    [
        UserWarning,
        ValueError,
        RuntimeError,
        StructuredDataError,
        KeyboardInterrupt,
    ],
)
def test_filter_warning_handler_errors_pass_through_before_graph_creation(
    core, kind
):
    asked = question(shop('python'))
    native = (
        kind('filter handler stopped', code='INVALID_CONFIGURATION')
        if kind is StructuredDataError
        else kind('filter handler stopped')
    )
    prior = LookupError('caller context')
    cause = KeyError('caller cause')
    native.__context__, native.__cause__ = prior, cause
    original_show = warnings.showwarning
    reached = []

    def fail(message, category, filename, lineno, file=None, line=None):
        if 'the entity filter' in str(message):
            reached.append('filter')
            raise native
        original_show(message, category, filename, lineno, file=file, line=line)

    with warnings.catch_warnings(), patch.object(warnings, 'showwarning', fail):
        warnings.simplefilter('always')
        with pytest.raises(kind) as raised:
            asked.materialize(
                indices=[7, 8],
                num_context_samples=1,
                num_neighbors=[3],
                num_hops=1,
                random_seed=7,
            )
    assert raised.value is native
    assert native.__context__ is prior
    assert native.__cause__ is cause
    assert reached == ['filter']
    assert asked._relational_materializer._model is None
    assert asked._relational_materializer._core_sampler_state == {}


def test_filter_handler_mutations_reach_native_source_indices_and_later_options(
    core,
):
    frames = shop('python')
    asked = question(frames)
    indices, neighbors = [7, 8], [3]
    config = {'num_estimators': 1}
    events = []
    original_show = warnings.showwarning

    def mutate(message, category, filename, lineno, file=None, line=None):
        value = str(message)
        if 'the entity filter' in value:
            events.append('filter')
            indices[:] = [5, 6]
            neighbors[:] = [1]
            config['num_estimators'] = 2
            frames['orders']['note'] = text(['filter-note'] * 64, 'python')
            frames['orders'].loc[1, 'order_id'] = 0
        elif value.startswith('the graph was built without'):
            events.append('graph')
        elif 'Received custom' in value:
            events.append('hops')
        original_show(message, category, filename, lineno, file=file, line=line)

    with (
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'python'
        ),
        warnings.catch_warnings(),
        patch.object(warnings, 'showwarning', mutate),
    ):
        warnings.simplefilter('always')
        request = asked.materialize(
            indices=indices,
            num_context_samples=1,
            num_neighbors=neighbors,
            num_hops=1,
            random_seed=7,
            inference_config=config,
        )
    assert events == ['filter', 'graph', 'hops']
    assert request.indices == [5, 6]
    assert request.predict.related_tables['users']['user_id'].tolist() == [5, 6]
    assert request.predict.related_tables['orders']['note'].tolist()
    assert set(request.predict.related_tables['orders']['note'].tolist()) == {
        'filter-note'
    }
    assert (
        request.payloads[0]['inference']['inference_config']['num_estimators']
        == 2
    )


@pytest.mark.parametrize('family', ['ranking', 'forecast'])
def test_filter_notices_are_emitted_by_reached_children_without_parent_duplication(
    core, family
):
    original = decomposed_question(family)
    asked = PredictiveQuery(
        original.text + ' WHERE users.user_id = 1', original.schema
    )
    events = []
    original_show = warnings.showwarning

    def observe(message, category, filename, lineno, file=None, line=None):
        value = str(message)
        if 'the entity filter' in value:
            events.append('filter')
        elif 'Received custom' in value:
            events.append('hops')
        original_show(message, category, filename, lineno, file=file, line=line)

    with (
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'python'
        ),
        warnings.catch_warnings(),
        patch.object(warnings, 'showwarning', observe),
    ):
        warnings.simplefilter('always')
        request = asked.materialize(
            indices=[1, 2],
            num_context_samples=3,
            num_neighbors=[3],
            num_hops=1,
            random_seed=7,
            **(
                {'candidates': pd.Series([0, 1])} if family == 'ranking' else {}
            ),
        )
    assert len(request.parts) == 2
    assert events == ['filter', 'hops', 'filter', 'hops']
