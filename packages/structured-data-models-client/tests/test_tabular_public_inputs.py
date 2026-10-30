# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public constructor admission preserves native validation order and owners."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pandas as pd
import pytest
from test_tabular_direct_core import DIRECT, Endpoint, client

from sdm_client import KumoTabular, Request, StructuredDataError


class HeaderGuard(pd.DataFrame):
    """A valid frame whose header must not precede argument validation."""

    def __getattribute__(self, name: str) -> Any:
        if name == 'columns' and self.__dict__.get('_guard_header', False):
            raise AssertionError('header read before argument refusal')
        return super().__getattribute__(name)


def guarded(values: dict[str, Any]) -> HeaderGuard:
    frame = HeaderGuard(values)
    object.__setattr__(frame, '_guard_header', True)
    return frame


class RelatedProbe(Mapping[str, Any]):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __len__(self) -> int:
        self.events.append('related truth')
        return 1

    def __getitem__(self, key: str) -> Any:
        raise AssertionError('related mapping copied before per-item validation')

    def __iter__(self) -> Iterator[str]:
        raise AssertionError('related keys consumed instead of reached items')

    def items(self) -> Any:
        self.events.append('first related item')
        yield 'bad', object()
        raise AssertionError('next related item reached after invalid frame')


@pytest.mark.parametrize('invalid', ['context', 'predict'])
def test_frame_refusal_precedes_headers_and_related(
    invalid: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        events: list[str] = []
        context = object() if invalid == 'context' else guarded({'target': [0, 1]})
        scoring = object() if invalid == 'predict' else guarded({'feature': [2]})
        with pytest.raises(StructuredDataError) as caught:
            Request(context, scoring, target='target', task='classification',
                    related_tables=RelatedProbe(events))
        assert events == []
        assert invalid in str(caught.value)
        outcomes.append((type(caught.value), caught.value.code))
    assert outcomes[0] == outcomes[1]


def test_related_frames_are_validated_before_next_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        events: list[str] = []
        with pytest.raises(StructuredDataError) as caught:
            Request(guarded({'target': [0, 1]}), guarded({'feature': [2]}),
                    target='target', task='not-a-task', related_tables=RelatedProbe(events))
        assert events == ['related truth', 'first related item']
        assert "related_tables['bad']" in str(caught.value)
        outcomes.append((type(caught.value), caught.value.code))
    assert outcomes[0] == outcomes[1]


def test_invalid_task_precedes_first_header(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        with pytest.raises(StructuredDataError) as caught:
            Request(guarded({'target': [0, 1]}), guarded({'feature': [2]}),
                    target='target', task='not-a-task')
        assert 'task must be' in str(caught.value)
        outcomes.append((type(caught.value), caught.value.code))
    assert outcomes[0] == outcomes[1]


def test_task_membership_precedes_header_and_retains_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeTask(str):
        def __new__(cls, owner: pd.DataFrame) -> NativeTask:
            task = super().__new__(cls, 'classification')
            task.owner = owner
            task.comparisons = []
            return task

        def __eq__(self, other: Any) -> bool:
            self.comparisons.append(other)
            object.__setattr__(self.owner, '_guard_header', False)
            return str.__eq__(self, other)

        __hash__ = str.__hash__

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = guarded({'feature': [1, 2], 'target': [0, 1]})
        task = NativeTask(context)
        request = Request(context, pd.DataFrame({'feature': [3]}),
                          target='target', task=task)
        assert task.comparisons[0] == 'classification'
        assert request.task.kind is task
        assert request.context.instance_table is context
        outcomes.append(task.comparisons)
    assert outcomes[0] == outcomes[1]


def test_plain_frame_subclasses_keep_owners_through_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeFrame(pd.DataFrame):
        pass

    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = NativeFrame({'feature': [1, 2], 'target': [0, 1]})
        scoring = NativeFrame({'feature': [3]})
        request = Request(context, scoring, target='target', task='classification')
        assert request.context.instance_table is context
        assert request.predict.instance_table is scoring
        assert request.schema[request.entity_table].data is context
        endpoint = Endpoint({'model': 'kumo-tabular', 'id': 'answer', 'metadata': {},
                             'predictions': [{'row_index': 0, 'prediction': 1}]})
        result = client(endpoint).predict(request, model=KumoTabular())
        payload = endpoint.calls[0]
        payload['metadata'].pop('request_id')
        outcomes.append((payload, result.items))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize(('case', 'message'), [
    ('task', 'task must be'),
    ('empty', 'context has no rows'),
    ('target', 'context has no target column'),
    ('duplicate', 'duplicate column name'),
    ('nonstring', 'column names must be strings'),
])
def test_label_refusal_is_reached_after_earlier_constructor_checks(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = []
    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        context = pd.DataFrame([[1, 2, 0], [3, 4, 1]], columns=[7, 7, 'target'])
        scoring = pd.DataFrame([[5]], columns=[7])
        task = 'not-a-task' if case == 'task' else 'classification'
        target = 'absent' if case == 'target' else 'target'
        if case == 'empty':
            context = context.iloc[:0]
        if case == 'nonstring':
            context = context.iloc[:, [0, 2]]
        with pytest.raises(StructuredDataError) as caught:
            Request(context, scoring, target=target, task=task)
        assert message in str(caught.value)
        outcomes.append((type(caught.value), caught.value.code))
    assert outcomes[0] == outcomes[1]


def test_string_subclass_column_label_retains_schema_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeLabel(str):
        pass

    for route in ('', DIRECT):
        monkeypatch.setenv('SDM_CORE_ROUTES', route)
        label = NativeLabel('feature')
        context = pd.DataFrame({label: [1, 2], 'target': [0, 1]})
        scoring = pd.DataFrame({label: [3]})
        # Pandas 3 normalizes inferred string indexes to built-in strings.
        # An explicit object Index retains the public label owner.
        context.columns = pd.Index([label, 'target'], dtype=object)
        scoring.columns = pd.Index([label], dtype=object)
        assert context.columns[0] is label
        assert scoring.columns[0] is label
        request = Request(context, scoring, target='target', task='classification')
        column = request.schema[request.entity_table][label]
        assert column.name is label
        assert request.context.instance_table is context
        assert request.predict.instance_table is scoring
