# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached constructor observations, with refusal decisions owned by Rust."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from sdm_client._rust import errors


def observe(call: Callable[..., Any], kind: str, **facts: Any) -> None:
    call(
        'check_direct_preparation',
        {
            'kind': 'admission',
            'observation': {'kind': kind, **facts},
        },
    )


def frame(value: Any, parameter: str, call: Callable[..., Any]) -> Any:
    valid = isinstance(value, pd.DataFrame)
    with errors.holding(direct_frame=lambda fields: 'pandas DataFrame'):
        observe(
            call,
            'frame',
            valid=valid,
            row_hint=False if valid else isinstance(value, (dict, pd.Series)),
            parameter=parameter,
            native_type='' if valid else type(value).__name__,
        )
    return value


def arguments(
    context: Any,
    scoring: Any,
    related_tables: Mapping[str, Any] | None,
    task: Any,
    call: Callable[..., Any],
) -> dict[str, Any]:
    from sdm_client.request import TASK_KINDS

    frame(context, 'context', call)
    frame(scoring, 'predict', call)
    related = {
        name: frame(value, f'related_tables[{name!r}]', call)
        for name, value in (related_tables or {}).items()
    }
    with errors.holding(direct_tables=lambda fields: sorted(related)):
        observe(call, 'related', present=bool(related))
    with errors.holding(direct_task=lambda fields: task):
        observe(call, 'task', accepted=task in TASK_KINDS)
    return related


def population(
    context: Any, scoring: Any, target: Any, call: Callable[..., Any]
) -> None:
    from sdm_client.request import _has_container_cells

    with errors.holding(direct_target=lambda fields: target):
        observe(
            call,
            'target_cells',
            containers=bool(
                target in context.columns
                and _has_container_cells(context[target])
            ),
        )
        observe(call, 'context_rows', empty=len(context) == 0)
        with errors.holding(
            direct_columns=lambda fields: sorted(map(str, context.columns))
        ):
            observe(call, 'context_target', present=target in context.columns)
        observe(call, 'predict_target', present=target in scoring.columns)
        observe(
            call,
            'known_target',
            present=not bool(context[target].notna().sum() == 0),
        )
    missing = [
        column for column in scoring.columns if column not in context.columns
    ]
    with errors.holding(direct_columns=lambda fields: missing):
        observe(call, 'extra_columns', present=bool(missing))


def names(context: Any, entity: Any, call: Callable[..., Any]) -> None:
    def duplicates(fields: Any) -> Any:
        return sorted(
            {str(c) for c in context.columns[context.columns.duplicated()]}
        )

    with errors.holding(
        direct_table=lambda fields: entity, direct_columns=duplicates
    ):
        observe(
            call,
            'duplicate_columns',
            present=bool(context.columns.has_duplicates),
        )
    non_strings = [
        name for name in context.columns if not isinstance(name, str)
    ]
    with errors.holding(
        direct_table=lambda fields: entity,
        direct_columns=lambda fields: non_strings,
    ):
        observe(call, 'column_types', non_strings=bool(non_strings))


def missing_features(
    context: Any, scoring: Any, target: Any, key: Any, call: Callable[..., Any]
) -> None:
    from sdm_client.request import _as_key_tuple

    identity = set(_as_key_tuple(key))
    absent = [
        column
        for column in context.columns
        if column != target
        and column not in identity
        and column not in scoring.columns
    ]
    with errors.holding(direct_columns=lambda fields: absent):
        observe(call, 'missing_features', present=bool(absent))


def constancy(
    context: Any, target: Any, task: Any, call: Callable[..., Any]
) -> None:
    from sdm_client.request import _CLASSIFICATION_KINDS

    answered = context[target].dropna()
    reached = call(
        'check_direct_preparation',
        {
            'kind': 'constancy_population',
            'empty': len(answered) == 0,
            'diverse': None,
        },
    )
    if not reached:
        return
    reached = call(
        'check_direct_preparation',
        {
            'kind': 'constancy_population',
            'empty': None,
            'diverse': bool(answered.nunique() >= 2),
        },
    )
    if not reached:
        return
    classification = task in _CLASSIFICATION_KINDS
    regression = None if classification else not bool(task != 'regression')
    reached = call(
        'check_direct_preparation',
        {
            'kind': 'constancy_kind',
            'classification': classification,
            'regression': regression,
        },
    )
    if not reached:
        return
    with errors.holding(direct_target=lambda fields: target):
        call(
            'check_direct_preparation',
            {
                'kind': 'constancy_notice',
                'classification': task in _CLASSIFICATION_KINDS,
            },
        )


def table_entry_name(entity: Any, call: Callable[..., Any]) -> None:
    with errors.holding(direct_name=lambda fields: entity):
        observe(
            call,
            'table_entry_name',
            valid=isinstance(entity, str) and bool(entity),
        )


def name(value: Any, call: Callable[..., Any]) -> None:
    from sdm_client.schema import _INVALID_NAME_CHARS

    with errors.holding(direct_name=lambda fields: value):
        observe(call, 'name_kind', valid=isinstance(value, str) and bool(value))
        present = [char in value for char in _INVALID_NAME_CHARS]
        characters = [
            char
            for char, found in zip(_INVALID_NAME_CHARS, present, strict=True)
            if found
        ]
        with errors.holding(direct_characters=lambda fields: characters):
            observe(call, 'name_characters', present=present)
