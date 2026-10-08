# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symbolic scalar inputs through public request and answer boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CASES = (
    'float',
    'float64',
    'longdouble',
    'complex',
    'complex128',
    'clongdouble',
)
ENTRIES = ('metadata', 'populated-metadata', 'inference', 'task_options')


def capture(case: str, entry: str) -> dict[str, Any]:
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client import PredictionResult, Request

    value = {
        'float': lambda: 1.25,
        'float64': lambda: np.float64(1.25),
        'longdouble': lambda: np.longdouble(1.25),
        'complex': lambda: 1 + 2j,
        'complex128': lambda: np.complex128(1 + 2j),
        'clongdouble': lambda: np.clongdouble(1 + 2j),
    }[case]()
    shared = {'value': value}
    options = {'extra': [shared, shared]}
    if entry in ('metadata', 'populated-metadata'):
        items = (
            [PredictionItem(row_index=0, prediction=0, metadata=options)]
            if entry == 'populated-metadata'
            else []
        )
        result = PredictionResult(
            model='kumo-tabular',
            request_id='scalar-probe',
            items=items,
            metadata=options,
            explanation=options,
            unscored_candidates=[value],
        )
        absent = result.absent(
            ['prediction', 'scores', 'explanation', 'unknown', 'scores']
        )
        assert absent == (('scores',) if items else ())
        assert result.explanation is options
        assert result.unscored_candidates[0] is value
        if items:
            assert items[0].metadata['extra'][0] is shared
        held = result.metadata
        outcome: dict[str, Any] = {'absent': list(absent)}
    else:
        request = Request(
            pd.DataFrame({'feature': [1, 2], 'target': [3.0, 4.0]}),
            pd.DataFrame({'feature': [3]}),
            target='target',
            task='regression',
            **{entry: options},
        )
        held = (
            request.inference if entry == 'inference' else request.task.options
        )
        outcome = {}
    outcome.update(
        value_preserved=bool(held['extra'][0]['value'] == value),
        scalar_identity=held['extra'][0]['value'] is value,
        shared_identity=held['extra'][0] is held['extra'][1] is shared,
    )
    assert all(
        outcome[name]
        for name in ('value_preserved', 'scalar_identity', 'shared_identity')
    )
    return outcome


def capture_scalar_events() -> dict[str, Any]:
    return {
        f'{entry}/{case}': capture(case, entry)
        for entry in ENTRIES
        for case in CASES
    }
