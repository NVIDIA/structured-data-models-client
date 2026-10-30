# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the client's test modules.

The models are small random-weight ones, standing in for the published ones so
that nothing is downloaded and no GPU is needed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import pytest

from sdm_client import Request


@pytest.fixture(scope='module')
def sdm():
    return pytest.importorskip('sdm', reason='needs the in-process extra')


@pytest.fixture(scope='module')
def tabular(sdm):
    import torch

    torch.manual_seed(0)
    return sdm.models.KumoTabular(size='small', pretrained=False, device='cpu')


@pytest.fixture(scope='module')
def relational(sdm):
    pytest.importorskip('sdm_relational_engine', reason='needs [relational]')
    import torch

    torch.manual_seed(0)
    return sdm.models.KumoRelational(pretrained=False, device='cpu')


@pytest.fixture
def tabular_request() -> Callable[..., Request]:
    """Build a single-table request: 20 labelled context rows, ``rows`` to score.

    A classification target alternates between ``classes`` (0/1 unless
    given); a regression target is a float. ``keyed=False`` drops the
    ``row_id`` column, so the rows have no key.
    """

    def build(
        rows: int = 1,
        task: str = 'classification',
        keyed: bool = True,
        classes: Sequence[Any] = (0, 1),
    ) -> Request:
        labels = [
            classes[i % 2] if task == 'classification' else float(i)
            for i in range(20)
        ]
        context = pd.DataFrame(
            {
                'row_id': range(20),
                'age': [20.0 + i for i in range(20)],
                'churn': labels,
            }
        )
        predict = pd.DataFrame(
            {
                'row_id': range(90, 90 + rows),
                'age': [33.0 + i for i in range(rows)],
            }
        )
        if not keyed:
            context = context.drop(columns='row_id')
            predict = predict.drop(columns='row_id')
        return Request(
            context=context, predict=predict, target='churn', task=task
        )

    return build


@pytest.fixture
def gradient_explanation() -> Callable[..., dict[str, Any]]:
    """Build one prediction record's ``explanation``, as a NIM sends it.

    The ``sdm_gradient_v1`` shape of the NIM user guides: ``row`` is the
    record's position in the predict table, ``gradients`` its own
    ``[output][column]`` (sent as ``[output][row][column]`` with one row), and
    ``related`` maps a table to ``(row_indices, columns, gradients)`` with
    gradients ``[output][row][column]``.
    """

    def build(
        *,
        outputs: Sequence[Mapping[str, Any]],
        row: int,
        columns: Sequence[str],
        gradients: Sequence[Sequence[float]],
        related: Mapping[str, tuple[list[int], list[str], list]] | None = None,
    ) -> dict[str, Any]:
        return {
            'format': 'sdm_gradient_v1',
            'details': {
                'input_space': 'preprocessed_numerical',
                'outputs': [dict(output) for output in outputs],
                'instance_table': {
                    'row_indices': [row],
                    'columns': list(columns),
                    'gradients': [[list(by_column)] for by_column in gradients],
                },
                'related_tables': {
                    name: {
                        'row_indices': list(indices),
                        'columns': list(names),
                        'gradients': values,
                    }
                    for name, (indices, names, values) in (
                        related or {}
                    ).items()
                },
            },
        }

    return build


@pytest.fixture
def fields_sent() -> Callable[[list[dict]], list[list[str]]]:
    """Read the output fields each recorded call asked for, in order."""

    def read(sent: list[dict]) -> list[list[str]]:
        return [payload['output']['fields'] for payload in sent]

    return read


@pytest.fixture
def hand_built_result() -> Callable[..., Any]:
    """Build a ``PredictionResult`` by hand, its explanations a plain dict.

    ``rows`` is ``(row_index, id)`` per item, or ``(row_index, id, step)``
    for a forecast; ``explanations`` maps each key to its entry as given.
    Nothing records the rows' positions, as ``predict()`` would.
    """
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client import PredictionResult

    def build(
        rows: Sequence[tuple[Any, ...]],
        explanations: Mapping[Any, dict[str, Any]],
    ) -> PredictionResult:
        items = [
            PredictionItem.from_dict(
                {
                    'row_index': row[0],
                    'id': row[1],
                    'prediction': 1.0,
                    **({'forecast_step': row[2]} if len(row) > 2 else {}),
                }
            )
            for row in rows
        ]
        return PredictionResult(
            model='kumo-tabular',
            request_id='r',
            items=items,
            explanations=dict(explanations),
            outputs=('prediction', 'explanation'),
        )

    return build
