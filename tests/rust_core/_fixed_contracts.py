# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact old refusals and governed native results for task 7b fixes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DECIMAL_MESSAGE = (
    "Encountered unsupported data type 'decimal128(10, 0)[pyarrow]' "
    "for column 'ENTITY' in task table. Please either manually override "
    "the columns's data type or remove the column from this table."
)
MULTILABEL_MESSAGE = (
    "Multilabel-classification queries based on 'LIST_DISTINCT' are not "
    'supported yet. If you planned to write a link prediction query '
    "instead, make sure to register 'product_id' as a foreign key."
)


def missing_index_warning(indices: int, tables: int) -> str:
    return (
        f'Missing {indices} indices in {tables} tables for optimal database '
        'querying. For improving runtime, we strongly suggest to create '
        'indices for primary and foreign keys, e.g., automatically by '
        'instantiating Kumo Relational via '
        '`KumoRelational(graph, optimize=True)`.'
    )


def fixed_replay(path: Path, reason: str | None) -> None:
    from _parity.corpus import check
    from _parity.nim import ReplayingSession
    from _parity.runner import run
    from _parity.tape import ReplayingTape
    from sdm_client import _impl, _request_core

    transcript = json.loads(path.read_text())
    name = transcript['name']
    listed = json.loads(
        (Path(__file__).parent / 'shadow/fix_changes.json').read_text()
    )
    if name == 'duckdb/regression#decimal-entity-keys':
        assert listed[name]['fix'] == 'decimal-entity-keys'
        message = 'RuntimeError: ' + DECIMAL_MESSAGE
        expected_reason = (
            'the run raised StructuredDataError: [INTERNAL_ERROR] '
            'the relational engine failed while building this request: '
            + message
        )
        assert reason is not None and re.fullmatch(
            re.escape(expected_reason)
            + r' at .*/sdm_client/materializers/relational\.py:\d+',
            reason,
        ), reason
        assert len(transcript['calls']) == 7
        assert transcript['calls'][4]['call'] == 'PredictiveQuery.materialize'
        transcript['calls'] = transcript['calls'][:5]
        transcript['steps'] = transcript['steps'][:25]
        expected_warnings = [
            {
                'category': 'builtins.UserWarning',
                'message': missing_index_warning(5, 3),
            }
        ] * 2
    elif name == 'memory/link-prediction#multilabel':
        assert listed[name]['fix'] == 'multilabel-link-prediction'
        message = 'NotImplementedError: ' + MULTILABEL_MESSAGE
        assert reason == (
            'the error differs at error.class: '
            '"sdm_client.errors.UnsupportedRequestError" became '
            '"sdm_client.errors.StructuredDataError"'
        ), reason
        expected_warnings = []
    else:
        raise AssertionError(f'No exact old replay contract for {name}')
    tape = ReplayingTape(transcript['steps'], path.parent)
    session = ReplayingSession([])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_impl, 'OLD', True)
        outcome = run(transcript, tape, session=session)
    assert outcome.error == {
        'class': 'sdm_client.errors.StructuredDataError',
        'code': 'INTERNAL_ERROR',
        'message': 'the relational engine failed while building this request: '
        + message,
    }
    assert outcome.warnings == expected_warnings
    assert outcome.payloads == [] and outcome.responses == []
    assert outcome.result is None and outcome.answer == {}
    assert tape.divergence is None and tape.leftover() is None
    assert session.divergence is None and session.leftover() is None
    started = []
    real = _request_core._native.materialize_run

    def observed(*args):
        started.append(args)
        return real(*args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_impl, 'OLD', False)
        patch.setattr(_request_core._native, 'materialize_run', observed)
        assert check(path) is None
    assert len(started) == 2
