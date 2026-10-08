# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'tests/rust_core/switch'))

from _switch import _lazy_engine_hooks, add_paths, error_chain  # noqa: E402

add_paths()


@pytest.mark.parametrize(
    ('indices', 'beneath'),
    [
        ([1, '2'], ['ValueError', 'ArrowInvalid', 'ArrowTypeError']),
        (['1', 1], ['ValueError', 'ArrowTypeError', 'ArrowTypeError']),
        ([1.5, 2**53 + 1], ['ValueError', 'ArrowInvalid', 'ArrowTypeError']),
        ([10_000_000_000], ['ValueError', 'ArrowInvalid']),
        ([2**63, '1'], ['OverflowError', 'ArrowInvalid']),
        ([2**128, '1'], ['OverflowError', 'ArrowInvalid']),
        ([-(2**128), '1'], ['OverflowError', 'ArrowInvalid']),
    ],
)
def test_public_decimal_indices_preserve_the_complete_chain(
    monkeypatch, indices, beneath
):
    from _parity import runner
    from _parity.nim import ReplayingSession
    from _parity.tape import ReplayingTape
    from _parity.transcript import load

    path = (
        REPO
        / 'tests/parity/transcripts/duckdb/regression--decimal-mixed-indices/transcript.json'
    )
    transcript = load(path)
    transcript['calls'][-1]['args']['indices'] = indices
    raised = []
    describe = runner.describe_error

    def observed(error):
        raised.append(error)
        return describe(error)

    monkeypatch.setattr(runner, 'describe_error', observed)
    monkeypatch.setattr(
        runner, 'engine_hooks', _lazy_engine_hooks(runner.engine_hooks)
    )
    tape = ReplayingTape(transcript['steps'], path.parent)
    outcome = runner.run(
        transcript,
        tape,
        session=ReplayingSession(transcript.get('responses') or []),
    )
    assert outcome.error is not None
    assert tape.divergence is None
    assert len(raised) == 1
    chain = [
        name
        for name in error_chain(raised[0])
        if not name.startswith(('sdm_relational_engine.', 'predictive_query.'))
    ]
    expected = ['sdm_client.errors.StructuredDataError'] + [
        ('pyarrow.lib.' if name.startswith('Arrow') else 'builtins.') + name
        for name in beneath
    ]
    assert chain == expected
