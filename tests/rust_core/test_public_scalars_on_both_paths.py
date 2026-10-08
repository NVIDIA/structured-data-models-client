# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public scalar calls finish and retain values and shared references."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CASES = (
    'float',
    'float64',
    'longdouble',
    'complex',
    'complex128',
    'clongdouble',
)


@pytest.mark.parametrize('selector', ['python', 'core'])
@pytest.mark.parametrize(
    'entry', ['metadata', 'populated-metadata', 'inference', 'task_options']
)
@pytest.mark.parametrize('case', CASES)
def test_public_scalar_calls_terminate(selector, entry, case):
    script = """
import json
import sys
sys.path.insert(0, 'scripts/parity')
from _scalar_events import capture
print(json.dumps(capture(*sys.argv[1:])))
"""
    completed = subprocess.run(
        [sys.executable, '-c', script, case, entry],
        cwd=REPO,
        env={**os.environ, 'SDM_CLIENT_IMPLEMENTATION': selector},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    outcome = json.loads(completed.stdout)
    assert outcome['value_preserved']
    assert outcome['scalar_identity']
    assert outcome['shared_identity']
    if 'metadata' in entry:
        assert outcome['absent'] == (
            ['scores'] if entry == 'populated-metadata' else []
        )


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('populated', [True, False])
def test_absence_ignores_opaque_and_cyclic_metadata(
    monkeypatch, old, populated
):
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client import PredictionResult, _impl

    class Opaque:
        def __repr__(self):
            raise RuntimeError('metadata must not be serialized')

    monkeypatch.setattr(_impl, 'OLD', old)
    shared = {'value': Opaque()}
    shared['self'] = shared
    metadata = {'extra': [shared, shared]}
    items = (
        [PredictionItem(row_index=0, metadata=metadata)] if populated else []
    )
    result = PredictionResult(
        model='kumo-tabular',
        request_id='opaque-probe',
        items=items,
        metadata=metadata,
        explanation=metadata,
        unscored_candidates=[shared],
    )
    assert result.absent(['prediction', 'explanation', 'prediction']) == (
        ('prediction',) if populated else ()
    )
    assert result.metadata['extra'][0] is result.metadata['extra'][1] is shared
    assert shared['self'] is shared
    assert result.explanation is metadata
    assert result.unscored_candidates[0] is shared
    if items:
        assert items[0].metadata['extra'][0] is shared


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('explanation', [None, {}, [], (), False, 0, ''])
@pytest.mark.parametrize('prediction', [None, False, 0, ''])
def test_absence_uses_presence_not_truthiness(
    monkeypatch, old, explanation, prediction
):
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client import PredictionResult, _impl

    monkeypatch.setattr(_impl, 'OLD', old)
    result = PredictionResult(
        model='kumo-tabular',
        request_id='presence-probe',
        items=[PredictionItem(row_index=0, prediction=prediction)],
        explanation=explanation,
    )
    expected = []
    if explanation is None:
        expected.append('explanation')
    if prediction is None:
        expected.append('prediction')
    assert result.absent(
        ['explanation', 'prediction', 'unknown', 'explanation']
    ) == tuple(expected)
    assert result.explanation is explanation


@pytest.mark.parametrize('selector', ['python', 'core'])
def test_scalar_recordings_match_both_selectors(selector):
    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            "import json, sys; sys.path.insert(0, 'scripts/parity'); from _scalar_events import capture_scalar_events; print(json.dumps(capture_scalar_events()))",
        ],
        cwd=REPO,
        env={**os.environ, 'SDM_CLIENT_IMPLEMENTATION': selector},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = json.loads(
        (REPO / 'tests/rust_core/shadow/host_events.json').read_text()
    )['scalars']
    assert json.loads(completed.stdout) == expected
