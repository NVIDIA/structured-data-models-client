# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Java's artifact lookup retries requests without retrying acceptance work."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION = 'actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333'
pytestmark = pytest.mark.tooling


def workflow() -> dict:
    return yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())


def test_java_artifact_lookup_uses_the_current_run_authenticated_rest_route() -> None:
    document = workflow()
    downloads = [
        step
        for step in document['jobs']['java']['steps']
        if str(step.get('uses', '')).startswith('actions/download-artifact@')
    ]
    assert len(downloads) == 1
    download = downloads[0]
    assert download['uses'] == ACTION
    assert download['with'] == {
        'name': 'java-jar',
        'path': 'jar',
        'github-token': '${{ github.token }}',
        'repository': '${{ github.repository }}',
        'run-id': '${{ github.run_id }}',
    }
    assert document['permissions']['actions'] == 'read'
    assert 'permissions' not in document['jobs']['java']
    assert 'continue-on-error' not in download
    assert 'if' not in download


def test_java_acceptance_remains_once_after_a_successful_download() -> None:
    job = workflow()['jobs']['java']
    steps = job['steps']
    download = next(
        i for i, step in enumerate(steps) if step.get('uses') == ACTION
    )
    names = (
        'The S0 cases on the Java host',
        'compare (java)',
        "The Java host's acceptance tests",
    )
    previous = download
    for name in names:
        selected = [
            (i, step)
            for i, step in enumerate(steps)
            if step.get('name') == name
        ]
        assert len(selected) == 1
        index, step = selected[0]
        assert previous < index
        assert 'continue-on-error' not in step
        assert 'if' not in step
        assert 'retry' not in step['run'].lower()
        previous = index
    assert job['timeout-minutes'] == 20
