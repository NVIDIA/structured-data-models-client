# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real workflow token audit accepts only its same-run Java download."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION = 'actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333'
pytestmark = pytest.mark.tooling


@pytest.fixture
def guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        't310_workflow_token_guard', ROOT / 'tests/test_ci_workflow.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(
    guard: ModuleType,
    document: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    extra: str = '',
) -> None:
    path = tmp_path / 'ci.yml'
    path.write_text(yaml.safe_dump(document, sort_keys=False) + extra)
    monkeypatch.setattr(guard, 'CI', path)
    guard.test_ci4_the_token_goes_only_to_ghcr_io(request)


def test_actual_same_run_java_download_passes_the_security_audit(
    guard: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    _audit(guard, guard._workflow(), tmp_path, monkeypatch, request)


@pytest.mark.parametrize(
    'mutation',
    [
        'pin',
        'repository',
        'run-id',
        'token',
        'name',
        'path',
        'extra-input',
        'global-write',
        'missing-actions-read',
        'job-write',
        'other-job',
        'extra-action',
        'duplicate-download',
        'shell-token',
        'env-token',
        'raw-token',
    ],
)
def test_unsafe_download_or_other_token_destination_still_fails(
    guard: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    mutation: str,
) -> None:
    document = guard._workflow()
    java = document['jobs']['java']
    download = next(step for step in java['steps'] if step.get('uses') == ACTION)
    extra = ''
    if mutation == 'pin':
        download['uses'] = 'actions/download-artifact@main'
    elif mutation in ('repository', 'run-id', 'name', 'path'):
        download['with'][mutation] = 'untrusted-target'
    elif mutation == 'token':
        download['with']['github-token'] = '${{ secrets.OTHER_TOKEN }}'
    elif mutation == 'extra-input':
        download['with']['url'] = 'https://example.com'
    elif mutation == 'global-write':
        document['permissions']['actions'] = 'write'
    elif mutation == 'missing-actions-read':
        document['permissions'].pop('actions')
    elif mutation == 'job-write':
        java['permissions'] = {'actions': 'write'}
    elif mutation == 'other-job':
        java['steps'].remove(download)
        document['jobs']['engine']['steps'].append(download)
    elif mutation == 'extra-action':
        java['steps'].append(
            {
                'uses': 'example/untrusted@main',
                'with': copy.deepcopy(download['with']),
            }
        )
    elif mutation == 'duplicate-download':
        java['steps'].append(copy.deepcopy(download))
    elif mutation == 'shell-token':
        java['steps'].append({'run': 'echo ${{ github.token }}'})
    elif mutation == 'env-token':
        java['steps'].append(
            {'run': 'echo hi', 'env': {'TOKEN': '${{ github.token }}'}}
        )
    elif mutation == 'raw-token':
        extra = 'stray:\n  github-token: ${{ github.token }}\n'
    else:
        raise AssertionError(mutation)
    with pytest.raises(AssertionError):
        _audit(guard, document, tmp_path, monkeypatch, request, extra)
