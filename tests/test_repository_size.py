# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for repository size enforcement and schema pruning."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'scripts/rust_core'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lock = load('contract_lock')
budget = load('repository_size')


def schemas(root):
    directory = root / lock.SCHEMA_DIR
    directory.mkdir(parents=True)
    (directory / 'contract.lock').write_text(
        '70 sha256:abc sdm-contract-70.json\n'
    )
    (directory / 'validated-query.lock').write_text(
        '2 sha256:def pql-validated-query-2.json\n'
    )
    for name in (
        'sdm-contract-69.json',
        'sdm-contract-70.json',
        'pql-validated-query-2.json',
        'state-payload.json',
    ):
        (directory / name).write_text('{}')
    return directory


def test_orphan_pruning_preserves_locked_and_embedded_schemas(tmp_path):
    directory = schemas(tmp_path)
    assert lock.unlocked_schemas(tmp_path) == [
        directory / 'sdm-contract-69.json'
    ]


@pytest.mark.parametrize(
    'name', ['contract/new.json', 'tests/compatibility/fixtures/new.json']
)
def test_budget_rejects_oversize_unknown_fixtures_missing_consumers_and_orphans(
    tmp_path, monkeypatch, name
):
    schemas(tmp_path)
    registry = tmp_path / budget.REGISTRY
    registry.parent.mkdir(parents=True)
    registry.write_text('{}')
    fixture = Path(name)
    (tmp_path / fixture).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / fixture).write_text(' ' * 251000)
    monkeypatch.setattr(
        budget.subprocess, 'check_output', lambda *a, **k: '1000000\t0\tfile\n'
    )
    monkeypatch.setattr(budget, 'MAX_BYTES', 250000)
    failures, _ = budget.check(tmp_path, [fixture], 'main')
    assert any('tracked bytes' in failure for failure in failures)
    assert any('added lines' in failure for failure in failures)
    assert any('named consumers' in failure for failure in failures)
    assert any('unlocked historical' in failure for failure in failures)
    registry.write_text(
        json.dumps(
            {
                str(fixture): {
                    'max_bytes': 250000,
                    'reason': 'large case',
                    'consumers': ['missing.py'],
                }
            }
        )
    )
    failures, _ = budget.check(tmp_path, [fixture], 'main')
    assert any('reviewed size' in failure for failure in failures)
    assert any('missing consumer' in failure for failure in failures)


def test_prune_refuses_invalid_lock_before_removing_any_schema(
    tmp_path, monkeypatch
):
    directory = schemas(tmp_path)
    monkeypatch.setattr(lock, 'REPO_ROOT', tmp_path)
    assert lock.main(['--no-base', '--prune-unlocked']) == 1
    assert (directory / 'sdm-contract-69.json').exists()


def test_parity_fixture_is_not_exempt_from_consumer_budget():
    assert budget.fixture(Path('tests/compatibility/fixtures/unreviewed.json'))
    assert not budget.fixture(Path('tests/compatibility/manifest.json'))
