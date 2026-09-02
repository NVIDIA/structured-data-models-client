# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vendored tree must stay byte-identical to upstream.

A repo-wide rename matches any text, including text inside a vendored file.
These tests are what turns that silent edit into a failure. See SOURCE.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TREE = PACKAGE_ROOT / 'src' / 'predictive_query'
MANIFEST = PACKAGE_ROOT / 'VENDOR_MANIFEST.json'

# Identifiers this repository introduces. None of them may appear inside the
# vendored tree, because upstream has never heard of them. This is deliberately
# not a ban on 'kumo': upstream legitimately carries NVIDIA copyright and
# historical references, so the assertion is "unchanged", not "sanitized".
FOREIGN_IDENTIFIERS = (
    'sdm_api',
    'sdm_client',
    'sdm_pql',
    'sdm_relational_engine',
    'sdm-client',
    'SDMClient',
)

_SKIP_DIRS = {'__pycache__', '.pytest_cache', '.ruff_cache'}


def vendored_files() -> list[Path]:
    return sorted(
        path
        for path in TREE.rglob('*')
        if path.is_file()
        and path.suffix not in {'.pyc', '.pyo'}
        and not any(part in _SKIP_DIRS for part in path.parts)
    )


def test_the_manifest_exists_and_is_not_empty():
    assert MANIFEST.exists(), 'record it with scripts/vendor_manifest.py record'
    recorded = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert recorded['files'], 'manifest records no files'
    assert recorded['upstream']['revision'], (
        'manifest names no upstream revision'
    )


def test_every_vendored_file_matches_its_recorded_digest():
    recorded = json.loads(MANIFEST.read_text(encoding='utf-8'))['files']
    current = {
        str(path.relative_to(TREE)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in vendored_files()
    }

    assert set(current) == set(recorded), (
        'the vendored file set changed; re-record only if the bump is '
        'deliberate'
    )
    mismatched = [
        name for name in sorted(current) if current[name] != recorded[name]
    ]
    assert not mismatched, (
        f'vendored files were modified in place: {mismatched}. A repo-wide '
        f'rename is the usual cause; exclude this path and revert.'
    )


@pytest.mark.parametrize('identifier', FOREIGN_IDENTIFIERS)
def test_no_identifier_from_this_repository_leaked_into_the_tree(identifier):
    guilty = [
        str(path.relative_to(TREE))
        for path in vendored_files()
        if identifier.encode() in path.read_bytes()
    ]
    assert not guilty, (
        f'{identifier!r} appears in vendored files {guilty}; upstream has '
        f'never heard of it, so a rename reached a path it should have skipped'
    )


def test_the_package_imports_and_exposes_its_public_surface():
    import predictive_query

    for name in (
        'PQLParser',
        'PredictiveQueryValidator',
        'ValidatedPredictiveQuery',
        'GraphDefinition',
        'TableDefinition',
        'Dtype',
        'Stype',
        'Capabilities',
        'Profile',
        'RELATIONAL',
        'TABULAR',
    ):
        assert hasattr(predictive_query, name), f'missing {name}'


def test_the_profiles_cover_the_models_we_serve_and_the_ones_we_plan_to():
    """Assert each profile's whole capability matrix, not a sample of it.

    A profile may only reject, so these flags are the entire statement of what
    a model will refuse before anything is sent. Checking two of them would let
    a silent widening through.
    """
    from predictive_query import ANOMALY, RELATIONAL, TABULAR, TIMESERIES

    def matrix(profile):
        capabilities = profile.capabilities
        return {
            name: getattr(capabilities, name)
            for name in (
                'relationships',
                'multiple_tables',
                'link_prediction',
                'forecasting',
                'static',
                'temporal',
            )
        }

    assert matrix(RELATIONAL) == {
        'relationships': True,
        'multiple_tables': True,
        'link_prediction': True,
        'forecasting': True,
        'static': True,
        'temporal': True,
    }
    # One table, no relationships. This is the whole of the tabular narrowing,
    # and it is what makes a relational aggregation fail before it is sent.
    assert matrix(TABULAR) == {
        'relationships': False,
        'multiple_tables': False,
        'link_prediction': False,
        'forecasting': False,
        'static': True,
        'temporal': True,
    }
    # Named here so a future model's arrival is a change to this test rather
    # than a surprise; see the architecture plan, "Adding a model".
    assert matrix(TIMESERIES) == {
        'relationships': False,
        'multiple_tables': False,
        'link_prediction': False,
        'forecasting': True,
        'static': False,
        'temporal': True,
    }
    assert matrix(ANOMALY) == {
        'relationships': False,
        'multiple_tables': False,
        'link_prediction': False,
        'forecasting': False,
        'static': True,
        'temporal': False,
    }
