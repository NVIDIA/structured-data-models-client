# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A RelBench dataset is identified by its archive name.

Only some published archives carry the ``rel-`` family prefix. Stripping a
fixed number of leading characters therefore mangles every name that does not
have it, ``'stackex'`` became ``'hex'``, and the download went looking for
an archive nobody publishes. The registry's own name is the canonical one; a
prefix the caller omits is added, never assumed.

The registry is stubbed, and so is the download: resolution is what is under
test here, and ``test_relbench_registry.py`` covers what is fetched.
"""

from __future__ import annotations

from typing import Any

import pytest
from sdm_relational_engine.exceptions import UnknownDatasetError
from sdm_relational_engine.rfm import relbench

_ARCHIVES = ['rel-f1', 'rel-amazon', 'rel-stack', 'stackex', 'ctu-financial']


class _FetchedError(Exception):
    r"""Stops the loader once the archive name is known."""

    def __init__(self, archive: str) -> None:
        self.archive = archive


def _registry() -> dict[str, Any]:
    return {'datasets': {name: {'tables': {}} for name in _ARCHIVES}}


def _fetch(archive: str, verbose: bool) -> None:
    raise _FetchedError(archive)


@pytest.fixture(autouse=True)
def stub_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(relbench, 'get_registry', _registry)
    monkeypatch.setattr(relbench, '_fetch', _fetch)


def _archive_for(dataset: str) -> str:
    with pytest.raises(_FetchedError) as info:
        relbench.from_relbench(dataset, verbose=False)
    return info.value.archive


@pytest.mark.parametrize(
    ('dataset', 'expected'),
    [
        ('f1', 'rel-f1'),
        ('rel-f1', 'rel-f1'),
        ('AMAZON', 'rel-amazon'),
        ('stackex', 'stackex'),
        ('ctu-financial', 'ctu-financial'),
    ],
)
def test_a_dataset_resolves_to_its_archive_name(
    dataset: str, expected: str
) -> None:
    assert _archive_for(dataset) == expected


def test_an_unprefixed_archive_is_not_confused_with_a_prefixed_one() -> None:
    r"""``stackex`` and ``rel-stack`` are different datasets."""
    assert _archive_for('stackex') != _archive_for('stack')
    assert _archive_for('stack') == 'rel-stack'


def test_an_unknown_dataset_lists_the_valid_ones() -> None:
    with pytest.raises(UnknownDatasetError) as info:
        relbench.from_relbench('nope', verbose=False)

    message = str(info.value)
    assert 'stackex' in message
    assert 'ctu-financial' in message
    assert 'f1' in message


def test_a_near_miss_is_suggested() -> None:
    with pytest.raises(UnknownDatasetError, match="Did you mean 'stackex'"):
        relbench.from_relbench('stackexx', verbose=False)
