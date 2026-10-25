# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release successful owned module fixtures after their complete test outcome."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@dataclass
class _ModuleDirs:
    paths: list[Path] = field(default_factory=list)
    finished: bool = False
    failed: bool = False


_OWNED = pytest.StashKey[dict[Path, _ModuleDirs]]()


@pytest.fixture(scope='module')
def owned_module_dir(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Callable[[str], Path]]:
    """Own only the allocated roots; retain their Git objects after any failure."""
    modules = request.config.stash.setdefault(_OWNED, {})
    owned = modules.setdefault(request.node.path, _ModuleDirs())

    def allocate(prefix: str) -> Path:
        path = tmp_path_factory.mktemp(prefix)
        owned.paths.append(path)
        return path

    try:
        yield allocate
    finally:
        # The last consumer's teardown report does not exist yet. In
        # particular, a later finalizer can still report a teardown failure.
        owned.finished = True


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()
    modules = item.config.stash.get(_OWNED, {})
    owned = modules.get(item.path)
    if owned is None:
        return
    owned.failed |= report.failed
    if report.when == 'teardown' and owned.finished and not owned.failed:
        # Every dependent setup, call and teardown has now reported. Do not
        # follow symlinks or Git alternates outside these exact owned roots.
        for path in owned.paths:
            shutil.rmtree(path)
        del modules[item.path]
