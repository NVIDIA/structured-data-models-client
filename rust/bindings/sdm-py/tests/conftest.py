# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The binding's tests, run against an installed client wheel.

They say which build they test, since some only a ``test-ops`` build can
pass: ``--native-build production`` for the wheel that ships, where only
``test_binding.py`` runs, and ``--native-build test-ops`` for the build
with the test-only operations, where every file runs.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--native-build',
        choices=('production', 'test-ops'),
        required=True,
        help='which build of sdm_client._native is installed',
    )


@pytest.fixture(scope='session')
def native_build(request: pytest.FixtureRequest) -> str:
    build: str = request.config.getoption('--native-build')
    return build
