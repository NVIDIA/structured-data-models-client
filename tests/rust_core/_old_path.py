# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Today's Python implementation, for a test that compares the core with it.

``sdm_client`` runs the core. The tests here that hold the core to today's
Python through public calls need those calls to run the old implementation,
which ``SDM_CLIENT_IMPLEMENTATION=python`` chooses for a whole process.
``python_path`` chooses it for one test, in this process: every object the
test makes is made on the old path, and none of them outlives it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture
def python_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    r"""Public calls run today's Python for the length of the test."""
    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', True)
    yield
