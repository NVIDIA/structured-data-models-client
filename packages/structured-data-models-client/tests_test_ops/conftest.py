# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python host's tests that need the core's test-only operations.

Only a ``test-ops`` build of ``sdm_client._native`` has them, and the wheel
that ships has not, so these run in ``native-test-ops``'s environment
and the ``test`` job does not collect them (decision E23). On any other
build they fail, never skip.
"""

from __future__ import annotations

from sdm_client import _native


def pytest_configure() -> None:
    if not _native.TEST_OPS:
        raise RuntimeError(
            'these tests need a test-ops build of sdm_client._native'
        )
