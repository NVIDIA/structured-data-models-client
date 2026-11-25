# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importing the client never loads the Rust core.

The wheel carries `sdm_client._native`, but only a core route reaches it,
lazily, through `sdm_client._rust`; every route is the old one by default,
so an import of the client, and of everything it exports, leaves the native
module unloaded (design, "Packaging").
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_client_does_not_load_the_native_module() -> None:
    script = (
        'import sys, sdm_client\n'
        'for name in sdm_client.__all__:\n'
        '    getattr(sdm_client, name)\n'
        'print("sdm_client._native" in sys.modules)\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'False'
