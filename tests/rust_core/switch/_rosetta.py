# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A pytest plugin for the binding's tests in cibuildwheel's Linux test
step (``accept_wheels.py``): under Rosetta emulation, and only there, it
skips the one test Rosetta cannot run.

``test_hostile.py::test_random_numbers_refuse_hostile_input_without_panicking``
asks for a 4 PB array. On Linux the allocation fails and the binding refuses
the call; Docker Desktop's Rosetta instead ends the process ("rosetta error:
could not find free space for allocation"). Rosetta is told apart by what it
alone shows an x86_64 process: ``vendor_id`` ``VirtualApple`` in
``/proc/cpuinfo``. A real x86_64 CPU, QEMU and native aarch64 never do, so
the test runs on all of them.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import pytest

FILE = 'test_hostile.py'
TEST = 'test_random_numbers_refuse_hostile_input_without_panicking'


def under_rosetta() -> bool:
    if platform.machine() != 'x86_64':
        return False
    try:
        text = Path('/proc/cpuinfo').read_text()
    except OSError:
        return False
    return any(
        key.strip() == 'vendor_id' and value.strip() == 'VirtualApple'
        for key, _, value in (line.partition(':') for line in text.splitlines())
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    if not under_rosetta():
        return
    for item in items:
        if item.path.name == FILE and item.name == TEST:
            item.add_marker(
                pytest.mark.skip(
                    reason='P9.wheels: Rosetta ends the process on a 4 PB '
                    'allocation instead of failing it'
                )
            )
