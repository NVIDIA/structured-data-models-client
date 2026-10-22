#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the converter's version-1 fixture, ``contract/tests/t33-capture-v1``
(T3.3's exit 4), once, with the capture writer at ``sdm-capture/1``:

    python contract/tests/t33_capture_v1_record.py

It writes :func:`content` (``t33_fixtures.capture()``) with no private
variable set, and refuses when the fixture exists or the writer's format is
not ``sdm-capture/1``: the fixture is never recorded again, so a later
format reads it through the converter (choice 2: "a later format change
converts captures offline with T3.3's converter, never re-records").
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / 't33-capture-v1'
VERSION = 'sdm-capture/1'


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def content() -> dict[str, Any]:
    r"""What the fixture holds."""
    return _load('t33_record_fixtures', HERE / 't33_fixtures.py').capture()


def main() -> int:
    captures = _load(
        't33_record_captures', ROOT / 'contract/runners/captures.py'
    )
    if captures.FORMAT != VERSION:
        print(
            f'the writer writes {captures.FORMAT}, not {VERSION}',
            file=sys.stderr,
        )
        return 1
    if FIXTURE.exists():
        print(
            f'{FIXTURE.name} exists: it is never recorded again',
            file=sys.stderr,
        )
        return 1
    for name in captures.PRIVATE:
        os.environ.pop(name, None)
    captures.write(FIXTURE, content())
    print(f'wrote {FIXTURE.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
