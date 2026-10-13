# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.32g's pending list, read by its acceptance tests (the package's
independent test author's).

``t132g-pending.json`` beside this file names, per group (``python-core``
and ``rust-host`` vector ids, and ``exits``, test ids), the tests that
failed when they were written, before the implementation. :func:`held`
runs a test's check: a test not listed must pass; a listed one must still
fail, and once it passes it fails, naming itself, until its id is taken
out of ``pending``. Nothing is skipped or marked: every check runs, every
time. The implementer removes ids as they pass and edits nothing else;
``pending`` stays within ``authored``, the test author's copy, so the list
only shrinks.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FILE = Path(__file__).with_name('t132g-pending.json')
GROUPS = ('python-core', 'rust-host', 'exits')


def gating() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FILE.read_text(encoding='utf-8'))
    return loaded


def pending(group: str) -> set[str]:
    return set(gating()['pending'][group])


def held(group: str, name: str, check: Callable[[], None]) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    if name not in pending(group):
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        f'contract/tests/t132g-pending.json pending[{group!r}]'
    )


def only_shrinks() -> list[str]:
    r"""What breaks the rule that ``pending`` stays within ``authored``."""
    loaded = gating()
    return [
        f'{group}: {name} is pending but was never authored'
        for group in GROUPS
        for name in sorted(
            set(loaded['pending'][group]) - set(loaded['authored'][group])
        )
    ]
