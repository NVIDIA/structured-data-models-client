# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.33's pending list, read by its acceptance tests (the package's
independent test author's).

``t133-pending.json`` beside this file names, per group, the tests that
failed when they were written, before the implementation: ``exits`` holds
the exits of T1.32f's real-data part, folded into T1.33, as ``py:<test
id>`` (``test_t133_adventureworks_cases.py`` and
``test_t133_relbench_recipe.py``). :func:`held` runs a test's check: a test
not listed must pass; a listed one must still fail, and once it passes it
fails, naming itself, until its id is taken out of ``pending``. Nothing is
skipped or marked: every check runs, every time. The implementer removes
ids as they pass and edits nothing else; ``pending`` stays within
``authored``, the test author's copy, so the list only shrinks. A later
group of T1.33's is added to :data:`GROUPS` and to both lists.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FILE = Path(__file__).with_name('t133-pending.json')
GROUPS = ('exits',)


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
        f'contract/tests/t133-pending.json pending[{group!r}]'
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
