# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.32h's pending list, read by its acceptance tests (the package's
independent test author's).

``t132h-pending.json`` beside this file names, per group, the tests that
failed when they were written, before the implementation: ``python-core``
and ``rust-host`` hold ``contract/given-vectors.json``'s vector ids (a
vector run as a case is ``case:<id>``), ``exits`` the other checks' test
ids (``py:`` the Python tests', ``rs:`` and ``rs-edit:`` the Rust ones').
A group named ``D3`` or ``D4`` holds the tests of that decision, still the
owner's: its own commit adds the group and can be dropped with it.

:func:`held` runs a test's check: a test not listed must pass; a listed one
must still fail, and once it passes it fails, naming itself, until its id
is taken out of ``pending``. Nothing is skipped or marked: every check
runs, every time. The implementer removes ids as they pass and edits
nothing else; ``pending`` stays within ``authored``, the test author's
copy, so the list only shrinks.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FILE = Path(__file__).with_name('t132h-pending.json')


def gating() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FILE.read_text(encoding='utf-8'))
    return loaded


def groups() -> list[str]:
    r"""The groups the test author wrote, ``authored``'s keys."""
    return sorted(gating()['authored'])


def pending(group: str) -> set[str]:
    loaded = gating()['pending']
    if group not in loaded:
        raise AssertionError(f't132h-pending.json: no pending group {group!r}')
    return set(loaded[group])


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
        f'contract/tests/t132h-pending.json pending[{group!r}]'
    )


def only_shrinks() -> list[str]:
    r"""What breaks the rule that ``pending`` stays within ``authored``."""
    loaded = gating()
    problems = []
    if set(loaded['pending']) != set(loaded['authored']):
        problems.append(
            f'the groups differ: pending {sorted(loaded["pending"])}, '
            f'authored {sorted(loaded["authored"])}'
        )
    for group in sorted(set(loaded['pending']) & set(loaded['authored'])):
        problems.extend(
            f'{group}: {name} is pending but was never authored'
            for name in sorted(
                set(loaded['pending'][group]) - set(loaded['authored'][group])
            )
        )
    return problems
