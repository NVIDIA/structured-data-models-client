# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.3's pending list, read by its acceptance tests (the package's
independent test author's).

``t33-pending.json`` beside this file names, as ``py:<function name>`` and
``rust:<test>``, the tests that failed when they were written, before the
implementation. :func:`held` runs a test's check: a test not listed must
pass; a listed one must still fail, and once it passes it fails, naming
itself, until it is taken out of ``pending``. Nothing is marked: every
check runs, every time. The implementer removes names as they pass and
edits nothing else; ``pending`` stays within ``authored``, the test
author's copy, so the list only shrinks. :func:`gated` wraps a test
function in :func:`held`, keeping its signature for its fixtures and
parameters.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FILE = Path(__file__).with_name('t33-pending.json')


def names(key: str) -> set[str]:
    return set(json.loads(FILE.read_text(encoding='utf-8'))[key])


def held(name: str, check: Callable[[], None]) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    if name not in names('pending'):
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of contract/tests/t33-pending.json'
        "'s pending"
    )


def gated(test: Callable[..., None]) -> Callable[..., None]:
    r"""``test``, its body run through :func:`held` as ``py:<its name>``."""

    @functools.wraps(test)
    def run(*args: Any, **kwargs: Any) -> None:
        held(f'py:{test.__name__}', lambda: test(*args, **kwargs))

    return run


def only_shrinks() -> list[str]:
    r"""What breaks the rule that ``pending`` stays within ``authored``."""
    return sorted(names('pending') - names('authored'))
