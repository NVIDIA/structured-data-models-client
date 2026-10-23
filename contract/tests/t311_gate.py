# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.11's pending list, read by its acceptance tests (the package's test
author's): ``contract/tests/test_t311_java.py`` and
``tests/test_t311_java_ci.py``.

``t311-pending.json`` beside this file names, by test function, the tests
that failed when they were written, before the implementation.
:func:`held` runs a test's check: a test not listed must pass; a listed one
must still fail, and once it passes it fails, naming itself, until its name
is taken out of ``pending``. Nothing is passed over or marked: every check
runs, every time it can. A test that needs what only some jobs have (a JAR,
a JDK, a Postgres server) says so with :func:`gated`'s ``when``; where that
is missing it has nothing to hold and returns, and a job that must have it
says so (``SDM_REQUIRE_JAVA``) and fails without it. The implementer
removes names as they pass and edits nothing else; ``pending`` stays within
``authored``, the test author's copy, so the list only shrinks.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FILE = Path(__file__).with_name('t311-pending.json')


def gating() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FILE.read_text(encoding='utf-8'))
    return loaded


def pending() -> set[str]:
    return set(gating()['pending'])


def held(name: str, check: Callable[[], None]) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    if name not in pending():
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        'contract/tests/t311-pending.json pending'
    )


def gated(
    when: Callable[[], bool] | None = None,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    r"""A test held by :func:`held` under its own name, run only when
    ``when`` says what it needs is here."""

    def wrap(test: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(test)
        def run(*args: Any, **kwargs: Any) -> None:
            if when is not None and not when():
                return
            held(test.__name__, lambda: test(*args, **kwargs))

        return run

    return wrap


def only_shrinks() -> list[str]:
    r"""What breaks the rule that ``pending`` stays within ``authored``."""
    loaded = gating()
    return [
        f'{name} is pending but was never authored'
        for name in sorted(set(loaded['pending']) - set(loaded['authored']))
    ]
