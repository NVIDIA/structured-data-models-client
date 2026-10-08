# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.diagnostics, over the fixtures: every warning and refusal site of
the ported code that Phase 8 credited to a fixture case is credited to one
the Rust runner runs, that is, one not excluded (``CONTRACT.md``,
"Fixtures").

The Rust runner (``accept_request_fixtures.rs``) holds each case it runs to
its class, code, message, ``details`` keys, and warning categories and order;
Phase 8's crediting (``tests/rust_core/request_net/CONTRACT.md``) says which
site each case reaches.
"""

from __future__ import annotations

import sys
from functools import cache
from typing import Any

from _request import (
    REPO,
    RUST_CORE,
    exclusions,
    is_core,
    ledger_rules,
    ported_snapshot,
    rule_destination,
    spans_of,
)

sys.path.insert(0, str(RUST_CORE / 'request_net'))

#: About 245 sites, most of them in ported code.
AT_LEAST = 150


@cache
def _spans(file: str) -> list[tuple[str, int, int]]:
    return spans_of(REPO / file)


@cache
def _snapshot() -> dict[tuple[str, str], str]:
    return ported_snapshot()


@cache
def _rules() -> list[Any]:
    return ledger_rules()


def _ported(file: str, line: int) -> bool:
    r"""Whether the innermost symbol holding ``line`` is ported code."""
    inside = [s for s in _spans(file) if s[1] <= line <= s[2]]
    symbol = min(inside, key=lambda s: s[2] - s[1])[0] if inside else '<module>'
    if is_core(_snapshot().get((file, symbol), '')):
        return True
    return is_core(rule_destination(_rules(), file, symbol))


def test_every_ported_site_is_credited_to_a_case_the_runner_runs() -> None:
    from accept_messages import Run

    run: Any = Run()
    skipped = exclusions()
    ported = [s for s in run.sites if _ported(s.file, s.line)]
    credited = [s for s in ported if run.credited.get(s)]
    assert len(credited) >= AT_LEAST, (
        f'only {len(credited)} ported sites are credited to a case'
    )
    missing = []
    for site in credited:
        cases = run.credited[site]
        if all(case in skipped for case in cases):
            missing.append(
                f'{site.where} ({site.kind} {site.cls}): every case credited '
                f'with it is excluded: {cases[:4]}'
            )
    assert not missing, (
        f'{len(missing)} ported site(s) are left to no case the Rust runner '
        f'runs:\n' + '\n'.join(missing[:60])
    )
