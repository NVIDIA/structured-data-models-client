# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P76.gaps: the sampling coverage gap list, and Phase 8's request gap and
unreachable-diagnostics lists, no longer say link prediction is
unreachable; the sampling list drops the link-prediction branches the
scenarios take and the multilabel refusal a scenario takes, and no text in
them cites a harness file (``CONTRACT.md``, "Gaps").

Phase 7.5's ``accept_driver_coverage.py``, run beside this file, checks
that every listed entry is a ported branch left untaken.
"""

from __future__ import annotations

import json
from typing import Any

from _link import (
    GAPS,
    LINK_BRANCHES,
    MULTILABEL_BRANCH,
    REPO,
    REQUEST_GAP_DOCS,
    RETIRED_REASONS,
    RFM,
    check_rfm_lines,
    cites_the_harness,
    says_link_prediction_is_never_gathered,
)


def _gaps() -> dict[str, list[dict[str, Any]]]:
    gaps = json.loads(GAPS.read_text('utf-8'))
    assert isinstance(gaps, dict), f'{GAPS} is not an object'
    return gaps


def _rfm_entries() -> dict[tuple[int, int], str]:
    return {
        tuple(entry['branch']): entry['reason']
        for entry in _gaps().get(RFM, [])
    }


def test_the_branches_are_where_the_checks_say() -> None:
    check_rfm_lines()


def test_no_reason_says_link_prediction_is_unreachable() -> None:
    stale = []
    for module, entries in _gaps().items():
        for entry in entries:
            reason = ' '.join(str(entry.get('reason', '')).lower().split())
            for retired in RETIRED_REASONS:
                if retired in reason:
                    stale.append(
                        f'{module} {entry["branch"]}: says {retired!r}'
                    )
    assert not stale, '\n'.join(stale)


def test_no_request_gap_says_the_client_never_gathers_link_prediction() -> None:
    stale = []
    for module, entries in _gaps().items():
        for entry in entries:
            if says_link_prediction_is_never_gathered(str(entry.get('reason'))):
                stale.append(f'{module} {entry["branch"]}')
    for doc in REQUEST_GAP_DOCS:
        for number, line in enumerate(doc.read_text('utf-8').splitlines(), 1):
            if line.lstrip().startswith('|') and (
                says_link_prediction_is_never_gathered(line)
            ):
                stale.append(f'{doc.relative_to(REPO)}:{number}')
    assert not stale, (
        'rows still say the client never gathers a link prediction:\n'
        + '\n'.join(stale)
    )


def test_the_branches_the_scenarios_take_are_not_listed() -> None:
    check_rfm_lines()
    listed = sorted(set(_rfm_entries()) & set(LINK_BRANCHES))
    assert not listed, (
        f'rfm.py still lists the link-prediction branches {listed}'
    )


def test_the_multilabel_branch_is_not_listed() -> None:
    r"""A scenario takes it (P76.multilabel holds that), so it is no gap."""
    check_rfm_lines()
    entries = _rfm_entries()
    assert MULTILABEL_BRANCH not in entries, (
        f'rfm.py {list(MULTILABEL_BRANCH)} (the multilabel '
        'NotImplementedError) is still listed: a scenario takes it, '
        f'{entries[MULTILABEL_BRANCH]!r}'
    )


def test_the_gap_lists_cite_no_harness_file() -> None:
    r"""Task 7: the reasons say what they mean, rather than cite a phase's
    report or other harness file, which is not in the repository."""
    cited = []
    for module, entries in _gaps().items():
        for entry in entries:
            found = cites_the_harness(str(entry.get('reason', '')))
            if found:
                cited.append(
                    f'{GAPS.name}: {module} {entry["branch"]}: {found!r}'
                )
    for doc in REQUEST_GAP_DOCS:
        for number, line in enumerate(doc.read_text('utf-8').splitlines(), 1):
            found = cites_the_harness(line)
            if found:
                cited.append(f'{doc.relative_to(REPO)}:{number}: {found!r}')
    assert not cited, (
        'gap texts cite files that are not in the repository:\n'
        + '\n'.join(cited[:30])
    )
