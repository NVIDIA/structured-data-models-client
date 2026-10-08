# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.unreached: the suite-rest command and ``run_corpus.py replay``, run on
the core, never reach code destined for a crate from the packages.

``_unreached.py``'s guard watches every start of ported code (``ported.tsv``)
in both runs and judges it by the nearest frame above it that is ported, test
or harness code, or package code (``CONTRACT.md``, "Ported code"): reached
from package code is a breach. Tests under ``tests/rust_core/`` (the shadow
harness, and the earlier phases' comparisons with today's Python) are not
judged. The guard must have seen ported code run, called directly by tests,
so it is shown to be watching.
"""

from __future__ import annotations

from _switch import replay_report, suite_report


def _breaches(report):
    return [
        f'{symbol} ({entry["count"]} times), reached from {entry["called_from"]}'
        f' in {entry["contexts"][:2]}'
        for symbol, entry in sorted(
            report['breaches'].items(), key=lambda kv: -kv[1]['count']
        )
    ]


def test_the_suite_never_reaches_ported_code_from_the_packages():
    report = suite_report()
    assert report['tests'] >= 4000, report.get('tail')
    assert report['counts']['direct'] > 0, (
        'the guard saw no ported code called by a test: it is not watching'
    )
    found = _breaches(report)
    assert not found, (
        f'{len(found)} ported symbols reached from package code:\n'
        + '\n'.join(found[:60])
    )


def test_the_replay_never_reaches_ported_code_from_the_packages():
    report = replay_report()
    assert report['exitstatus'] == 0, report.get('tail')
    found = _breaches(report)
    assert not found, (
        f'{len(found)} ported symbols reached from package code:\n'
        + '\n'.join(found[:60])
    )
