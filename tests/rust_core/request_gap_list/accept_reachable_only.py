# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P87.reachable-only: the whole request gap list says, by machine, what
holds each branch (``CONTRACT.md``, "The whole list").

- Every entry of ``scripts/parity/request_coverage_gaps.json`` whose reason
  says "reachable only" cites, by pytest node id, at least one host test;
  each it cites is collected under ``tests/rust_core``, passes, calls into
  ``sdm_core._native``, and takes the entry's branch in today's Python.
- No stored transcript and no fixture case, of any operation, excluded or
  not, takes the branch of any entry.
- No host test, that is no test pytest collects under ``tests/rust_core``,
  takes the branch of an entry whose reason does not say "reachable only".
"""

from __future__ import annotations

from _gaplist import (
    cited,
    corpus_measure,
    host_measure,
    host_problems,
    listed,
    says_reachable_only,
)


def _name(module: str, branch: tuple[int, int]) -> str:
    return f'{module.rsplit("/src/", 1)[-1]} {list(branch)}'


def test_the_list_and_the_measures_are_there() -> None:
    assert len(listed()) >= 100, len(listed())
    found = corpus_measure()
    assert not found['failed_transcripts'], found['failed_transcripts']
    assert found['transcripts'] and found['held_cases'] and found['other_cases']
    hosts = host_measure()
    ran = [node for node, body in hosts.items() if body['outcomes']]
    assert len(ran) >= 300, f'only {len(ran)} host tests ran'


def test_each_reachable_only_entry_cites_a_host_test_that_holds_it() -> None:
    entries = [
        (module, branch, reason)
        for (module, branch), reason in sorted(listed().items())
        if says_reachable_only(reason)
    ]
    problems = [
        f"{_name(module, branch)}: says 'reachable only' and cites no host "
        f'test by pytest node id (tests/rust_core/<path>.py::<test>'
        f'[<param id>])'
        for module, branch, reason in entries
        if not cited(reason)
    ]
    if any(cited(reason) for _, _, reason in entries):
        hosts = host_measure()
        for module, branch, reason in entries:
            for node in cited(reason):
                problems += host_problems(
                    _name(module, branch), branch, module, node, hosts
                )
    assert not problems, '\n'.join(problems)


def test_no_recorded_case_takes_a_listed_branch() -> None:
    taken = corpus_measure()['all']
    found = [
        _name(module, branch)
        for (module, branch) in sorted(listed())
        if branch in taken.get(module, set())
    ]
    assert not found, (
        'a stored transcript or a fixture case takes these listed branches, '
        'so they are no gap:\n' + '\n'.join(found)
    )


def test_no_host_test_takes_a_branch_not_said_reachable_only() -> None:
    hosts = host_measure()
    problems = []
    for (module, branch), reason in sorted(listed().items()):
        if says_reachable_only(reason):
            continue
        takers = sorted(
            node
            for node, body in hosts.items()
            if branch in body['executed'].get(module, set())
        )
        if takers:
            problems.append(
                f"{_name(module, branch)}: the reason does not say 'reachable "
                f"only', yet {', '.join(takers[:3])} take"
                f'{"s" if len(takers) == 1 else ""} it'
            )
    assert not problems, '\n'.join(problems)
