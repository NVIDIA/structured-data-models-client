# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.errors: on the core, every transcript and public fixture case raises
only public ``sdm_client`` error types, and nothing of the binding escapes.

- Every error raised is an instance of a class ``sdm_client`` exports, of the
  ``sdm_client`` package.
- No error escapes the operation unhandled (a ``BaseException`` such as a
  pyo3 panic), and no class of ``sdm_core`` or ``pyo3_runtime`` is anywhere
  in the chain a traceback shows.
- The chain beneath each error (``__cause__``, or an unsuppressed
  ``__context__``) is today's, leaving out classes of the two ported
  packages, which only the old path has: ``explain_error``'s chaining kept.
"""

from __future__ import annotations

from _switch import (
    CORE,
    OLD,
    PORTED_PACKAGES,
    by_key,
    capture,
    core_in_use,
    fix_changes,
    is_core_error_class,
)


def _chain(record):
    return [
        c
        for c in record.get('error_chain') or []
        if not c.startswith(PORTED_PACKAGES)
    ]


def _judge(what, key):
    core = capture(CORE, what)
    old = by_key(capture(OLD, what), key)
    listed = fix_changes()
    problems = [f'not on the core: {p}' for p in core_in_use(core)]
    raised = 0
    for rec in core['records']:
        name = rec[key]
        if rec.get('crashed'):
            problems.append(f'{name}: escaped unhandled: {rec["crashed"]}')
            continue
        error = rec.get('error')
        if not error:
            continue
        raised += 1
        cls = error.get('class', '')
        if not cls.startswith('sdm_client.') or not rec.get('error_public'):
            problems.append(
                f'{name}: raised {cls}, not a public sdm_client error'
            )
        leaked = [
            c for c in rec.get('error_chain') or [] if is_core_error_class(c)
        ]
        if leaked:
            problems.append(f'{name}: the binding shows in the chain {leaked}')
        if (
            name not in listed
            and name in old
            and _chain(rec) != _chain(old[name])
        ):
            problems.append(
                f'{name}: the chain is {_chain(rec)}, today it is {_chain(old[name])}'
            )
    return problems, raised


def test_every_transcript_error_is_a_public_sdm_client_error():
    problems, raised = _judge('transcripts', 'name')
    assert not problems, '\n'.join(problems[:60])
    assert raised >= 5, f'only {raised} transcripts raised'


def test_every_public_fixture_error_is_a_public_sdm_client_error():
    problems, raised = _judge('fixtures', 'id')
    assert not problems, '\n'.join(problems[:60])
    assert raised >= 200, f'only {raised} fixture cases raised'
