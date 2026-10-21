# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Regressions for T2.14b's fix rounds on ``python-core`` (the package's
test author's, from review round 2's probes).

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t214b_regressions.py -q

Each scenario of ``contract/tools/t214b_regressions_record.py`` runs on
python-core with the core routes forced, by the recorder's own
``observe``, and is held to what the reference gave
(``contract/t214b-regression-vectors.json``):

- repeated and equal temporal ranking candidates (fix 1, Codex P2-1; fix
  2's typed kinds): equal dates are one candidate, aware timestamps of one
  instant at two offsets are one (as ``dict.fromkeys`` keeps them), a
  naive timestamp beside an aware one stays two: the parts, each question
  asked and the rankings are the reference's;
- a column's meaning set to ``5`` (fix 1, Codex P3-1): refused with the
  reference's class, code and words;
- an aggregation's filter reading a double past float32's range (fix 1,
  Claude P3-2, FX-1): pandas' ``RuntimeWarning`` in the reference's words;
- Postgres ``json`` documents escaping lone surrogates (fix 1, Codex
  P2-3), through ``from_source``, ``materialize`` and ``predict``: each
  request the endpoint receives is the reference's.

The Postgres provider's own regressions (a declared numeric's refusal,
``json[]`` elements) are in ``tests_test_ops/test_t214b_postgres_regressions.py``,
and the typed candidate as plain data in ``sdm-request``'s tests.

Gating: ``t214b-pending.json``'s ``pending`` names, as ``regress:<name>``,
the tests that failed when written. Each must still fail (strict); once
one passes it fails, naming itself, until it is taken out. Where
``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES`` is
not set, the Postgres scenario has nothing to hold and returns.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / 'contract/tools/t214b_regressions_record.py'
PENDING = Path(__file__).with_name('t214b-pending.json')
PREFIX = 'regress:'


def recorder() -> ModuleType:
    name = 't214b_regressions_record'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, RECORDER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


NAMES = (
    *(f'candidates:{name}' for name in recorder().CANDIDATES),
    'stype',
    'where',
    'json',
)


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert name in gating['authored'], f'{name} is not authored'
    if name not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        'contract/tests/t214b-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote; this
    file's authored keys are its scenarios."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    mine = {k for k in gating['authored'] if k.startswith(PREFIX)}
    assert mine == {f'{PREFIX}{name}' for name in NAMES}


def _dsn() -> str | None:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        return None
    return dsn


@functools.cache
def vectors() -> dict[str, Any]:
    return recorder().load()


@functools.cache
def core() -> dict[str, Any]:
    r"""python-core's observation of every scenario, on the core routes."""
    return recorder().observe(core=True, dsn=_dsn())


def test_the_vectors_are_the_reference_s_and_exercise_what_they_name() -> None:
    r"""Control: recorded from the reference commit the contract names, by
    the committed recorder; the candidates repeat as named (two dates one,
    two instants one, a naive and an aware two), the meaning's refusal names
    the number, the filter warns, and a document's lone surrogate is sent
    escaped."""
    document = vectors()
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {document["reference"]}' in env
    assert document['recorded_by'] == (
        'contract/tools/t214b_regressions_record.py'
    )
    parts = {k: v['parts'] for k, v in document['candidates'].items()}
    assert parts == {
        'dates-repeated': 2,
        'dates-distinct': 2,
        'aware-one-instant': 2,
        'aware-distinct': 2,
        'zone-fold': 2,
        'zone-against-utc': 3,
        'times-one-instant': 2,
        'naive-beside-aware': 3,
    }
    assert document['stype']['refusal']['message'].startswith(
        '[INVALID_REQUEST] 5 is not a semantic type; use one of '
    )
    assert document['where']['runtime_warnings'] == [
        'overflow encountered in cast'
    ]
    sent = json.dumps(document['json']['bodies'])
    assert document['json']['refusal'] is None
    assert '\\ud800' in sent and '\\udc00' in sent


@pytest.mark.parametrize('name', list(recorder().CANDIDATES))
def test_temporal_candidates_are_asked_as_the_reference_keeps_them(
    name: str,
) -> None:
    r"""Fix 1 (Codex P2-1) and fix 2: a date, time or timestamp candidate
    given twice, or as one instant at two offsets, is one question, as the
    reference's ``dict.fromkeys`` keeps it; a naive and an aware timestamp
    are two. The parts, each question and the rankings are the
    reference's."""

    def check() -> None:
        want = vectors()['candidates'][name]
        got = core()['candidates'][name]
        assert got == want, (
            f'{name}: {json.dumps(got)[:400]} where python-ref has '
            f'{json.dumps(want)[:400]}'
        )

    held(f'{PREFIX}candidates:{name}', check)


def test_a_number_as_a_meaning_is_refused_in_the_reference_s_words() -> None:
    r"""Fix 1 (Codex P3-1): ``stype = 5`` is refused as the reference
    refuses it, naming the value and the meanings there are."""

    def check() -> None:
        assert core()['stype'] == vectors()['stype']

    held(f'{PREFIX}stype', check)


def test_a_filter_reading_a_double_past_float32_warns_as_the_reference() -> (
    None
):
    r"""Fix 1 (Claude P3-2, FX-1): an aggregation's WHERE reading a double
    past float32's range gives pandas' ``RuntimeWarning`` in the
    reference's words, and materialize answers as there."""

    def check() -> None:
        assert core()['where'] == vectors()['where']

    held(f'{PREFIX}where', check)


def test_json_documents_escaping_lone_surrogates_are_sent_as_the_reference() -> (
    None
):
    r"""Fix 1 (Codex P2-3): a Postgres ``json`` column whose documents
    escape a lone surrogate (in a value, in a key, in a list beside a
    surrogate pair) goes through ``from_source``, ``materialize`` and
    ``predict``, and each request the endpoint receives is the
    reference's, every document as the reference writes it."""
    if _dsn() is None:
        return

    def check() -> None:
        want = vectors()['json']
        got = core()['json']
        assert got['refusal'] == want['refusal'], got['refusal']
        assert got['bodies'] == want['bodies'], (
            'the requests differ from the reference'
        )

    held(f'{PREFIX}json', check)
