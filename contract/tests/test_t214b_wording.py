# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.14b's wording exits on ``python-core`` (the package's test author's):
the host wording rows G15, M4, FX-1 and FX-4 (KO2-32), the core refusals
that reach Python as their code's generic sentence, and a binary target
with one class in its sample.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t214b_wording.py -q

Written from the design's T2.14b row, migration-plan.md's "Host error and
warning wording" rows and the carried notes (T2.12 round 1: core refusals
reach Python users as "the request is not one the core accepts" where the
reference names the table, the key and an example; T2.13's tracker, every
other refusal still said as its code's sentence; T2.8 round 3, a binary
target whose sample holds one class), against the reference (python-ref at
``contract/reference-env.txt``'s commit), before the implementation.

Evidence, all from the reference: its committed lines for the
``wording.*`` cases (``run_python.py``) and
``contract/t214b-wording-vectors.json`` (``wording_record.py``: each
case's outcome, a refusal's message and details, every warning by class and
text, and FX-4's probes), which ``python-core`` is observed against by the
same code (``wording_record.observe`` and ``probe``).

- G15: a key or time column ``from_source`` refuses (``wording.key-*``,
  ``wording.time-*``) and the schema edits the ``edit.*`` cases refuse are
  said with the table, the column and what the table has, as the
  reference says them.
- M4: the sampler's warnings in the reference's words: a context reaching
  back before the data (``CONTEXT_BEFORE_DATA``) and an anchor after it
  (``ANCHOR_AFTER_DATA``), each naming the times, and the missing-index
  warning beside them (E41(i): all three the core's words until now).
- FX-1: pandas' ``RuntimeWarning`` from casting a double past float32's
  range, beside the refusal, as the reference gives it
  (``wording.float32-*``).
- FX-4: a whole table's MIN that is an infinity, shown in the
  constant-column warning as the reference shows it (DuckDB's ``fetchall``
  hands it over as ``datetime.max`` or ``datetime.min``).
- The generic refusals: every committed S1 case the core refuses as the
  reference does but in its code's sentence, said naming what the
  reference's message names (:data:`NAMED`), and never the bare sentence.
- A binary target with one class (``wording.binary-one-class``,
  ``decimal.compare-*``), explained as the reference explains it.

Gating: ``t214b-pending.json``'s ``pending`` names, as
``wording:named:<case key>``, ``wording:warned:<label>:<case key>`` and
``wording:fx4:<probe>``, the keys whose test failed when written. Each must
still fail (strict); once one passes it fails, naming itself, until it is
taken out. ``pending`` stays within ``authored``. Where
``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES`` is
not set, the Postgres keys have nothing to hold and return without
asserting; with it set, a missing server fails (H26).
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
RECORDER = ROOT / 'contract/tools/wording_record.py'
CASES = ROOT / 'contract/cases/s1'
PENDING = Path(__file__).with_name('t214b-pending.json')
PREFIX = 'wording:'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def recorder() -> ModuleType:
    return _module('wording_record', RECORDER)


def runner() -> ModuleType:
    return recorder().options().runner()


#: What each refusal's message names, by case, as the reference's names it
#: (each word must be in the reference's recorded message for the case key
#: to be held; the control below holds the DuckDB key of every case to
#: that): the table, the column or key, the value given and an example of
#: what is taken. An empty list holds only that the message is not its
#: code's bare sentence (the reference's message there names nothing of
#: the request).
NAMED: dict[str, list[str]] = {
    # G15, at from_source.
    'wording.key-missing-table': ["'nope'", "['orders', 'users']"],
    'wording.key-none': ["'users'", 'empty primary key'],
    'wording.key-number': ['primary key', '5'],
    'wording.time-key': ["'user_id'", "'users'", 'primary key'],
    'wording.time-missing-column': [
        "'orders'",
        "'nope'",
        "['order_id', 'price', 'ts', 'user_id']",
    ],
    'wording.time-number': [
        "'orders'",
        '5',
        "['order_id', 'price', 'ts', 'user_id']",
    ],
    # G15, at the schema's edits.
    'edit.key-missing-column': ["'users'", "'nope'"],
    'edit.key-time-column': ["'ts'", 'primary key'],
    'edit.leave-out-the-key': ['orders.order_id', 'primary key'],
    'edit.link-not-the-key': ['orders.user_id', 'users.age', "'user_id'"],
    'edit.meaning-not-carried': ["'country'", "'numerical'"],
    'edit.meaning-unknown': ["'banana'", 'semantic type'],
    'edit.time-column-not-a-time': ["'age'", "'timestamp'"],
    # The generic refusals at materialize and predict.
    'date.day-first-then-packed': ['99999-01-01'],
    'empty.indices': ['indices'],
    # The value past int64 is a context example's key read from the data,
    # which the core keeps out (KO2-86, register P2D-4): the column, the
    # row and the remedy, as the reference names them.
    'key.uint64-above-int64-max': [
        "Column 'instance_table.__kumo_entity_ref_0' row 0",
        'outside the range int64 represents',
        'send the column as strings',
    ],
    'rank.entities-by-candidates.at': ["'users'", '499', 'candidate 0'],
    'rule.composite-key-given': ['orders'],
    'sample.anchor-entity.context-time': ["'entity'", 'context'],
    'sample.anchor-entity.no-time-column': ["'users'", 'time column'],
    'sample.anchor-entity.temporal': ["'entity'", 'static'],
    'sample.context-anchor-time.entity': ["'entity'", 'context'],
    'sample.context-anchor-time.too-early': [
        "'2022-01-01 00:00:00'",
        "'2023-01-01 00:00:00'",
    ],
    'sample.exclude-cols.key': ["'user_id'", "'users'", "['age', 'country']"],
    'sample.exclude-cols.list': ['exclude_cols_dict', 'list'],
    'sample.exclude-cols.text-columns': ["exclude_cols_dict['users']", 'str'],
    'sample.exclude-cols.unknown-column': [
        "'nope'",
        "'users'",
        "['age', 'country']",
    ],
    'sample.exclude-cols.unknown-table': ["'nope'", "['orders', 'users']"],
    'sample.max-pq-iterations.text': [],
    'sample.max-pq-iterations.zero': ['max_pq_iterations', '0'],
    'sample.num-hops.bool': ['num_hops', 'True'],
    'sample.num-hops.fraction': ['num_hops', '1.5'],
    'sample.num-hops.seven': ['7 hops', '6'],
    'sample.num-hops.text': ['num_hops', "'2'"],
    'sample.num-hops.zero': ['num_hops', 'num_neighbors=[8, 8]'],
    'sample.num-neighbors.empty': ['num_neighbors', 'no hop'],
    'sample.num-neighbors.number': ['num_neighbors', 'sequence'],
    'sample.num-neighbors.seven-hops': ['7 hops', '6'],
    'sample.seed.negative': ['random_seed', '-1'],
    'wording.rank-text-candidate': ['buys.item_id', "'1'"],
    'wording.float32-classification': ['non-finite', 'inf'],
    'wording.float32-regression': ['non-finite', 'inf'],
    # A binary target with one class in its sample (T2.8 round 3).
    'wording.binary-one-class': [
        "'PREDICT COUNT(orders.*, 0, 30, days) > 1000 FOR EACH users.user_id'",
        'two classes',
    ],
    'decimal.compare-finer-eq': ['users.d', 'two classes'],
    'decimal.compare-past-precision': ['users.d', 'two classes'],
    'decimal.compare-past-precision-gt': ['users.d', 'two classes'],
}
#: The warnings held word for word, by label: the class, and how the
#: reference's text opens or the core's opened when written (so a warning
#: the core gives in other words is still counted as the same warning).
LABELS: dict[str, tuple[str, tuple[str, ...]]] = {
    # M4.
    'context-before-data': (
        'UserWarning',
        (
            'Context anchor timestamp is too early',
            "the context's anchor time is too early",
        ),
    ),
    'anchor-after-data': (
        'UserWarning',
        ('Anchor timestamp ', 'the anchor time is after'),
    ),
    'missing-indices': ('UserWarning', ('Missing ',)),
    # FX-1.
    'runtime-warning': ('RuntimeWarning', ('',)),
}
#: The cases each label is held on, on every database the case runs on.
WARNED: dict[str, tuple[str, ...]] = {
    'context-before-data': (
        'target.filter.sum',
        'shape.star.customers',
        'wording.binary-one-class',
    ),
    'anchor-after-data': ('wording.anchor-after-data',),
    'missing-indices': ('wording.anchor-after-data', 'shape.star.customers'),
    'runtime-warning': (
        'wording.float32-classification',
        'wording.float32-regression',
    ),
}
#: What FX-4's warning shows for each probe's whole-table MIN, as the
#: reference shows it.
INFINITY_SHOWN = {
    'infinity.timestamptz.positive': '9999-12-31 23:59:59.999999',
    'infinity.timestamp.positive': '9999-12-31 23:59:59.999999',
    'infinity.timestamptz.negative': '0001-01-01 00:00:00',
    'infinity.timestamp.negative': '0001-01-01 00:00:00',
}
#: The sentences the core route says a refusal in when it has no words of
#: its own for it (``_rust/errors.py``'s ``ERRORS``).
GENERIC = (
    'the request is not one the core accepts',
    'the request is valid, and not one the core supports yet',
    'the schema is not one the core can work with',
)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The cases by id, each with its digest, as the runner loads them."""
    loaded = runner().load_cases(
        [CASES / f'{c}.json' for c in recorder().CASE_IDS]
    )
    return {case['id']: (case, digest) for case, digest in loaded}


def keys_of(case_id: str) -> list[str]:
    case, _ = cases()[case_id]
    return [key(case_id, d) for d in runner().case_databases(case)]


@functools.cache
def vectors() -> dict[str, Any]:
    return recorder().load()


def _named_keys() -> list[str]:
    r"""The case keys whose message is held: each of :data:`NAMED`'s cases
    on each database where the reference refuses with every named word in
    its message."""
    out = []
    document = vectors()['vectors']
    for case_id, words in NAMED.items():
        for case_key in keys_of(case_id):
            want = document[case_key]
            if want['outcome']['result'] == 'refusal' and all(
                word in (want['message'] or '') for word in words
            ):
                out.append(case_key)
    return out


NAMED_KEYS = tuple(_named_keys())
WARNED_KEYS = tuple(
    (label, case_key)
    for label, case_ids in WARNED.items()
    for case_id in case_ids
    for case_key in keys_of(case_id)
)


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
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
    r"""Control: ``pending`` names only keys the test author wrote, and
    the ``wording:`` keys ``authored`` names are every key held here."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    authored = {k for k in gating['authored'] if k.startswith(PREFIX)}
    assert authored == (
        {f'{PREFIX}named:{k}' for k in NAMED_KEYS}
        | {f'{PREFIX}warned:{label}:{k}' for label, k in WARNED_KEYS}
        | {f'{PREFIX}fx4:{name}' for name in recorder().PROBES}
    )


def test_every_case_held_here_is_recorded_and_named() -> None:
    r"""Control (no vacuous key): every case here has a vector on each of
    its databases at its case file, every case of :data:`NAMED` is held on
    DuckDB at least (the reference refusing there with every named word),
    and :data:`WARNED`'s cases give each label's warning on DuckDB on the
    reference."""
    document = vectors()
    assert document['recorded_by'] == 'contract/tools/wording_record.py'
    for case_id in recorder().CASE_IDS:
        _, digest = cases()[case_id]
        for case_key in keys_of(case_id):
            vector = document['vectors'].get(case_key)
            assert vector is not None, f'{case_key}: no vector'
            assert vector['case'] == digest, f'{case_key}: another case file'
    assert set(NAMED) | {c for ids in WARNED.values() for c in ids} == set(
        recorder().CASE_IDS
    )
    for case_id in NAMED:
        assert case_id in NAMED_KEYS, f'{case_id}: not held on DuckDB'
    for label, case_ids in WARNED.items():
        for case_id in case_ids:
            assert _labelled(label, document['vectors'][case_id]['warnings'])


def test_python_ref_s_line_is_at_each_wording_case_file() -> None:
    r"""The ``wording.*`` cases: python-ref's line for each, on each
    database, recorded at the case file, ends as the case expects, as the
    vector does."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    assert vectors()['reference'] == commit
    lines = {
        json.loads(line)['case']: json.loads(line)
        for line in (ROOT / f'contract/reference/{commit}.jsonl')
        .read_text(encoding='utf-8')
        .splitlines()
    }
    for case_id in recorder().OWN:
        case, digest = cases()[case_id]
        expect = case['expect']
        want = (
            expect['outcome'],
            expect['stage'],
            expect.get('error'),
            expect.get('code'),
        )
        for case_key in keys_of(case_id):
            line = lines.get(case_key)
            assert line is not None, f'{case_key}: no python-ref line'
            assert line['identity']['inputs']['case'] == digest
            outcome = line['outcome']
            got = (
                outcome['result'],
                outcome['stage'],
                outcome.get('error'),
                outcome.get('code'),
            )
            assert got == want, case_key
            assert vectors()['vectors'][case_key]['outcome'] == outcome


# --- python-core against the reference ---------------------------------------------


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required (H26)."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


def _skipped(case_key: str) -> bool:
    return case_key.endswith('@postgres') and not _server()


@functools.cache
def core_observed(case_key: str) -> dict[str, Any]:
    r"""python-core's observation of the case, by the recorder's code."""
    case_id, _, database = case_key.partition('@')
    case, _ = cases()[case_id]
    return recorder().observe(case, database or 'duckdb', core=True)


def _outcome(outcome: dict[str, Any]) -> tuple[Any, ...]:
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


def same_outcome(want: tuple[Any, ...], got: tuple[Any, ...]) -> bool:
    r"""Two outcomes the same, but that the core route raises a refusal
    coded ``UNSUPPORTED_REQUEST`` as ``UnsupportedRequestError``, a
    subclass of the reference's ``StructuredDataError`` (register P1B-20,
    approved by the owner, E78)."""
    if want == got:
        return True
    return (
        want[0] == got[0] == 'refusal'
        and want[1:2] == got[1:2]
        and want[3] == got[3] == 'UNSUPPORTED_REQUEST'
        and (want[2], got[2])
        == ('StructuredDataError', 'UnsupportedRequestError')
    )


def _words(message: str | None) -> str:
    r"""A refusal's text with its ``[CODE] `` prefix left out."""
    return recorder().CODE_PREFIX.sub('', message or '')


@pytest.mark.parametrize('case_key', NAMED_KEYS)
def test_python_core_names_what_the_reference_names(case_key: str) -> None:
    r"""G15, the carried generic-refusal notes and T2.8 round 3:
    python-core refuses as the reference does, and its message names what
    the reference's names (the table, the column or key, the value given
    and an example of what is taken), never only its code's sentence."""
    if _skipped(case_key):
        return
    case_id = case_key.partition('@')[0]

    def check() -> None:
        want = vectors()['vectors'][case_key]
        got = core_observed(case_key)
        outcome = (_outcome(want['outcome']), _outcome(got['outcome']))
        assert same_outcome(*outcome), (
            f'{case_key}: {outcome[1]} where python-ref has {outcome[0]}'
        )
        said = _words(got['message'])
        assert not said.startswith(GENERIC), (
            f'{case_key}: said as its code: {said!r:.300}'
        )
        missing = [w for w in NAMED[case_id] if w not in said]
        assert not missing, (
            f'{case_key}: the refusal does not name {missing}: {said!r:.300}'
            f'; the reference says {_words(want["message"])!r:.300}'
        )

    held(f'{PREFIX}named:{case_key}', check)


def _labelled(label: str, shown: list[list[str]]) -> list[list[str]]:
    kind, openings = LABELS[label]
    return [
        list(warning)
        for warning in shown
        if warning[0] == kind and warning[1].startswith(openings)
    ]


@pytest.mark.parametrize(
    ('label', 'case_key'),
    WARNED_KEYS,
    ids=[f'{label}:{k}' for label, k in WARNED_KEYS],
)
def test_python_core_warns_in_the_reference_s_words(
    label: str, case_key: str
) -> None:
    r"""M4 and FX-1: python-core gives the sampler's warnings in the
    reference's words (a context reaching back before the data and an
    anchor after it, each naming the times, and the missing indexes), and
    pandas' ``RuntimeWarning`` beside a double past float32's range, each
    by class and text (a core code's prefix left out), where the reference
    gives it and nowhere else."""
    if _skipped(case_key):
        return

    def check() -> None:
        want = _labelled(label, vectors()['vectors'][case_key]['warnings'])
        got = _labelled(label, core_observed(case_key)['warnings'])
        assert got == want, (
            f'{case_key}: {label}: python-core warns {got} where python-ref '
            f'warns {want}'
        )

    held(f'{PREFIX}warned:{label}:{case_key}', check)


@pytest.mark.parametrize('name', sorted(INFINITY_SHOWN))
def test_an_infinite_minimum_is_shown_as_the_reference_shows_it(
    name: str,
) -> None:
    r"""FX-4: a timestamp column null in the rows the preview reads and an
    infinity after them is passed over as constant on both routes, and the
    warning shows the whole table's MIN as the reference shows it
    (``9999-12-31 23:59:59.999999`` or ``0001-01-01 00:00:00``), word for
    word."""
    want = vectors()['probes'][name]
    shown = INFINITY_SHOWN[name]
    guessed = [w for w in want['warnings'] if 'time column' in w[1]]
    assert guessed and all(shown in w[1] for w in guessed), want['warnings']

    def check() -> None:
        got = recorder().probe(name, core=True)
        assert got['outcome'] == want['outcome']
        assert [w for w in got['warnings'] if 'time column' in w[1]] == (
            guessed
        ), f'{name}: python-core warns {got["warnings"]}'

    held(f'{PREFIX}fx4:{name}', check)
