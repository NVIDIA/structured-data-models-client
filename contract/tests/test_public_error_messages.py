# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public errors and warnings preserve useful messages for invalid requests.

Current-client checks use retained expected values and scripted endpoint responses.
No archived client runtime is executed.
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
RECORDER = ROOT / 'contract/support/error_wording_scenarios.py'
CASES = ROOT / 'contract/cases/s1'

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
    return _module('error_wording_scenarios', RECORDER)


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
    # which the core keeps out (D-4): the column, the
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
    # A binary target with one class in its sample ( round 3).
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
    r"""The case keys whose message is held: each of:data:`NAMED`'s cases
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


def test_every_case_held_here_is_recorded_and_named() -> None:
    r"""Control (no vacuous key): every case here has a vector on each of
    its databases at its case file, every case of:data:`NAMED` is held on
    DuckDB at least (the reference refusing there with every named word),
    and:data:`WARNED`'s cases give each label's warning on DuckDB on the
    reference."""
    document = vectors()
    for case_id in recorder().CASE_IDS:
        _, digest = cases()[case_id]
        for case_key in keys_of(case_id):
            vector = document['vectors'].get(case_key)
            assert vector is not None, f'{case_key}: no vector'
    assert set(NAMED) | {c for ids in WARNED.values() for c in ids} == set(
        recorder().CASE_IDS
    )
    for case_id in NAMED:
        assert case_id in NAMED_KEYS, f'{case_id}: not held on DuckDB'
    for label, case_ids in WARNED.items():
        for case_id in case_ids:
            assert _labelled(label, document['vectors'][case_id]['warnings'])


def test_expected_wording_outcomes_match_cases() -> None:
    """Independent expected facts agree with this case's stated behavior."""
    lines = {
        json.loads(line)['case']: json.loads(line)
        for line in (
            ROOT / 'contract/fixtures/maintained/request-expectations.jsonl'
        )
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
            assert line is not None, f'{case_key}: no expected facts'
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
    required."""
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
    return recorder().observe(case, database or 'duckdb')


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
    subclass of the reference's ``StructuredDataError`` (B-20,
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
    r"""G15, the carried generic-refusal notes and round 3:
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

    check()


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

    check()


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
        got = recorder().probe(name)
        assert got['outcome'] == want['outcome']
        assert [w for w in got['warnings'] if 'time column' in w[1]] == (
            guessed
        ), f'{name}: python-core warns {got["warnings"]}'

    check()
