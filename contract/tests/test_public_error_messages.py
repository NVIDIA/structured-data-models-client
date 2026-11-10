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
    'sample.num-hops.text': ['num_hops', "'2'"],
    'sample.num-hops.zero': ['num_hops', 'num_neighbors=[8, 8]'],
    'sample.num-neighbors.empty': ['num_neighbors', 'no hop'],
    'sample.num-neighbors.number': ['num_neighbors', 'sequence'],
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


def _named_keys() -> list[str]:
    """Every authored diagnostic case on each supported database."""
    return [case_key for case_id in NAMED for case_key in keys_of(case_id)]


NAMED_KEYS = tuple(_named_keys())
WARNED_KEYS = tuple(
    (label, case_key)
    for label, case_ids in WARNED.items()
    for case_id in case_ids
    for case_key in keys_of(case_id)
)


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
def test_public_refusal_names_the_invalid_input(case_key: str) -> None:
    """Errors identify the supplied table, column or value, not just an error code."""
    if _skipped(case_key):
        pytest.skip('Postgres test DSN is not configured')
    case_id = case_key.partition('@')[0]
    case, _ = cases()[case_id]
    expect = case['expect']
    got = core_observed(case_key)
    expected = (
        expect['outcome'],
        expect['stage'],
        expect.get('error'),
        expect.get('code'),
    )
    # SQLite reads decimal source columns as strings, so type validation
    # precedes target-class validation. Wide keys fail at its integer boundary.
    required_words = NAMED[case_id]
    if case_key.endswith('@sqlite') and case_id.startswith('decimal.compare-'):
        expected = (
            'refusal',
            'materialize',
            'StructuredDataError',
            'INVALID_REQUEST',
        )
        required_words = ['users.d', 'type string', 'constant', 'quotes']
    elif case_key == 'key.uint64-above-int64-max@sqlite':
        # The maintained SQLite path now handles this input successfully.
        assert _outcome(got['outcome']) == ('success', 'predict', None, None)
        return
    elif case_key == 'key.uint64-above-int64-max@postgres':
        expected = ('refusal', 'from_source', 'OverflowError', None)
        required_words = ['int']
    assert same_outcome(expected, _outcome(got['outcome'])), case_key
    said = _words(got['message'])
    assert not said.startswith(GENERIC), (
        f'{case_key}: generic message {said!r:.300}'
    )
    missing = [word for word in required_words if word not in said]
    assert not missing, f'{case_key}: missing {missing}: {said!r:.300}'


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
def test_public_warnings_name_source_bounds_and_index_requirements(
    label: str, case_key: str
) -> None:
    """Warn with the source bounds and index requirements that explain the issue."""
    if _skipped(case_key):
        pytest.skip('Postgres test DSN is not configured')
    case_id = case_key.partition('@')[0]
    case, _ = cases()[case_id]
    expect = case['expect']
    got = core_observed(case_key)
    expected = (
        expect['outcome'],
        expect['stage'],
        expect.get('error'),
        expect.get('code'),
    )
    assert same_outcome(expected, _outcome(got['outcome'])), case_key
    warnings = _labelled(label, got['warnings'])
    postgres = case_key.endswith('@postgres')
    count = (
        0 if postgres and label in ('missing-indices', 'runtime-warning') else 1
    )
    assert len(warnings) == count, (case_key, label, got['warnings'])
    if not count:
        return
    message = warnings[0][1]
    if label == 'context-before-data':
        bounds = (
            ('2023-12-03', '2024-01-01')
            if case_id == 'shape.star.customers'
            else ('2022-12-31', '2023-01-01')
        )
        assert all(bound in message for bound in bounds), message
    elif label == 'anchor-after-data':
        assert '2030-01-01' in message and '2024-07-07' in message
    elif label == 'missing-indices':
        counts = ('7', '4') if case_id == 'shape.star.customers' else ('3', '2')
        assert (
            f'Missing {counts[0]} indices in {counts[1]} tables' in message
        ), message
        assert 'primary' in message and 'foreign' in message, message
    else:
        assert message == 'overflow encountered in cast'


@pytest.mark.parametrize('name', sorted(INFINITY_SHOWN))
def test_infinite_timestamp_bounds_have_actionable_warning(name: str) -> None:
    """Full-column bounds, including infinities beyond preview rows, are named."""
    got = recorder().probe(name)
    assert got['outcome'] == {
        'result': 'success',
        'stage': 'from_source',
        'error': None,
        'code': None,
    }
    guessed = [
        warning for warning in got['warnings'] if 'time column' in warning[1]
    ]
    assert len(guessed) == 1
    assert guessed[0][0] == 'SDMWarning'
    assert INFINITY_SHOWN[name] in guessed[0][1]


def test_diagnostic_cases_and_connector_selectors_are_complete() -> None:
    assert set(NAMED) | {
        case for cases in WARNED.values() for case in cases
    } == set(recorder().CASE_IDS)
    assert set(NAMED_KEYS) == {key for case in NAMED for key in keys_of(case)}
    assert set(LABELS) == set(WARNED)
    assert all(case in cases() for case in recorder().CASE_IDS)
