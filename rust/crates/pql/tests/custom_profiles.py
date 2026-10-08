# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Record ``custom_profiles.json``: today's Python answer for queries
validated with capabilities and validator options that none of the five
named profiles has, for ``custom_profiles.rs``.

The PQL corpus names its profile, so it can only reach the five that
``predictive_query`` defines. But ``Profile`` and ``Capabilities`` are
public, and so is ``PredictiveQueryValidator`` with its
``allow_array_targets`` and ``allow_timestamp_arrays`` options, so a caller
can reach checks that no named profile does. This table covers them.

``profiles`` rows run ``predictive_query.validate`` with a custom
``Profile`` over probe queries: each capability turned off on its own, and a
model that declares only ``temporal`` and ``static``, in inference mode; and
every capability and none, in both modes. ``options`` rows run
``PredictiveQueryValidator(schema, mode, allow_array_targets=...,
allow_timestamp_arrays=...)`` on the parsed query, with default
capabilities. Each answer is recorded as the corpus records one
(``tests/parity/test_pql_corpus.py``'s ``resolve``).

Run from the repository root with the repository's packages installed:
``python rust/crates/pql/tests/custom_profiles.py [--check]``. ``--check``
writes nothing and fails if the stored table differs from what today's
Python gives.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TABLE = HERE / 'custom_profiles.json'

sys.path.insert(0, str(REPO / 'tests' / 'rust_core' / 'pql'))

from _pql import corpus_module, schemas  # noqa: E402

CAPABILITIES = (
    'relationships',
    'multiple_tables',
    'temporal',
    'static',
    'entity_filters',
    'counterfactuals',
    'derived_targets',
    'link_prediction',
    'forecasting',
    'key_columns_in_filters',
    'timestamp_columns_in_filters',
)

USERS = 'FOR EACH users.user_id'

#: Queries over the corpus's shop schema, each touching what at least one
#: capability governs. None has a date literal, so no answer depends on the
#: platform's ``strftime`` (``DATES.md``).
PROBES = {
    'static-column': f'PREDICT users.age {USERS}',
    'static-condition': f'PREDICT users.age > 30 {USERS}',
    'count': f'PREDICT COUNT(orders.*, 0, 30, days) {USERS}',
    'count-condition': f'PREDICT COUNT(orders.*, 0, 30, days) > 2 {USERS}',
    'assuming': f'PREDICT SUM(orders.amount, 0, 30, days) {USERS} '
    'ASSUMING COUNT(orders.*, 0, 7, days) > 2',
    'forecast': 'PREDICT SUM(orders.amount, 0, 1, days) FORECAST 5 '
    f'TIMEFRAMES {USERS}',
    'key-in-target-filter': 'PREDICT COUNT(orders.* WHERE orders.product_id '
    f'= 3, 0, 30, days) {USERS}',
    'key-in-entity-filter': f'PREDICT users.age {USERS} WHERE users.user_id '
    '> 3',
    'timestamp-in-filter': 'PREDICT COUNT(orders.* WHERE orders.ordered_at '
    f'IS NOT NULL, 0, 30, days) {USERS}',
    'entity-filter': f'PREDICT COUNT(orders.*, 0, 30, days) {USERS} WHERE '
    'users.age > 30',
    'entity-filter-aggregation': 'PREDICT FIRST(orders.status, 0, 30, days) '
    f'{USERS} WHERE COUNT(reviews.*, -30, 0, days) > 1',
    'rank-foreign-key': 'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, '
    f'days) RANK TOP 5 {USERS}',
    'entity-ids': 'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id '
    'IN (1, 2)',
    'single-table': 'PREDICT orders.amount FOR EACH orders.order_id',
}

#: Capability sets, by the id rows use: what to start from (every
#: capability, or none) and which to set.
VARIANTS: dict[str, dict[str, Any]] = {
    'all': {'base': 'all', 'set': {}},
    'none': {'base': 'nothing', 'set': {}},
    **{
        f'no-{name}': {'base': 'all', 'set': {name: False}}
        for name in CAPABILITIES
    },
    'only-temporal-and-static': {
        'base': 'nothing',
        'set': {'temporal': True, 'static': True},
    },
}

#: Queries for the validator's options: ``LIST_DISTINCT`` of a timestamp, an
#: ID and a category, as a target and inside one.
OPTION_PROBES = {
    'timestamps': 'PREDICT LIST_DISTINCT(orders.ordered_at, 0, 30, days) '
    f'{USERS}',
    'timestamps-ranked': 'PREDICT LIST_DISTINCT(orders.ordered_at, 0, 30, '
    f'days) RANK TOP 3 {USERS}',
    'ids-ranked': 'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) '
    f'RANK TOP 5 {USERS}',
    'ids': f'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) {USERS}',
    'categories': f'PREDICT LIST_DISTINCT(orders.status, 0, 30, days) {USERS}',
    'categories-classified': 'PREDICT LIST_DISTINCT(orders.status, 0, 30, '
    f'days) CLASSIFY {USERS}',
    'compared': 'PREDICT LIST_DISTINCT(orders.status, 0, 30, days) = '
    f"'shipped' {USERS}",
}

MODEL = 'a custom model'
MODES = ('INFERENCE', 'TRAINING')


@contextlib.contextmanager
def _on_package(name: str, value: Any) -> Iterator[None]:
    r"""``predictive_query.<name>`` is ``value`` until the block ends, then
    what it was, or nothing."""
    import predictive_query

    missing = object()
    original = getattr(predictive_query, name, missing)
    setattr(predictive_query, name, value)
    try:
        yield
    finally:
        if original is missing:
            delattr(predictive_query, name)
        else:
            setattr(predictive_query, name, original)


def _answer(query: str, schema: dict[str, Any], profile: Any) -> Any:
    r"""What ``resolve`` records for ``query`` under ``profile``, which it
    looks up by name on the package."""
    with _on_package('CUSTOM_PROFILE', profile):
        got = corpus_module().resolve(
            {'query': query, 'schema': schema, 'profile': 'CUSTOM_PROFILE'}
        )
    return json.loads(json.dumps(got))


def capabilities(variant: dict[str, Any]) -> Any:
    from predictive_query import Capabilities

    if variant['base'] == 'all':
        base = Capabilities(model=MODEL)
    else:
        base = Capabilities.nothing(MODEL)
    return base.also(**variant['set'])


def profile_answer(
    query: str, schema: dict[str, Any], mode: str, variant: dict[str, Any]
) -> Any:
    from predictive_query import Mode, Profile

    profile = Profile(mode=Mode(mode), capabilities=capabilities(variant))
    return _answer(query, schema, profile)


def option_answer(
    query: str,
    schema: dict[str, Any],
    mode: str,
    allow_array_targets: bool,
    allow_timestamp_arrays: bool,
) -> Any:
    r"""``PredictiveQueryValidator`` with the options, on the parsed query,
    in place of ``predictive_query.validate``."""
    import predictive_query
    from predictive_query import Mode
    from predictive_query.parser.parser import PQLParser
    from predictive_query.validator.predictive_query_validator import (
        PredictiveQueryValidator,
    )

    def validate(text: str, schema: Any, profile: Any) -> Any:
        parsed = PQLParser(mode=profile.mode).to_parsed_predictive_query(text)
        resolved, response = PredictiveQueryValidator(
            schema,
            mode=profile.mode,
            allow_array_targets=allow_array_targets,
            allow_timestamp_arrays=allow_timestamp_arrays,
        ).validate_predictive_query(parsed)
        if resolved is None or not response.ok:
            return None, response
        return resolved, response

    profile = predictive_query.Profile(mode=Mode(mode))
    with _on_package('validate', validate):
        return _answer(query, schema, profile)


def record() -> dict[str, Any]:
    shop = schemas()[0]
    assert sorted(shop['tables']) == ['orders', 'products', 'reviews', 'users']
    profiles = []
    for variant_id, variant in VARIANTS.items():
        for mode in MODES if variant_id in ('all', 'none') else MODES[:1]:
            for probe_id, query in PROBES.items():
                profiles.append(
                    {
                        'id': f'{variant_id}/{mode.lower()}/{probe_id}',
                        'capabilities': variant_id,
                        'mode': mode,
                        'query': query,
                        'python': profile_answer(query, shop, mode, variant),
                    }
                )
    options = []
    for arrays in (False, True):
        for timestamps in (True, False):
            for mode in MODES:
                for probe_id, query in OPTION_PROBES.items():
                    options.append(
                        {
                            'id': f'arrays-{arrays}/timestamps-{timestamps}/'
                            f'{mode.lower()}/{probe_id}',
                            'allow_array_targets': arrays,
                            'allow_timestamp_arrays': timestamps,
                            'mode': mode,
                            'query': query,
                            'python': option_answer(
                                query, shop, mode, arrays, timestamps
                            ),
                        }
                    )
    for row in profiles + options:
        assert 'exception' not in row['python'], row
    ids = [row['id'] for row in profiles + options]
    assert len(ids) == len(set(ids)), 'row ids are not unique'
    return {
        'model': MODEL,
        'capabilities': VARIANTS,
        'schema': shop,
        'profiles': profiles,
        'options': options,
    }


def render(table: dict[str, Any]) -> str:
    return json.dumps(table, indent=1, ensure_ascii=False) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = render(record())
    if args.check:
        if not TABLE.exists() or TABLE.read_text(encoding='utf-8') != text:
            print(
                f"{TABLE} is not what today's Python gives; rerun without "
                '--check',
                file=sys.stderr,
            )
            return 1
        print(f'{TABLE.name}: up to date')
        return 0
    TABLE.write_text(text, encoding='utf-8')
    print(f'wrote {TABLE}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
