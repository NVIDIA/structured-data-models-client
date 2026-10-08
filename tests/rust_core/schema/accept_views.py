# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.views: every view fixture and recorded view, plus 10,000 generated
metric views and 10,000 generated semantic views, give an equal spec and
equal messages in the Rust core and today's Python.

- Metric views, through ``metric_view_parse``: the definitions the engine's
  tests parse and the corpus's metric-view scenarios parse, each with and
  without trust, and generated ones (``_views.py``, seed 20261008), against
  ``parse_metric_view``: the spec, its messages, or the refusal.
- Semantic views, through ``schema_from_source`` over the stand-in warehouse
  (``_warehouse.py``): the YAML the engine's tests read and the corpus
  recorded, each with and without trust, and generated ones (seed 20261009),
  against today's ``Schema.from_source``: the outcome, its conversion
  messages included, and the statements asked.
  ``semantic_view_parse`` answers each of them without a panic.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest
import yaml
from _schema import difference, loads, native, python_outcome, rust_from_source
from _views import (
    YAML11,
    engine_test_views,
    is_yaml_error,
    json_of,
    metric_views,
    python_metric_view,
    recorded_views,
    semantic_views,
)
from _warehouse import Warehouse, source, tables_of

COUNT = 10_000
VIEW = 'SV'
QUALIFIED = f'TEST.PARITY_SHOP.{VIEW}'


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[Any]]:
    engine = engine_test_views(tmp_path_factory.mktemp('views'))
    recorded = recorded_views()
    return {
        'metric': engine['metric'] + recorded['metric'],
        'semantic': engine['semantic'] + recorded['semantic'],
        'recorded-metric': recorded['metric'],
        'recorded-semantic': recorded['semantic'],
    }


def test_the_view_fixtures_are_found(fixtures: dict[str, list[Any]]) -> None:
    assert len(fixtures['metric']) >= 40
    assert len(fixtures['semantic']) >= 8
    assert fixtures['recorded-metric'], 'no recorded metric view'
    assert fixtures['recorded-semantic'], 'no recorded semantic view'


# --- Metric views -------------------------------------------------------------------


def rust_metric_view(definition: str, trust: bool) -> dict[str, Any]:
    got = native('metric_view_parse').metric_view_parse(definition, trust)
    found = loads(got, 'metric_view_parse')
    assert isinstance(found, dict) and (set(found) in ({'spec'}, {'error'})), (
        f'metric_view_parse gave {str(found)[:200]}'
    )
    return found


def _compare_metric(views: list[tuple[str, bool]]) -> list[str]:
    failures = []
    for i, (definition, trust) in enumerate(views):
        want = json_of(python_metric_view(definition, trust))
        got = rust_metric_view(definition, trust)
        problem = difference(want, got)
        if problem:
            failures.append(
                f'view #{i} (trust={trust}): {problem}\n{definition}'
            )
    return failures


def test_metric_view_fixtures(fixtures: dict[str, list[Any]]) -> None:
    views = [
        (definition, trust)
        for definition, given in fixtures['metric']
        for trust in (given, not given)
    ]
    failures = _compare_metric(views)
    assert not failures, (
        f'{len(failures)} of {len(views)} differ:\n' + '\n'.join(failures[:20])
    )


def test_generated_metric_views() -> None:
    views = list(metric_views(COUNT))
    assert not [d for d, _ in views if is_yaml_error(d)], (
        'invalid YAML generated'
    )
    answers = Counter(
        next(iter(python_metric_view(d, t))) for d, t in views[:2000]
    )
    assert answers['spec'] >= 1200 and answers['error'] >= 200, answers
    spelled = Counter(s for s in YAML11 for d, _ in views if f' {s}\n' in d)
    assert len(spelled) >= len(YAML11) - 2, (
        f'YAML 1.1 spellings used: {spelled}'
    )
    assert any('\n  on:' in d or '\non:' in d or ' on:' in d for d, _ in views)
    failures = _compare_metric(views)
    assert not failures, f'{len(failures)} of {COUNT} differ:\n' + '\n'.join(
        failures[:20]
    )


# --- Semantic views ---------------------------------------------------------------


def _python_semantic(
    text: str, tables: dict[Any, Any], trust: bool
) -> tuple[dict[str, Any], list[Any]]:
    from sdm_client import Schema

    warehouse = Warehouse({QUALIFIED: text}, tables)
    model = source(warehouse).semantic_model(VIEW, trust_expressions=trust)
    outcome = python_outcome(lambda: Schema.from_source(model))
    assert not warehouse.unknown, warehouse.unknown
    return outcome, warehouse.log


def _rust_semantic(
    text: str, tables: dict[Any, Any], trust: bool
) -> tuple[dict[str, Any], list[Any]]:
    warehouse = Warehouse({QUALIFIED: text}, tables)
    model = source(warehouse).semantic_model(VIEW, trust_expressions=trust)
    outcome, asked = rust_from_source(model, {})
    return outcome, asked


def _statements(log: list[Any]) -> Counter[str]:
    r"""The statements of a log, as a multiset; no parameters and an empty
    list of them are the same statement."""
    return Counter(json.dumps([sql, params or None]) for sql, params in log)


def _compare_semantic(
    views: list[tuple[str, dict[Any, Any], bool]],
) -> list[str]:
    module = native('schema_from_source', 'semantic_view_parse')
    failures = []
    for i, (text, tables, trust) in enumerate(views):
        where = f'view #{i} (trust={trust})'
        want, said = _python_semantic(text, tables, trust)
        got, asked = _rust_semantic(text, tables, trust)
        problem = difference(want, got)
        if problem:
            failures.append(f'{where}: {problem}\n{text}')
        elif _statements(said) != _statements(asked):
            failures.append(f"{where}: other statements than Python's\n{text}")
        try:
            parsed = module.semantic_view_parse(text, VIEW)
        except Exception:  # noqa: BLE001, S110 - a refusal is allowed
            continue
        if not isinstance(parsed, str):
            failures.append(f'{where}: semantic_view_parse gave {type(parsed)}')
    return failures


def test_semantic_view_fixtures(fixtures: dict[str, list[Any]]) -> None:
    views = [
        (text, tables_of(yaml.safe_load(text)), trust)
        for text in fixtures['semantic']
        for trust in (False, True)
    ]
    failures = _compare_semantic(views)
    assert not failures, (
        f'{len(failures)} of {len(views)} differ:\n' + '\n'.join(failures[:20])
    )


def test_generated_semantic_views() -> None:
    views = list(semantic_views(COUNT))
    failures = _compare_semantic(views)
    assert not failures, f'{len(failures)} of {COUNT} differ:\n' + '\n'.join(
        failures[:20]
    )


def test_generated_semantic_views_read_and_refuse() -> None:
    r"""The generated views are not all refused: today's Python reads most of
    them, and drops parts of many with a message."""
    answers: Counter[str] = Counter()
    for text, tables, trust in semantic_views(1000):
        outcome, _ = _python_semantic(text, tables, trust)
        answers['error' if 'error' in outcome else 'schema'] += 1
        if outcome.get('warnings'):
            answers['messages'] += 1
        if 'schema' in outcome and outcome['schema']['relationships']:
            answers['links'] += 1
    assert answers['schema'] >= 800, answers
    assert answers['messages'] >= 300, answers
    assert answers['links'] >= 100, answers
    assert answers['error'] >= 20, answers


def test_the_stand_in_reads_a_recorded_view(
    fixtures: dict[str, list[Any]],
) -> None:
    text = fixtures['recorded-semantic'][0]
    outcome, _ = _python_semantic(text, tables_of(yaml.safe_load(text)), False)
    assert 'schema' in outcome, outcome
    assert outcome['schema']['relationships'], outcome
