# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A warehouse leg is compared like any database (the owner's decision
B-lite, T3.3 fix 4): against the reference line recorded on it, by every
rule, with no stand-in verdict; a missing statement or another result
fails, and nothing is set aside as live only."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'base.binary'
CASE = 't33.leg.binary'
LEG = 'databricks-sql'


def _compare() -> Any:
    name = 't33_compare_legs_compare'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'scripts/rust_core/compare.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Leg:
    r"""A warehouse-only copy of an S1 case, the reference line recorded on
    ``LEG`` and python-core's line equal to it, under ``tmp``."""

    def __init__(self, tmp: Path, database: str = LEG) -> None:
        self.database = database
        case = json.loads(
            (ROOT / f'contract/cases/s1/{SOURCE}.json').read_text(
                encoding='utf-8'
            )
        )
        case.pop('core')
        case.update(id=CASE, slice='S2', warehouses=[database])
        self.cases = tmp / 'cases'
        self.cases.mkdir()
        data = json.dumps(case, indent=2).encode()
        (self.cases / f'{CASE}.json').write_bytes(data)
        [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
        line = next(
            json.loads(text)
            for text in path.read_text(encoding='utf-8').splitlines()
            if json.loads(text)['case'] == SOURCE
        )
        line['case'] = f'{CASE}@{database}'
        line['identity']['inputs']['case'] = hashlib.sha256(data).hexdigest()
        self.reference = tmp / 'reference.jsonl'
        self.reference.write_text(json.dumps(line) + '\n', encoding='utf-8')
        core = copy.deepcopy(line)
        core['identity'].update(
            runner='python-core',
            reference=None,
            native={'sha256': '0' * 64},
            native_loaded=True,
            operations=[],
        )
        core['core'] = dict.fromkeys(
            ('accepted', 'described', 'materialized', 'predicted', 'error')
        ) | {'inputs': [], 'fetches': []}
        self.core = core
        self.observed = tmp / 'python-core.jsonl'

    def judge(self, line: dict[str, Any]) -> Any:
        self.observed.write_text(json.dumps(line) + '\n', encoding='utf-8')
        return _compare().compare(
            [self.cases],
            self.reference,
            [self.observed],
            database=self.database,
        )


@pytest.fixture
def leg(tmp_path: Path) -> Leg:
    return Leg(tmp_path)


def test_t33_an_equal_warehouse_line_passes_with_no_stand_in_verdict(
    leg: Leg,
) -> None:
    report = leg.judge(copy.deepcopy(leg.core))
    assert report.failures == []
    assert not hasattr(report, 'live_only')


def test_t33_a_missing_statement_on_a_warehouse_leg_fails(leg: Leg) -> None:
    line = copy.deepcopy(leg.core)
    statements = line['driver']['capture']['statements']
    assert statements, 'not vacuous: the reference line ran statements'
    statements.pop()
    line['driver']['query_log']['statements'] -= 1
    report = leg.judge(line)
    assert any(rule == 'sql.count' for rule, _ in report.failures), (
        report.failures
    )


def test_t33_another_result_on_a_warehouse_leg_fails(leg: Leg) -> None:
    line = copy.deepcopy(leg.core)
    line['public']['result']['digest'] = '0' * 64
    report = leg.judge(line)
    assert report.failures, 'a changed result digest must fail'


def test_t33_a_stand_in_verdict_is_no_longer_an_observation_field(
    leg: Leg,
) -> None:
    line = copy.deepcopy(leg.core)
    line['standin'] = {'live_only': 'B1', 'flagged': []}
    line['public']['result']['digest'] = '0' * 64
    report = leg.judge(line)
    # Not set aside as live only: the verdict is refused and the result
    # still differs.
    assert report.failures


def _provider_report(
    tmp: Path,
    database: str,
    provider: dict[str, Any],
    *,
    given: dict[str, Any] | None = None,
    pinned: dict[str, Any] | None = None,
    changed: tuple[str, Any] | None = None,
) -> Any:
    """Synthetic start arguments, judged through compare, not live evidence.

    Expected providers below are independent literals from the source contract.
    Keep the committed generic core specifications rather than learning them
    from the runner's inputs or deleting them as the older Leg fixture does.
    """
    case = json.loads((ROOT / f'contract/cases/s1/{SOURCE}.json').read_text())
    case.update(id=CASE, until='materialize')
    case['expect'] = {'outcome': 'success', 'stage': 'materialize'}
    case['core'].pop('predict')
    case['input'].update(given or {})
    if database in ('duckdb', 'sqlite', 'postgres'):
        case['databases'] = [database]
    else:
        case.update(slice='S2', warehouses=[database])
    if pinned is not None:
        for spec in case['core'].values():
            spec['provider'] = copy.deepcopy(pinned)
    cases = tmp / 'cases'
    cases.mkdir()
    data = json.dumps(case).encode()
    (cases / f'{CASE}.json').write_bytes(data)
    [reference] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    line = next(
        json.loads(text)
        for text in reference.read_text().splitlines()
        if json.loads(text)['case'] == SOURCE
    )
    line['case'] = CASE if database == 'duckdb' else f'{CASE}@{database}'
    line['identity'].update(
        runner='python-core',
        reference=None,
        native={'sha256': '0' * 64},
        native_loaded=True,
        operations=['describe_source', 'materialize'],
    )
    line['identity']['inputs']['case'] = hashlib.sha256(data).hexdigest()
    line['outcome'] = {
        'result': 'success',
        'stage': 'materialize',
        'error': None,
        'code': None,
    }
    line['public'].update(payloads=[], result=None)
    described = {'tables': [], 'links': []}
    starts = [
        {'op': op, **copy.deepcopy(spec), 'provider': copy.deepcopy(provider)}
        for op, spec in case['core'].items()
    ]
    starts[1].update(schema=described, query=case['input']['pql'])
    if changed is not None:
        field, value = changed
        if field.startswith('provider.'):
            name = field.removeprefix('provider.')
            for start in starts:
                if value is None:
                    start['provider'].pop(name, None)
                else:
                    start['provider'][name] = value
        else:
            starts[1][field] = value
    line['core'] = {
        'inputs': starts,
        'fetches': [{'request': {}, 'plan': {}}],
        'accepted': None,
        'described': described,
        'materialized': {
            'version': 1,
            'query': case['input']['pql'],
            'payload': {},
        },
        'predicted': None,
        'error': None,
    }
    observed = tmp / 'python-core.jsonl'
    observed.write_text(json.dumps(line) + '\n')
    return _compare().compare([cases], None, [observed], database=database)


@pytest.mark.parametrize(
    'database,given,provider',
    [
        ('duckdb', {}, {'provider': 'sql', 'dialect': 'duckdb'}),
        ('postgres', {}, {'provider': 'sql', 'dialect': 'postgres'}),
        (
            'lakebase',
            {},
            {
                'provider': 'sql',
                'dialect': 'postgres',
                'namespace': ['{schema}'],
            },
        ),
        (
            'snowflake',
            {},
            {
                'provider': 'sql',
                'dialect': 'snowflake',
                'namespace': ['{catalog}', '{schema}'],
            },
        ),
        (
            'databricks-sql',
            {},
            {
                'provider': 'sql',
                'dialect': 'databricks',
                'namespace': ['{catalog}', '{schema}'],
            },
        ),
        (
            'databricks-spark',
            {},
            {
                'provider': 'sql',
                'dialect': 'databricks',
                'namespace': ['{catalog}', '{schema}'],
            },
        ),
        (
            'snowflake',
            {'named_part': 'schema'},
            {
                'provider': 'sql',
                'dialect': 'snowflake',
                'namespace': [None, '{schema}'],
            },
        ),
        (
            'databricks-sql',
            {'named_part': 'catalog'},
            {
                'provider': 'sql',
                'dialect': 'databricks',
                'namespace': ['{catalog}', None],
            },
        ),
        (
            'lakebase',
            {'connect': 'lakebase'},
            {'provider': 'sql', 'dialect': 'postgres'},
        ),
        (
            'snowflake',
            {'connect': 'snowpark'},
            {'provider': 'sql', 'dialect': 'snowflake'},
        ),
    ],
)
def test_generic_provider_inputs_follow_declared_source_names(
    tmp_path: Path,
    database: str,
    given: dict[str, Any],
    provider: dict[str, Any],
) -> None:
    report = _provider_report(tmp_path, database, provider, given=given)
    assert report.failures == [], report.texts


@pytest.mark.parametrize(
    'changed',
    [
        ('provider.dialect', 'lakebase'),
        ('provider.dialect', None),
        ('provider.namespace', None),
        ('provider.namespace', ['wrong_schema']),
        ('provider.namespace', ['{catalog}', '{schema}']),
        ('indices', [{'t': 'i64', 'v': '999'}]),
        ('anchor_time', None),
        ('query', 'PREDICT users.age FOR EACH users.user_id'),
    ],
)
def test_provider_adaptation_does_not_copy_or_excuse_changed_inputs(
    tmp_path: Path, changed: tuple[str, Any]
) -> None:
    report = _provider_report(
        tmp_path,
        'lakebase',
        {'provider': 'sql', 'dialect': 'postgres', 'namespace': ['{schema}']},
        changed=changed,
    )
    assert not any('runner failure' in text for _, text in report.failures), (
        report.texts
    )
    assert any(
        rule == 'inputs' and 'core.inputs' in text
        for rule, text in report.failures
    ), report.texts


def test_explicit_provider_and_view_fields_are_not_rewritten(
    tmp_path: Path,
) -> None:
    provider = {
        'provider': 'sql',
        'dialect': 'snowflake',
        'namespace': ['pinned_catalog', 'pinned_schema'],
        'view': {'name': 'pinned_view', 'trust_expressions': False},
    }
    report = _provider_report(tmp_path, 'snowflake', provider, pinned=provider)
    assert report.failures == [], report.texts


def _with_unused_count(
    leg: Leg, *, suffix: str = '', parameters: Any = None
) -> None:
    """Seed the exact existing KO2-58 policy shape in this private pair."""
    reference = json.loads(leg.reference.read_text())
    statements = reference['driver']['capture']['statements']
    count = copy.deepcopy(
        next(s for s in statements if s['stage'] == 'materialize')
    )
    count.update(
        sql='SELECT COUNT(*) FROM "owned"."users"' + suffix,
        parameters=parameters,
    )
    statements.insert(
        next(
            i for i, s in enumerate(statements) if s['stage'] == 'materialize'
        ),
        count,
    )
    reference['driver']['query_log']['statements'] += 1
    leg.reference.write_text(json.dumps(reference) + '\n')


def test_lakebase_uses_only_the_existing_unused_postgres_count_policy(
    tmp_path: Path,
) -> None:
    leg = Leg(tmp_path, 'lakebase')
    _with_unused_count(leg)
    report = leg.judge(copy.deepcopy(leg.core))
    assert report.failures == [], report.texts


@pytest.mark.parametrize(
    'suffix,parameters',
    [
        (' WHERE user_id > 0', None),
        (' UNION ALL SELECT 1', None),
        ('', [{'type': 'str', 'value': 'bound'}]),
    ],
)
def test_lakebase_does_not_filter_other_or_parameterized_counts(
    tmp_path: Path,
    suffix: str,
    parameters: Any,
) -> None:
    leg = Leg(tmp_path, 'lakebase')
    _with_unused_count(leg, suffix=suffix, parameters=parameters)
    report = leg.judge(copy.deepcopy(leg.core))
    assert any(rule == 'sql.count' for rule, _ in report.failures), report.texts


def test_other_warehouse_does_not_inherit_postgres_unused_read_policy(
    tmp_path: Path,
) -> None:
    leg = Leg(tmp_path)
    _with_unused_count(leg)
    report = leg.judge(copy.deepcopy(leg.core))
    assert any(rule == 'sql.count' for rule, _ in report.failures), report.texts


@pytest.mark.parametrize('mutation', ['missing', 'changed_result'])
def test_lakebase_unused_count_policy_keeps_every_real_read_strict(
    tmp_path: Path,
    mutation: str,
) -> None:
    leg = Leg(tmp_path, 'lakebase')
    _with_unused_count(leg)
    line = copy.deepcopy(leg.core)
    statements = line['driver']['capture']['statements']
    at = next(
        i
        for i, s in enumerate(statements)
        if s['stage'] == 'materialize' and s['results']
    )
    if mutation == 'missing':
        statements.pop(at)
        line['driver']['query_log']['statements'] -= 1
        expected = 'sql.count'
    else:
        values = statements[at]['results'][0]['values']
        assert values['ordered'] != '0' * 64 and values['unordered'] != '0' * 64
        values.update(ordered='0' * 64, unordered='0' * 64)
        expected = 'sql.result'
    report = leg.judge(line)
    assert any(rule == expected for rule, _ in report.failures), report.texts
