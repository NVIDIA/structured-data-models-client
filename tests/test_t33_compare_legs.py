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

    def __init__(self, tmp: Path) -> None:
        case = json.loads(
            (ROOT / f'contract/cases/s1/{SOURCE}.json').read_text(
                encoding='utf-8'
            )
        )
        case.pop('core')
        case.update(id=CASE, slice='S2', warehouses=[LEG])
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
        line['case'] = f'{CASE}@{LEG}'
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
            [self.cases], self.reference, [self.observed], database=LEG
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
