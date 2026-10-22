# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""compare.py's fixes of T3.2's first review round (Codex r1, 4 to 6):

- a live-only verdict reports every rule's difference of the runner's
  comparison, the outcome's included, not only those after the host fields;
- Java's prediction payload is held to the reference as rust-host's is, and
  Java agrees with python-core on the core fields;
- live H9 admits the warehouse drivers' reads the recording mode records,
  judged by their own evidence (the Arrow schema, or the rows);
- a live-only item is still held to its inputs, route and reach (Claude
  P2-3), and a line whose typed result digest is excused and which holds no
  text digest fails until recorded again (Claude P2-4).
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't32_r1_compare')
LIVE = _module(ROOT / 'tests/test_t32_live_only.py', 't32_r1_live_fixture')
#: What run_python.py's warehouse captures record a read as.
ARROW_METHODS = (
    'fetchall_arrow',
    'fetchmany_arrow',
    'fetch_arrow_all',
    'fetch_pandas_all',
    'toArrow',
    'toPandas',
)
ROW_METHODS = ('collect', 'first', 'stream', 'copy')
REFUSED = {
    'result': 'refusal',
    'stage': 'from_source',
    'error': 'StructuredDataError',
    'code': 'LIVE_ONLY',
}


# --- 4: a live-only verdict covers the outcome --------------------------------------


def _refused(pair: Any, standin: dict[str, Any]) -> Any:
    line = pair.answered(standin)
    line['outcome'] = dict(REFUSED)
    line['public'] = {'schema': None, 'payloads': [], 'result': None}
    return pair.judge(line)


@pytest.mark.parametrize(
    'standin',
    [
        {'live_only': 'B1', 'flagged': []},
        {'live_only': 'B2', 'flagged': []},
        {'live_only': None, 'flagged': ['B4']},
    ],
)
def test_a_live_only_refusal_leaves_no_blocking_difference(
    tmp_path: Path, standin: dict[str, Any]
) -> None:
    pair = LIVE.Pair(tmp_path, 'databricks-sql')
    report = _refused(pair, standin)
    assert pair.key in report.live_only, report.texts
    assert [r for r, _ in report.failures if r is not None] == [], report.texts


def test_an_unflagged_refusal_still_fails(tmp_path: Path) -> None:
    pair = LIVE.Pair(tmp_path, 'databricks-sql')
    report = _refused(pair, {'live_only': None, 'flagged': []})
    assert not report.live_only
    assert 'outcome' in [r for r, _ in report.failures], report.texts


# --- 5: Java's payload and core fields ---------------------------------------------


@pytest.mark.parametrize('who', ['rust-host', 'java'])
def test_a_core_host_s_changed_payload_fails(who: str) -> None:
    report = COMPARE.Report()
    COMPARE._core_payload(
        report,
        {'id': 'probe'},
        who,
        {'core': {'predicted': {'body': {'value': 999}}}},
        {'public': {'payloads': [{'parsed': {'value': 1}}]}},
    )
    assert [rule for rule, _ in report.failures] == ['payload'], report.texts
    assert who in report.failures[0][1]


def test_java_agrees_with_python_core_on_the_core_fields() -> None:
    core = {
        'inputs': [],
        'fetches': [],
        'described': None,
        'materialized': None,
        'predicted': {'body': {'value': 1}},
        'accepted': None,
        'error': None,
    }
    python = {
        'identity': {'operations': ['predict'], 'contract': 1},
        'core': core,
    }
    java = copy.deepcopy(python)
    java['core']['predicted'] = {'body': {'value': 999}}
    report = COMPARE.Report()
    COMPARE._core_agree(report, {'id': 'probe'}, python, java, host='java')
    assert [rule for rule, _ in report.failures] == ['core'], report.texts
    assert 'python-core against java' in report.failures[0][1]
    # compare() pairs Java's observation with python-core's.
    assert "observed.get('java'" in inspect.getsource(COMPARE.compare)


# --- 6: the warehouse drivers' reads -------------------------------------------------


def _arrow_read(method: str, kind: str = 'int64') -> dict[str, Any]:
    return {
        'error': None,
        'results': [
            {
                'method': method,
                'rows': 2,
                'digest': 'd',
                'schema': [{'name': 'user_id', 'type': kind}],
                'values': {
                    'columns': ['user_id'],
                    'types': None,
                    'unencodable': [],
                    'ordered': 'o',
                    'unordered': 'u',
                },
            }
        ],
    }


def _row_read(method: str) -> dict[str, Any]:
    read = _arrow_read(method)
    del read['results'][0]['schema']
    return read


@pytest.mark.parametrize('method', ARROW_METHODS)
def test_equal_warehouse_arrow_reads_compare_clean_and_a_type_change_fails(
    method: str,
) -> None:
    report = COMPARE.Report()
    read = _arrow_read(method)
    COMPARE._results(
        report, 'at', read, copy.deepcopy(read), False,
        ('python-core', 'python-ref'), database='snowflake',
    )  # fmt: skip
    assert report.failures == [], report.texts
    report = COMPARE.Report()
    COMPARE._results(
        report, 'at', _arrow_read(method, 'int32'), read, False,
        ('python-core', 'python-ref'), database='snowflake',
    )  # fmt: skip
    assert [text for _, text in report.failures] == [
        'at result 0: Arrow schema differs'
    ]


@pytest.mark.parametrize('method', ROW_METHODS)
def test_equal_warehouse_row_reads_compare_clean_and_changed_rows_fail(
    method: str,
) -> None:
    report = COMPARE.Report()
    read = _row_read(method)
    COMPARE._results(
        report, 'at', read, copy.deepcopy(read), False,
        ('python-core', 'python-ref'), database='databricks-spark',
    )  # fmt: skip
    assert report.failures == [], report.texts
    other = _row_read(method)
    other['results'][0]['values']['unordered'] = 'v'
    report = COMPARE.Report()
    COMPARE._results(
        report, 'at', other, read, False,
        ('python-core', 'python-ref'), database='databricks-spark',
    )  # fmt: skip
    assert [r for r, _ in report.failures] == ['sql.result'], report.texts


def test_a_warehouse_arrow_read_without_its_schema_fails() -> None:
    read = _row_read('fetch_arrow_all')
    report = COMPARE.Report()
    COMPARE._results(
        report, 'at', read, copy.deepcopy(read), False,
        ('python-core', 'python-ref'), database='snowflake',
    )  # fmt: skip
    assert any('no Arrow schema' in text for _, text in report.failures)


# --- Claude P2-3: a live-only item keeps its inputs, route and reach ----------------


def test_a_live_only_item_with_a_stale_input_still_fails(
    tmp_path: Path,
) -> None:
    pair = LIVE.Pair(tmp_path, 'databricks-sql')
    line = pair.answered({'live_only': 'B1', 'flagged': []})
    line['outcome'] = dict(REFUSED)
    line['public'] = {'schema': None, 'payloads': [], 'result': None}
    line['identity']['inputs']['case'] = '0' * 64
    report = pair.judge(line)
    assert pair.key in report.live_only, report.texts
    assert 'inputs' in [r for r, _ in report.failures], report.texts
    assert 'outcome' not in [r for r, _ in report.failures], report.texts


# --- Claude P2-4: an excused typed digest needs the text digest ----------------------


def _public(digest: str, text: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {'columns': [], 'rows': 1, 'digest': digest}
    if text is not None:
        result['text_digest'] = text
    return {
        'public': {'schema': None, 'payloads': [], 'result': result},
        'host': {'warnings': []},
    }


@pytest.mark.parametrize('excused', [True, False])
def test_an_excused_digest_with_no_text_digest_fails_and_an_ordinary_line_passes(
    excused: bool,
) -> None:
    report = COMPARE.Report({'result': 'gate', 'warnings': 'gate'})
    if excused:
        report.more_exact = {
            ('k@postgres', 'python-core'): ('public.result/digest',)
        }
    COMPARE._host_fields(
        report,
        {'id': 'k@postgres'},
        'python-core',
        _public('a' * 64 if excused else 'b' * 64, 'c' * 64),
        _public('b' * 64, None),
    )
    rules = [rule for rule, _ in report.failures]
    assert rules == (['result'] if excused else []), report.texts
