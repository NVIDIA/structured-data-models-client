# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""compare.py's fixes of T3.2's first review round (Codex r1, 4 to 6):

- a refusal on a warehouse leg fails on the outcome, as on any database
  (the live-only verdict it once covered went with the stand-in, B-lite);
- Java's prediction payload is held to the reference as rust-host's is, and
  Java agrees with python-core on the core fields;
- live H9 admits the warehouse drivers' reads the recording mode records,
  judged by their own evidence (the Arrow schema, or the rows);
- a line whose typed result digest is excused and which holds no text
  digest fails until recorded again (Claude P2-4).
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
    'code': 'INVALID_REQUEST',
}


# --- 4: a refusal on a warehouse leg fails as anywhere -----------------------------


def test_a_refusal_on_a_warehouse_leg_fails(tmp_path: Path) -> None:
    r"""A warehouse leg is judged against its recorded reference by every
    rule (B-lite): a refusal where the old client answered fails on the
    outcome."""
    pair = LIVE.Pair(tmp_path, 'databricks-sql')
    line = pair.answered(None)
    line['outcome'] = dict(REFUSED)
    line['public'] = {'schema': None, 'payloads': [], 'result': None}
    report = pair.judge(line)
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


def _numeric_core_agreement(
    field: str,
    mine: Any,
    theirs: Any,
    *,
    shared: bool = False,
    state_part: str | None = None,
    held: bool = False,
) -> Any:
    """Compare captured numeric tokens without parsing them through floats."""
    core: dict[str, Any] = {
        'inputs': [{'op': 'predict', 'state': {'payload': mine}}],
        'fetches': [],
        'described': None,
        'materialized': None,
        'predicted': None,
        'accepted': None,
        'error': None,
    }
    state_key = 'held' if held else 'state'
    if field == 'inputs':
        state = core['inputs'][0].pop('state')
        if state_part == 'ranking':
            state = {'payload': None, 'ranking': {'parts': [state]}}
        elif state_part is not None:
            state = {'payload': None, state_part: [state]}
        core['inputs'][0][state_key] = state
    elif field == 'materialized':
        core['inputs'] = []
        core['materialized'] = {'payload': mine}
    else:
        core['inputs'] = []
        core['predicted'] = {'body': {'value': mine}}
        if field == 'calls':
            core['predicted']['body'] = {'value': None}
            core['predicted']['calls'] = [{'value': mine}]
    python = {
        'identity': {'operations': ['predict'], 'contract': 30},
        'core': core,
    }
    rust = copy.deepcopy(python)
    if field == 'inputs':
        state = rust['core']['inputs'][0][state_key]
        if state_part == 'ranking':
            state = state['ranking']['parts'][0]
        elif state_part is not None:
            state = state[state_part][0]
        state['payload'] = theirs
    elif field == 'materialized':
        rust['core']['materialized']['payload'] = theirs
    elif field == 'calls':
        rust['core']['predicted']['calls'][0]['value'] = theirs
    else:
        rust['core']['predicted']['body']['value'] = theirs
    if shared:
        rust['identity']['operations'].append('predict')
        rust['core']['inputs'].append({'op': 'predict', 'state': {}})
    report = COMPARE.Report()
    COMPARE._core_agree(
        report,
        {'id': 'numeric-core'},
        python,
        rust,
        stop='predict' if shared else None,
    )
    return report


@pytest.mark.parametrize(
    'field', ['inputs', 'materialized', 'predicted', 'calls']
)
@pytest.mark.parametrize(
    ('mine', 'theirs'),
    [('-3e-06', '-3e-6'), ('1.000e-5', '0.00001'), ('-0.00', '-0e1')],
)
def test_core_fields_agree_on_exact_float_values_with_different_spelling(
    field: str, mine: str, theirs: str
) -> None:
    report = _numeric_core_agreement(
        field, {'number': mine}, {'number': theirs}
    )
    assert report.failures == [], report.texts


@pytest.mark.parametrize(
    'field', ['inputs', 'materialized', 'predicted', 'calls']
)
@pytest.mark.parametrize(
    ('mine', 'theirs'),
    [
        ({'number': '1'}, {'number': '1.0'}),
        ({'number': '-0.0'}, {'number': '0e0'}),
        ({'number': '1e-6'}, {'number': '1.0000000000000001e-6'}),
        ({'number': '9007199254740992.0'}, {'number': '9007199254740993.0'}),
        ('1e-06', '1e-6'),
    ],
)
def test_core_numeric_agreement_keeps_kind_sign_exact_value_and_strings(
    field: str, mine: Any, theirs: Any
) -> None:
    report = _numeric_core_agreement(field, mine, theirs)
    assert [rule for rule, _ in report.failures] == ['core'], report.texts


@pytest.mark.parametrize(
    ('mine', 'theirs', 'agrees'),
    [('3e-06', '3e-6', True), ('1', '1.0', False)],
)
def test_shared_core_inputs_use_the_same_lossless_numeric_rule(
    mine: str, theirs: str, agrees: bool
) -> None:
    report = _numeric_core_agreement(
        'inputs', {'number': mine}, {'number': theirs}, shared=True
    )
    assert (report.failures == []) is agrees, report.texts


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('part', [None, 'forecast', 'batches', 'ranking'])
def test_later_and_held_request_payloads_keep_exact_numeric_agreement(
    part: str | None, held: bool
) -> None:
    for theirs, agrees in [('3e-6', True), ('3.000000000000001e-6', False)]:
        report = _numeric_core_agreement(
            'inputs',
            {'number': '3e-06'},
            {'number': theirs},
            state_part=part,
            held=held,
        )
        assert (report.failures == []) is agrees, report.texts


@pytest.mark.parametrize('field', ['source', 'state', 'metadata'])
def test_numeric_payload_rule_does_not_rewrite_other_core_maps(
    field: str,
) -> None:
    python = {
        'identity': {'operations': ['predict'], 'contract': 30},
        'core': {
            'inputs': [],
            'fetches': [],
            'described': None,
            'materialized': None,
            'predicted': None,
            'accepted': None,
            'error': None,
        },
    }
    before, after = {'number': '1e-06'}, {'number': '1e-6'}
    if field == 'metadata':
        python['core']['predicted'] = {'metadata': before}
    else:
        python['core']['inputs'] = [
            {'op': 'predict', field: {'primary_keys': before}}
        ]
    rust = copy.deepcopy(python)
    if field == 'metadata':
        rust['core']['predicted']['metadata'] = after
    else:
        rust['core']['inputs'][0][field]['primary_keys'] = after
    report = COMPARE.Report()
    COMPARE._core_agree(report, {'id': 'numeric-map'}, python, rust)
    assert [rule for rule, _ in report.failures] == ['core'], report.texts


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
