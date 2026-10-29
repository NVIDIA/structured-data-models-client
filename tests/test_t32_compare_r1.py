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
import hashlib
import importlib.util
import inspect
import json
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


def _append_legacy_prediction(
    line: dict[str, Any], value: Any, *, calls: bool = False
) -> None:
    """Finite authored Predict evidence, with unread Arrow/reply left absent.

    Each body has one numeric-token or string cell, so its original JSON
    bytes can be written independently of the observation snapshot codec.
    These are comparator fixtures, not records of an executed SDK call.
    """
    core = line['core']
    run = len(line['identity']['operations'])
    line['identity']['operations'].append('predict')
    if type(value) is int:
        value = {'number': str(value)}
    values = [{'number': '0.0'}, value] if calls else [value]
    bodies = [{'value': copy.deepcopy(item)} for item in values]
    for body, item in zip(bodies, values, strict=True):
        token = (
            item['number']
            if isinstance(item, dict) and set(item) == {'number'}
            else json.dumps(item)
        )
        raw = ('{"value":' + token + '}').encode()
        wire_hash = hashlib.sha256(raw).hexdigest()
        core['model_calls'].append(
            {
                'run': run,
                'body': body,
                'body_sha256': wire_hash,
                'reply': None,
                'model_endpoint': None,
            }
        )
        line['public']['payloads'].append(
            {'parsed': copy.deepcopy(body), 'sha256': wire_hash}
        )
    state = {
        'version': 1,
        'query': 'PREDICT users.age FOR EACH users.user_id',
        'payload': copy.deepcopy(bodies[0]),
    }
    core['inputs'].append(
        {'op': 'predict', 'model': 'kumo-relational', 'state': state}
    )
    product = {
        'body': copy.deepcopy(bodies[-1]),
        **({'calls': copy.deepcopy(bodies)} if calls else {}),
        'model': 'kumo-relational',
        'request_id': f'numeric-{run}',
        'metadata': None,
        'result': None,
        'outputs': ['prediction'],
    }
    core['legacy_predictions'].append(
        {
            'run': run,
            'completion': 'done',
            'result': product,
            'error': None,
        }
    )
    core['predicted'] = copy.deepcopy(product)


def _legacy_observation(value: Any, *, calls: bool = False) -> dict[str, Any]:
    line = {
        'identity': {
            'operations': [],
            'contract': 30,
            'native_loaded': True,
            'native': {'sha256': '0' * 64},
        },
        'public': {'payloads': []},
        'core': {
            'inputs': [],
            'fetches': [],
            'described': None,
            'materialized': None,
            'predicted': None,
            'accepted': None,
            'error': None,
            'model_calls': [],
            'legacy_predictions': [],
            'capture_capabilities': {
                'model_replies': False,
                'native_error_control': False,
            },
        },
    }
    _append_legacy_prediction(line, value, calls=calls)
    return line


def test_sparse_numeric_fixture_has_owned_products_without_mutating_it() -> (
    None
):
    line = _legacy_observation({'number': '3e-06'}, calls=True)
    before = copy.deepcopy(line)
    view = COMPARE._prediction_view(line)
    assert view['operations'] == ['predict']
    assert view['inputs'] == [{'op': 'predict', 'model': 'kumo-relational'}]
    [event] = view['events']
    assert event['position'] == 0
    assert event['result'] == line['core']['legacy_predictions'][0]['result']
    assert event['calls'] == line['core']['predicted']['calls']
    assert all(attempt['reply'] is None for attempt in view['attempts'])
    assert all(
        attempt['model_endpoint'] is None for attempt in view['attempts']
    )
    COMPARE._prediction_reader().check_receipts(
        view['attempts'], line['public']['payloads']
    )
    assert line == before


@pytest.mark.parametrize('changed', ['body', 'hash', 'receipt'])
def test_sparse_numeric_fixture_keeps_independent_body_hash_receipts(
    changed: str,
) -> None:
    line = _legacy_observation({'number': '-3e-06'})
    reader = COMPARE._prediction_reader()
    reader.check_receipts(
        COMPARE._prediction_view(line)['attempts'], line['public']['payloads']
    )
    if changed == 'body':
        line['core']['model_calls'][0]['body']['value'] = {'number': '7.0'}
    elif changed == 'hash':
        line['core']['model_calls'][0]['body_sha256'] = '0' * 64
    else:
        line['public']['payloads'][0]['sha256'] = '0' * 64
    with pytest.raises(reader.ObservationError):
        reader.check_receipts(
            COMPARE._prediction_view(line)['attempts'],
            line['public']['payloads'],
        )


@pytest.mark.parametrize(
    'changed', ['input', 'owner', 'journal', 'product', 'state', 'endpoint']
)
def test_sparse_numeric_fixture_refuses_incomplete_or_foreign_run_evidence(
    changed: str,
) -> None:
    line = _legacy_observation({'number': '3e-06'})
    if changed == 'state':
        line['core']['materialized'] = copy.deepcopy(
            line['core']['inputs'][0]['state']
        )
    assert len(COMPARE._prediction_view(line)['events']) == 1
    if changed == 'input':
        line['core']['inputs'][0].pop('model')
    elif changed == 'owner':
        line['core']['model_calls'][0]['run'] = 1
    elif changed == 'journal':
        line['core']['legacy_predictions'].clear()
    elif changed == 'product':
        line['core']['legacy_predictions'][0]['result'].pop('request_id')
    elif changed == 'state':
        line['core']['inputs'][0]['state']['payload'] = {'value': 'changed'}
    else:
        line['core']['model_calls'][0]['model_endpoint'] = {
            'endpoint': 'unknown'
        }
    with pytest.raises(COMPARE._prediction_reader().ObservationError):
        COMPARE._prediction_view(line)


def test_shared_numeric_fixture_keeps_its_first_product_separate_from_the_last() -> (
    None
):
    line = _legacy_observation({'number': '3e-06'})
    first = copy.deepcopy(line['core']['legacy_predictions'][0]['result'])
    _append_legacy_prediction(line, {'number': '99.0'})
    view = COMPARE._prediction_view(line)
    assert [event['position'] for event in view['events']] == [0, 1]
    assert view['events'][0]['result'] == first
    assert view['events'][1]['result'] == line['core']['predicted']
    assert view['events'][0]['result'] != view['events'][1]['result']
    line['core'].pop('legacy_predictions')
    with pytest.raises(COMPARE._prediction_reader().ObservationError):
        COMPARE._prediction_view(line)


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
    observation = _legacy_observation(999)
    clean = COMPARE.Report()
    COMPARE._core_payload(
        clean,
        {'id': 'probe'},
        who,
        observation,
        {'public': {'payloads': [{'parsed': {'value': {'number': '999'}}}]}},
    )
    assert clean.failures == [], clean.texts
    report = COMPARE.Report()
    COMPARE._core_payload(
        report,
        {'id': 'probe'},
        who,
        observation,
        {'public': {'payloads': [{'parsed': {'value': {'number': '1'}}}]}},
    )
    assert [rule for rule, _ in report.failures] == ['payload'], report.texts
    assert who in report.failures[0][1]


def test_java_agrees_with_python_core_on_the_core_fields() -> None:
    python = _legacy_observation(1)
    java = _legacy_observation(999)
    report = COMPARE.Report()
    COMPARE._core_agree(report, {'id': 'probe'}, python, java, host='java')
    assert [rule for rule, _ in report.failures] == ['payload', 'core'], (
        report.texts
    )
    assert all(
        'python-core against java' in text for _, text in report.failures
    )
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
) -> tuple[Any, bool]:
    """Check numeric boundaries alongside admitted, owned event observations.

    The common Predict input intentionally retains only its canonical model.
    Exercise the raw state/held payload normalizer explicitly, rather than
    claiming that a removed common input field was compared by the reader.
    """
    input_agrees = True
    if field == 'inputs':
        probes = []
        for value in (mine, theirs):
            state = {'payload': value}
            if state_part == 'ranking':
                state = {'payload': None, 'ranking': {'parts': [state]}}
            elif state_part is not None:
                state = {'payload': None, state_part: [state]}
            probes.append(
                [{'op': 'predict', 'held' if held else 'state': state}]
            )
        input_agrees = COMPARE._core_field('inputs', probes[0]) == (
            COMPARE._core_field('inputs', probes[1])
        )
        python = _legacy_observation(mine)
        rust = _legacy_observation(theirs)
    else:
        python = _legacy_observation(mine, calls=field == 'calls')
        rust = _legacy_observation(theirs, calls=field == 'calls')
        if field == 'materialized':
            for line in (python, rust):
                core = line['core']
                core['materialized'] = copy.deepcopy(core['inputs'][0]['state'])
                core['predicted'] = None
                core['model_calls'] = []
                line['public']['payloads'] = []
                core['legacy_predictions'][0].update(
                    completion='unobserved', result=None
                )
    if shared:
        _append_legacy_prediction(rust, {'number': '99.0'})
        assert [
            event['position']
            for event in COMPARE._prediction_view(rust)['events']
        ] == [0, 1]
    report = COMPARE.Report()
    COMPARE._core_agree(
        report,
        {'id': 'numeric-core'},
        python,
        rust,
        stop='predict' if shared else None,
    )
    return report, input_agrees


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
    report, input_agrees = _numeric_core_agreement(
        field, {'number': mine}, {'number': theirs}
    )
    assert report.failures == [], report.texts
    assert input_agrees


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
    report, input_agrees = _numeric_core_agreement(field, mine, theirs)
    if field == 'inputs':
        assert not input_agrees
    assert [rule for rule, _ in report.failures] == (
        ['core'] if field == 'materialized' else ['payload', 'core']
    ), report.texts


@pytest.mark.parametrize(
    ('mine', 'theirs', 'agrees'),
    [('3e-06', '3e-6', True), ('1', '1.0', False)],
)
def test_shared_core_inputs_use_the_same_lossless_numeric_rule(
    mine: str, theirs: str, agrees: bool
) -> None:
    report, input_agrees = _numeric_core_agreement(
        'inputs', {'number': mine}, {'number': theirs}, shared=True
    )
    assert (report.failures == []) is agrees, report.texts
    assert input_agrees is agrees


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('part', [None, 'forecast', 'batches', 'ranking'])
def test_later_and_held_request_payloads_keep_exact_numeric_agreement(
    part: str | None, held: bool
) -> None:
    for theirs, agrees in [('3e-6', True), ('3.000000000000001e-6', False)]:
        report, input_agrees = _numeric_core_agreement(
            'inputs',
            {'number': '3e-06'},
            {'number': theirs},
            state_part=part,
            held=held,
        )
        assert (report.failures == []) is agrees, report.texts
        assert input_agrees is agrees


@pytest.mark.parametrize('field', ['source', 'state', 'metadata'])
def test_numeric_payload_rule_does_not_rewrite_other_core_maps(
    field: str,
) -> None:
    python = _legacy_observation(0)
    rust = copy.deepcopy(python)
    before, after = {'number': '1e-06'}, {'number': '1e-6'}
    if field == 'metadata':
        for line, value in ((python, before), (rust, after)):
            line['core']['predicted']['metadata'] = value
            line['core']['legacy_predictions'][0]['result']['metadata'] = value
    else:
        # Pure raw input boundary controls stay explicit: these maps are not
        # canonical-model fields in a current common prediction observation.
        probes = [
            [{'op': 'predict', field: {'primary_keys': value}}]
            for value in (before, after)
        ]
        assert COMPARE._core_field('inputs', probes[0]) != (
            COMPARE._core_field('inputs', probes[1])
        )
    report = COMPARE.Report()
    COMPARE._core_agree(report, {'id': 'numeric-map'}, python, rust)
    assert [rule for rule, _ in report.failures] == (
        ['core'] if field == 'metadata' else []
    ), report.texts


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
