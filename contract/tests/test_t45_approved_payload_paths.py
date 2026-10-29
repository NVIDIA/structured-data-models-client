# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual call journals retain the existing approved semantic path scopes.

Compact independently authored DTOs exercise the retained R54 failure classes.
They are not captures, new value approvals, or per-row sampling-validity proof.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_t44_prediction_observation import MODEL, actual_observation, reader
from test_t136_forecast_capture import _module

ROOT = Path(__file__).parents[2]
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
COMPARE = _module(
    't45_payload_path_compare', ROOT / 'scripts/rust_core/compare.py'
)
AVERAGE = 'lag.average-empty-windows@sqlite'
CONDITION = 'lag.condition-per-call@sqlite'
# Case, physical FK column, exact affected row-cell index, completed type change.
FOREIGN_KEYS = [
    (
        'sample.parent-table.decimal-keys@sqlite',
        'accounts',
        'owner_id',
        3,
        True,
    ),
    ('sample.parent-table@sqlite', 'orders', 'user_id', 4, True),
    (
        'sample.parent-table.relbench-avito@sqlite',
        'AdsInfo',
        'LocationID',
        3,
        False,
    ),
    (
        'sample.parent-table.relbench-stack@sqlite',
        'posts',
        'ParentId',
        5,
        False,
    ),
    ('sample.parent-table.static@sqlite', 'orders', 'user_id', 3, True),
    ('shape.diamond-from-source@sqlite', 'orders', 'cid', 7, True),
    ('shape.diamond.addresses@sqlite', 'orders', 'cid', 7, True),
    ('shape.two-paths.keyed@sqlite', 'details', 'hid', 4, True),
    ('shape.two-paths.keyless@sqlite', 'details', 'hid', 5, True),
]


def snapshot(raw):
    return json.loads(
        raw,
        parse_int=lambda token: {'number': token},
        parse_float=lambda token: {'number': token},
    )


def payload(position, request_id):
    # Non-excused IDs, neighboring values and call position distinguish rows
    # and calls even when every approved lag/FK scalar is removed.
    return {
        'model': MODEL,
        'metadata': {'request_id': request_id, 'call_position': position},
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': [
                    'instance_id',
                    'TARGET',
                    'entity',
                    'neighbor',
                    'lag0',
                    'lag1',
                ],
                'rows': [
                    [0, 21, 7, 10, None, None],
                    [1, 22, 8, 11, None, 10.0],
                ],
            }
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': [
                    'instance_id',
                    'entity',
                    'neighbor',
                    'lag0',
                    'lag1',
                ],
                'rows': [[2, 7, 12, 34.0, None], [3, 8, 13, 35.5, None]],
            }
        },
        'schema': {'instance_table': {'primary_key': 'instance_id'}},
        'output': {'fields': ['prediction']},
    }


def gathered(count):
    """One entry, genuine typed call starts, and one selected full aggregate."""
    core, operations = actual_observation()
    if count == 1:
        return core, operations
    assert count == 2
    second, _ = actual_observation('sdm-000000000002', 'endpoint-r2')
    entry, other = core['predictions'][0], second['predictions'][0]
    final = entry['results'][0]
    call = copy.deepcopy(other['calls'][0])
    call['run'] = 4
    entry['calls'].append(call)
    entry['runs'] = [2, 3, 4, 5]
    final['run'] = 5
    aggregation = final['input']['input']['aggregation']
    aggregation['parts'].append(
        copy.deepcopy(
            other['results'][0]['input']['input']['aggregation']['parts'][0]
        )
    )
    aggregation['counted'] = [2, 2]
    final['data']['predicted'].update(
        model=call['scope']['model'], request_id=call['scope']['request_id']
    )
    final['data']['answer_details'] = {'calls_answered': 2, 'rows_answered': 4}
    final['data']['lineage'].extend(
        copy.deepcopy(other['results'][0]['data']['lineage'])
    )
    for row in copy.deepcopy(other['results'][0]['result']['rows']):
        row[0] = {'i64': str(int(row[0]['i64']) + 2)}
        row[3] = {'i64': str(int(row[3]['i64']) + 2)}
        final['result']['rows'].append(row)
    second_start = copy.deepcopy(second['inputs'][3])
    second_start['run'] = 4
    finish_start = copy.deepcopy(core['inputs'][4])
    finish_start.update(
        run=5,
        input=copy.deepcopy(final['input']),
        result=copy.deepcopy(final['data']),
    )
    core['inputs'] = [*core['inputs'][:4], second_start, finish_start]
    return core, ['describe_source', 'materialize'] + ['prepare_request'] * 4


def fixture(bodies, producer):
    core, operations = gathered(len(bodies))
    entry = core['predictions'][0]
    receipts = []
    for call, body in zip(entry['calls'], bodies, strict=True):
        body = copy.deepcopy(body)
        body['metadata']['request_id'] = call['request_id']
        raw = json.dumps(body, separators=(', ', ': ')).encode()
        call.update(
            input_body_base64=base64.b64encode(raw).decode('ascii'),
            input_body_sha256=hashlib.sha256(raw).hexdigest(),
            body=snapshot(raw),
            body_sha256=hashlib.sha256(raw).hexdigest(),
        )
        receipts.append(
            {
                'method': 'POST',
                'path': '/v1/predictions',
                'parsed': copy.deepcopy(call['body']),
                'sha256': call['body_sha256'],
            }
        )
    final = entry['results'][0]
    predicted = final['data']['predicted']
    product = {
        'body': copy.deepcopy(entry['calls'][-1]['body']),
        'model': predicted['model'],
        'request_id': predicted['request_id'],
        'metadata': copy.deepcopy(final['metadata']),
        'result': copy.deepcopy(final['result']),
        'outputs': copy.deepcopy(predicted['outputs']),
    }
    if len(bodies) == 2:
        product['calls'] = [
            copy.deepcopy(call['body']) for call in entry['calls']
        ]
    core['predicted'] = product
    if producer == 'legacy':
        calls = entry['calls']
        core.pop('predictions')
        core['inputs'] = [
            *copy.deepcopy(core['inputs'][:2]),
            {
                'op': 'predict',
                'model': MODEL,
                'state': copy.deepcopy(core['materialized']),
            },
        ]
        operations = ['describe_source', 'materialize', 'predict']
        core['model_calls'] = [
            {
                'run': 2,
                'body': copy.deepcopy(call['body']),
                'body_sha256': call['body_sha256'],
                'reply': copy.deepcopy(call['reply']),
                'model_endpoint': None,
            }
            for call in calls
        ]
        core['capture_capabilities'] = {
            'model_replies': True,
            'native_error_control': True,
        }
        core['legacy_predictions'] = [
            {
                'run': 2,
                'completion': 'done',
                'result': copy.deepcopy(product),
                'error': None,
                'answered': None,
            }
        ]
    else:
        assert producer == 'gathered'
    return {
        'identity': {'operations': operations, 'contract': 35},
        'core': core,
        'public': {'payloads': receipts},
    }


def reference(bodies):
    return {
        'public': {
            'payloads': [
                {
                    'method': 'POST',
                    'path': '/v1/predictions',
                    'parsed': snapshot(json.dumps(body).encode()),
                    'sha256': hashlib.sha256(
                        json.dumps(body).encode()
                    ).hexdigest(),
                }
                for body in bodies
            ]
        }
    }


def judge(case, observed, expected, who='rust-host'):
    untouched = copy.deepcopy((observed, expected))
    report = COMPARE.Report(paths=(('metadata', 'request_id'),))
    report.more_exact = COMPARE.more_exact(REGISTER)
    COMPARE._core_payload(report, {'id': case}, who, observed, expected)
    assert (observed, expected) == untouched
    return report


def lag_pair(case):
    count = 2 if case == CONDITION else 1
    bodies = [payload(at, f'fixture-{at}') for at in range(count)]
    wanted = copy.deepcopy(bodies)
    for at, (actual, old) in enumerate(zip(bodies, wanted, strict=True)):
        old['context']['instance_table']['rows'][0][4] = (
            True if count == 2 else 34.0
        )
        old['context']['instance_table']['rows'][1][5] = None
        if count == 2:
            actual['context']['instance_table']['rows'][1][5] = False
            old['context']['instance_table']['rows'][0][5] = False
            actual['predict']['instance_table']['rows'][0][3 + at] = at == 0
        old['predict']['instance_table']['rows'][0][
            3 + (at if count == 2 else 0)
        ] = None
    return bodies, wanted


def foreign_key_pair(spec):
    case, table, column, index, dtype_change = spec
    body = payload(0, 'fixture-fk')
    names = ['kept_' + str(at) for at in range(index + 2)]
    names[index] = column
    rows = [[at * 10 + cell for cell in range(index + 2)] for at in range(2)]
    first, old_first = {
        'sample.parent-table.decimal-keys@sqlite': (0, 0.0),
        'sample.parent-table@sqlite': (0, 0.0),
        'sample.parent-table.relbench-avito@sqlite': (0.0, None),
        'sample.parent-table.relbench-stack@sqlite': (8.0, None),
        'sample.parent-table.static@sqlite': (0, 0.0),
        'shape.diamond-from-source@sqlite': (16, None),
        'shape.diamond.addresses@sqlite': (16, None),
        'shape.two-paths.keyed@sqlite': (13, None),
        'shape.two-paths.keyless@sqlite': (0, None),
    }[case]
    rows[0][index] = first
    for section in ('context', 'predict'):
        body[section]['related_tables'] = {
            table: {
                'format': 'arrays',
                'columns': names,
                'rows': copy.deepcopy(rows),
            }
        }
    body['schema']['related_tables'] = {
        table: {
            'columns': {
                column: {'dtype': 'int64' if dtype_change else 'float64'}
            }
        }
    }
    wanted = copy.deepcopy(body)
    wanted['context']['related_tables'][table]['rows'][0][index] = old_first
    wanted['predict']['related_tables'][table]['rows'][1][index] = None
    if dtype_change:
        wanted['schema']['related_tables'][table]['columns'][column][
            'dtype'
        ] = 'float64'
    return case, [body], [wanted]


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
@pytest.mark.parametrize('case', [AVERAGE, CONDITION])
def test_existing_own_row_lag_fields_apply_at_the_observed_single_or_two_call_root(
    producer, case
):
    bodies, wanted = lag_pair(case)
    observed = fixture(bodies, producer)
    view = reader().view(observed['core'], observed['identity']['operations'])
    assert len(view['attempts']) == (2 if case == CONDITION else 1)
    assert len(view['events']) == 1
    if case == CONDITION:
        assert len(view['events'][0]['result']['calls']) == 2
        assert len(observed['core']['predicted']['calls']) == 2
    else:
        assert 'calls' not in view['events'][0]['result']
    root = (
        'core.predicted.calls' if case == CONDITION else 'core.predicted.body'
    )
    paths = COMPARE.more_exact(REGISTER)[case, 'rust-host']
    assert root + '/*/context/instance_table/rows/*/4' in paths
    assert judge(case, observed, reference(bodies)).failures == []
    assert judge(case, observed, reference(wanted)).failures == []


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
@pytest.mark.parametrize(
    'spec', FOREIGN_KEYS, ids=[item[0] for item in FOREIGN_KEYS]
)
def test_existing_sampled_fk_fields_keep_their_scalar_null_and_dtype_scope(
    producer, spec
):
    case, bodies, wanted = foreign_key_pair(spec)
    observed = fixture(bodies, producer)
    reader().view(observed['core'], observed['identity']['operations'])
    paths = COMPARE.more_exact(REGISTER)[case, 'rust-host']
    assert (
        f'core.predicted.body/*/context/related_tables/{spec[1]}/rows/*/{spec[3]}'
        in paths
    )
    assert judge(case, observed, reference(bodies)).failures == []
    assert judge(case, observed, reference(wanted)).failures == []


@pytest.mark.parametrize('case', [AVERAGE, CONDITION])
@pytest.mark.parametrize(
    'changed', ['neighbor', 'container', 'row-count', 'row-order', 'metadata']
)
def test_lag_scalar_exceptions_do_not_hide_neighbor_structure_or_order(
    case, changed
):
    bodies, wanted = lag_pair(case)
    observed = fixture(bodies, 'gathered')
    assert judge(case, observed, reference(bodies)).failures == []
    table = wanted[0]['context']['instance_table']
    if changed == 'neighbor':
        table['rows'][0][3] = 99
    elif changed == 'container':
        table['rows'][0][4] = [True]
    elif changed == 'row-count':
        table['rows'].append(copy.deepcopy(table['rows'][0]))
    elif changed == 'row-order':
        table['rows'].reverse()
    else:
        wanted[0]['metadata']['unapproved'] = 'changed'
    assert judge(case, observed, reference(wanted)).failures


@pytest.mark.parametrize(
    'changed',
    ['neighbor', 'container', 'dtype-container', 'columns', 'row-count'],
)
def test_fk_exceptions_do_not_extend_to_neighboring_cells_or_containers(
    changed,
):
    spec = FOREIGN_KEYS[1]
    case, bodies, wanted = foreign_key_pair(spec)
    observed = fixture(bodies, 'legacy')
    assert judge(case, observed, reference(bodies)).failures == []
    table = wanted[0]['context']['related_tables']['orders']
    if changed == 'neighbor':
        table['rows'][0][spec[3] + 1] = 99
    elif changed == 'container':
        table['rows'][0][spec[3]] = {'unexpected': 0}
    elif changed == 'dtype-container':
        wanted[0]['schema']['related_tables']['orders']['columns']['user_id'][
            'dtype'
        ] = ['int64']
    elif changed == 'columns':
        table['columns'][spec[3]] = 'another_column'
    else:
        table['rows'].pop()
    assert judge(case, observed, reference(wanted)).failures


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
@pytest.mark.parametrize('changed', ['extra', 'missing', 'reordered'])
def test_approved_multi_call_scalars_preserve_actual_call_cardinality_and_order(
    producer, changed
):
    bodies, wanted = lag_pair(CONDITION)
    observed = fixture(bodies, producer)
    assert judge(CONDITION, observed, reference(bodies)).failures == []
    if changed == 'extra':
        wanted.append(copy.deepcopy(wanted[0]))
    elif changed == 'missing':
        wanted.pop()
    else:
        wanted.reverse()
    assert judge(CONDITION, observed, reference(wanted)).failures


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
@pytest.mark.parametrize(
    'changed',
    ['hash', 'receipt-hash', 'receipt-body', 'missing-receipt', 'missing-call'],
)
def test_scoped_exceptions_never_replace_original_hash_and_receipt_conservation(
    producer, changed
):
    bodies, _ = lag_pair(CONDITION)
    observed = fixture(bodies, producer)
    assert (
        judge(
            CONDITION, observed, reference(bodies), who='python-core'
        ).failures
        == []
    )
    calls = (
        observed['core']['predictions'][0]['calls']
        if producer == 'gathered'
        else observed['core']['model_calls']
    )
    if changed == 'hash':
        calls[0]['body_sha256'] = '0' * 64
    elif changed == 'receipt-hash':
        observed['public']['payloads'][0]['sha256'] = '0' * 64
    elif changed == 'receipt-body':
        observed['public']['payloads'][0]['parsed'] = {}
    elif changed == 'missing-receipt':
        observed['public']['payloads'].pop()
    else:
        calls.pop()
    assert judge(
        CONDITION, observed, reference(bodies), who='python-core'
    ).failures


def test_approved_single_body_paths_cannot_cover_a_two_call_product_or_the_reverse():
    bodies, wanted = lag_pair(CONDITION)
    assert judge(
        AVERAGE, fixture(bodies, 'gathered'), reference(wanted)
    ).failures
    bodies, wanted = lag_pair(AVERAGE)
    assert judge(
        CONDITION, fixture(bodies, 'gathered'), reference(wanted)
    ).failures


@pytest.mark.parametrize('producer', ['gathered', 'legacy'])
def test_reordering_actual_call_journal_is_not_an_approved_scalar_change(
    producer,
):
    bodies, wanted = lag_pair(CONDITION)
    observed = fixture(bodies, producer)
    assert judge(CONDITION, observed, reference(bodies)).failures == []
    calls = (
        observed['core']['predictions'][0]['calls']
        if producer == 'gathered'
        else observed['core']['model_calls']
    )
    calls.reverse()
    assert judge(CONDITION, observed, reference(wanted)).failures


def test_real_case_and_runner_authorization_is_required_without_new_exemptions():
    case, bodies, wanted = foreign_key_pair(FOREIGN_KEYS[1])
    observed = fixture(bodies, 'gathered')
    assert judge(case, observed, reference(bodies)).failures == []
    assert judge('unapproved.case@sqlite', observed, reference(wanted)).failures
    scope = COMPARE.more_exact(REGISTER)
    assert (case, 'python-old') not in scope
    # Existing counterpart grants python-core alone, not either host that
    # _core_payload compares directly with a reference endpoint recording.
    unsupported = 'sample.parent-table@databricks-spark'
    assert (unsupported, 'python-core') in scope
    for who in ('rust-host', 'java'):
        assert (unsupported, who) not in scope
        assert judge(unsupported, observed, reference(wanted), who=who).failures
