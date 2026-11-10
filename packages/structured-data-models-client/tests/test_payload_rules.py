# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public payload rules checked against source rows and explicit contracts."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from sdm_client._rust import run as _rust_run

HERE = Path(__file__).resolve().parent


def _recorder() -> Any:
    name = 'payload_rules_record'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / 'payload_rules_record.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _recorder()
LEAKED = re.compile(r'([\d,]+) of the rows sampled as context are the rows')


def _leaked(notices: list[str]) -> list[int]:
    r"""The counts the context-leakage notices give."""
    return sorted(
        int(m.group(1).replace(',', ''))
        for m in (LEAKED.search(n) for n in notices)
        if m
    )


def _on_the_core(
    spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], list[bytes]]:
    r"""The case observed on the core route, and the payload attachments'
    bytes as the core's runs gave them."""
    taken: list[bytes] = []
    take = _rust_run.Finished.take

    def recording(self: Any, attachment: int) -> Any:
        got = take(self, attachment)
        state = (
            self.result.get('state') if isinstance(self.result, dict) else None
        )
        if isinstance(state, dict) and attachment == state.get('payload'):
            taken.append(bytes(got))
        return got

    monkeypatch.setattr(_rust_run.Finished, 'take', recording)
    return R.observe(spec), taken


DIRECT_RULES = {R.REGRESSION_FIELDS, R.REFUSALS, R.LIMITS, R.AVG_NULL}


def _size_case(length: int) -> dict[str, Any]:
    return {
        'name': f'payload-size-{length}',
        'covers': R.LIMITS,
        'query': R.REGRESSION,
        'setup': R.padded(length),
        'materialize': {'indices': [0, 1], 'anchor_time': R.ANCHOR},
        'measure': True,
        'call': False,
    }


@pytest.mark.parametrize('length, accepted', [(275689, True), (275690, False)])
def test_public_materialization_enforces_model_payload_byte_limit(
    length: int, accepted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    got, attachments = _on_the_core(_size_case(length), monkeypatch)
    if not accepted:
        assert got['outcome'] == 'refused', got
        assert got['code'] == 'INVALID_REQUEST', got
        assert '30MB' in got['message'], got
        assert not attachments
        return
    assert got['outcome'] == 'payload', got
    assert len(attachments) == 1
    state = json.loads(attachments[0])
    assert state.pop('client_generated_keys') == {'orders': '__node_id'}
    # Independently serialize only the model body, not client provenance.
    model_bytes = json.dumps(
        state, allow_nan=False, separators=(',', ':')
    ).encode('utf-8')
    assert len(attachments[0]) == got['bytes']
    prefix = b'{"client_generated_keys":{"orders":"__node_id"},'
    assert attachments[0].startswith(prefix)
    assert model_bytes == b'{' + attachments[0][len(prefix) :]
    assert len(model_bytes) <= 30 * 1024 * 1024


@pytest.mark.parametrize('padding, accepted', [(5, True), (8, False)])
def test_client_only_provenance_does_not_consume_model_payload_limit(
    padding, accepted, monkeypatch
):
    spec = _size_case(275689)
    spec['setup'].append(
        f"UPDATE users SET note=note||'{'a' * padding}' WHERE user_id=0"
    )
    got, attachments = _on_the_core(spec, monkeypatch)
    if accepted:
        assert got['outcome'] == 'payload', got
        state = bytes(attachments[0])
        prefix = b'{"client_generated_keys":{"orders":"__node_id"},'
        assert state.startswith(prefix)
        model_bytes = b'{' + state[len(prefix) :]
        assert len(model_bytes) <= R.PAYLOAD_LIMIT < len(state)
    else:
        assert got['outcome'] == 'refused'
        assert '30MB' in got['message']


REFUSAL_CODES = {
    'base.regression.statistic-unknown': 'INVALID_REQUEST',
    'base.binary.statistic': 'INVALID_REQUEST',
    'base.regression.num-context-0': 'INVALID_REQUEST',
    'base.regression.num-context-10001': 'UNSUPPORTED_REQUEST',
    'base.regression.entities-1001': 'INVALID_REQUEST',
    'star.tables-16': 'INVALID_REQUEST',
    'empty.indices': 'INVALID_REQUEST',
    'reserved.instance-id': 'INVALID_SCHEMA',
    'reserved.anchor-time': 'INVALID_SCHEMA',
}
DIRECT_CASES = [c for c in R.CASES if c['covers'] in DIRECT_RULES]


@pytest.mark.parametrize(
    'spec', DIRECT_CASES, ids=[c['name'] for c in DIRECT_CASES]
)
def test_materialization_options_and_admission_rules(
    spec: dict[str, Any],
) -> None:
    got = R.observe(spec)
    name = spec['name']
    if name == 'avg-equals-null':
        assert got['outcome'] == 'refused', got
        assert got['code'] == 'UNSUPPORTED_REQUEST', got
        assert got['rule'] == 'average_null', got
        assert got['message'], got
        return
    if name in REFUSAL_CODES:
        assert got['outcome'] == 'refused', got
        assert got['code'] == REFUSAL_CODES[name], got
        assert got['message'], got
        return
    assert got['outcome'] == 'payload', got
    assert got['parts'] == 1
    assert got['task']['kind'] == 'regression'
    assert got['task']['target']['column_name'] == 'TARGET'
    assert got['task']['target']['dtype'] == 'float32'
    arguments = spec['materialize']
    samples = arguments.get('num_context_samples', 1000)
    assert got['inference']['run_mode'] == (
        'fast' if samples <= 1000 else 'best'
    )
    expected_output = {'fields': ['prediction']}
    if 'prediction_statistic' in arguments:
        expected_output['prediction_statistic'] = arguments[
            'prediction_statistic'
        ]
    assert got['output'] == expected_output
    requested = len(arguments['indices'])
    assert got['predict'] == requested
    assert got['metadata']['num_prediction_examples'] == requested
    if name.startswith('base.regression'):
        assert got['context'] == min(samples, 112)
        assert got['metadata']['num_context_examples'] == min(samples, 112)
    if name == 'star.tables-15':
        assert len(got['context_related']) == 15
        assert set(got['context_related']) == set(got['predict_related'])


CLASSIFICATION_CASES = [c for c in R.CASES if c['covers'] == R.CLASSIFICATION]


@pytest.mark.parametrize(
    'spec', CLASSIFICATION_CASES, ids=[c['name'] for c in CLASSIFICATION_CASES]
)
def test_classification_labels_and_leakage_follow_source_rows(
    spec: dict[str, Any],
) -> None:
    import duckdb

    got = R.observe(spec)
    assert got['outcome'] == 'payload', got
    name = spec['name']
    temporal = name == 'base.binary' or name.endswith(
        ('.binary-count', '.binary-sum')
    )
    condition = name.endswith('.binary-static-condition')
    binary = temporal or condition
    column = (
        'age'
        if condition
        else 'active'
        if name.endswith('.boolean-static')
        else 'segment'
        if name.endswith('.multiclass-static')
        else 'country'
    )
    with duckdb.connect() as source:
        for statement in spec['setup']:
            source.execute(statement)
        target = got['task']['target']
        assert got['task']['kind'] == (
            'binary_classification' if binary else 'multiclass_classification'
        )
        assert target['column_name'] == 'TARGET'
        assert target['dtype'] == ('bool' if binary else 'string')
        if binary:
            assert target['classes'] == ['false', 'true']
            assert target['positive_class'] == 'true'
        else:
            classes = {
                str(row[0])
                for row in source.execute(
                    f'SELECT DISTINCT {column} FROM users'
                ).fetchall()
            }
            assert set(target['classes']) == classes
            assert len(target['classes']) == len(classes)
            assert target.get('positive_class') is None
            if column == 'active':
                assert target['classes'] == (
                    ['True', 'False']
                    if name.startswith('generated-13.')
                    else ['False', 'True']
                )
            if name == 'base.multiclass':
                assert target['classes'] == ['IN', 'DE', 'US']
        requested = spec['materialize']['indices']
        prediction = [
            dict(zip(got['predict']['columns'], row, strict=True))
            for row in got['predict']['rows']
        ]
        assert [row['__kumo_entity_ref_0'] for row in prediction] == requested
        assert got['metadata']['num_prediction_examples'] == len(requested)
        context = [
            dict(zip(got['context']['columns'], row, strict=True))
            for row in got['context']['rows']
        ]
        assert context
        for row in context:
            entity = row['__kumo_entity_ref_0']
            if temporal:
                summed = name.endswith('.binary-sum')
                aggregate = 'SUM(price)' if summed else 'COUNT(*)'
                days = 60 if summed else 30
                threshold = 150 if summed else 0 if name == 'base.binary' else 2
                anchor = row['__kumo_anchor_time']
                value = source.execute(
                    f'SELECT COALESCE({aggregate}, 0) > {threshold} FROM orders '
                    f'WHERE user_id=? AND ts>CAST(? AS TIMESTAMP) '
                    f'AND ts<=CAST(? AS TIMESTAMP)+INTERVAL {days} DAY',
                    [entity, anchor, anchor],
                ).fetchone()[0]
            else:
                assert entity not in requested
                expression = 'age > 40' if condition else column
                value = source.execute(
                    f'SELECT {expression} FROM users WHERE user_id=?', [entity]
                ).fetchone()[0]
                if not binary:
                    value = str(value)
            assert type(row['TARGET']) is type(value)
            assert row['TARGET'] == value
        for half, rows in [('context', context), ('predict', prediction)]:
            ids = {row['instance_id'] for row in rows}
            assert len(ids) == len(rows)
            related = got[half + '_related']
            assert set(related) == {'users', 'orders'}
            assert set(related['users']['instance_ids']) == ids
            eligible = {
                row['instance_id']
                for row in rows
                if source.execute(
                    'SELECT COUNT(*) FROM orders WHERE user_id=? AND ts<=CAST(? AS TIMESTAMP)',
                    [row['__kumo_entity_ref_0'], row['__kumo_anchor_time']],
                ).fetchone()[0]
            }
            assert set(related['orders']['instance_ids']) == eligible
            for table in related.values():
                assert set(table['instance_ids']) <= ids
            if not temporal:
                assert column not in related['users']['columns']
        if not temporal:
            count = source.execute('SELECT COUNT(*) FROM users').fetchone()[0]
            assert len(context) == count - len(set(requested))
            assert got['metadata']['num_context_examples'] == count
            assert _leaked(got['notices']) == [len(set(requested))]


@pytest.mark.parametrize(
    'spec',
    [c for c in R.CASES if c['covers'] == R.INFERENCE],
    ids=lambda c: c['name'],
)
def test_inference_options_reach_endpoint_or_refuse_before_send(
    spec, monkeypatch
):
    from contextlib import contextmanager

    sent = []
    endpoint = R._refusing_endpoint

    @contextmanager
    def capture():
        with endpoint() as (url, received):
            yield url, received
            sent.extend(received)

    monkeypatch.setattr(R, '_refusing_endpoint', capture)
    got = R.observe(spec)
    assert got['outcome'] == 'payload', got
    options = spec['materialize']['inference_config']
    config = got['inference']['inference_config']
    for key, value in options.items():
        assert config[key] == value
    assert got['inference']['run_mode'] == 'fast'
    if spec['name'].endswith('-unread'):
        assert got['call_refusal']['code'] == 'UNSUPPORTED_REQUEST'
        assert sent == []
        return
    assert len(sent) == 1
    body = json.loads(sent[0])
    assert body['model'] == 'kumo-relational'
    assert body['inference']['inference_config'] == options
    assert body['inference']['run_mode'] == 'fast'
    fields = (
        ['prediction', 'probabilities']
        if '.binary.' in spec['name']
        else ['prediction']
    )
    expected_output = {'fields': fields}
    if '.binary.' not in spec['name']:
        expected_output['prediction_statistic'] = 'median'
    assert body['output'] == expected_output


def test_wide_decimal_features_preserve_every_digit_in_both_halves(monkeypatch):
    spec = next(c for c in R.CASES if c['covers'] == R.DECIMAL)
    got, attachments = _on_the_core(spec, monkeypatch)
    assert got['outcome'] == 'payload', got
    assert len(attachments) == 1
    payload = json.loads(attachments[0])
    feature = payload['schema']['related_tables']['users']['columns']['f']
    assert feature['dtype'] == 'string'
    assert feature['stype'] == 'categorical'
    for half in ['context', 'predict']:
        table = payload[half]['related_tables']['users']
        assert table['rows']
        for values in table['rows']:
            row = dict(zip(table['columns'], values, strict=True))
            assert row['f'] == str(
                12345678901234567890123456789 + row['user_id']
            )
