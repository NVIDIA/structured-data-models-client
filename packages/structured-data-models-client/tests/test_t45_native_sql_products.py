# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Useful SQL products consume one actual tagged value and explicit owners."""

from __future__ import annotations

import copy
import json

import pytest
from test_t44_materialize_result_readers import _materialized

from sdm_client._rust import native_sql_products as products
from sdm_client._rust import results
from sdm_client._rust.codec import CodecError
from sdm_client._rust.native_products import Bindings, Product
from sdm_client._rust.run import Finished
from sdm_client.errors import StructuredDataError

RELATION = {'source': ['events'], 'found': False, 'columns': []}
COLUMN = {'source': RELATION, 'column': 'value'}
RANGE = {'start': 0, 'stop': 2}


def _data(kind):
    if kind == 'source':
        return {
            'source': 'actual source',
            'identity': 'source identity',
            'owner': 4,
            'target': COLUMN,
            'tables': [
                {'source': RELATION, 'read': RELATION, 'columns': ['value']}
            ],
        }
    if kind == 'population':
        return {
            'source': 'actual source',
            'identity': 'population identity',
            'owner': 4,
            'rows': 2,
            'batch_rows': 10,
            'target': COLUMN,
        }
    if kind == 'label_inputs':
        return {
            'population': 'population identity',
            'range': RANGE,
            'owner': 4,
            'source_read': 4,
            'target': COLUMN,
            'form': 'filtered',
            'condition_columns': [COLUMN, COLUMN],
        }
    if kind == 'target_column':
        return {
            'input': _data('label_inputs'),
            'owner': 5,
            'derivation': 'conditioned',
            'rows': 2,
            'condition_values': [
                {
                    'column': COLUMN,
                    'converted': {'rows': 6, 'presence': 7, 'encoding': None},
                },
                {
                    'column': COLUMN,
                    'converted': {
                        'rows': 6,
                        'presence': None,
                        'encoding': None,
                    },
                },
            ],
        }
    if kind == 'label_groups':
        return {
            'population': 'population identity',
            'range': RANGE,
            'owner': 4,
            'native_target': COLUMN,
            'target_encoding': None,
        }
    if kind == 'label_batch':
        return {
            'population': 'population identity',
            'range': RANGE,
            'owner': 4,
            'complete': True,
        }
    if kind == 'column_meaning':
        return {
            'source': 'source identity',
            'column': COLUMN,
            'input': {'storage': 'cell_rows'},
            'scalar': {'meaning': 'nullable', 'physical': {'type': 'int64'}},
        }
    if kind == 'lookup':
        return {
            'owner': 4,
            'read': 4,
            'source': RELATION,
            'key': 'event_id',
            'range': RANGE,
            'context_rows': 1,
            'scored_rows': 1,
        }
    if kind == 'lookup_rows':
        return {'positions': 4}
    raise AssertionError(kind)


class Handle:
    def __init__(self, attachments):
        self.attachments = dict(attachments)
        self.takes = []

    def take_attachment(self, attachment):
        self.takes.append(attachment)
        return self.attachments.pop(attachment)


@pytest.mark.parametrize(
    'kind', ['label_inputs', 'target_column', 'label_groups']
)
def test_condition_history_rebinds_only_explicit_owners_and_keeps_arrow_distinct(
    kind,
):
    child = _data('label_inputs')
    child.update(owner=20, source_role='entity', eligibility=None)
    history = {
        'columns': [child],
        'values': [{'column': COLUMN, 'owner': 21}],
    }
    data = _data(kind)
    if kind == 'target_column':
        data['input']['eligibility'] = history
        data['condition_values'][0]['converted']['times'] = 8
        expected = [4, 20, 21, 5]
        ids = expected + [6, 7, 8]
    elif kind == 'label_inputs':
        data['eligibility'] = history
        expected = [4, 20, 21]
        ids = expected
    else:
        data.update(what_if=history, restored={'rows': 22, 'encoding': None})
        expected = [4, 20, 21]
        ids = expected + [22]
    original = copy.deepcopy(data)
    assert products.owner_ids(kind, data) == expected
    owners = {number: object() for number in ids}
    finished, handle = _finished(kind, data, owners)
    value = products.adopt(finished, kind)
    assert set(value.owners) == set(ids)
    assert handle.takes == [17, *dict.fromkeys(ids)]
    bindings = Bindings()
    bound = products.bind(value, bindings)
    assert [
        bindings.attachments[key] for key in products.owner_ids(kind, bound)
    ] == [owners[key] for key in expected]
    assert data == original
    assert value.owners == owners


@pytest.mark.parametrize('slot', ['column', 'value', 'restored'])
def test_bad_nested_history_attachment_is_rejected_before_any_native_owner_take(
    slot,
):
    data = _data('label_groups')
    child = _data('label_inputs')
    data.update(
        what_if={
            'columns': [child],
            'values': [{'column': COLUMN, 'owner': 21}],
        },
        restored={'rows': 22, 'encoding': None},
    )
    if slot == 'column':
        child['owner'] = True
    elif slot == 'value':
        data['what_if']['values'][0]['owner'] = 2**32
    else:
        data['restored']['rows'] = -1
    finished, handle = _finished('label_groups', data)
    with pytest.raises(CodecError):
        products.adopt(finished, 'label_groups')
    assert handle.takes == [17]


def _finished(kind, data=None, owners=None):
    raw = json.dumps(
        {'value': kind, 'data': _data(kind) if data is None else data}
    ).encode()
    handle = Handle({17: raw, **(owners or {})})
    return Finished({'op': 'prepare_materialize', 'value': 17}, handle), handle


@pytest.mark.parametrize(
    ('kind', 'ids'),
    [
        ('source', [4]),
        ('population', [4]),
        ('label_inputs', [4]),
        ('target_column', [4, 5, 6, 7]),
        ('label_groups', [4]),
        ('label_batch', [4]),
        ('column_meaning', []),
        ('lookup_rows', [4]),
    ],
)
def test_read_and_adopt_take_exact_explicit_owners_once_in_contract_order(
    kind, ids
):
    for reader in (
        products.read,
        lambda finished: products.adopt(finished, kind),
    ):
        owners = {number: object() for number in ids}
        finished, handle = _finished(kind, owners=owners)
        product = reader(finished)
        assert type(product) is Product
        assert product.kind == kind
        assert product.data == _data(kind)
        assert set(product.owners) == set(ids)
        assert all(product.owners[number] is owners[number] for number in ids)
        assert handle.takes == [17, *ids]
        assert handle.attachments == {}


@pytest.mark.parametrize(
    'kind',
    [
        'source',
        'population',
        'label_inputs',
        'label_groups',
        'label_batch',
        'lookup',
    ],
)
def test_bind_rewrites_only_owner_and_preserves_run_local_read_numbers(kind):
    data = _data(kind)
    before = copy.deepcopy(data)
    owner = object()
    product = Product(kind, data, {4: owner})
    bindings = Bindings()
    bindings.add(b'already bound')
    written = products.bind(product, bindings)
    assert written == {**before, 'owner': 3}
    assert data == before
    assert product.owners == {4: owner}
    assert bindings.attachments == {2: b'already bound', 3: owner}
    if kind == 'lookup':
        assert written['read'] == 4
    if kind == 'label_inputs':
        assert written['source_read'] == 4


def test_target_bind_keeps_aliases_and_rebinds_each_named_nested_slot_only():
    data = _data('target_column')
    before = copy.deepcopy(data)
    owners = {number: object() for number in (4, 5, 6, 7)}
    product = Product('target_column', data, owners)
    bindings = Bindings()
    written = products.bind(product, bindings)
    assert written['input'] == {**before['input'], 'owner': 2}
    assert written['owner'] == 3
    assert [item['converted'] for item in written['condition_values']] == [
        {'rows': 4, 'presence': 5, 'encoding': None},
        {'rows': 4, 'presence': None, 'encoding': None},
    ]
    assert bindings.attachments == {
        2: owners[4],
        3: owners[5],
        4: owners[6],
        5: owners[7],
    }
    assert data == before
    assert product.owners == owners


def test_meaning_has_no_owner_and_positions_are_an_explicit_arrow_owner_slot():
    bindings = Bindings()
    meaning = _data('column_meaning')
    assert (
        products.bind(Product('column_meaning', meaning, {}), bindings)
        == meaning
    )
    assert bindings.attachments == {}
    actual = object()
    assert products.bind(
        Product('lookup_rows', {'positions': 4}, {4: actual}), bindings
    ) == {'positions': 2}
    assert bindings.attachments == {2: actual}


@pytest.mark.parametrize('kind', ['ordinary', 'forecast', 'batches', 'ranking'])
def test_final_request_reads_nested_materialized_once_without_consuming_state_attachments(
    kind,
):
    nested = _materialized(kind)
    before = copy.deepcopy(nested)
    for reader in (products.read, products.request):
        payload = object()
        finished, handle = _finished(
            'request', nested, {11: payload, 12: b'query'}
        )
        actual = reader(finished)
        assert actual == results.Gathered.from_contract(before)
        assert handle.takes == [17]
        assert handle.attachments == {11: payload, 12: b'query'}
        assert finished.result == {'op': 'prepare_materialize', 'value': 17}
        assert finished.take(11) is payload
    assert nested == before


def test_task_dispatch_handles_source_and_request_as_actual_single_pass_alternatives():
    owner = object()
    source, source_handle = _finished('source', owners={4: owner})
    request, request_handle = _finished('request', _materialized())
    assert isinstance(products.read(source), Product)
    assert isinstance(products.read(request), results.Gathered)
    assert source_handle.takes == [17, 4]
    assert request_handle.takes == [17]


@pytest.mark.parametrize(
    'reader',
    [lambda finished: products.adopt(finished, 'population'), products.request],
)
def test_wrong_expected_tag_is_refused_before_owner_takes(reader):
    owner = object()
    finished, handle = _finished('source', owners={4: owner})
    with pytest.raises(CodecError):
        reader(finished)
    assert handle.takes == [17]
    assert handle.attachments == {4: owner}


def test_wrong_operation_is_refused_before_any_attachment_read():
    handle = Handle({17: b'not read'})
    finished = Finished({'op': 'prepare_frames', 'value': 17}, handle)
    with pytest.raises(StructuredDataError) as caught:
        products.read(finished)
    assert caught.value.code == 'INTERNAL_ERROR'
    assert handle.takes == []


@pytest.mark.parametrize(
    'raw',
    [
        b'broken JSON',
        b'[]',
        b'{"value":"source"}',
        b'{"value":"source","data":null}',
        b'{"value":"source","data":{},"extra":1}',
        b'{"value":"unknown","data":{"owner":4}}',
        b'{"value":"lookup","data":{"owner":4}}',
    ],
)
def test_malformed_or_unknown_envelopes_never_take_named_owners(raw):
    owner = object()
    handle = Handle({17: raw, 4: owner})
    with pytest.raises(CodecError):
        products.read(
            Finished({'op': 'prepare_materialize', 'value': 17}, handle)
        )
    assert handle.takes == [17]
    assert handle.attachments == {4: owner}


@pytest.mark.parametrize(
    'malformed',
    [
        'missing',
        'extra',
        'bool_owner',
        'negative',
        'oversized',
        'bad_input',
        'bad_condition',
        'bad_presence',
        'json_owner',
    ],
)
def test_complete_product_structure_is_validated_before_taking_any_owner(
    malformed,
):
    data = copy.deepcopy(_data('target_column'))
    if malformed == 'missing':
        del data['rows']
    elif malformed == 'extra':
        data['action'] = 'advance'
    elif malformed == 'bool_owner':
        data['owner'] = True
    elif malformed == 'negative':
        data['owner'] = -1
    elif malformed == 'oversized':
        data['owner'] = 2**32
    elif malformed == 'bad_input':
        data['input'] = []
    elif malformed == 'bad_condition':
        data['condition_values'][1]['converted']['extra'] = 0
    elif malformed == 'bad_presence':
        data['condition_values'][1]['converted']['presence'] = False
    else:
        data['owner'] = 17
    owners = {number: object() for number in (4, 5, 6, 7)}
    finished, handle = _finished('target_column', data, owners)
    with pytest.raises(CodecError):
        products.read(finished)
    assert handle.takes == [17]
    assert handle.attachments == owners


@pytest.mark.parametrize(
    'change', ['unknown', 'missing_owner', 'extra_owner', 'bad_shape']
)
def test_invalid_bind_does_not_partially_publish_attachments(change):
    kind, data, owners = 'source', copy.deepcopy(_data('source')), {4: object()}
    if change == 'unknown':
        kind = 'request'
    elif change == 'missing_owner':
        owners = {}
    elif change == 'extra_owner':
        owners[99] = object()
    else:
        del data['target']
    bindings = Bindings()
    bindings.add(b'untouched')
    with pytest.raises(CodecError):
        products.bind(Product(kind, data, owners), bindings)
    assert bindings.attachments == {2: b'untouched'}


def test_malformed_terminal_request_keeps_state_owners_unread():
    data = _materialized()
    data['state']['payload'] = True
    finished, handle = _finished('request', data, {11: object()})
    with pytest.raises(CodecError):
        products.request(finished)
    assert handle.takes == [17]


def test_source_temporal_meaning_and_completed_notice_have_no_owner_slots():
    source = _data('source')
    source['tables'][0]['temporal_columns'] = ['value']
    actual = object()
    bindings = Bindings()
    rebound = products.bind(Product('source', source, {4: actual}), bindings)
    assert rebound['tables'] == source['tables']
    assert bindings.attachments == {2: actual}
    meaning = _data('column_meaning')
    meaning['notice'] = {
        'code': 'TIME_VALUES_UNREAD',
        'sentence': 'native_time_values_unread',
        'message': 'native time conversion',
        'fields': {'rows': {'t': 'u64', 'v': '4'}},
    }
    before = copy.deepcopy(meaning)
    bindings = Bindings()
    assert (
        products.bind(Product('column_meaning', meaning, {}), bindings)
        == before
    )
    assert meaning == before
    assert bindings.attachments == {}


@pytest.mark.parametrize(
    'notice',
    [
        3,
        {'code': 'TIME_VALUES_UNREAD'},
        {
            'code': 'TIME_VALUES_UNREAD',
            'sentence': 'native_time_values_unread',
            'message': 'actual',
            'extra': True,
        },
    ],
)
def test_completed_notice_reader_rejects_malformed_data_before_any_owner_take(
    notice,
):
    data = _data('column_meaning')
    data['notice'] = notice
    bindings = Bindings()
    with pytest.raises(CodecError):
        products.bind(Product('column_meaning', data, {}), bindings)
    assert bindings.attachments == {}
