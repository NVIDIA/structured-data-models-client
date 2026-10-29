# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit SQL timestamp transport preserves native Arrow values and fields."""

from __future__ import annotations

import copy

import pyarrow as pa
import pytest

from sdm_client._rust.providers.result_columns import result_columns


def conversion(column=0, name='when', source_kind='wall_time'):
    return {
        'conversion': 'timestamp_micros',
        'column': column,
        'name': name,
        'source_kind': source_kind,
    }


@pytest.mark.parametrize('source_kind', ['wall_time', 'instant'])
def test_timestamp_transport_preserves_complete_signed_microseconds_and_nulls(
    source_kind,
):
    counts = [
        -(2**63),
        -(2**63) + 1,
        -(2**53) - 1,
        -1,
        None,
        0,
        2**53 + 1,
        2**63 - 1,
    ]
    field = pa.field(
        'when',
        pa.int64(),
        nullable=True,
        metadata={b'column-note': b'original metadata'},
    )
    schema = pa.schema(
        [field, pa.field('id', pa.int64(), nullable=False)],
        metadata={
            b'session-zone': b'Pacific/Kiritimati',
            b'table-note': b'exact',
        },
    )
    values = pa.chunked_array(
        [
            pa.array(counts[:3], type=pa.int64()),
            pa.array(counts[3:6], type=pa.int64()),
            pa.array(counts[6:], type=pa.int64()),
        ]
    )
    original = pa.Table.from_arrays(
        [values, pa.array(list(range(len(counts))), type=pa.int64())],
        schema=schema,
    )
    descriptor = [conversion(source_kind=source_kind)]
    before = copy.deepcopy(descriptor)
    restored = result_columns(original, descriptor)
    assert descriptor == before
    assert original.schema == schema
    assert original.column(0).to_pylist() == counts
    assert restored.column_names == ['when', 'id']
    assert restored.num_rows == original.num_rows
    assert restored.schema.metadata == schema.metadata
    assert restored.schema.field(0) == field.with_type(pa.timestamp('us'))
    assert restored.schema.field(0).metadata == field.metadata
    assert restored.schema.field(0).nullable is True
    assert restored.schema.field(0).type.tz is None
    assert restored.column(0).cast(pa.int64()).to_pylist() == counts
    assert restored.column(0).null_count == original.column(0).null_count == 1
    assert restored.schema.field(1) == schema.field(1)
    assert restored.column(1).equals(original.column(1))


def test_reordered_conversions_address_duplicate_names_by_actual_ordinal():
    fields = [
        pa.field('same', pa.int64(), nullable=False, metadata={b'at': b'0'}),
        pa.field('same', pa.int64(), nullable=True, metadata={b'at': b'1'}),
        pa.field('integer', pa.int64()),
    ]
    source = pa.Table.from_arrays(
        [
            pa.array([2**53 + 1, -1]),
            pa.array([None, 9], type=pa.int64()),
            pa.array([7, 8]),
        ],
        schema=pa.schema(fields, metadata={b'order': b'kept'}),
    )
    restored = result_columns(
        source,
        [
            conversion(1, 'same', 'instant'),
            conversion(0, 'same', 'wall_time'),
        ],
    )
    assert restored.column_names == source.column_names
    assert restored.schema.metadata == source.schema.metadata
    for at in (0, 1):
        assert restored.schema.field(at) == fields[at].with_type(
            pa.timestamp('us')
        )
        assert restored.schema.field(at).metadata == fields[at].metadata
        assert restored.column(at).cast(pa.int64()).equals(source.column(at))
    assert restored.schema.field(2) == fields[2]
    assert restored.column(2).equals(source.column(2))


@pytest.mark.parametrize('source_kind', ['wall_time', 'instant'])
def test_empty_and_all_null_bigint_results_keep_the_selected_timestamp_meaning(
    source_kind,
):
    for counts in ([], [None, None]):
        source = pa.table({'when': pa.array(counts, type=pa.int64())})
        restored = result_columns(source, [conversion(source_kind=source_kind)])
        assert restored.schema.field(0).type == pa.timestamp('us')
        assert restored.column(0).cast(pa.int64()).to_pylist() == counts
        assert restored.num_rows == len(counts)


def test_absent_conversion_returns_the_actual_table_without_timestamp_guessing():
    for source in (
        pa.table({}),
        pa.table({'when': pa.array([0, 123], type=pa.int64())}),
        pa.table({'when': pa.array([0, 123], type=pa.timestamp('us'))}),
    ):
        assert result_columns(source, []) is source


@pytest.mark.parametrize(
    'invalid',
    [
        None,
        [],
        {},
        {'conversion': 'timestamp_micros', 'column': 0, 'name': 'when'},
        {'conversion': 'zoned_text_utc'},
        {**conversion(), 'conversion': 'guess_timestamp'},
        {**conversion(), 'column': True},
        {**conversion(), 'column': -1},
        {**conversion(), 'column': 1},
        {**conversion(), 'column': 2**64},
        {**conversion(), 'column': 0.5},
        {**conversion(), 'column': '0'},
        {**conversion(), 'name': 'another'},
        {**conversion(), 'name': 0},
        {**conversion(), 'source_kind': 'session_time'},
        {**conversion(), 'source_kind': None},
        {**conversion(), 'source_kind': False},
        {**conversion(), 'provider': 'an-origin-is-not-a-conversion'},
    ],
)
def test_malformed_or_unknown_descriptors_fail_without_changing_the_input(
    invalid,
):
    source = pa.table({'when': pa.array([0, None], type=pa.int64())})
    schema = source.schema
    with pytest.raises(ValueError):
        result_columns(source, [invalid])
    assert source.schema == schema
    assert source.column(0).to_pylist() == [0, None]


def test_duplicate_ordinals_are_rejected_even_with_the_same_matching_name():
    source = pa.table({'when': pa.array([0], type=pa.int64())})
    with pytest.raises(ValueError):
        result_columns(
            source, [conversion(), conversion(source_kind='instant')]
        )
    assert source.column(0).type == pa.int64()


@pytest.mark.parametrize(
    'array',
    [
        pa.array([0], type=pa.int32()),
        pa.array([0], type=pa.uint64()),
        pa.array([0.0], type=pa.float64()),
        pa.array(['0']),
        pa.nulls(1),
        pa.array([0], type=pa.timestamp('us')),
    ],
)
def test_timestamp_conversion_requires_actual_bigint_storage(array):
    source = pa.table({'when': array})
    with pytest.raises(ValueError):
        result_columns(source, [conversion()])
    assert source.column(0).type == array.type


def test_an_invalid_later_descriptor_does_not_publish_a_partial_conversion():
    source = pa.table(
        {
            'first': pa.array([0], type=pa.int64()),
            'second': pa.array([1], type=pa.int32()),
        }
    )
    with pytest.raises(ValueError):
        result_columns(
            source, [conversion(0, 'first'), conversion(1, 'second')]
        )
    assert source.schema.types == [pa.int64(), pa.int32()]
