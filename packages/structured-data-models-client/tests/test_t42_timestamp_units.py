# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumed native timestamp wire facts, with no new duration or time-of-day units."""

import pytest

from sdm_client._rust import codec


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('count', [-(2**63), -(2**63) + 1, -1, 0, 1, 2**63 - 1])
@pytest.mark.parametrize('zone', [None, 'America/Los_Angeles'])
def test_timestamp_codec_preserves_native_unit_zone_and_raw_i64(
    unit, count, zone
):
    wire = {'t': 'ts', 'v': str(count), 'unit': unit, 'zone': zone}
    assert codec.timestamp(count, unit, zone) == wire
    value = codec.decode_scalar(wire)
    assert value.tag == 'ts'
    assert value.value == count
    assert value.unit == unit
    assert value.zone == zone


@pytest.mark.parametrize('unit', ['s', 'ms'])
@pytest.mark.parametrize('kind', ['duration', 'time64'])
def test_coarse_timestamp_units_do_not_expand_other_temporal_codecs(unit, kind):
    with pytest.raises(codec.CodecError):
        codec.decode_scalar({'t': kind, 'v': '1', 'unit': unit})


@pytest.mark.parametrize(
    'unit', ['', 'h', 'MS', 'US', 'nanosecond', 'microsecond']
)
def test_unknown_timestamp_units_are_refused_by_both_codec_directions(unit):
    with pytest.raises(codec.CodecError):
        codec.timestamp(1, unit, None)
    with pytest.raises(codec.CodecError):
        codec.decode_scalar({'t': 'ts', 'v': '1', 'unit': unit, 'zone': None})
