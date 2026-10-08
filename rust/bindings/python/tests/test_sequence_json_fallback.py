# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import MappingProxyType

import pytest
from sdm_core.replay import replay_snapshot


def typed(value):
    if isinstance(value, float):
        return {'float': value.hex()}
    if isinstance(value, dict):
        return {
            'mapping': [
                [typed(key), typed(item)] for key, item in value.items()
            ]
        }
    if isinstance(value, list):
        return {'list': [typed(item) for item in value]}
    return {'scalar': value}


def replay_values(values):
    document = {
        'format': 'p97-generators-v1',
        'cases': [
            {
                'id': 'sequence-json',
                'depends_on': [],
                'expected': 'not an answer',
                'replay_input': {
                    'generator': {
                        'name': 'rfm.sequences',
                        'stream_seed': 0,
                        'ordinal': 0,
                    },
                    'generator_input': typed(
                        {
                            'calls': [
                                {
                                    'series': 'arrow',
                                    'values': values,
                                    'dtype': 'stringlist',
                                    'stype': None,
                                }
                            ]
                        }
                    ),
                },
            }
        ],
    }
    files = MappingProxyType(
        {
            'tests/parity/frozen-generators/sequence-json.json': json.dumps(
                document
            ).encode(),
        }
    )
    result = replay_snapshot(files, case_id='sequence-json')['sequence-json']
    assert result['family'] == 'original-generator-v1'
    assert result['observation']['error'] is None
    assert result['observation']['warnings'] == []
    return result['observation']['result']


def projected(values):
    import pandas as pd

    return [
        value if isinstance(value, list) else None if pd.isna(value) else value
        for value in values
    ]


@pytest.mark.parametrize(
    'text',
    [
        'Infinity',
        'NaN',
        '-Infinity',
        'true',
        'null',
        '"a"',
        '1.5',
        '9223372036854775808',
        '1e400',
        '{"a":1,"a":2}',
        '[true,null,"x"]',
        '"NaN"',
        '"Infinity"',
        '-0.0',
        '{"nested":[{"$float":"nan"},null]}',
    ],
)
def test_sequence_json_fallback_preserves_complete_mixed_values(text):
    values = [text, '[1]']
    assert replay_values(values) == typed(
        [
            {
                'output': {
                    'dtype': 'object',
                    'values': projected(
                        [json.loads(value) for value in values]
                    ),
                }
            },
        ]
    )


@pytest.mark.parametrize(
    'text', ['[1,]', '{"a":1,}', 'NaNx', '[1] trailing', '"bad\ntext"']
)
def test_malformed_json_retains_the_whole_column(text):
    values = ['[1]', text]
    assert replay_values(values) == typed(
        [
            {'output': {'dtype': 'string[pyarrow]', 'values': values}},
        ]
    )


@pytest.mark.parametrize(
    'values,dtype',
    [
        (['1', '2'], 'int64'),
        (['1', 'null'], 'float64'),
        (['true', 'false'], 'bool'),
        (['true', 'null'], 'object'),
        (['9223372036854775808', '1'], 'uint64'),
        (['18446744073709551616', '1.5'], 'object'),
        (['170141183460469231731687303715884105727', '1.5'], 'object'),
        (['9223372036854775808', '1.5'], 'float64'),
        (['-1', '9223372036854775808', '1.5'], 'object'),
        (['-9223372036854775809', '1.5'], 'object'),
        (['18446744073709551615', '1.5'], 'float64'),
        (['18446744073709551615', 'null'], 'float64'),
    ],
)
def test_json_mapping_preserves_pandas_numeric_inference(values, dtype):
    import pandas as pd
    import pyarrow as pa

    actual_original = pd.Series(values, dtype=pd.ArrowDtype(pa.string())).map(
        json.loads
    )
    assert str(actual_original.dtype) == dtype
    assert replay_values(values) == typed(
        [
            {
                'output': {
                    'dtype': dtype,
                    'values': projected(actual_original.tolist()),
                }
            },
        ]
    )


@pytest.mark.parametrize('missing', ['NaN', 'null'])
@pytest.mark.parametrize('other', ['[NaN]', 'Infinity', '-Infinity'])
def test_top_level_missing_projection_preserves_nested_and_infinite_values(
    missing, other
):
    import pandas as pd
    import pyarrow as pa

    values = [missing, other]
    original = pd.Series(values, dtype=pd.ArrowDtype(pa.string())).map(
        json.loads
    )
    assert replay_values(values) == typed(
        [
            {
                'output': {
                    'dtype': str(original.dtype),
                    'values': projected(original.tolist()),
                }
            },
        ]
    )
