# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict optional inputs authenticated by a separate original host witness."""

from __future__ import annotations

from support import require, same_json

DTYPES = {'object', 'int64', 'float64', 'bool', 'string', 'str'} | {
    f'datetime64[{unit}]' for unit in ('s', 'ms', 'us', 'ns')
}
UNITS = ('s', 'ms', 'us', 'ns')
FAMILIES = {'request.results', 'request.merge'}
SOURCES = {
    'packages/structured-data-models-client/src/sdm_client/result.py',
    'rust/bindings/python/python/sdm_core/answers.py',
    'rust/bindings/python/python/sdm_core/describe.py',
}


def fields(value, names, code):
    require(type(value) is dict and set(value) == set(names), code)


def integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def descriptor(host):
    fields(
        host,
        {'year', 'today', 'pads_year', 'tznames', 'timestamp_series_units'},
        'host.fields',
    )
    require(integer(host['year'], -(2**63), 2**63 - 1), 'host.year')
    today = host['today']
    require(
        type(today) is list
        and len(today) == 3
        and integer(today[0], -(2**63), 2**63 - 1)
        and all(integer(value, 0, 2**32 - 1) for value in today[1:]),
        'host.today',
    )
    require(type(host['pads_year']) is bool, 'host.pads_year')
    require(
        type(host['tznames']) is list
        and all(type(value) is str for value in host['tznames']),
        'host.tznames',
    )
    units = host['timestamp_series_units']
    require(
        type(units) is list
        and len(units) == 4
        and all(
            unit in UNITS and UNITS.index(unit) >= position
            for position, unit in enumerate(units)
        ),
        'host.timestamp_series_units',
    )


def primitive(entry):
    fields(
        entry, {'operation', 'arguments', 'outcome'}, 'host.primitive_fields'
    )
    operation, arguments, outcome = (
        entry['operation'],
        entry['arguments'],
        entry['outcome'],
    )
    require(
        type(operation) is str
        and operation in {'infer_column', 'column_order', 'index_constructor'},
        'host.operation',
    )
    if operation == 'index_constructor':
        fields(
            arguments, {'values', 'name', 'row_count'}, 'host.index_arguments'
        )
        fields(arguments['values'], {'list'}, 'host.index_values')
        names = arguments['values']['list']
        require(type(names) is list, 'host.index_names')
        for name in names:
            fields(name, {'scalar'}, 'host.index_name')
            require(type(name['scalar']) is str, 'host.index_name_type')
        require(
            same_json(arguments['name'], {'scalar': None})
            and integer(arguments['row_count'], 0, 2**63 - 1),
            'host.index_options',
        )
        fields(
            outcome,
            {'status', 'dtype', 'class', 'length'},
            'host.index_outcome',
        )
        require(
            outcome['status'] == 'success'
            and type(outcome['length']) is int
            and outcome['length'] == len(names),
            'host.index_length',
        )
        allowed = (
            {('Index', 'str'), ('Index', 'object')}
            if names
            else {('RangeIndex', 'int64')}
            if arguments['row_count'] == 0
            else {('Index', 'object')}
        )
        require(
            type(outcome['class']) is str
            and type(outcome['dtype']) is str
            and (outcome['class'], outcome['dtype']) in allowed,
            'host.index_representation',
        )
        return
    if operation == 'infer_column':
        fields(arguments, {'list'}, 'host.infer_arguments')
        require(type(arguments['list']) is list, 'host.infer_cells')
        fields(
            outcome,
            {'status', 'dtype', 'index_class', 'length'},
            'host.infer_outcome',
        )
        require(
            outcome['status'] == 'success'
            and type(outcome['dtype']) is str
            and outcome['dtype'] in DTYPES
            and outcome['index_class'] == 'RangeIndex'
            and type(outcome['length']) is int
            and outcome['length'] == len(arguments['list']),
            'host.infer_representation',
        )
        return
    fields(
        arguments,
        {'column', 'stable', 'ascending', 'missing_last'},
        'host.order_arguments',
    )
    require(
        all(
            type(arguments[key]) is bool
            for key in ('stable', 'ascending', 'missing_last')
        ),
        'host.order_flags',
    )
    fields(arguments['column'], {'series'}, 'host.order_column')
    column = arguments['column']['series']
    fields(
        column,
        {'dtype', 'name', 'index', 'attrs', 'values', 'categories', 'ordered'},
        'host.order_series',
    )
    fields(column['values'], {'list'}, 'host.order_values')
    require(type(column['values']['list']) is list, 'host.order_cells')
    length = len(column['values']['list'])
    fields(column['index'], {'index'}, 'host.order_index')
    index = column['index']['index']
    fields(
        index,
        {'class', 'dtype', 'names', 'values', 'levels', 'codes'},
        'host.order_index_fields',
    )
    require(
        same_json(
            index['values'],
            {'list': [{'scalar': position} for position in range(length)]},
        ),
        'host.order_row_labels',
    )
    fields(outcome, {'status', 'permutation'}, 'host.order_outcome')
    positions = outcome['permutation']
    require(
        outcome['status'] == 'success'
        and type(positions) is list
        and len(positions) == length
        and all(type(position) is int for position in positions)
        and sorted(positions) == list(range(length)),
        'host.order_permutation',
    )


def validate(value, witness):
    fields(value, {'host', 'primitives'}, 'host.input_fields')
    descriptor(value['host'])
    require(type(value['primitives']) is list, 'host.primitives')
    for entry in value['primitives']:
        primitive(entry)
    require(same_json(value, witness), 'host.original_witness')


def input_join(replay, coordinate, original):
    require(
        type(replay) is dict
        and set(replay)
        in (
            {'generator', 'generator_input'},
            {'generator', 'generator_input', 'host_inputs'},
        ),
        'host.replay_fields',
    )
    require(
        same_json(replay['generator'], coordinate)
        and same_json(replay['generator_input'], original['input']),
        'generator.original_input_join',
    )
    if 'host_inputs' in replay:
        require(coordinate['name'] in FAMILIES, 'host.family')
        require('host_inputs_witness' in original, 'host.missing_witness')
        validate(replay['host_inputs'], original['host_inputs_witness'])


def witness(root, name, value):
    require(name in FAMILIES, 'host.family')
    from generator_host_capture import collect_host_inputs

    result = collect_host_inputs(root, name, value)
    validate(result, result)
    return result


def joined_case(case, coordinate, original):
    replay = case['replay_input']
    input_join(replay, coordinate, original)
    if 'host_inputs' not in replay:
        return case
    return {
        **case,
        'replay_input': {
            'generator': replay['generator'],
            'generator_input': replay['generator_input'],
        },
    }
