# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate descriptor-only families without extending primitive rules."""

from __future__ import annotations

import generator_host
from support import require, same_json

FAMILIES = {'labels.question_batches', 'schema.date_columns', 'rfm.sequences'}
SOURCES = {'rust/bindings/python/python/sdm_core/describe.py'}


def validate(value, witness, name):
    require(name in FAMILIES, 'descriptor.family')
    generator_host.fields(
        value, {'host', 'primitives'}, 'descriptor.input_fields'
    )
    require(
        type(value['primitives']) is list and not value['primitives'],
        'descriptor.no_primitives',
    )
    host = value['host']
    if name == 'rfm.sequences':
        require(
            type(host) is dict and 'numpy_text_prefix' in host,
            'descriptor.sequence_capability',
        )
        require(
            type(host['numpy_text_prefix']) is bool,
            'descriptor.sequence_boolean',
        )
        host = {
            key: item
            for key, item in host.items()
            if key != 'numpy_text_prefix'
        }
    generator_host.descriptor(host)
    require(same_json(value, witness), 'descriptor.original_witness')


def witness(root, name, value):
    require(name in FAMILIES, 'descriptor.family')
    from generator_descriptor_capture import collect_descriptor

    measured = collect_descriptor(root, name)
    validate(measured, measured, name)
    return measured


def joined_case(case, coordinate, original):
    replay = case['replay_input']
    if coordinate['name'] not in FAMILIES:
        return case
    require(
        type(replay) is dict
        and set(replay)
        in (
            {'generator', 'generator_input'},
            {'generator', 'generator_input', 'host_inputs'},
        ),
        'descriptor.replay_fields',
    )
    require(
        same_json(replay['generator'], coordinate)
        and same_json(replay['generator_input'], original['input']),
        'descriptor.original_input',
    )
    if 'host_inputs' not in replay:
        return case
    require('host_inputs_witness' in original, 'descriptor.missing_witness')
    validate(
        replay['host_inputs'],
        original['host_inputs_witness'],
        coordinate['name'],
    )
    return {
        **case,
        'replay_input': {
            'generator': replay['generator'],
            'generator_input': replay['generator_input'],
        },
    }
