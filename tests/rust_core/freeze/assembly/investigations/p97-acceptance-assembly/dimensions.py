# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite reviewed matrix templates bind constructor, column and public query."""

from __future__ import annotations

from support import INVESTIGATIONS, contract, load, require, same_json


def template_binding(recipe: dict, templates: dict) -> dict | None:
    if 'matrix' not in recipe:
        return None
    matrix = recipe['matrix']
    require(
        set(matrix) == {'template', 'dtype', 'values'},
        'dimension.matrix_fields',
    )
    require(matrix['template'] in templates, 'dimension.unreviewed_template')
    template = templates[matrix['template']]
    require(
        set(template)
        == {
            'entrypoint',
            'operation',
            'role',
            'backend',
            'task',
            'table',
            'column',
            'input',
            'dtype_ids',
        },
        'dimension.template_fields',
    )
    require(
        template['role'] in contract.ROLES
        and template['backend'] in contract.BACKENDS
        and template['task'] in contract.TASKS
        and matrix['dtype'] in template['dtype_ids'],
        'dimension.template_scope',
    )
    require(
        recipe['entrypoint'] == template['entrypoint']
        and recipe['operation'] == template['operation'],
        'dimension.entrypoint',
    )
    fixed = {
        key: value for key, value in recipe['input'].items() if key != 'tables'
    }
    require(
        same_json(fixed, template['input'])
        and fixed['backend'] == template['backend'],
        'dimension.query_schema_backend',
    )
    require(
        type(matrix['values']) is list and bool(matrix['values']),
        'dimension.values',
    )
    return template


def construct(
    recipe: dict, templates: dict, frames: dict, operations
) -> dict | None:
    template = template_binding(recipe, templates)
    if template is None:
        return None
    import pandas as pandas

    matrix = recipe['matrix']
    probe = load(
        '_p97_matrix_dtype_constructors',
        INVESTIGATIONS / 'p97-base-contract-staging/probe_dtypes.py',
    )
    constructors = dict(probe.cases())
    require(matrix['dtype'] in constructors, 'dimension.constructor_missing')
    prototype = constructors[matrix['dtype']]()
    series = pandas.Series(
        operations.decode(matrix['values']), dtype=prototype.dtype
    )
    table, column = template['table'], template['column']
    require(
        table in frames
        and column in frames[table]
        and len(series) == len(frames[table]),
        'dimension.actual_column',
    )
    frames[table][column] = series
    require(
        frames[table][column].dtype == prototype.dtype, 'dimension.actual_dtype'
    )
    return {
        'dtype': matrix['dtype'],
        'role': template['role'],
        'backend': template['backend'],
        'task': template['task'],
        'input_data_sha256': contract.digest(
            {
                'frames': {
                    name: operations._frame_out(frame)
                    for name, frame in frames.items()
                },
                'request': template['input'],
            }
        ),
        'template': matrix['template'],
        'table': table,
        'column': column,
    }
