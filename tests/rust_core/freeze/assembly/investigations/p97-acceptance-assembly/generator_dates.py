# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use only the finite approved date policy and authenticated actual lane samples."""

import generator_date_policy as policy
from support import checked_path, require, same_json, unique_json

POLICY = 'tests/rust_core/freeze/date-reference-policy.json'


def original_lanes(receipt):
    key = f'{policy.FAMILY}/{policy.SEED}'
    result = {}
    for major, expected in (('2', '2.3.3'), ('3', '3.0.6')):
        executions = [
            execution
            for execution in receipt['executions']
            if execution.get('pandas_major') == major
            and key in execution.get('streams', {})
        ]
        require(bool(executions), 'date.authenticated_executions')
        observed = {}
        for execution in executions:
            require(
                execution['failure'] is None
                and execution['pandas_version'] == expected
                and execution['metadata']['pandas_version'] == expected
                and execution['python_version']
                == execution['metadata']['python_version']
                and execution['native_sha256']
                == execution['metadata']['native_sha256'],
                'date.authenticated_runtime',
            )
            require(
                not set(observed) & set(execution['streams'][key]),
                'date.duplicate_original_coordinate',
            )
            observed.update(execution['streams'][key])
        require(
            same_json(observed, receipt['samples'][major][key]),
            'date.executed_original_samples',
        )
        result[major] = {
            'sources': receipt['sources'],
            'pandas_version': expected,
            'samples': observed,
        }
    return result


def validate(root, case, receipt):
    correction = case['corrected_reference']
    if correction is None or correction['dv'] != 'DV114':
        return
    coordinate = case['replay_input']['generator']
    require(coordinate['name'] == policy.FAMILY, 'date.correction_family')
    require(root is not None, 'date.source_root')
    bound = unique_json(checked_path(root, POLICY).read_bytes())
    lanes = original_lanes(receipt)
    require(
        same_json(bound, policy.generate(lanes['2'], lanes['3'])),
        'date.approved_finite_policy',
    )
    policy.verify(case, lanes['2'], lanes['3'], bound)
