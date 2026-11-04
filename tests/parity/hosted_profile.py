# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hosted execution policy, separate from the supported capability matrix."""

import argparse
import json

from contract import cells, manifest

LOCAL = ('memory', 'sqlite', 'duckdb', 'postgres', 'csv', 'parquet', 'zip')
EMULATED = ('s3-csv', 's3-parquet')
WAREHOUSE = (
    'databricks-sql',
    'databricks-spark',
    'databricks-metric',
    'snowflake',
    'snowflake-semantic',
)
OPTIONAL = ('lakebase',)


def profile(data=None):
    data = manifest() if data is None else data
    declared = (*LOCAL, *EMULATED, *WAREHOUSE, *OPTIONAL)
    if len(set(declared)) != len(declared):
        raise ValueError('Connector has more than one execution policy')
    if set(declared) != set(data['connectors']):
        raise ValueError('Every supported connector needs an execution policy')
    return {
        'version': 1,
        'matrix_mode': 'materialization-only',
        'model_evidence': 'separate-pretrained-cpu-smoke',
        'required': [
            {
                'connector': connector,
                'service': service,
                'cells': sum(cell[1] == connector for cell in cells(data)),
            }
            for connectors, service in (
                (LOCAL, 'local'),
                (EMULATED, 'emulated-s3'),
                (WAREHOUSE, 'real-warehouse'),
            )
            for connector in connectors
        ],
        'optional': {
            'lakebase': 'Owner excluded managed Lakebase from required CI.',
            'real-s3': 'Local emulation does not certify AWS IAM or service behavior.',
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', action='store_true')
    args = parser.parse_args()
    value = profile()
    print(json.dumps({'include': value['required']} if args.matrix else value))
