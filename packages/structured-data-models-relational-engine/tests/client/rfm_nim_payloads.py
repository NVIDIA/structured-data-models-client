# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

NIM_V1_PREDICTION_PATH = '/v1/predictions'
NIM_V1_SESSIONS_PATH = '/v1/sessions'
NIM_HEALTH_READY_PATH = '/v1/health/ready'
SDK_CONFIG_PATH = '/config'
SDK_V1_CONNECTORS_PATH = '/v1/connectors'
SDK_V1_RFM_PARSE_QUERY_PATH = '/v1/rfm/parse_query'
SDK_V1_RFM_VALIDATE_QUERY_PATH = '/v1/rfm/validate_query'

_NIM_V1_SMOKE_PAYLOAD: dict[str, Any] = {
    'model': 'kumo-relational',
    'task': {
        'kind': 'classification',
        'target': {
            'column_name': 'status',
            'dtype': 'bool',
        },
        'entity_table_names': ['accounts'],
    },
    'schema': {
        'instance_table': {
            'columns': {
                'instance_id': {
                    'dtype': 'int64',
                    'stype': 'ID',
                    'nullable': False,
                },
                'status': {
                    'dtype': 'bool',
                    'stype': 'categorical',
                },
                'anchor_time': {
                    'dtype': 'timestamp[us]',
                    'stype': 'timestamp',
                },
            },
            'primary_key': 'instance_id',
        },
        'related_tables': {
            'accounts': {
                'columns': {
                    'instance_id': {
                        'dtype': 'int64',
                        'stype': 'ID',
                        'nullable': False,
                    },
                    'segment': {
                        'dtype': 'string',
                        'stype': 'categorical',
                    },
                },
                'primary_key': 'instance_id',
            },
        },
        'relationships': [
            {
                'source_columns': ['instance_id'],
                'target_table': 'accounts',
                'target_columns': ['instance_id'],
            }
        ],
    },
    'context': {
        'instance_table': {
            'format': 'arrays',
            'columns': ['instance_id', 'status', 'anchor_time'],
            'rows': [
                [501, True, '2025-01-01T00:00:00Z'],
                [502, False, '2025-01-02T00:00:00Z'],
            ],
        },
        'related_tables': {
            'accounts': {
                'format': 'arrays',
                'columns': ['instance_id', 'segment'],
                'rows': [
                    [501, 'enterprise'],
                    [502, 'startup'],
                ],
            },
        },
    },
    'predict': {
        'instance_table': {
            'format': 'arrays',
            'columns': ['instance_id', 'anchor_time'],
            'rows': [[601, '2025-02-01T00:00:00Z']],
        },
        'related_tables': {
            'accounts': {
                'format': 'arrays',
                'columns': ['instance_id', 'segment'],
                'rows': [[601, 'enterprise']],
            },
        },
    },
    'output': {
        'fields': ['prediction', 'probabilities'],
    },
    'inference': {
        'run_mode': 'best',
    },
}


def nim_v1_smoke_payload() -> dict[str, Any]:
    """Payload mirrored from kumo_rfm_nim/scripts/run_container_smoke.sh."""
    return deepcopy(_NIM_V1_SMOKE_PAYLOAD)


def nim_v1_text_stringlist_payload() -> dict[str, Any]:
    """Exercise automatic text token payloads and GloVe initialization."""
    payload = nim_v1_smoke_payload()
    accounts_schema = payload['schema']['related_tables']['accounts']
    accounts_schema['columns']['description'] = {
        'dtype': 'stringlist',
        'stype': 'text',
    }
    for split, values in (
        ('context', [['enterprise'], ['startup']]),
        ('predict', [['enterprise']]),
    ):
        accounts = payload[split]['related_tables']['accounts']
        accounts['columns'].append('description')
        for row, value in zip(accounts['rows'], values, strict=True):
            row.append(value)
    return payload


def nim_v1_prediction_only_output_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['output']['fields'] = ['prediction']
    return payload


def nim_v1_fast_run_mode_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['inference']['run_mode'] = 'fast'
    return payload


def nim_v1_without_inference_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload.pop('inference')
    return payload


def nim_v1_two_predict_rows_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['predict']['instance_table']['rows'] = [
        [601, '2025-02-01T00:00:00Z'],
        [602, '2025-02-02T00:00:00Z'],
    ]
    payload['predict']['related_tables']['accounts']['rows'] = [
        [601, 'enterprise'],
        [602, 'startup'],
    ]
    return payload


def nim_v1_reordered_predict_rows_payload() -> dict[str, Any]:
    payload = nim_v1_two_predict_rows_payload()
    payload['predict']['instance_table']['rows'] = [
        [602, '2025-02-02T00:00:00Z'],
        [601, '2025-02-01T00:00:00Z'],
    ]
    return payload


def nim_v1_empty_predict_rows_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['predict']['instance_table']['rows'] = []
    payload['predict']['related_tables']['accounts']['rows'] = []
    return payload


def nim_v1_explicit_utc_offset_timestamp_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['instance_table']['rows'] = [
        [501, True, '2025-01-01T00:00:00.123456+00:00'],
        [502, False, '2025-01-02T03:04:05-05:00'],
    ]
    payload['predict']['instance_table']['rows'] = [
        [601, '2025-02-01T12:30:45.000001+02:00'],
    ]
    return payload


def nim_v1_regression_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['task'] = {
        'kind': 'regression',
        'target': {
            'column_name': 'score',
            'dtype': 'float64',
        },
        'entity_table_names': ['accounts'],
    }
    payload['schema']['instance_table']['columns'] = {
        'instance_id': {
            'dtype': 'int64',
            'stype': 'ID',
            'nullable': False,
        },
        'score': {
            'dtype': 'float64',
            'stype': 'numerical',
        },
        'score_hint': {
            'dtype': 'float64',
            'stype': 'numerical',
        },
        'anchor_time': {
            'dtype': 'timestamp[us]',
            'stype': 'timestamp',
        },
    }
    payload['context']['instance_table'] = {
        'format': 'arrays',
        'columns': ['instance_id', 'score', 'score_hint', 'anchor_time'],
        'rows': [
            [501, 0.25, 1.0, '2025-01-01T00:00:00Z'],
            [502, 0.75, 2.0, '2025-01-02T00:00:00Z'],
        ],
    }
    payload['predict']['instance_table'] = {
        'format': 'arrays',
        'columns': ['instance_id', 'score_hint', 'anchor_time'],
        'rows': [[601, 1.5, '2025-02-01T00:00:00Z']],
    }
    payload['output']['fields'] = ['prediction']
    return payload


def nim_v1_multiclass_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['task'] = {
        'kind': 'multiclass_classification',
        'target': {
            'column_name': 'tier',
            'dtype': 'string',
            'classes': ['bronze', 'silver'],
        },
        'entity_table_names': ['accounts'],
    }
    payload['schema']['instance_table']['columns'] = {
        'instance_id': {
            'dtype': 'int64',
            'stype': 'ID',
            'nullable': False,
        },
        'tier': {
            'dtype': 'string',
            'stype': 'categorical',
        },
        'score_hint': {
            'dtype': 'float64',
            'stype': 'numerical',
        },
        'anchor_time': {
            'dtype': 'timestamp[us]',
            'stype': 'timestamp',
        },
    }
    payload['context']['instance_table'] = {
        'format': 'arrays',
        'columns': ['instance_id', 'tier', 'score_hint', 'anchor_time'],
        'rows': [
            [501, 'bronze', 1.0, '2025-01-01T00:00:00Z'],
            [502, 'silver', 2.0, '2025-01-02T00:00:00Z'],
        ],
    }
    payload['predict']['instance_table'] = {
        'format': 'arrays',
        'columns': ['instance_id', 'score_hint', 'anchor_time'],
        'rows': [[601, 1.5, '2025-02-01T00:00:00Z']],
    }
    return payload


def sdk_v1_smoke_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['version'] = 'v1'
    payload['metadata'] = {
        'source_query': 'rfm-sdk-nim-contract-smoke',
    }
    return payload


def nim_v1_session_create_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload.pop('predict')
    payload.pop('output')
    payload.pop('inference')
    return payload


def nim_v1_session_predict_minimal_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    return {
        'predict': payload['predict'],
        'output': payload['output'],
    }
