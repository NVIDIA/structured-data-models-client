# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-client boundary pairs against a dedicated, configured deployment.

Run with -m live_nim and SDM_LIMIT_TEST_MODEL=kumo-tabular or kumo-relational.
Set SDM_LIMIT_TEST_URL for NIM, or SDM_LIMIT_TEST_ENDPOINT and
SDM_LIMIT_TEST_PROFILE for Databricks. Set SDM_LIMIT_TEST_CONTEXT_ROWS and
SDM_LIMIT_TEST_PREDICT_ROWS to the deployment's confirmed effective limits.
Use small limits on dedicated endpoints. This suite never configures a server.
Each test predicts at the limit before asking for one more row.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Schema,
    SDMClient,
)
from sdm_client.errors import NimRequestError

pytestmark = pytest.mark.live_nim


def request_for(context_rows, predict_rows, model):
    count = context_rows + predict_rows
    indices = list(range(context_rows, count))
    tables = {
        'users': pd.DataFrame(
            {
                'id': range(count),
                'feature': [i * 2.5 for i in range(count)],
                'value': [i * 1.25 for i in range(context_rows)]
                + [None] * predict_rows,
            }
        )
    }
    keys = {'users': 'id'}
    if isinstance(model, KumoRelational):
        tables['events'] = pd.DataFrame(
            {
                'event': range(count),
                'id': range(count),
                'feature': [i * 2.5 for i in range(count)],
            }
        )
        keys['events'] = 'event'
    schema = Schema.from_data(tables, primary_keys=keys)
    schema['users']['value'].stype = 'numerical'
    schema['users']['feature'].stype = 'numerical'
    options = (
        {'rows_per_call': predict_rows}
        if isinstance(model, KumoRelational)
        else {}
    )
    request = PredictiveQuery(
        'PREDICT users.value FOR EACH users.id', schema
    ).materialize(
        model=model,
        indices=indices,
        num_context_samples=context_rows,
        **options,
    )
    # Prevent sampling or automatic batching from hiding the tested boundary.
    assert request.context_rows == context_rows
    assert request.predict_rows == predict_rows
    if isinstance(model, KumoRelational):
        assert len(request.payloads) == 1
    return request, indices


@pytest.mark.parametrize('axis', ['context', 'predict'])
def test_exact_row_limit_succeeds_then_one_more_is_refused(axis):
    name = os.environ.get('SDM_LIMIT_TEST_MODEL')
    limit_text = os.environ.get(f'SDM_LIMIT_TEST_{axis.upper()}_ROWS')
    if not name or not limit_text:
        pytest.skip('set the dedicated model and confirmed row limit')
    models = {'kumo-tabular': KumoTabular, 'kumo-relational': KumoRelational}
    assert name in models, 'use an actual supported model, not a prototype'
    limit = int(limit_text)
    assert limit >= 2
    model = models[name]()
    endpoint = os.environ.get('SDM_LIMIT_TEST_ENDPOINT')
    if endpoint:
        assert endpoint not in {'kumo-score', 'kumo-score-probe'}
        from databricks.sdk import WorkspaceClient

        client = SDMClient.for_databricks_serving(
            endpoint,
            workspace_client=WorkspaceClient(
                profile=os.environ['SDM_LIMIT_TEST_PROFILE']
            ),
        )
    else:
        client = SDMClient(url=os.environ['SDM_LIMIT_TEST_URL'])
    with client:
        counts = {
            'context': min(
                8, int(os.environ.get('SDM_LIMIT_TEST_CONTEXT_ROWS', '8'))
            ),
            'predict': 1,
        }
        counts[axis] = limit
        request, indices = request_for(
            counts['context'], counts['predict'], model
        )
        frame = client.predict(request, model=model).to_pandas()
        assert len(frame) == counts['predict']
        assert frame['row_index'].tolist() == list(range(counts['predict']))
        assert frame['id'].astype(str).tolist() == [str(i) for i in indices]
        assert np.isfinite(frame['prediction'].to_numpy(dtype=float)).all()

        counts[axis] += 1
        oversized, _ = request_for(counts['context'], counts['predict'], model)
        with pytest.raises(NimRequestError) as refused:
            client.predict(oversized, model=model)
        assert refused.value.code == 'UNSUPPORTED_SHAPE'
        if endpoint:
            assert refused.value.status_code >= 400
            assert endpoint in str(refused.value)
        else:
            assert refused.value.status_code == 422
            assert f'{axis}.instance_table' in str(refused.value)
            assert f'exceeds limit {limit}' in str(refused.value)
