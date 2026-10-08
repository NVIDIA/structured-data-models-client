# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's context run where the engine checks a task's references.

``KumoRelational._get_context`` checks that every entity a task names has a
row when ``_validate_references`` is set. sdm_client never sets it
(``rfm.py``, ``materialize_task``), so neither the corpus nor the driver
fixtures (``test_driver_fixtures.py``) reach it, and this test asks the
engine directly, watching the call as ``test_rfm_runs.py`` watches it: the
core's ``context`` run, driven from the call's inputs, must raise the same
class with the same text.

The other questions the engine refuses are driver fixtures now
(``tests/parity/fixtures/driver.json``), where the coverage measure sees
them. Without the core or the engine installed, as in CI's test job, it
skips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip('sdm_core')
pytest.importorskip('sdm_relational_engine')

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from test_rfm_runs import Boundary, _compare, _model_describer  # noqa: E402

QUERY_COUNT = (
    'PREDICT COUNT(purchases.*, 0, 30, days) FOR EACH customers.customer_id'
)


def test_unknown_reference() -> None:
    r"""A task naming a key no row holds, given to ``_get_context`` with its
    references checked."""
    from _parity import data as datagen
    from sdm_client import Schema
    from sdm_client.materializers.relational import RelationalMaterializer
    from sdm_relational_engine.rfm.rfm import KumoRelational

    frames = datagen.generate('marketplace', 0)
    schema = Schema.from_data(
        dict(frames), primary_keys={'stores': 'store_code'}
    )
    model = RelationalMaterializer(schema)._get_model()
    task = model._get_task_table(
        query=model._parse_query(QUERY_COUNT), indices=[1], random_seed=7
    )
    task._pred_df['ENTITY'] = [999]
    boundary = Boundary(KumoRelational, '_get_context', lambda m: m._sampler)
    with boundary.installed(_model_describer('_get_context')):
        with pytest.raises(Exception, match='unresolved row'):
            model._get_context(task, random_seed=7)
    assert len(boundary.calls) == 1
    raised = boundary.calls[0]['error']
    assert type(raised).__name__ == 'TaskReferenceError', repr(raised)
    count, problems = _compare(
        'context', boundary, lambda call: None, lambda v: None
    )
    assert count == 1
    assert not problems, '\n'.join(problems)
