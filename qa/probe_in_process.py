# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDMClient.in_process() against the NIMs, request for request.

Each request goes to a NIM and to the model in this process, and the two
answers are compared column by column. Against a NIM running the same
structured-data-models release on the same device, CPU or GPU, every number is
identical. With this process on a CPU and the NIM on a GPU, the two compute at
different precision and draw different random numbers, so set
SDM_IN_PROCESS_TOLERANCE to the largest difference to accept.
"""

from __future__ import annotations

import os

import data
import numpy as np
from harness import RELATIONAL, TABULAR, fail, note, probe, save
from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Schema,
    SDMClient,
)

AREA = 'in-process'
TOLERANCE = float(os.environ.get('SDM_IN_PROCESS_TOLERANCE', '0'))


def agree(entry, hosted, local) -> None:
    a, b = hosted.to_pandas(), local.to_pandas()
    if list(a.columns) != list(b.columns):
        fail(entry, f'columns {list(a.columns)} and {list(b.columns)}')
        return
    worst = 0.0
    for column in a.columns:
        if a[column].dtype.kind == 'f':
            x = a[column].to_numpy(dtype=float)
            y = b[column].to_numpy(dtype=float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                fail(entry, f'{column} is missing on one side only')
                return
            differ = (x != y) & ~np.isnan(x)
            if differ.any():
                worst = max(worst, float(np.abs(x - y)[differ].max()))
        elif a[column].astype(str).tolist() != b[column].astype(str).tolist():
            fail(entry, f'{column} differs')
            return
    if worst > TOLERANCE:
        fail(entry, f'largest difference {worst:.3g} exceeds {TOLERANCE:g}')
    else:
        note(entry, f'largest difference {worst:.3g}')


def tabular_requests():
    for name, frame, key, target in (
        ('adult', data.adult().head(1500), 'person_id', 'income'),
        (
            'wine',
            data.wine().sample(1500, random_state=0),
            'wine_id',
            'quality',
        ),
        ('titanic', data.titanic(), 'passenger_id', 'survived'),
        ('bank', data.bank().sample(2000, random_state=0), 'contact_id', 'y'),
        ('students', data.students(), 'student_id', 'Target'),
    ):
        schema = Schema.from_data({name: frame}, primary_keys={name: key})
        for column in schema[name]:
            if schema[name][column].stype == 'text':
                schema[name][column].stype = 'categorical'
        query = PredictiveQuery(
            f'PREDICT {name}.{target} FOR EACH {name}.{key}', schema
        )
        yield name, query.materialize(indices=frame[key].head(40).tolist())


def relational_requests():
    schema = data.northwind_ready()
    customers = data.northwind('Customers')['CustomerID'].head(12).tolist()
    for text in (
        'PREDICT COUNT(Orders.*, 0, 90, days) FOR EACH Customers.CustomerID',
        'PREDICT SUM(Orders.Freight, 0, 90, days) FOR EACH Customers.CustomerID',
        'PREDICT COUNT(Orders.*, 0, 90, days) > 0 FOR EACH Customers.CustomerID',
    ):
        yield text, PredictiveQuery(text, schema).materialize(indices=customers)


def run() -> None:
    with (
        SDMClient(url=TABULAR, timeout=900) as hosted,
        SDMClient.in_process() as local,
    ):
        for name, request in tabular_requests():
            with probe(
                AREA, f'tabular {name}', 'the answer the NIM gives'
            ) as e:
                agree(
                    e,
                    hosted.predict(request, model=KumoTabular()),
                    local.predict(request, model=KumoTabular()),
                )
    with (
        SDMClient(url=RELATIONAL, timeout=900) as hosted,
        SDMClient.in_process() as local,
    ):
        for text, request in relational_requests():
            with probe(AREA, text, 'the answer the NIM gives') as e:
                agree(
                    e,
                    hosted.predict(request, model=KumoRelational()),
                    local.predict(request, model=KumoRelational()),
                )
            with probe(AREA, f'{text}, in a session', 'the same answer') as e:
                with (
                    hosted.fit(request, model=KumoRelational()) as pinned,
                    local.fit(request, model=KumoRelational()) as held,
                ):
                    agree(e, pinned.predict(request), held.predict(request))


if __name__ == '__main__':
    run()
    save('in_process.json')
