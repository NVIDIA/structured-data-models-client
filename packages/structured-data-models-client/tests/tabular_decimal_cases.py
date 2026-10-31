# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Decimal comparison journeys with standard contexts and frames."""

import warnings
from decimal import (
    Decimal,
    FloatOperation,
    Inexact,
    InvalidOperation,
    localcontext,
)
from typing import Any

import numpy as np
import pandas as pd
from tabular_numeric_pure_cases import Endpoint

CASES = [
    f'{family}/{entry}/{trap}/{direction}'
    for family in (
        'finite',
        'boundary',
        'float_nan',
        'infinity',
        'quiet_nan',
        'signaling_nan',
    )
    for entry in ('indices', 'predict')
    for trap in ('clear', 'trapped')
    for direction in ('forward', 'reverse')
]
CASES += [
    f'{family}/{entry}/{action}/forward'
    for family in ('warning_first', 'decimal_first')
    for entry in ('indices', 'predict')
    for action in ('always', 'error')
]
CASES += [f'sticky/{entry}/clear/forward' for entry in ('indices', 'predict')]


def observe_case(name: str) -> dict[str, Any]:
    from sdm_client import KumoTabular, PredictiveQuery, SDMClient

    family, entry, mode, direction = name.split('/')
    n = 81 if entry == 'indices' else 10002
    with localcontext() as context:
        context.traps[FloatOperation] = False
        context.traps[InvalidOperation] = False
        pairs = {
            'finite': [Decimal('1.1'), 1.1],
            'boundary': [Decimal('9007199254740993'), float(2**53)],
            'float_nan': [Decimal('1.1'), float('nan')],
            'infinity': [Decimal('1e10000'), float('inf')],
            'quiet_nan': [Decimal('NaN'), Decimal('1')],
            'signaling_nan': [Decimal('sNaN'), Decimal('1')],
            'sticky': [Decimal('1.1'), 1.1],
            'warning_first': [Decimal('1.1'), 1.1],
            'decimal_first': [Decimal('1.1'), 1.1],
        }
        pair = pairs[family]
        if direction == 'reverse':
            pair = pair[::-1]
        labels = np.empty(n, dtype=object)
        for position in range(n):
            labels[position] = pair[position % 2]
        index = pd.Index(labels, dtype=object)
        if family in ('warning_first', 'decimal_first'):
            decimals = np.empty(2, dtype=object)
            decimals[:] = pair
            numbers = np.empty(2, dtype=object)
            numbers[:] = [np.float32(1), 1e100]
            levels = [pd.Index(numbers), pd.Index(decimals)]
            if family == 'decimal_first':
                levels.reverse()
            index = pd.MultiIndex(
                levels=levels,
                codes=[np.arange(n) % 2, np.arange(n) % 2],
                verify_integrity=False,
            )
        frame = pd.DataFrame(
            {
                'id': range(n),
                'feature': np.arange(n) * 0.5,
                'target': np.arange(n) + 100.0,
            },
            index=index,
        )
        query = PredictiveQuery(
            'PREDICT data.target FOR EACH data.id', {'data': frame}
        )
        arguments = (
            {'indices': [n - 1], 'num_context_samples': 40}
            if entry == 'indices'
            else {'predict': frame.iloc[[-1]].drop(columns=['target'])}
        )
        context.traps[FloatOperation] = mode == 'trapped'
        context.traps[InvalidOperation] = mode == 'trapped'
        context.clear_flags()
        if family == 'sticky':
            context.flags[Inexact] = True
            context.flags[FloatOperation] = True
        endpoint = Endpoint()
        client = SDMClient.__new__(SDMClient)
        client._start(endpoint)
        outcome: dict[str, Any] = {}
        try:
            with (
                np.errstate(over='warn'),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter('always')
                if mode == 'error':
                    warnings.simplefilter('error', RuntimeWarning)
                try:
                    request = query.materialize(**arguments, task='regression')
                    outcome['context_ids'] = request.context.instance_table[
                        'id'
                    ].tolist()
                    result = client.predict(request, model=KumoTabular())
                    outcome['encoded_ids'] = [
                        row[0]
                        for row in endpoint.calls[0]['context'][
                            'instance_table'
                        ]['rows']
                    ]
                    outcome['predictions'] = result.to_pandas().to_dict(
                        orient='records'
                    )
                except Exception as error:  # noqa: BLE001 - public differential outcome
                    outcome['error'] = {
                        'type': type(error).__name__,
                        'message': str(error),
                    }
                outcome['flags'] = sorted(
                    signal.__name__
                    for signal, active in context.flags.items()
                    if active
                )
                # The owner excludes incidental pandas warning repetition.
                # Keep category/message presence; warning-as-error stays exact.
                outcome['warnings'] = [
                    list(event)
                    for event in sorted(
                        {
                            (warning.category.__name__, str(warning.message))
                            for warning in caught
                            if not warning.category.__module__.startswith(
                                'pandas'
                            )
                        }
                    )
                ]
        finally:
            client.close()
        return outcome
