# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tables the in-process and backend-parity suites ask about."""

from __future__ import annotations

import pandas as pd


def people(start: int = 0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'person_id': range(start, start + 60),
            'age': [20.0 + (i * 7) % 40 for i in range(60)],
            'segment': ['north' if i % 3 else 'south' for i in range(60)],
            'visits': [(i * 5) % 11 for i in range(60)],
        }
    )
    frame['churn'] = (frame['visits'] > 5).astype(int)
    frame['plan'] = [
        'pro' if age > 40 else 'plus' if visits > 3 else 'free'
        for age, visits in zip(frame['age'], frame['visits'], strict=True)
    ]
    frame['spend'] = frame['age'] * 2.5 + frame['visits']
    return frame


def graph() -> dict[str, pd.DataFrame]:
    users = pd.DataFrame(
        {
            'user_id': range(40),
            'tier': [('gold', 'silver', 'bronze')[i % 3] for i in range(40)],
            'age': [20.0 + (i * 7) % 30 for i in range(40)],
        }
    )
    items = pd.DataFrame(
        {'item_id': range(6), 'price': [5.0 + i for i in range(6)]}
    )
    rows = []
    for user in range(40):
        busy = user % 2 == 0
        for k in range(40 if busy else 4):
            rows.append(
                {
                    'order_id': len(rows),
                    'user_id': user,
                    'item_id': (user + k) % 6,
                    'ts': pd.Timestamp('2024-01-01')
                    + pd.Timedelta(days=(k * 5 if busy else k * 50) + user % 9),
                }
            )
    return {'users': users, 'items': items, 'orders': pd.DataFrame(rows)}
