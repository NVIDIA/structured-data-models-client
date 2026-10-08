# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real datasets, loaded once, as the SDK would receive them."""

from __future__ import annotations

import functools
import pathlib
import sqlite3

import pandas as pd

DATA = pathlib.Path.home() / 'sdm-qa' / 'data'

ADULT_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income',
]


@functools.cache
def adult() -> pd.DataFrame:
    """48k rows, 14 mixed-type features, '?' for missing, skewed target."""
    frame = pd.read_csv(
        DATA / 'adult' / 'adult.data',
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values=['?'],
    )
    frame.insert(0, 'person_id', range(len(frame)))
    frame['income'] = (frame['income'].str.strip() == '>50K').astype(int)
    return frame


@functools.cache
def wine() -> pd.DataFrame:
    """Numeric features and an ordinal 3..9 target: a real multiclass case."""
    red = pd.read_csv(DATA / 'wine' / 'winequality-red.csv', sep=';')
    red['colour'] = 'red'
    white = pd.read_csv(DATA / 'wine' / 'winequality-white.csv', sep=';')
    white['colour'] = 'white'
    frame = pd.concat([red, white], ignore_index=True)
    frame.columns = [c.replace(' ', '_') for c in frame.columns]
    frame.insert(0, 'wine_id', range(len(frame)))
    return frame


@functools.cache
def titanic() -> pd.DataFrame:
    """Small, nulls in two columns, bool and category dtypes on load."""
    frame = pd.read_csv(DATA / 'titanic.csv')
    frame.insert(0, 'passenger_id', range(len(frame)))
    return frame


def northwind(table: str) -> pd.DataFrame:
    with sqlite3.connect(DATA / 'northwind.sqlite') as connection:
        return pd.read_sql(f'select * from "{table}"', connection)


@functools.cache
def northwind_graph(order_limit: int = 4_000) -> dict[str, pd.DataFrame]:
    """A real relational schema, including a table name with a space in it."""
    customers = northwind('Customers')
    orders = northwind('Orders').head(order_limit)
    details = northwind('Order Details')
    details = details[details['OrderID'].isin(set(orders['OrderID']))]
    products = northwind('Products')
    # Northwind's later rows carry dates this parser does not read, and a row
    # with no time cannot be placed in a temporal graph. Dropping them here
    # keeps the fixture honest: coercing to NaT silently loses 3,170 of 4,000.
    orders['OrderDate'] = pd.to_datetime(orders['OrderDate'], errors='coerce')
    orders = orders[orders['OrderDate'].notna()].reset_index(drop=True)
    return {
        'Customers': customers,
        'Orders': orders,
        'Order Details': details.reset_index(drop=True),
        'Products': products,
    }


def northwind_ready(order_limit: int = 4_000):
    """Northwind with the semantic types this deployment will actually take.

    Free text is refused by the relational endpoint, and the columns here are
    short labels rather than prose, so they read as categorical.
    """
    from sdm_client import Schema

    schema = Schema.from_data(
        northwind_graph(order_limit),
        primary_keys={
            'Customers': 'CustomerID',
            'Orders': 'OrderID',
            'Products': 'ProductID',
        },
        time_columns={'Orders': 'OrderDate'},
    )
    for table in schema:
        for column in schema[table]:
            if schema[table][column].stype == 'text':
                schema[table][column].stype = 'categorical'
    return schema


@functools.cache
def bank() -> pd.DataFrame:
    """41k rows, 20 features, a heavily imbalanced binary target.

    Different in shape from Adult: more categorical columns, several of them
    with the string 'unknown' rather than a null, and float features that come
    from a macroeconomic series rather than the individual.
    """
    frame = pd.read_csv(
        DATA / 'bank' / 'bank-additional' / 'bank-additional-full.csv', sep=';'
    )
    frame.columns = [c.replace('.', '_') for c in frame.columns]
    frame.insert(0, 'contact_id', range(len(frame)))
    frame['y'] = (frame['y'] == 'yes').astype(int)
    return frame


@functools.cache
def students() -> pd.DataFrame:
    """4.4k rows, 36 mostly-numeric features, a three-class target.

    Column names carry spaces, slashes and parentheses, which is the point of
    including it: a real export whose headers were never meant for code.
    """
    frame = pd.read_csv(DATA / 'students' / 'data.csv', sep=';')
    frame.insert(0, 'student_id', range(len(frame)))
    return frame
