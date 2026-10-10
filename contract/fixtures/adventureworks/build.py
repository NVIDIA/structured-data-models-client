#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Builds the ``adventureworks`` base of ``contract/bases.yaml`` (T1.21b).

    python -I contract/fixtures/adventureworks/build.py --source DIR [--write]

E27's AdventureWorks baseline case runs the journey of
``examples/databricks/adventureworks_relational.ipynb`` on local DuckDB. Its
data are a small subset of Microsoft's AdventureWorks sample, read from the
``microsoft/sql-server-samples`` repository at a fixed commit, the files and
commit the Databricks quickstart (PR #29's ``kumo_relational_quickstart``
notebook) loads, and loaded as that notebook loads them: each file read by
pandas as tab-separated text with nullable dtypes, every ``*Date`` column
parsed as a timestamp, and each order line given its order's ``CustomerID``
and ``OrderDate``. The licence is MIT; README.md beside this file records
the source, the commit and the licence, and LICENSE.txt is the licence's
text as the source repository carries it.

``--source DIR`` holds the five source files; a missing one is downloaded
from GitHub, the only time this script touches the network, and every one
is checked against its SHA-256 below before it is read. CI never runs this
script: the base it writes is committed.

The subset is deterministic, so the output is byte-stable: customers are
taken in the order of the MD5 of their id's decimal text, the order the
quickstart samples its customers in, and each customer with at least one
order is kept, with every order and order line of theirs and every product
those lines name, when the rows kept stay within ``ROWS`` in all; a
customer whose rows would pass it is passed over. Each table's rows are in
key order, and its columns are the source file's, the order lines' two
added last as the quickstart adds them. A column's DuckDB type is the one
its pandas dtype maps to (``Int64`` to ``BIGINT``, ``Float64`` to
``DOUBLE``, ``string`` to ``VARCHAR``, a timestamp to ``TIMESTAMP``), as
Spark maps the quickstart's frames.

Without ``--write`` the base is built and compared with the committed one,
and the exit status says whether they are the same.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASES = HERE.parent.parent / 'bases.yaml'

COMMIT = 'b47eadc85277f72843647223cde53aa1b85ffcb8'
URL = (
    f'https://raw.githubusercontent.com/microsoft/sql-server-samples/{COMMIT}/'
)
DIRECTORY = 'samples/databases/adventure-works/oltp-install-script/'

#: Each source file's path in the repository and its SHA-256.
FILES = {
    'Customer.csv': (
        DIRECTORY + 'Customer.csv',
        '4704aa5b46aed326599014524c821aca50e938587e6a1133c0f6d9aae5c5efc6',
    ),
    'Product.csv': (
        DIRECTORY + 'Product.csv',
        'df0379a7cba8b11b97ea5d104cf1313ef9f683137b6426f2fac6bb8c74c2a2ea',
    ),
    'SalesOrderHeader.csv': (
        DIRECTORY + 'SalesOrderHeader.csv',
        'cf637c7d58006ebbc13daf0b8ed6cf7f738165d298ae0b5479a59237429c1882',
    ),
    'SalesOrderDetail.csv': (
        DIRECTORY + 'SalesOrderDetail.csv',
        '220129977aa63e34fa4a1d66c2604bd2184373151efb09c98b343371ba77ab8a',
    ),
    'license.txt': (
        'license.txt',
        '201f3705979e932c92cd3c6ddbddf4dc3c71dcc5a82089a1e4018227cfe1cf09',
    ),
}

#: Each table, its source file, and the file's columns in order, as the
#: quickstart names them.
TABLES = {
    'customers': (
        'Customer.csv',
        'CustomerID PersonID StoreID TerritoryID AccountNumber rowguid '
        'ModifiedDate',
    ),
    'products': (
        'Product.csv',
        'ProductID Name ProductNumber MakeFlag FinishedGoodsFlag Color '
        'SafetyStockLevel ReorderPoint StandardCost ListPrice Size '
        'SizeUnitMeasureCode WeightUnitMeasureCode Weight DaysToManufacture '
        'ProductLine Class Style ProductSubcategoryID ProductModelID '
        'SellStartDate SellEndDate DiscontinuedDate rowguid ModifiedDate',
    ),
    'sales_order_headers': (
        'SalesOrderHeader.csv',
        'SalesOrderID RevisionNumber OrderDate DueDate ShipDate Status '
        'OnlineOrderFlag SalesOrderNumber PurchaseOrderNumber AccountNumber '
        'CustomerID SalesPersonID TerritoryID BillToAddressID ShipToAddressID '
        'ShipMethodID CreditCardID CreditCardApprovalCode CurrencyRateID '
        'SubTotal TaxAmt Freight TotalDue Comment rowguid ModifiedDate',
    ),
    'sales_order_details': (
        'SalesOrderDetail.csv',
        'SalesOrderID SalesOrderDetailID CarrierTrackingNumber OrderQty '
        'ProductID SpecialOfferID UnitPrice UnitPriceDiscount LineTotal '
        'rowguid ModifiedDate',
    ),
}

#: Each table's key, which orders its rows.
KEYS = {
    'customers': ['CustomerID'],
    'products': ['ProductID'],
    'sales_order_headers': ['SalesOrderID'],
    'sales_order_details': ['SalesOrderDetailID'],
}

#: The most rows the base holds, over its four tables.
ROWS = 1000

#: Rows per INSERT statement.
CHUNK = 50

#: The line before the base in bases.yaml, from which this script writes to
#: the end of the file.
MARKER = (
    "  # E27's AdventureWorks base (T1.21b), written by "
    'contract/fixtures/adventureworks/build.py; do not edit it by hand.'
)
HEADER = [
    MARKER,
    "  # A subset of Microsoft's AdventureWorks sample, from",
    f'  # github.com/microsoft/sql-server-samples at {COMMIT},',
    '  # MIT licensed; source, commit and licence in',
    '  # contract/fixtures/adventureworks/README.md and LICENSE.txt.',
    '  adventureworks:',
]


class BuildFailure(Exception):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(source: Path) -> None:
    r"""Each source file in ``source``, downloaded when missing, and each
    checked against its SHA-256."""
    source.mkdir(parents=True, exist_ok=True)
    for name, (path, expected) in FILES.items():
        target = source / name
        if not target.exists():
            url = URL + path
            if not url.startswith('https://'):
                raise BuildFailure(f'{name}: not an https URL')
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - https, checked above
                target.write_bytes(response.read())
        found = digest(target.read_bytes())
        if found != expected:
            raise BuildFailure(f'{name}: SHA-256 {found}, expected {expected}')


def frames(source: Path) -> dict[str, Any]:
    r"""The four tables as the quickstart loads them."""
    import pandas as pd

    loaded = {}
    for table, (name, columns) in TABLES.items():
        frame = pd.read_csv(
            source / name,
            sep='\t',
            header=None,
            names=columns.split(),
            dtype_backend='numpy_nullable',
        )
        for column in frame.columns:
            if column.endswith('Date'):
                frame[column] = pd.to_datetime(frame[column])
        loaded[table] = frame
    orders = loaded['sales_order_headers'][
        ['SalesOrderID', 'CustomerID', 'OrderDate']
    ]
    loaded['sales_order_details'] = loaded['sales_order_details'].merge(
        orders, on='SalesOrderID', validate='many_to_one'
    )
    return loaded


def subset(loaded: dict[str, Any]) -> dict[str, Any]:
    r"""The customers kept, with their orders, order lines and products."""
    customers = loaded['customers']
    headers = loaded['sales_order_headers']
    details = loaded['sales_order_details']
    orders = headers.groupby('CustomerID').size().to_dict()
    lines = details.groupby('CustomerID').size().to_dict()
    bought = details.groupby('CustomerID')['ProductID'].apply(set).to_dict()
    ordered = sorted(
        (int(value) for value in customers['CustomerID']),
        key=lambda value: hashlib.md5(  # noqa: S324 - an order, not security
            str(value).encode()
        ).hexdigest(),
    )
    kept: list[int] = []
    products: set[int] = set()
    rows = 0
    for customer in ordered:
        if customer not in orders:
            continue
        more = products | {int(value) for value in bought[customer]}
        total = (
            rows
            + 1
            + orders[customer]
            + lines[customer]
            + len(more)
            - len(products)
        )
        if total > ROWS:
            continue
        kept.append(customer)
        products = more
        rows = total
    chosen = {
        'customers': customers[customers['CustomerID'].isin(kept)],
        'products': loaded['products'][
            loaded['products']['ProductID'].isin(products)
        ],
        'sales_order_headers': headers[headers['CustomerID'].isin(kept)],
        'sales_order_details': details[details['CustomerID'].isin(kept)],
    }
    if sum(len(frame) for frame in chosen.values()) != rows:
        raise BuildFailure('the rows kept are not the rows counted')
    return {
        table: frame.sort_values(KEYS[table]).reset_index(drop=True)
        for table, frame in chosen.items()
    }


def duckdb_type(dtype: Any) -> str:
    text = str(dtype)
    if text == 'Int64':
        return 'BIGINT'
    if text == 'Float64':
        return 'DOUBLE'
    if text in ('string', 'str'):
        return 'VARCHAR'
    if text.startswith('datetime64['):
        return 'TIMESTAMP'
    raise BuildFailure(f'no DuckDB type for the dtype {text}')


def literal(value: Any, kind: str) -> str:
    import pandas as pd

    if pd.isna(value):
        return 'NULL'
    if kind == 'BIGINT':
        return str(int(value))
    if kind == 'DOUBLE':
        number = float(value)
        if number != number or number in (float('inf'), float('-inf')):
            raise BuildFailure('a number DuckDB text cannot hold')
        return repr(number)
    if kind == 'TIMESTAMP':
        return "'" + pd.Timestamp(value).isoformat(sep=' ') + "'"
    text = str(value)
    if any(ord(character) < 0x20 for character in text):
        raise BuildFailure('a text value holds a control character')
    return "'" + text.replace("'", "''") + "'"


def statements(chosen: dict[str, Any]) -> list[str]:
    written = []
    for table, frame in chosen.items():
        name = f'adventureworks_{table}'
        kinds = [duckdb_type(frame[column].dtype) for column in frame.columns]
        columns = ', '.join(
            f'"{column}" {kind}'
            for column, kind in zip(frame.columns, kinds, strict=True)
        )
        written.append(f'CREATE TABLE {name} ({columns})')
        rows = [
            '('
            + ', '.join(
                literal(value, kind)
                for value, kind in zip(row, kinds, strict=True)
            )
            + ')'
            for row in frame.itertuples(index=False, name=None)
        ]
        for start in range(0, len(rows), CHUNK):
            written.append(
                f'INSERT INTO {name} VALUES '
                + ', '.join(rows[start : start + CHUNK])
            )
    for statement in written:
        # contract/bases.yaml holds plain one-line YAML scalars only, which
        # both runners' readers take as the text written (sdm-testkit's
        # bases.rs refuses anything else).
        if ': ' in statement or ' #' in statement or '\n' in statement:
            raise BuildFailure('a statement is not a plain one-line scalar')
    return written


def block(written: list[str]) -> str:
    return '\n'.join([*HEADER, *(f'  - {line}' for line in written)]) + '\n'


def committed() -> tuple[str, str]:
    r"""bases.yaml before the base, and the base as committed."""
    text = BASES.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    marker = MARKER + '\n'
    if marker not in lines:
        return text, ''
    at = lines.index(marker)
    return ''.join(lines[:at]), ''.join(lines[at:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    arguments = parser.parse_args(argv)
    try:
        fetch(arguments.source)
        chosen = subset(frames(arguments.source))
        built = block(statements(chosen))
    except BuildFailure as failure:
        print(f'build.py: {failure}', file=sys.stderr)
        return 1
    counts = ', '.join(
        f'{table} {len(frame)}' for table, frame in chosen.items()
    )
    before, now = committed()
    if arguments.write:
        BASES.write_text(before + built, encoding='utf-8')
        print(f'wrote the adventureworks base: {counts}')
        return 0
    if now != built:
        print('the committed adventureworks base differs from the build')
        return 1
    print(f'the committed adventureworks base is the build: {counts}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
