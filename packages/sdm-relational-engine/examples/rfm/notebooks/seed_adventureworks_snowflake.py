# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Seed the AdventureWorks tables the Snowflake notebook reads.

The peer of whatever loaded ``main.kumo_rfm.aw_*`` for the Databricks notebook.
Source is Microsoft's own OLTP install script, which ships the tables as
headerless tab-separated files, so the column names come from the AdventureWorks
DDL rather than from the files.

Two shaping decisions carry over from the Databricks tables:

``sales_order_details`` gets ``OrderDate`` and ``CustomerID``. AdventureWorks
keeps both on the header alone. The notebook treats the detail table as a
timeline and aggregates ``OrderQty`` over a window, which needs a time column on
the table being aggregated; and it ranks products for a customer, which is link
prediction and needs the detail table to carry a foreign key to the entity being
predicted for, not merely a two-hop path through the header. Both are joined
from the header rather than invented.

``rowguid`` is kept. It is a per-row UUID that cannot inform a prediction, and
the notebook removes it from the graph explicitly, showing that leaving it on
``sales_order_details`` drives the demand forecast to exactly 0.00. Dropping it
here instead would delete the lesson.

Three load-time decisions are not obvious from the code:

``COLUMN_TYPES`` drives an explicit ``CREATE TABLE`` rather than write_pandas'
``auto_create_table``, which maps a datetime64 column onto NUMBER holding epoch
microseconds. The graph then reads ``OrderDate`` as an integer and refuses it as
a time column, so every temporal query in the notebook becomes unavailable.
Nullable integer keys have the matching problem: inferred, they land as FLOAT,
and a key that renders as ``29825.0`` neither joins nor reads as an identifier.

``PRIMARY_KEYS`` are declared because the graph infers a primary key from
uniqueness, and ``sales_order_headers`` has two unique columns: ``SalesOrderID``
and ``rowguid``. The inference declines to choose between them, which leaves the
header unlinkable and the detail table with nothing to aggregate through.

Dates are verified after loading, by reading them back as text and comparing
years against the source. A rescaled timestamp loads without complaint and is
caught only much later, by whatever first reads the column.

Usage::

    python seed_adventureworks_snowflake.py --database KUMO_RFM_SPCS --schema ADVENTUREWORKS

Column names are upper-cased on load. AdventureWorks ships them in mixed case,
which Snowflake preserves only as a quoted identifier, and a quoted identifier
cannot be read back out of a semantic view: ``from_snowflake_semantic_view``
takes the name verbatim and looks for a column literally called
``"OrderDate"``, quotes included. Upper case keeps every identifier unquoted, so
the notebook and the Cortex agent share one vocabulary.

Credentials are read from the environment: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER,
SNOWFLAKE_PASSWORD, SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

SOURCE_BASE = (
    'https://raw.githubusercontent.com/microsoft/sql-server-samples'
    '/master/samples/databases/adventure-works/oltp-install-script'
)

CUSTOMER_COLUMNS = [
    'CustomerID',
    'PersonID',
    'StoreID',
    'TerritoryID',
    'AccountNumber',
    'rowguid',
    'ModifiedDate',
]

PRODUCT_COLUMNS = [
    'ProductID',
    'Name',
    'ProductNumber',
    'MakeFlag',
    'FinishedGoodsFlag',
    'Color',
    'SafetyStockLevel',
    'ReorderPoint',
    'StandardCost',
    'ListPrice',
    'Size',
    'SizeUnitMeasureCode',
    'WeightUnitMeasureCode',
    'Weight',
    'DaysToManufacture',
    'ProductLine',
    'Class',
    'Style',
    'ProductSubcategoryID',
    'ProductModelID',
    'SellStartDate',
    'SellEndDate',
    'DiscontinuedDate',
    'rowguid',
    'ModifiedDate',
]

ORDER_HEADER_COLUMNS = [
    'SalesOrderID',
    'RevisionNumber',
    'OrderDate',
    'DueDate',
    'ShipDate',
    'Status',
    'OnlineOrderFlag',
    'SalesOrderNumber',
    'PurchaseOrderNumber',
    'AccountNumber',
    'CustomerID',
    'SalesPersonID',
    'TerritoryID',
    'BillToAddressID',
    'ShipToAddressID',
    'ShipMethodID',
    'CreditCardID',
    'CreditCardApprovalCode',
    'CurrencyRateID',
    'SubTotal',
    'TaxAmt',
    'Freight',
    'TotalDue',
    'Comment',
    'rowguid',
    'ModifiedDate',
]

ORDER_DETAIL_COLUMNS = [
    'SalesOrderID',
    'SalesOrderDetailID',
    'CarrierTrackingNumber',
    'OrderQty',
    'ProductID',
    'SpecialOfferID',
    'UnitPrice',
    'UnitPriceDiscount',
    'LineTotal',
    'rowguid',
    'ModifiedDate',
]

SOURCES = {
    'Customer.csv': CUSTOMER_COLUMNS,
    'Product.csv': PRODUCT_COLUMNS,
    'SalesOrderHeader.csv': ORDER_HEADER_COLUMNS,
    'SalesOrderDetail.csv': ORDER_DETAIL_COLUMNS,
}

DATE_COLUMNS = {
    'OrderDate',
    'DueDate',
    'ShipDate',
    'ModifiedDate',
    'SellStartDate',
    'SellEndDate',
    'DiscontinuedDate',
}

COLUMN_TYPES = {
    'CustomerID': 'NUMBER(38,0)',
    'PersonID': 'NUMBER(38,0)',
    'StoreID': 'NUMBER(38,0)',
    'TerritoryID': 'NUMBER(38,0)',
    'AccountNumber': 'VARCHAR(20)',
    'rowguid': 'VARCHAR(36)',
    'ProductID': 'NUMBER(38,0)',
    'Name': 'VARCHAR(100)',
    'ProductNumber': 'VARCHAR(40)',
    'MakeFlag': 'NUMBER(1,0)',
    'FinishedGoodsFlag': 'NUMBER(1,0)',
    'Color': 'VARCHAR(30)',
    'SafetyStockLevel': 'NUMBER(38,0)',
    'ReorderPoint': 'NUMBER(38,0)',
    'StandardCost': 'NUMBER(19,4)',
    'ListPrice': 'NUMBER(19,4)',
    'Size': 'VARCHAR(10)',
    'SizeUnitMeasureCode': 'VARCHAR(6)',
    'WeightUnitMeasureCode': 'VARCHAR(6)',
    'Weight': 'NUMBER(19,4)',
    'DaysToManufacture': 'NUMBER(38,0)',
    'ProductLine': 'VARCHAR(4)',
    'Class': 'VARCHAR(4)',
    'Style': 'VARCHAR(4)',
    'ProductSubcategoryID': 'NUMBER(38,0)',
    'ProductModelID': 'NUMBER(38,0)',
    'SalesOrderID': 'NUMBER(38,0)',
    'RevisionNumber': 'NUMBER(38,0)',
    'Status': 'NUMBER(38,0)',
    'OnlineOrderFlag': 'NUMBER(1,0)',
    'SalesOrderNumber': 'VARCHAR(30)',
    'PurchaseOrderNumber': 'VARCHAR(30)',
    'SalesPersonID': 'NUMBER(38,0)',
    'BillToAddressID': 'NUMBER(38,0)',
    'ShipToAddressID': 'NUMBER(38,0)',
    'ShipMethodID': 'NUMBER(38,0)',
    'CreditCardID': 'NUMBER(38,0)',
    'CreditCardApprovalCode': 'VARCHAR(30)',
    'CurrencyRateID': 'NUMBER(38,0)',
    'SubTotal': 'NUMBER(19,4)',
    'TaxAmt': 'NUMBER(19,4)',
    'Freight': 'NUMBER(19,4)',
    'TotalDue': 'NUMBER(19,4)',
    'Comment': 'VARCHAR(256)',
    'SalesOrderDetailID': 'NUMBER(38,0)',
    'CarrierTrackingNumber': 'VARCHAR(30)',
    'OrderQty': 'NUMBER(38,0)',
    'SpecialOfferID': 'NUMBER(38,0)',
    'UnitPrice': 'NUMBER(19,4)',
    'UnitPriceDiscount': 'NUMBER(19,4)',
    'LineTotal': 'NUMBER(38,6)',
}

PRIMARY_KEYS = {
    'aw_customers': 'CustomerID',
    'aw_products': 'ProductID',
    'aw_sales_order_headers': 'SalesOrderID',
    'aw_sales_order_details': 'SalesOrderDetailID',
}


def download(cache: Path) -> dict[str, Path]:
    """Fetch each source file once, into ``cache``."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SOURCES:
        target = cache / name
        if not target.exists():
            print(f'  downloading {name}')
            urllib.request.urlretrieve(f'{SOURCE_BASE}/{name}', target)
        paths[name] = target
    return paths


def read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one headerless tab-separated AdventureWorks extract.

    ``quoting=3`` disables quote handling: the files are not quoted, and a bare
    double quote inside a product name would otherwise swallow the rest of the
    row. ``keep_default_na=False`` then ``replace`` keeps that explicit, since
    an empty field means NULL here and the string "NA" is a real Size value.
    """
    frame = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=columns,
        quoting=3,
        keep_default_na=False,
        dtype=str,
        encoding='utf-8',
        encoding_errors='replace',
        on_bad_lines='warn',
    )
    frame = frame.replace({'': None})
    for column in frame.columns:
        if column in DATE_COLUMNS:
            frame[column] = pd.to_datetime(frame[column], errors='coerce')
    return frame


def dates_as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Render datetime columns as ISO text for the load.

    ``write_pandas`` ships the frame as parquet, and the timestamp arrives
    scaled wrong: nanoseconds are read as microseconds, putting AdventureWorks
    in the year 52410985, and writing microseconds instead only moves the error
    rather than fixing it. Both load without complaint and are caught much
    later, by whatever first reads the column.

    Text is unambiguous. The target column is declared TIMESTAMP_NTZ, so
    Snowflake parses on the way in and the stored type is still a timestamp.
    """
    frame = frame.copy()
    for column in frame.columns:
        if column in DATE_COLUMNS:
            rendered = frame[column].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            frame[column] = rendered.where(frame[column].notna(), None)
    return frame


def coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parse the listed columns, keeping whole numbers whole.

    ``Int64`` rather than the default: a nullable integer parsed as float comes
    back from Snowflake as one, and a key that renders as ``29825.0`` neither
    joins nor reads as an identifier.
    """
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors='coerce')
        if COLUMN_TYPES.get(column, '').startswith('NUMBER') and COLUMN_TYPES[
            column
        ].endswith(',0)'):
            values = values.astype('Int64')
        frame[column] = values
    return frame


def upper_case_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename every column to upper case; see the module docstring."""
    return frame.rename(columns=str.upper)


def column_type(column: str) -> str:
    """The Snowflake type for one AdventureWorks column.

    Raises rather than defaulting: an unmapped column would otherwise be
    created as whatever the fallback happened to be, which is how the dates
    became NUMBER in the first place.
    """
    if column in DATE_COLUMNS:
        return 'TIMESTAMP_NTZ'
    try:
        return COLUMN_TYPES[column]
    except KeyError:
        raise SystemExit(
            f'column {column!r} has no declared Snowflake type; add it to '
            'COLUMN_TYPES rather than letting it be inferred'
        ) from None


def create_table(
    cursor, database: str, schema: str, name: str, columns: list[str]
) -> None:
    """Create the table from the declared types, replacing any earlier load."""
    definitions = ', '.join(f'{c.upper()} {column_type(c)}' for c in columns)
    cursor.execute(
        f'CREATE OR REPLACE TABLE {database}.{schema}.{name} ({definitions})'
    )


def build() -> dict[str, pd.DataFrame]:
    """Read the four extracts and shape them the way the notebook expects."""
    paths = download(Path(os.environ.get('AW_CACHE', '/tmp/aw')))
    frames = {
        name: read_table(paths[name], columns)
        for name, columns in SOURCES.items()
    }

    customers = coerce_numeric(
        frames['Customer.csv'],
        ['CustomerID', 'PersonID', 'StoreID', 'TerritoryID'],
    )
    products = coerce_numeric(
        frames['Product.csv'],
        [
            'ProductID',
            'MakeFlag',
            'FinishedGoodsFlag',
            'SafetyStockLevel',
            'ReorderPoint',
            'StandardCost',
            'ListPrice',
            'Weight',
            'DaysToManufacture',
            'ProductSubcategoryID',
            'ProductModelID',
        ],
    )
    headers = coerce_numeric(
        frames['SalesOrderHeader.csv'],
        [
            'SalesOrderID',
            'RevisionNumber',
            'Status',
            'OnlineOrderFlag',
            'CustomerID',
            'SalesPersonID',
            'TerritoryID',
            'BillToAddressID',
            'ShipToAddressID',
            'ShipMethodID',
            'CreditCardID',
            'CurrencyRateID',
            'SubTotal',
            'TaxAmt',
            'Freight',
            'TotalDue',
        ],
    )
    details = coerce_numeric(
        frames['SalesOrderDetail.csv'],
        [
            'SalesOrderID',
            'SalesOrderDetailID',
            'OrderQty',
            'ProductID',
            'SpecialOfferID',
            'UnitPrice',
            'UnitPriceDiscount',
            'LineTotal',
        ],
    )

    details = details.merge(
        headers[['SalesOrderID', 'OrderDate', 'CustomerID']],
        on='SalesOrderID',
        how='left',
    )
    unmatched = int(details['OrderDate'].isna().sum())
    if unmatched:
        raise SystemExit(
            f'{unmatched} order detail rows have no matching header, so their '
            'OrderDate would be null and the detail timeline would be wrong'
        )

    return {
        'aw_customers': customers,
        'aw_products': products,
        'aw_sales_order_headers': headers,
        'aw_sales_order_details': details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--schema', required=True)
    args = parser.parse_args()

    tables = build()
    for name, frame in tables.items():
        print(f'  {name}: {len(frame):,} rows x {len(frame.columns)} columns')

    connection = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        role=os.environ['SNOWFLAKE_ROLE'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
    )
    try:
        cursor = connection.cursor()
        cursor.execute(f'CREATE DATABASE IF NOT EXISTS {args.database}')
        cursor.execute(
            f'CREATE SCHEMA IF NOT EXISTS {args.database}.{args.schema}'
        )
        for name, frame in tables.items():
            create_table(
                cursor,
                args.database,
                args.schema,
                name.upper(),
                list(frame.columns),
            )
            success, _, rows, _ = write_pandas(
                connection,
                upper_case_columns(dates_as_text(frame)),
                name.upper(),
                database=args.database,
                schema=args.schema,
                auto_create_table=False,
                quote_identifiers=False,
            )
            print(f'  loaded {name.upper()}: success={success} rows={rows:,}')

        print('\n  verifying keys and dates')
        for name, frame in tables.items():
            table = f'{args.database}.{args.schema}."{name.upper()}"'
            key = PRIMARY_KEYS[name]
            total, distinct = cursor.execute(
                f'SELECT COUNT(*), COUNT(DISTINCT {key.upper()}) FROM {table}'
            ).fetchone()
            if total != distinct:
                raise SystemExit(
                    f'{name}: {key} is not unique ({distinct:,} of {total:,}), '
                    'so it cannot be the primary key'
                )

            for column in (c for c in frame.columns if c in DATE_COLUMNS):
                low, high = cursor.execute(
                    f'SELECT TO_VARCHAR(MIN({column.upper()})), '
                    f'TO_VARCHAR(MAX({column.upper()})) FROM {table}'
                ).fetchone()
                if low is None:
                    continue
                expected = frame[column].dropna()
                if expected.empty:
                    continue
                loaded_years = {int(low[:4]), int(high[:4])}
                source_years = {expected.min().year, expected.max().year}
                if loaded_years != source_years:
                    raise SystemExit(
                        f'{name}.{column}: loaded {sorted(loaded_years)} but '
                        f'the source has {sorted(source_years)}; the timestamp '
                        'was rescaled on the way in'
                    )
            print(f'    {name}: {total:,} rows, {key} unique, dates match')
    finally:
        connection.close()


if __name__ == '__main__':
    main()
