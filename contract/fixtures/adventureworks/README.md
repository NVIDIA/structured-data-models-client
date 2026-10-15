# The AdventureWorks fixture

The `adventureworks` base of [`contract/bases.yaml`](../../bases.yaml) is a small subset of Microsoft's AdventureWorks sample database, for E27's AdventureWorks baseline case (`contract/cases/s1/base.adventureworks.json`, T1.21b).
It is test data in this repository and ships in no distribution.

## Source

- Repository: <https://github.com/microsoft/sql-server-samples>
- Commit: `b47eadc85277f72843647223cde53aa1b85ffcb8` (2025-11-14), the commit the Databricks quickstart notebook loads (PR #29's `examples/databricks/kumo_relational_quickstart.ipynb`).
- Files, under `samples/databases/adventure-works/oltp-install-script/`, with their SHA-256:

| File | SHA-256 |
|---|---|
| `Customer.csv` | `4704aa5b46aed326599014524c821aca50e938587e6a1133c0f6d9aae5c5efc6` |
| `Product.csv` | `df0379a7cba8b11b97ea5d104cf1313ef9f683137b6426f2fac6bb8c74c2a2ea` |
| `SalesOrderHeader.csv` | `cf637c7d58006ebbc13daf0b8ed6cf7f738165d298ae0b5479a59237429c1882` |
| `SalesOrderDetail.csv` | `220129977aa63e34fa4a1d66c2604bd2184373151efb09c98b343371ba77ab8a` |

## Licence

MIT. The repository's `license.txt` at that commit (SHA-256 `201f3705979e932c92cd3c6ddbddf4dc3c71dcc5a82089a1e4018227cfe1cf09`) is the MIT License under the heading "Microsoft SQL Server Sample Code" and "Copyright (c) Microsoft Corporation"; it is copied byte for byte as [LICENSE.txt](LICENSE.txt), which carries the copyright and permission notice the licence asks copies to keep.
GitHub's licence detection reports the repository as "Other" because of that heading; the text is the MIT License.
The quickstart notebook names the same licence for the same files.
The repository's [NOTICE](../../../NOTICE) lists the fixture.

## How it is built

[`build.py`](build.py) writes the base, and is run by hand:

```bash
python -I contract/fixtures/adventureworks/build.py --source DIR --write
```

`DIR` holds the five files above; a missing one is downloaded from the pinned commit, and each is checked against its SHA-256 before it is read.
Without `--write` it builds the base and says whether it is the committed one.
CI never runs it and never downloads anything: the base is committed.

It loads the tables as the quickstart does: each file read by pandas as tab-separated text with nullable dtypes, every `*Date` column parsed as a timestamp, and each order line given its order's `CustomerID` and `OrderDate`.
The subset is deterministic: customers in the order of the MD5 of their id's decimal text (the order the quickstart samples customers in), each customer with at least one order kept, with all of their orders, their order lines and the products those lines name, while the rows kept stay within 1,000 in all; a customer whose rows would pass that is passed over.
Each table is `adventureworks_<name>` with the source file's columns (the order lines' `CustomerID` and `OrderDate` last), each typed as its pandas dtype maps: `Int64` to `BIGINT`, `Float64` to `DOUBLE`, `string` to `VARCHAR`, a timestamp to `TIMESTAMP`.

| Table | Rows |
|---|---|
| `adventureworks_customers` | 142 |
| `adventureworks_products` | 110 |
| `adventureworks_sales_order_headers` | 215 |
| `adventureworks_sales_order_details` | 533 |
| In all | 1,000 |

Built in `ref-pandas3` (Python 3.12.13, pandas 3.0.6); a second build gave the same bytes.
`contract/tests/test_refusal_map.py::test_the_fixture_holds_its_readme_s_rows_and_at_most_2000` holds the committed base to at most 2,000 rows, and to this table.
