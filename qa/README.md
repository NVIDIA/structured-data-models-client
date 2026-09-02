# QA probes

The end-to-end pass that produced [`../bugs`](../bugs). Kept because every
remaining failure is a filed bug, so the suite doubles as a regression check:
when a bug is fixed, its probe flips to `ok`.

These are not part of the test suite. They need live endpoints, real
credentials and datasets that are downloaded rather than vendored, so they are
run by hand.

## Setup

```bash
mkdir -p ~/sdm-qa/data && cd ~/sdm-qa/data
curl -sSLO https://archive.ics.uci.edu/static/public/2/adult.zip
curl -sSLO https://archive.ics.uci.edu/static/public/186/wine+quality.zip
curl -sSLO https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
curl -sSL -o northwind.sqlite \
  https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db
unzip -q adult.zip -d adult && unzip -q wine+quality.zip -d wine
```

## Running

```bash
export SDM_TABULAR_URL=http://localhost:8000
export SDM_RELATIONAL_URL=http://localhost:8001

python probe_tabular.py       # kumo-tabular on Adult, Wine, Titanic
python probe_relational.py    # kumo-relational on Northwind
python probe_connectors.py    # local files, SQLite, DuckDB
python probe_edges.py         # awkward values, shapes, the error contract
python probe_cloud.py         # S3   (needs AWS credentials and /tmp/qa_bucket)
python probe_warehouse.py     # Snowflake and Databricks (needs SNOWFLAKE_*, DATABRICKS_*)
```

Each writes a JSON record to `~/sdm-qa/findings/` and prints a summary. A probe
never stops the run: it records `PASS`, `FAIL` or `RAISED` and continues, so one
broken thing does not hide the next.

`probe_cloud.py` and `probe_warehouse.py` create objects and tables and remove
them before they exit.

## Expected failures

These fail on purpose, and each one names the bug it is waiting on.

| Probe | Bug |
|---|---|
| `empty predict frame` (tabular, edges) | [5](../bugs/005-empty-frames-are-not-refused.md) |
| `a Decimal from a warehouse` | [1](../bugs/001-decimal-columns-lose-precision.md) |
| `a uint64 past int64` | [4](../bugs/004-uint64-declared-as-int64.md) |
| `a duplicated primary key` | [2](../bugs/002-non-unique-primary-key-accepted.md) |
| `an anchor time before the data starts` | [3](../bugs/003-engine-exceptions-escape-the-error-surface.md) |
| `more entities than the endpoint scores` | [3](../bugs/003-engine-exceptions-escape-the-error-surface.md) |
| `an entity id that is not in the table` | [3](../bugs/003-engine-exceptions-escape-the-error-surface.md) |
| `a free-text column` | [8](../bugs/008-endpoint-limits-are-not-declared-locally.md) |
| `materializing prints nothing to stdout` | [7](../bugs/007-materialize-prints-to-stdout.md) |
| `the graph sent matches the schema` | [15](../bugs/015-the-schemas-relationships-are-additive-only.md) |
| `asserting there are no relationships` | [15](../bugs/015-the-schemas-relationships-are-additive-only.md) |
| `the schema can draw itself` | [16](../bugs/016-visualize-is-documented-but-not-reachable.md) |
| `csv, a large integer with a blank in the column` | [9](../bugs/009-csv-reader-rounds-large-integers.md) |
| `duckdb, a SUM over a BIGINT column` | [14](../bugs/014-a-sum-over-integers-in-duckdb-is-rounded.md) |
