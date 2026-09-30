# Installation

| Command | You get |
|---|---|
| `pip install sdm-client` | The client, the language, and the tabular model. Runs anywhere. |
| `pip install sdm-client[relational]` | Adds the relational model, pulling in the native engine. **Python 3.10–3.13 only** — see [prerequisites](prerequisites.md). |
| `pip install sdm-client[snowflake]` | Reads tables from a warehouse. Also `[sqlite]`, `[duckdb]`, `[postgres]`, `[databricks]`, `[lakebase]`, `[s3]`. |
| `pip install sdm-client[all]` | The relational model and every warehouse backend. |
| `pip install sdm-client[databricks-serving]` | Reaches a model served on Databricks Model Serving, through `SDMClient.for_databricks_serving`. Not in `[all]`: it is a way to reach a model, not a backend. |

What ships in the base wheel is decided by **dependency weight, not
preference**. A model that does heavy client-side work — the relational one,
with its graph building, native sampling and query resolution — is an opt-in
extra. Warehouse drivers are opt-in the same way.

## On Databricks

Install into the notebook with `%pip`, then restart Python, each step in its own cell:

```text
import numpy
keep_numpy = f'numpy=={numpy.__version__}'
```

```text
%pip install --only-binary=:all: "sdm-client[relational,databricks,databricks-serving]" $keep_numpy
```

```text
dbutils.library.restartPython()
```

Four things differ from a laptop, each found by running on every LTS runtime and serverless environment:

- **Keep the runtime's own NumPy.**
  Databricks builds its preinstalled packages against the NumPy it ships.
  Installing can upgrade pandas and, through it, NumPy beneath them, after which the Python process cannot start.
  Pinning NumPy to the version already there leaves pip to choose a pandas that fits it.
- **Write version bounds with `~=`, not `<` or `>`.**
  A classic cluster's `%pip` reads its line through a shell, which takes `<` and `>` as redirections: `numpy<2` fails, and `pyarrow>=10` is silently dropped.
- **Sampling through the notebook's own Spark needs Databricks Runtime 17.3 LTS or serverless environment 4.**
  `DatabricksSource(spark, ...)` reads results with `DataFrame.toArrow()`, which arrived in Spark 4.
  On an older runtime, name a SQL warehouse instead: `DatabricksSource(warehouse(warehouse_id, workspace=WorkspaceClient()), ...)`, with `warehouse` from `sdm_connectors.databricks`.
- **For Lakebase, install the `lakebase` extra rather than `postgres`: `sdm-client[relational,lakebase,databricks-serving]`.**
  It brings plain `psycopg`, and the Databricks SDK that `sdm_connectors.databricks.lakebase()` mints the connection's token with.
  The `postgres` extra's `psycopg[binary]` bundles an OpenSSL that fails its FIPS self-test on Databricks' aarch64 machines and aborts the process.
  Plain `psycopg` uses the runtime's own libpq, which works on both architectures.
  Reach Lakebase from serverless compute, which connects on a private address; a classic cluster goes over the public internet and connects only if the workspace admits its egress address.

## Packages

Five distributions, released together on one version.

| Distribution | Import | What it is |
|---|---|---|
| `sdm-client` | `sdm_client` | The public API |
| `sdm-api` | `sdm_api` | The Universal TFM wire types. **No third-party dependencies** |
| `sdm-pql` | `predictive_query` | The Predictive Query Language |
| `sdm-connectors` | `sdm_connectors` | Reading tables from sqlite, duckdb, PostgreSQL, Snowflake, Databricks, S3 |
| `sdm-relational-engine` | `sdm_relational_engine` | The relational driver (native) |
