# Installation

| Command | You get |
|---|---|
| `pip install structured-data-models-client` | The client, the language, and the tabular model. Runs anywhere. |
| `pip install 'structured-data-models-client[relational]'` | Adds the relational model and native engine. **Python 3.10 through 3.13 only**. See [prerequisites](prerequisites.md). |
| `pip install 'structured-data-models-client[snowflake]'` | Reads tables from a warehouse. Also `[sqlite]`, `[duckdb]`, `[postgres]`, `[databricks]`, `[lakebase]`, `[s3]`. |
| `pip install 'structured-data-models-client[all]'` | The relational model and every warehouse backend. |
| `pip install 'structured-data-models-client[databricks-serving]'` | Reaches a model served on Databricks Model Serving, through `SDMClient.for_databricks_serving`. Not in `[all]`: it is a way to reach a model, not a backend. |
| `pip install 'structured-data-models-client[in-process]'` | Runs the models in this Python process with structured-data-models, through `SDMClient.in_process`. Brings PyTorch. The relational model also needs `[relational]`, which gathers its requests. **Python 3.11 or later**: on 3.10 the extra installs nothing, and `in_process()` says why. Not in `[all]`, for the same reason as `[databricks-serving]`. |

**What you install is not what you import.** The five distributions are
`structured-data-models-client`, `-api`, `-pql`, `-connectors` and
`-relational-engine`; the modules they install are `sdm_client`, `sdm_api`,
`predictive_query`, `sdm_connectors` and `sdm_relational_engine`. Only the
first of each pair is ever typed at a prompt.

**Dependency weight determines what ships in the base wheel.** The relational
model does heavy client-side work: graph building, native sampling and query
resolution. It is an opt-in extra, as are warehouse drivers.

## On Databricks

Install into the notebook with `%pip`, then restart Python, each step in its own cell:

```text
import numpy
keep_numpy = f'numpy=={numpy.__version__}'
```

```text
%pip install --only-binary=:all: "structured-data-models-client[relational,databricks,databricks-serving]" $keep_numpy
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
- **For Lakebase, install the `lakebase` extra rather than `postgres`: `structured-data-models-client[relational,lakebase,databricks-serving]`.**
  It brings plain `psycopg`, and the Databricks SDK that `sdm_connectors.databricks.lakebase()` mints the connection's token with.
  The `postgres` extra's `psycopg[binary]` bundles an OpenSSL that fails its FIPS self-test on Databricks' aarch64 machines and aborts the process.
  Plain `psycopg` uses the runtime's own libpq, which works on both architectures.
  For the same reason, do not install `[all]` on a Databricks aarch64 machine: it includes the `postgres` extra, and with it `psycopg[binary]`.
  Reach Lakebase from serverless compute, which connects on a private address; a classic cluster goes over the public internet and connects only if the workspace admits its egress address.

Databricks requires a partner product to identify every connection and API request it opens, so each one structured-data-models-client opens names it as `nvidia_structured-data-models-client/<version>`.
Model Serving requests and SQL warehouse connections carry it in the User-Agent, and Lakebase connections carry it as `application_name`, each ahead of any you pass.
The `WorkspaceClient` requests structured-data-models-client makes carry it through a copy of your client, which is left unchanged.
Statements on a warehouse session structured-data-models-client opens are also tagged `structured_data_models_client_purpose: sampling` in query history.
A connection or Spark session you open yourself and pass to a source is used as it is, without either.

## Packages

Five distributions, released together on one version.

| Distribution | Import | What it is |
|---|---|---|
| `structured-data-models-client` | `sdm_client` | The public API |
| `structured-data-models-api` | `sdm_api` | The Universal TFM wire types. **No third-party dependencies** |
| `structured-data-models-pql` | `predictive_query` | The Predictive Query Language |
| `structured-data-models-connectors` | `sdm_connectors` | Reading tables from sqlite, duckdb, PostgreSQL, Snowflake, Databricks, S3 |
| `structured-data-models-relational-engine` | `sdm_relational_engine` | The relational driver (native) |

`structured-data-models-pql` is an internal dependency of `structured-data-models-client`, not a supported API, and its `predictive_query` package can change in any release.
Write PQL with `sdm_client.PredictiveQuery`.
