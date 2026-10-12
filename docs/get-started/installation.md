# Installation

| Command | You get |
|---|---|
| `pip install structured-data-models-client` | The client, the language, and the tabular model. Runs anywhere. |
| `pip install 'structured-data-models-client[relational]'` | Adds the relational model and native engine. **Python 3.10 through 3.13 only**. See [prerequisites](prerequisites.md). |
| `pip install 'structured-data-models-client[snowflake]'` | Reads tables from a warehouse. Also `[sqlite]`, `[duckdb]`, `[postgres]`, `[databricks]`, `[lakebase]`, `[s3]`. |
| `pip install 'structured-data-models-client[all]'` | The relational model and every warehouse backend. |
| `pip install 'structured-data-models-client[databricks-serving]'` | Reaches a model served on Databricks Model Serving, through `SDMClient.for_databricks_serving`. Not in `[all]`: it is a way to reach a model, not a backend. |

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
major, minor = (int(part) for part in numpy.__version__.split('.')[:2])
keep_numpy = f'numpy=={numpy.__version__}' if (major, minor) >= (1, 26) else 'numpy~=1.26.0'
```

```text
%pip install --only-binary=:all: "structured-data-models-client[relational,databricks,databricks-serving]" $keep_numpy
```

```text
dbutils.library.restartPython()
```

Four things differ from a laptop, each found by running on every LTS runtime and serverless environment, except the one path marked to be validated:

- **Keep the runtime's own NumPy where it is 1.26 or later, and NumPy 1.26 where it is older.**
  Databricks builds its preinstalled packages against the NumPy it ships.
  Installing with NumPy left unbounded can let pandas take NumPy beneath them to a new major version, after which the Python process cannot start; within NumPy 1, a newer release still runs packages built against an older one.
  Where the runtime ships NumPy 1.26 or later, pinning it leaves pip to choose a pandas that fits it.
  Where it ships an older one (DBR 14.3 LTS and 15.4 LTS, serverless environments 1 and 2, each NumPy 1.23.5), the client's floor of 1.26 cannot be met by the runtime's own, so the install upgrades NumPy, and pandas and pyarrow with it; `numpy~=1.26.0` holds NumPy to the 1.26 line. That path is pending validation.
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
