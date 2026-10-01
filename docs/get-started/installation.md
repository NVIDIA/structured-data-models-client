# Installation

| Command | You get |
|---|---|
| `pip install sdm-client` | The client, the language, and the tabular model. Runs anywhere. |
| `pip install sdm-client[relational]` | Adds the relational model and native engine. **Python 3.10 through 3.13 only**. See [prerequisites](prerequisites.md). |
| `pip install sdm-client[snowflake]` | Reads tables from a warehouse. Also `[sqlite]`, `[duckdb]`, `[postgres]`, `[databricks]`, `[s3]`. |
| `pip install sdm-client[all]` | The relational model and every warehouse backend. |

**Dependency weight determines what ships in the base wheel.** The relational
model does heavy client-side work: graph building, native sampling and query
resolution. It is an opt-in extra, as are warehouse drivers.

## Packages

Five distributions, released together on one version.

| Distribution | Import | What it is |
|---|---|---|
| `sdm-client` | `sdm_client` | The public API |
| `sdm-api` | `sdm_api` | The Universal TFM wire types. **No third-party dependencies** |
| `sdm-pql` | `predictive_query` | The Predictive Query Language |
| `sdm-connectors` | `sdm_connectors` | Reading tables from sqlite, duckdb, PostgreSQL, Snowflake, Databricks, S3 |
| `sdm-relational-engine` | `sdm_relational_engine` | The relational driver (native) |
