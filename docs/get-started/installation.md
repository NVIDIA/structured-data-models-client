# Installation

| Command | You get |
|---|---|
| `pip install sdm-client` | The client, the language, and the tabular model. Runs anywhere. |
| `pip install sdm-client[relational]` | Adds the relational model, pulling in the native engine. |
| `pip install sdm-client[snowflake]` | Reads tables from a warehouse. Also `[sqlite]`, `[duckdb]`, `[postgres]`, `[databricks]`, `[s3]`. |
| `pip install sdm-client[all]` | The relational model and every warehouse backend. |

What ships in the base wheel is decided by **dependency weight, not
preference**. A model that does heavy client-side work — the relational one,
with its graph building, native sampling and query resolution — is an opt-in
extra. Warehouse drivers are opt-in the same way.

## Packages

Five distributions, released together on one version.

| Distribution | Import | What it is |
|---|---|---|
| `sdm-client` | `sdm_client` | The public API |
| `sdm-api` | `sdm_api` | The Universal TFM wire types. **No third-party dependencies** |
| `sdm-pql` | `predictive_query` | The Predictive Query Language |
| `sdm-connectors` | `sdm_connectors` | Reading tables from sqlite, duckdb, PostgreSQL, Snowflake, Databricks, S3 |
| `sdm-relational-engine` | `sdm_relational_engine` | The relational driver (native) |
