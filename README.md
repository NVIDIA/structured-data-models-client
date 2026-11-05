<p align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/images/logo_light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_dark.svg">
  <img src="docs/images/logo_light.svg" alt="Structured Data Models" width="125">
</picture>
</p>

<h1 align="center">NVIDIA Structured Data Models Client

[![PyPI](https://img.shields.io/pypi/v/structured-data-models-client?style=flat&color=76B900)](https://pypi.org/project/structured-data-models-client)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg?style=flat&color=76B900)](https://www.python.org/downloads)
[![License: Apache 2.0](https://img.shields.io/badge/license-apache%202.0-brightgreen.svg?style=flat&color=76B900)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&color=76B900)](CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat&color=76B900)](docs/index.md)

</h1>

**Ask a predictive question of one table, or of a whole relational schema, and
get an answer without training a model for each dataset.**

```bash
pip install 'structured-data-models-client[relational]'
```

## Overview

Structured-data foundation models predict in context. You give the model
labelled rows and the rows you want scored, and it answers in one pass. There
is no training loop, no hyperparameter search, and no per-dataset artifact to
manage.

This client talks to those models. It builds the request, sends it to a NIM
inference endpoint, and gives you back a `PredictionResult`, which
`to_pandas()` turns into a DataFrame. Inference runs in the NIM, so this
package downloads no weights and needs no GPU.

Two models are available through the client:

| Model | For | Asked with |
| --- | --- | --- |
| `kumo-tabular` | rows of a single table | `KumoTabular()` |
| `kumo-relational` | several related tables, joined as a graph | `KumoRelational()` |

**The model is an argument, not a method.** Both use the same call.

## Getting started

You need a running NIM to predict against. Point the client at it and ask:

```python
import pandas as pd
from sdm_client import KumoRelational, PredictiveQuery, SDMClient, Schema

users = pd.DataFrame({'user_id': range(12)})
orders = pd.DataFrame(
    {
        'order_id': range(120),
        'user_id': [i % 12 for i in range(120)],
        'price': [10.0 + i for i in range(120)],
        'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
    }
)

schema = Schema.from_data({'users': users, 'orders': orders})
query = PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
)
request = query.materialize(indices=[1, 2, 3])

with SDMClient(url='http://localhost:8000') as client:
    result = client.predict(request, model=KumoRelational())
    print(result.to_pandas())
```

The [quickstart](docs/get-started/quickstart.md) runs this end to end, and
[prerequisites](docs/get-started/prerequisites.md) covers what you need first.

Models can also run in the Python process with `SDMClient.in_process(...)` and the optional `in-process` extra (Python 3.11+).
Rust prepares requests and assembles results on both paths; the Python model runtime replaces the HTTP endpoint.
See [the local-model quickstart](docs/get-started/quickstart.md).

## Installation

| Command | You get |
| --- | --- |
| `pip install structured-data-models-client` | The client and shared Rust core for both models |
| `pip install 'structured-data-models-client[relational]'` | Compatibility extra; both models are already included |
| `pip install 'structured-data-models-client[snowflake]'` | Reads tables from a warehouse. Also `[sqlite]`, `[duckdb]`, `[postgres]`, `[databricks]`, `[lakebase]`, `[s3]` |
| `pip install 'structured-data-models-client[all]'` | Both models and every warehouse backend |

[Installation](docs/get-started/installation.md) has the rest, including what
differs on Databricks.

## Requirements

Python 3.10 through 3.14.
Both models use the native Rust core included in the client wheel.
Release builds target Linux x86-64 and aarch64, macOS Apple silicon and Intel, and 64-bit Windows.
A matching CPython wheel is required; see [prerequisites](docs/get-started/prerequisites.md).

## Packages

Three distributions are released together and share one version, and a given
version of one is only tested against the same version of the others:

| Package | Import | What it is |
| --- | --- | --- |
| `structured-data-models-client` | `sdm_client` | the public API: client, `Schema`, `Request`, model handles, sessions and results |
| `structured-data-models-api` | `sdm_api` | the wire types. No third-party dependencies |
| `structured-data-models-connectors` | `sdm_connectors` | reads source tables from SQLite, DuckDB, PostgreSQL, Snowflake, Databricks and S3 |

The import name is not the package name, and is not derived from it.

Write PQL with `sdm_client.PredictiveQuery`; parsing and validation run in the Rust core.

Release history is in [`CHANGELOG.md`](CHANGELOG.md), and what a version bump
means is in [stability](docs/reference/stability.md).

## Documentation

Full documentation is under [`docs/`](docs/index.md):

- [Overview](docs/about/overview.md) and
  [architecture](docs/about/architecture.md)
- [Prerequisites](docs/get-started/prerequisites.md),
  [installation](docs/get-started/installation.md),
  [quickstart](docs/get-started/quickstart.md)
- [Prediction output](docs/reference/prediction-output.md) and
  [limits](docs/reference/limits.md)
- [Stability](docs/reference/stability.md): what is public, and what a version
  bump means

[`skills/`](skills/README.md) is the same material written for a coding agent.
Every example in it that can run is run against a live endpoint by the test
suite, and compared to the output printed beneath it.

## Repository layout

The client uses the shared Rust core under `rust/`.
The compatibility ledger in `tests/compatibility/manifest.json` records supported APIs and required coverage.

### Product (ships to users)

- `packages/structured-data-models-client/`: the public `sdm_client` API.
  - `src/sdm_client/_rust/`: the Python host of the Rust core: the run loop, the message codec, the error table, and `providers/`, which run the core's SQL on the user's DuckDB, SQLite or Postgres connection.
- `packages/structured-data-models-connectors/`: `sdm_connectors`, the `Source` classes and the driver helpers in `backends/`.
- `packages/structured-data-models-api/`: `sdm_api`, the wire types generated from the model's OpenAPI file.
- `rust/`: the Rust core; see [`rust/README.md`](rust/README.md).
  - `rust/crates/`: the library crates (all but `sdm-testkit` ship).
  - `rust/bindings/sdm-py/`: the Python module `sdm_client._native`.
- `docs/get-started/`, `docs/about/`, `docs/reference/`: user documentation.
- `examples/`: notebooks, in `local/` and `databricks/`.
- `skills/`: the same material written for a coding agent.
- `scripts/` (top-level files): repository checks and generators, such as the hygiene check, the `sdm_api` generator and the public API snapshot.
- `.github/workflows/`: CI and publishing.

### Product tests

- `packages/<package>/tests/`: each package's tests; `packages/structured-data-models-client/tests_test_ops/` needs a core built with test-only operations.
- `rust/crates/<crate>/tests/` and `rust/bindings/sdm-py/tests/`: the Rust and binding tests.
- `rust/testdata/`: shared Rust test data, including a vendored time zone database.
- `tests/golden/public_api.json` and `tests/golden/wire_payloads.json`: snapshots of the public API and the wire payloads.
- `tests/test_release_contract.py`, `tests/test_skills.py`, `tests/test_native_wheel.py`, `tests/test_wheel_smoke.py`, `tests/test_rust_binding_edges.py`, `tests/test_rust_contract.py`: release, skills, native build and installed-wheel checks.
- `qa/`: manual probes against live endpoints and warehouses.

### Shared test infrastructure and development tools

- `contract/`: shared cases, databases, expected observations and regression vectors used by current Python, Java and Rust tests.
  Recorded fixtures preserve useful regression coverage without requiring the old client at test time.
- `rust/crates/sdm-testkit/`: a test host and provider conformance runner for the Rust core.
- `tests/`: API, packaging, CI, architecture and integration checks.
  `tests/compatibility/` records supported capabilities and their coverage across languages, models and connectors.
- `scripts/rust_core/`: native wheel/JAR builds, installed-wheel checks, Rust attribution generation and architecture checks.
- `scripts/live/`: endpoint and warehouse validation tools, including resource write guards.
- `docs/development/`: the [change guide](docs/development/change-guide.md), [engine boundaries](docs/development/engine-boundaries.md) and [validation evidence](docs/development/validation.md).

Retained fixtures and product checks remain part of ordinary development; see [CONTRIBUTING.md](CONTRIBUTING.md) for the commands.

### How a prediction flows

The Rust core never opens a connection or a socket.
It hands the host steps, "run these SQL statements" or "send this request body", and the host answers with Arrow tables or response bytes.
The paths below are under `packages/` for Python and under `rust/crates/` for Rust.

1. **Connect a source.**
   Python: `sdm_connectors/source.py` (`DuckDBSource`, `SQLiteSource`, `PostgresSource`, `DatabricksSource`, `SnowflakeSource`).
   Rust: none; the connection stays in the host, and `sdm_client/_rust/providers/` runs the core's SQL on it.
2. **Describe the schema** (`Schema.from_source`).
   Python: `sdm_client/schema.py` and the source adapters pass schema inputs and Arrow data to the core.
   Rust: `sdm-core/src/describe.rs` drives, `sdm-data` writes the catalog SQL, `sdm-schema` infers types, keys, time columns and links.
3. **Validate the PQL** (`PredictiveQuery`).
   Python: `sdm_client/query.py` passes the query and schema to the core.
   Rust: `sdm-pql`, reached only through `sdm-core/src/pql.rs`.
4. **Materialize the examples** (`query.materialize`: the entities, the anchor time, the labels of past rows).
   Python: `sdm_client/materializers/relational.py` prepares inputs and drives core operations.
   Rust: `sdm-core/src/materialize.rs`, with time, random streams and labels in `sdm-engine`.
5. **Sample each entity's neighbors.**
   Python: source adapters execute requested reads and return Arrow data; sampling policy stays in Rust.
   Rust: `sdm-core/src/subgraph.rs` and `sdm-engine/src/sample.rs`, with the SQL from `sdm-data`, run by the host; `sdm-core/src/fallback.rs` samples in memory.
6. **Build the request.**
   Python: `sdm_client/request.py` holds the public request and passes inputs to the core.
   Rust: `sdm-request` (payload and options) over `sdm-wire` (the model's JSON types).
7. **Call the model** (`SDMClient.predict`).
   Python: `sdm_client/client.py` and `sdm_client/core/transport.py`.
   Rust: `sdm-core/src/predict.rs` and `sdm-request/src/predict.rs` build the body; the host sends it from `sdm_client/_rust/predict.py`.
8. **Return the result** (`PredictionResult.to_pandas()`).
   Python: `sdm_client/result.py`.
   Rust: `sdm-wire/src/response.rs` reads the answer and `sdm-request` lays out the rows; errors carry codes from `sdm-contract/src/error.rs`, which `sdm_client/_rust/errors.py` turns into exceptions.

Python and Java use the shared Rust core for validation, sampling and request construction.
The host language owns connections, transport and conversion to its public result types.
There is no legacy runtime switch.

## Contributing

Contributions are welcome under the Developer Certificate of Origin. Start with
[`CONTRIBUTING.md`](CONTRIBUTING.md), which has the development setup and the
test commands, and please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

```bash
git clone https://github.com/NVIDIA/structured-data-models-client
cd structured-data-models-client
python -m pip install -e './packages/structured-data-models-client[test]'
python -m pytest packages/structured-data-models-client/tests -m 'not live_nim' -q
```

## Governance and support

- Who decides what, and how a change gets merged:
  [`GOVERNANCE.md`](GOVERNANCE.md)
- Maintainers and the escalation path: [`MAINTAINERS.md`](MAINTAINERS.md)
- Support level and where to ask: [`SUPPORT.md`](SUPPORT.md)

## Security

Please do not open a public issue for a security problem. Vulnerability
reporting goes to NVIDIA PSIRT. See [`SECURITY.md`](SECURITY.md), which also
records what is a deliberate design decision rather than a vulnerability.

## License

Apache-2.0. See [`LICENSE`](LICENSE). Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES.

Each package under `packages/` carries its own copy of `LICENSE` so its
`pyproject.toml` can reference it, which is what puts the license inside each
wheel. Those copies are the same terms as the root file, not additional ones:
[api](packages/structured-data-models-api/LICENSE),
[client](packages/structured-data-models-client/LICENSE),
[connectors](packages/structured-data-models-connectors/LICENSE).

The client wheel includes compiled Rust dependencies and their generated attribution.
See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for retained legal notices.
Optional connector dependencies are resolved at installation time.
Review their license terms before use.
