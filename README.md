# NVIDIA Structured Data Models Client

Ask a predictive question of one table, or of a whole relational schema, and
get an answer without training a model for each dataset.

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

## Installation

| Command | You get |
| --- | --- |
| `pip install structured-data-models-client` | The client, the language, and the tabular model |
| `pip install 'structured-data-models-client[relational]'` | Adds the relational model and its native engine |
| `pip install 'structured-data-models-client[snowflake]'` | Reads tables from a warehouse. Also `[sqlite]`, `[duckdb]`, `[postgres]`, `[databricks]`, `[lakebase]`, `[s3]` |
| `pip install 'structured-data-models-client[all]'` | The relational model and every warehouse backend |

[Installation](docs/get-started/installation.md) has the rest, including what
differs on Databricks.

## Requirements

Python 3.10 through 3.14. The client, the language and the connectors are pure
Python and install anywhere.

The relational engine is a native build, so it ships wheels rather than a
source distribution: `manylinux_2_28` for x86-64 and aarch64, macOS 11 or newer
for Apple silicon and Intel, and 64-bit Windows, each on **CPython 3.10 through
3.14**. `[relational]` resolves on all of those and nowhere else. There is **no
source distribution**, so anywhere without a matching wheel pip has nothing to
fall back to: musl, an older glibc, a 32-bit interpreter, or a Python newer
than the engine has built for yet.

## Packages

Five distributions are released together and share one version, and a given
version of one is only tested against the same version of the others:

| Package | Import | What it is |
| --- | --- | --- |
| `structured-data-models-client` | `sdm_client` | the public API: client, `Schema`, `Request`, model handles, sessions and results |
| `structured-data-models-api` | `sdm_api` | the wire types. No third-party dependencies |
| `structured-data-models-pql` | `predictive_query` | the Predictive Query Language: grammar, parser, validator, model profiles |
| `structured-data-models-connectors` | `sdm_connectors` | reads source tables from SQLite, DuckDB, PostgreSQL, Snowflake, Databricks and S3 |
| `structured-data-models-relational-engine` | `sdm_relational_engine` | the relational driver: graph building, samplers, the neighbor sampler |

The import name is not the package name, and is not derived from it.

`structured-data-models-pql` is an internal dependency of the client, not a
supported API: its `predictive_query` package can change in any release. Write
PQL with `sdm_client.PredictiveQuery`.

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
[connectors](packages/structured-data-models-connectors/LICENSE),
[pql](packages/structured-data-models-pql/LICENSE),
[relational-engine](packages/structured-data-models-relational-engine/LICENSE).

One third-party component is vendored rather than resolved at install time:
`mermaid.min.js`, bundled in the relational engine so `Graph.visualize()` draws
offline, since Snowflake and Databricks notebooks block external scripts. It is
MIT licensed and is itself a bundle, so the components inside it are enumerated
with their copyrights in the third-party section of [`LICENSE`](LICENSE). Its
bytes are pinned by digest and checked in CI.

Everything else is resolved at install time rather than shipped here. This
project will download and install additional third-party open source software
projects. Review the license terms of these open source projects before use.
