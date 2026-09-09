# sdm-client

One Python client for NVIDIA's structured-data foundation models.

Structured-data foundation models predict in context. You give the model
labelled rows and the rows you want scored, and it answers in one pass. There
is no training loop, no hyperparameter search, and no per-dataset artifact to
manage.

This client builds the request, sends it to a NIM, and gives you back a
DataFrame. Inference runs in the NIM, so this package downloads no weights and
needs no GPU.

```python
from sdm_client import (
    SDMClient,
    Schema,
    PredictiveQuery,
    KumoRelational,
)

schema = Schema.from_data({'users': users_df, 'orders': orders_df})
query = PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
)
request = query.materialize(indices=[1, 2, 3])

with SDMClient(url='http://localhost:8000') as client:
    result = client.predict(request, model=KumoRelational())
    print(result.to_pandas())
```

**The model is an argument, not a method.** The same request runs on any model
whose profile accepts it.

## Status

All seven phases are built and tested end to end against live NIMs. The
architecture is written down in
[`docs/about/architecture.md`](docs/about/architecture.md), and the packages follow it.
Nothing has been released yet.

| Phase | | |
|---|---|---|
| 1 | Foundation — `sdm-api`, `sdm-pql`, CI | ✅ |
| 2 | Client and the tabular model | ✅ |
| 3 | Connectors | ✅ |
| 4 | The relational model | ✅ |
| 5 | `fit` and explanations | ✅ |
| 6 | Docs and end-to-end | ✅ |
| 7 | PQL for tabular | ✅ |

## Packages

Five distributions, released together on one version.

| Distribution | Import | Kind | What it is |
|---|---|---|---|
| `sdm-client` | `sdm_client` | pure python | The public API: client, `Schema`, `Request`, model handles, transport, sessions, results |
| `sdm-api` | `sdm_api` | pure python | The Universal TFM wire types. No third-party dependencies |
| `sdm-pql` | `predictive_query` | pure python | The Predictive Query Language: grammar, parser, validator, AST, model profiles |
| `sdm-connectors` | `sdm_connectors` | pure python | Reading source tables from SQLite, DuckDB, PostgreSQL, Snowflake, Databricks, S3 |
| `sdm-relational-engine` | `sdm_relational_engine` | native | The relational driver: graph building, samplers, the neighbor sampler |

Only `sdm-relational-engine` is native, so the tabular model works anywhere and
the relational model needs Linux x86-64.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install \
  -e './packages/sdm-api[test]' \
  -e './packages/sdm-pql[test]' \
  -e './packages/sdm-client[test]' \
  -e './packages/sdm-connectors[sqlite,duckdb,postgres,databricks,test]'
.venv/bin/python -m pytest \
  packages/sdm-api/tests packages/sdm-pql/tests \
  packages/sdm-client/tests packages/sdm-connectors/tests -q
```

The relational engine is a native build and is installed and tested on its
own, from its own directory:

```bash
.venv/bin/pip install -e './packages/sdm-relational-engine'
cd packages/sdm-relational-engine && ../../.venv/bin/python -m pytest tests -q
```

Lint and format with `ruff`; the formatter owns line length.

## Vendored code

One tree is vendored, and it is pinned by digest and checked in CI, because a
repo-wide rename has silently corrupted a vendored artifact in this family of
repositories before.

`mermaid.min.js` is bundled in the relational engine so `Graph.visualize()`
draws offline: Snowflake and Databricks notebooks block external scripts, so a
diagram fetched from a CDN would not draw. Upstream bytes, SHA-256 pinned,
components enumerated in `LICENSE`.

Everything else here, including the Predictive Query Language in `sdm-pql`, is
first-party source maintained in this repository.

## License

Apache-2.0. Each package carries its own copy of `LICENSE` so its
`pyproject.toml` can reference it; those copies are the same terms as the root
file, not additional ones.

## Documentation

Full documentation is under [`docs/`](docs/index.md):

- [Overview](docs/about/overview.md) and [Architecture](docs/about/architecture.md)
- [Prerequisites](docs/get-started/prerequisites.md), [Installation](docs/get-started/installation.md), [Quickstart](docs/get-started/quickstart.md)
- [Prediction output](docs/reference/prediction-output.md) and [Limits](docs/reference/limits.md)

## Testing

```bash
# everything that needs nothing running, and opens no sockets
python -m pytest packages/sdm-api/tests packages/sdm-pql/tests \
                packages/sdm-client/tests packages/sdm-connectors/tests \
  -q -m 'not live_nim and not live_warehouse and not socket'

# the native engine, from the repository root
python -m pytest packages/sdm-relational-engine/tests -q

# against a NIM you are running
SDM_TABULAR_URL=http://localhost:8000 \
SDM_RELATIONAL_URL=http://localhost:8001 \
  python -m pytest packages/sdm-client/tests -m live_nim -q

# against real warehouses
python -m pytest packages/sdm-connectors/tests -m live_warehouse -q
```
