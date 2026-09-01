# sdm-client

One Python client for NVIDIA's structured-data foundation models.

Structured-data foundation models predict in context. You give the model
labelled rows and the rows you want scored, and it answers in one pass. There
is no training loop, no hyperparameter search, and no per-dataset artifact to
manage.

This client builds the request, sends it to a NIM, and gives you back a
DataFrame. Inference runs in the NIM, so this package downloads no weights and
needs no GPU.

> The relational path in this sample arrives in phase 4. The tabular path
> below it works today against a running NIM.

```python
from sdm_client import (
    StructuredDataClient,
    Schema,
    PredictiveQuery,
    KumoRelational,
)

schema = Schema.from_data({'users': users_df, 'orders': orders_df})
query = PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
)
request = query.materialize(indices=[1, 2, 3])

with StructuredDataClient(url='http://localhost:8000') as client:
    result = client.predict(request, model=KumoRelational())
    print(result.to_pandas())
```

**The model is an argument, not a method.** The same request runs on any model
whose profile accepts it.

## Status

Under construction. The architecture is settled and written down in
[`docs/architecture.md`](docs/architecture.md); the packages are being built
against it phase by phase. Nothing has been released.

| Phase | | |
|---|---|---|
| 1 | Foundation — `sdm-api`, `sdm-pql`, CI | ✅ |
| 2 | Client and the tabular model | ✅ |
| 3 | Connectors | in progress |
| 4 | The relational model | |
| 5 | `fit` and explanations | |
| 6 | Docs and end-to-end | |
| 7 | PQL for tabular | |

## Packages

Five distributions, released together on one version.

| Distribution | Import | Kind | What it is |
|---|---|---|---|
| `sdm-client` | `sdm_client` | pure python | The public API: client, `Schema`, `Request`, model handles, transport, sessions, results |
| `sdm-api` | `sdm_api` | pure python | The Universal TFM wire types. No third-party dependencies |
| `sdm-pql` | `predictive_query` | pure python | The Predictive Query Language: grammar, parser, validator, AST, model profiles |
| `sdm-connectors` | `sdm_connectors` | pure python | Reading source tables from SQLite, DuckDB, Snowflake, Databricks, S3 |
| `sdm-relational-engine` | `sdm_relational_engine` | native | The relational driver: graph building, samplers, the neighbor sampler |

Only `sdm-relational-engine` is native, so the tabular model works anywhere and
the relational model needs Linux x86-64.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e './packages/sdm-api[test]' -e './packages/sdm-pql[test]'
.venv/bin/python -m pytest packages/*/tests -q
```

Lint and format with `ruff`; the formatter owns line length.

## Vendored code

`sdm-pql` is a verbatim copy of the `predictive_query` package from
`kumo/predictive-query-language`, carried only until that repository publishes
to an index. Its bytes are pinned per file in `VENDOR_MANIFEST.json` and
asserted in CI — a repo-wide rename has silently corrupted a vendored artifact
in this family of repositories before. See
[`packages/sdm-pql/SOURCE.md`](packages/sdm-pql/SOURCE.md).

## License

Apache-2.0. Each package carries its own copy of `LICENSE` so its
`pyproject.toml` can reference it; those copies are the same terms as the root
file, not additional ones.
