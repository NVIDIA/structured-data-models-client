# sdm_relational_engine

The Kumo Relational driver for the [`sdm-client`](../sdm-client/README.md) client.

This distribution provides the heavy, client-side machinery a Kumo Relational prediction
needs before a request reaches a NIM: the relational `Graph`/`Table` abstractions,
the neighbor samplers (local native `relationallib`, plus DuckDB / SQLite / Snowflake /
Databricks backends), the PQL parser, and the HTTP client that talks to a Universal
TFM NIM.

It is imported as `sdm_relational_engine` and is normally installed transitively via the client's
`relational` extra rather than on its own:

```bash
pip install sdm-client[relational]      # pulls the sdm_relational_engine driver
```

It can also be installed on its own, for the graph, sampler and PQL machinery.
Predicting is not available that way: the engine's entry points refuse a direct
call, so a prediction has to go through `sdm-client`.

```bash
pip install sdm-relational-engine
# optional data backends:
pip install "sdm-relational-engine[duckdb]"      # or [sqlite] / [snowflake] / [databricks]
# optional features:
pip install "sdm-relational-engine[explain]"     # natural-language explanation summaries
pip install "sdm-relational-engine[relbench]"    # Graph.from_relbench() dataset loading
```

Release wheels are built for CPython 3.10, 3.11, 3.12 and 3.13
(`manylinux_2_28` x86-64); no source distribution is published.

Application code should import the client's neutral surface, not this package
directly:

```python
from sdm_client import PredictiveQuery, SDMClient, KumoRelational

query = PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
    {'users': users_df, 'orders': orders_df},
)
request = query.materialize(indices=[1, 2, 3])

with SDMClient(url='http://localhost:8000') as client:
    result = client.predict(request, model=KumoRelational())
```

## Local development

```bash
pip install -e ".[test,databricks]"
pytest tests -m "not live_nim"
```

The live-NIM suite (`scripts/run_rfm_nim_live_tests.sh --full`) includes durable,
ticket-free rejection regressions for fixed NIM defects; every rejection probe is
followed by a known-good prediction to detect service degradation.

The native neighbor sampler (`sdm_relational_engine.relationallib`) is built from
`src/sdm_relational_engine/csrc/neighbor_sampler.cpp` via CMake/scikit-build-core. Set
`WITH_RELATIONALLIB=0` to skip the native build for a pure-Python editable install.
