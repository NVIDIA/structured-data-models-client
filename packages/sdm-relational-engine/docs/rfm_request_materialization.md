# RFM Request Materialization

`KumoRelational.materialize_task()` generates the Universal TFM requests that
`predict_task()` would send without resolving an API client or contacting a
NIM. Both methods use the same private request iterator, so batching,
neighborhood sampling, exclusions, serialization, and validation cannot
diverge.

## Request records

Each immutable `MaterializedPredictionRequest` contains:

- The final payload mapping passed to the live RFM API.
- A zero-based batch index.
- A half-open `prediction_start` / `prediction_stop` range into the
  original ordered prediction rows.
- The serialized request size in bytes.

The record fields are frozen. The nested payload is deliberately the exact
mapping used by live prediction, rather than a reconstructed or deeply frozen
copy. Consumers should treat it as read-only.

Materialization accepts the same request-affecting task options as
`predict_task()`, including explain configuration, embeddings, run mode,
neighbors, inference configuration, exclusions, prediction time, and top-k.
It performs no retries because it performs no network operation.

## Deterministic neighborhood sampling

`materialize_task()` and `predict_task()` accept `random_seed`, which defaults
to `42`. For local graphs, reusing a seed resets the native neighborhood
sampler for every generated request batch. Given the same client commit, graph
data, ordered task rows, batching, and request options, repeated calls produce
semantically identical payloads and batch provenance. Live prediction and
materialization share this seed path. Passing `None` opts out of reseeding and
allows the sampler's random state to advance between calls.

## Link-prediction context candidates

For link prediction, related-table rows in a materialized request are the RHS
candidates returned by relational neighborhood sampling. Context target lists
remain supervision on the instance table; serialization does not turn target
IDs that were absent from the sampled neighborhood into synthetic related-table
rows. This matches the original KumoRelational client behavior and avoids assigning an
unsampled target a feature row from another instance or anchor time.

Consequently, a sampled context neighborhood can contain no positive RHS
candidate even when its target list is non-empty. Materialization preserves
that sampled result rather than silently changing the effective candidate set
or class balance.

A future positive-coverage policy should be implemented explicitly before
serialization: add missing targets as supervised readout candidates, fetch
their RHS features as of each context example's anchor time, and distinguish
them from historically sampled neighbors. That design can improve positive
coverage without fabricating a historical edge or copying request-wide
features across point-in-time boundaries.

## Local installation prerequisite

Materializing from a local `Graph` performs native neighborhood sampling and
therefore requires the compiled `sdm_relational_engine.relationallib` extension. Install the client
with its default native build enabled; `WITH_RELATIONALLIB=0` is only suitable for
metadata-only workflows and cannot run local materialization.

For a source checkout using the published `kumo-api` dependency:

```bash
python -m pip install --editable .
python -c "import sdm_relational_engine.relationallib"
```

CI runners that materialize local tasks should perform the import check before
running benchmarks so a metadata-only installation fails immediately.

## Replay correlation identity

Canonical replay identifies an evaluation row by the pair:

```text
(SHA-256 of the canonical request, PredictionItem.row_index)
```

The existing Universal TFM schema defines `row_index` as the zero-based row
position in `predict.instance_table.rows`. A materialized request guarantees
that the number of those rows equals `prediction_stop - prediction_start`.
Consequently, a response item maps to the original ordered `TaskTable`
prediction row at:

```text
materialized_request.prediction_start + prediction_item.row_index
```

The NIM must return exactly one `PredictionItem` for each request prediction
row and populate `row_index` with a complete, unique permutation of
`0..N-1`, where `N` is the number of rows in the request's predict instance
table. Every item must also set `id` to the string form of the transport
`instance_id` in that predict row. Response array order is not part of the
identity contract.

`RFMAPI.predict()` validates response cardinality, complete/unique/in-range
`row_index` values, and the transport `id` at every row index. It then
restores request order. The request-local `instance_ids` passed to that method
are generated transport keys used only for correlation validation; they are
not user entity values.

Interactive `ENTITY` values come from the client's private, ordered
`entity_ids` mapping. Native driver entity output and response `id` do not
determine public entity identity. Replay tooling should retain the raw
`row_index` and validate the raw response `id` against the request's
transport `instance_id`.

The client does not hash payloads because canonical JSON encoding, artifact
immutability, and hashing belong to the benchmark artifact producer. The
producer must canonicalize and hash the exact materialized payload before
replay and reject responses with missing, duplicate, out-of-range, or
incomplete row indexes. SHA-256 collisions are treated as cryptographically
negligible.

## Canonical payload identity and opaque names

The Universal payload uses `instance_id` as the non-null, request-local
transport key for every instance row. Related occurrences carry the same-split
`instance_id`, and related-table primary keys include it. Entity roots are
declared through explicit instance-table relationships.

Entity-reference columns use collision-safe names beginning with
`__kumo_entity_ref`. Those spellings carry no semantic authority: consumers
must follow the declared relationship instead of inferring meaning from
`ENTITY`, `ENTITY_n`, `TARGET_PRED`, quantile-like names, or internal-name
prefixes. User-provided names and values remain opaque; only the Universal
`instance_table` and `instance_id` names are reserved.

The task's anchor time is declared by `task.anchor_time_column` and remains
explicit regardless of the `use_prediction_time` inference option.

## Sanitization diagnostics

`KumoRelational.sanitization_report` exposes backend-neutral diagnostics. Local
graphs report mutually exclusive row-removal counts in this precedence order:

1. Null primary key.
2. Duplicate primary key.
3. Null required timestamp.

For each table, the reason counts sum to the total dropped rows. Backends that
cannot observe sanitization decisions return `not_available`; they never use
zero counts to mean unknown. Task entity references are validated against the
fully sanitized local graph before any request batch is generated or sent.

## Request limits

Every serialized context and prediction instance or related table is checked
independently against the Universal TFM 10,000-row limit. This check is
separate from the existing 30 MiB payload limit. The client reports the batch,
table path, actual count, and limit and does not truncate or resample data.
