# sdm_relational_engine.api: provenance

This tree began as a copy of the internal `kumo-api` wire-type package,
internalized so the client is self-contained (no external `kumo-api` dependency)
for the open-source release.

**Upstream is gone.** `kumo-api` has been deleted, so this is now the only
copy and the sole maintained source. There is nothing left to re-sync from;
the script that used to do it has been removed along with it. Edit these files
directly, as first-party code.

This file is kept as the attribution and provenance record for the
internalized code, not as an instruction for re-syncing.

## Baseline

| | |
|---|---|
| Upstream package | `kumo-api` |
| Version | `0.92.0` |
| Tag / commit | `v0.92.0`, commit `9ee1c96` |

## Changes made against that baseline

- All internal `kumoapi.*` imports rewritten to `sdm_relational_engine.api.*`.
- `rfm/protos/` (protobuf `.proto` definitions) dropped: the client uses the JSON
  wire format only and never exercises the protobuf code paths, and the
  generated `_pb2` modules are build-time artifacts not shipped in source.
- The methods that imported `rfm/protos/` dropped with it, since every one of
  them raised `ModuleNotFoundError`: `Context.{fill_protobuf_,to_protobuf,
  serialize,from_protobuf,from_bytes,get_memory_stats}` and the parquet codec
  helpers they used (`rfm/context.py`); `InferenceConfig.{fill_protobuf,
  from_protobuf,_target_transform_from_proto}` (`rfm/inference.py`);
  `RFMPredictRequest.{to_protobuf,serialize,from_bytes}` and
  `RFMEvaluateRequest.{to_protobuf,serialize,from_bytes}` (`rfm/requests.py`).
- Kumo Relational Enterprise platform modules dropped as unreachable from the RFM path:
  `jobs.py`, `online_serving.py`, `distilled_model_plan.py`,
  `data_snapshot.py`, `rbac.py`. Nothing in `sdm_relational_engine`, `sdm_client` or
  `sdm_connectors` imports them, and they describe batch jobs, serving
  endpoints, snapshots and RBAC, none of which this client offers.
- `model_plan.py`, `encoder.py` and `train.py` dropped: they describe model
  architectures, encoders and training jobs for a service that trains models,
  and this client sends in-context examples to a NIM and trains nothing.
- `rfm/pquery.py`, `rfm/explain.py` and four of the five `explain/` modules
  dropped: server-side query and explanation types that nothing outside `api/`
  names. Only `explain/gradient.py` is reachable, it backs
  `Explanation.feature_importance`.
- `column_analysis.py` and `subgraph.py` dropped for the same reason.
- Internal identifiers removed from shipped docstrings and comments: a private
  planning-tool URL on `BigQueryCredentials`, and four TODOs naming individual
  engineers.

## Files this client owns outright

These were never upstream's to define, each names only the subset of `api/`
this client actually uses, and the re-sync used to preserve them explicitly:

- `rfm/__init__.py` and `explain/__init__.py`: re-export only the subset used
  here.
- `rfm/requests.py`: keeps only the prediction request and response. Upstream
  also defined validate/parse/evaluate types; this client only ever POSTs a
  prediction.
- `pquery/AST/column.py`: renders a column back to the query text that names
  it, quoting a name the bare identifier cannot spell. See `../pql/SOURCE.md`.

`RunMode` and `MissingType` were the only two names reachable from the dropped
`model_plan.py`. They live in `sdm_relational_engine/runmode.py`, outside this tree, because
the re-sync used to replace `api/` wholesale.
