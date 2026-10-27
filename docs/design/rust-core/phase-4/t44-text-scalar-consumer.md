# T4.4 text features and scalar classification

This development-seam substep extends the numeric and typed-time frame consumer with nonkey text features and scalar text classification targets.
The public journey is `Schema.from_data → scalar PQL → materialize → inspect → scripted KumoRelational predict → PredictionResult.to_pandas → merge`.
Numeric scalar graph keys retain their existing semantics and default routing stays unchanged.

## Shared behavior

The host supplies the existing detached Arrow owners without interpreting feature meanings or choosing samples.
`sdm-engine::frames` admits UTF8 and LargeUTF8 nonkey arrays and reuses the existing Arrow take kernels for row selection.
Categorical values keep their original strings, nulls and source row order.
The frame orchestrator reuses the existing `subgraph::words` helper for a feature whose semantic type is `text`, then the existing typed `Words` payload primitive.
That helper reads both Arrow offset widths directly and delegates normalization to `sdm-engine::words`; no offset narrowing, whole-array text cast or second normalization algorithm is introduced.

Scalar categorical targets reuse the existing static label and class kernels, request task validation and result layout.
No model policy, wire encoding or result conversion is copied into the frame consumer.
The shared request layer continues to own one-class refusals and the ordered model bodies.

The existing Arrow RowConverter fingerprint is unchanged.
It fingerprints the original logical strings and validity before text normalization, independently of Arrow chunks and call-local attachment ids.
Checkpoint owners remain detached from caller mutation, survive a later task lookup failure and are rebound through the existing checkpoint format.
The normal numeric and typed-time sampling paths and their random streams are unchanged.

## Frozen evidence and boundaries

The tests were authored and pushed before production edits.
Their receipt is copied byte for byte from public calls against frozen reference `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1`, with source byte identity, environment identity, blocked sockets and a local scripted transport.
The successful cases cover built-in pandas StringDtype with Python and PyArrow storage and exact Arrow UTF8 arrays.
They include categorical and text features, scalar text classification, Unicode, embedded NUL, empty strings, nulls, all-null feature columns, chunked input and multiple model calls.
Tests compare complete materialized payloads, request bodies, ordered warnings, predictions and merged rows; captured request ids are replayed as host facts.
Plain-data HostRun twins exercise both physical text offset widths without fetches or native semantic callbacks, including fresh-id checkpoint rebinding, rechunking and a same-length logical text mutation that must invalidate a checkpoint.
These authored tests are not a claim that candidate execution or package verification has passed.

Exact built-in ArrowDtype LargeUTF8 has a different frozen native boundary from StringDtype with PyArrow storage.
Its constructor succeeds, but its old relational materializer refuses `large_string[pyarrow]` while building the graph.
That exact native materialization family remains explicitly unfinished in this seam and is retained as a frozen failure control.
Physical LargeUTF8 remains available for supported native StringDtype conversion and is accepted by the language-neutral core.

Both text key offset widths remain explicit unfinished frame-consumer paths.
Admitting text features does not admit text primary keys or foreign keys through the numeric identifier reader.
Object, Decimal, categorical-extension and custom-hook inputs, text-time interpretation, supplied prediction rows, ranking, forecasting, link integration and fitted sessions remain outside this bounded substep.
Typed timestamps retain the previously documented temporal and no-time boundaries.
No contract, generated artifact, version, lock, Java feature API or default routing change is part of this package.

The narrow native LargeUTF8 guard runs only on fresh input, after the existing converter validates the entire current table and returns its native dtype facts.
Earlier invalid labels, duplicate labels and unfinished conversion families therefore retain their current precedence.
A reused checkpoint does not revisit changed caller frames or apply this guard to a snapshot's supported physical LargeUTF8 owners.
This guard marks unfinished development scope and does not replace the old materializer's diagnostic as a final compatibility policy.
