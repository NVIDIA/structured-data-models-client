# T4.4 categorical text features and scalar classification

This development-seam substep extends the numeric and typed-time frame consumer with categorical text features and scalar text classification targets.
The public journey is `Schema.from_data → scalar PQL → materialize → inspect → scripted KumoRelational predict → PredictionResult.to_pandas → merge`.
Numeric scalar graph keys retain their existing semantics and default routing stays unchanged.

## Shared behavior

The host supplies the existing detached Arrow owners without interpreting feature meanings or choosing samples.
`sdm-engine::frames` admits UTF8 and LargeUTF8 nonkey arrays and reuses the existing Arrow take kernels for row selection.
Categorical values keep their original strings, nulls and source row order.
Scalar categorical targets reuse the existing static label and class kernels, request task validation and result layout.
No model policy, wire encoding or result conversion is copied into the frame consumer.
The shared request layer continues to own one-class refusals and ordered model bodies.

The existing Arrow RowConverter fingerprint is unchanged.
It fingerprints original logical strings and validity independently of Arrow chunks and call-local attachment ids.
Checkpoint owners remain detached from caller mutation, survive a later task lookup failure and are rebound through the existing checkpoint format.
The normal numeric and typed-time sampling paths and their random streams are unchanged.

Text identifying columns are refused at the physical UTF8/LargeUTF8 key-reader boundary.
Logical schema dtype `string` does not identify a physical text key: the constructor also uses it for UInt64 columns with values above i64 maximum, and callers may edit numeric key metadata.
The consumer therefore retains numeric identifying arrays whose schema metadata names string.
Frozen public positive witnesses cover ordinary numeric keys after a dtype edit and safe selected UInt64 keys beside an unsampled wide key with matching foreign keys.
An all-wide selected-key witness separately reaches the old payload's int64 entity-reference range diagnostic; it is not evidence of unrestricted wide-key wire support.

## Required remaining Text normalization work

The whole new frame `Text` semantic family remains explicitly unfinished until the core has an adequate native execution profile.
This is required later Phase 4 work, not a scope exclusion or a final approved refusal.
The categorical string feature and scalar classification paths in this package do not normalize feature contents.

Actual frozen public model bodies differ across supported native runtimes even when the source input uses built-in StringDtype or exact Arrow UTF8.
For selected feature values `ΟΣ` and `İ`, pandas 2.2.3 sends `['ος']` and `['i̇']`, while pandas 3.0.6 sends `['οσ']` and `['i']`.
The old engine's `fillna('').astype(str).str.lower()` selects Python full/contextual lowercase in the first recorded cell and Arrow simple lowercase in the second.
The existing provider `subgraph::words` helper and engine normalization algorithm remain unchanged.
No content-based Unicode gate, host semantic normalizer, blanket extension fallback or global normalization policy is introduced.
The current frame manifests do not represent the required execution profile; later work must establish adequate native facts and preserve both observed model inputs in the shared core.

## Frozen evidence and boundaries

Tests were authored and pushed before production edits and corrective tests were pushed before the corresponding repair.
Their receipts are copied byte for byte from public calls against frozen reference `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1`, with source/native byte identity, environment identity, blocked sockets and a local scripted transport.
The original forty-case pandas 3 receipt is preserved unchanged, including its native Text successes.
New pandas 2 and pandas 3 receipts retain the Unicode execution-profile evidence, numeric metadata-edit success and safe selected UInt64 success.

Successful categorical feature and classification cases cover built-in pandas StringDtype with Python and PyArrow storage and exact Arrow UTF8 arrays.
They include Unicode, embedded NUL, empty strings, nulls, all-null feature columns, chunked input and multiple model calls.
Tests compare complete materialized payloads, request bodies, ordered warnings, predictions and merged rows; captured request ids are replayed as host facts.
Plain-data HostRun twins exercise both physical text offset widths without fetches or native semantic callbacks, including fresh-id checkpoint rebinding, rechunking and a same-length logical text mutation that must invalidate a checkpoint.
These authored tests are not a claim that candidate execution or package verification has passed.

Exact built-in ArrowDtype LargeUTF8 has a different frozen native boundary from StringDtype with PyArrow storage.
Its constructor succeeds, but its old relational materializer refuses `large_string[pyarrow]` while building the graph.
That exact native materialization family remains explicitly unfinished in this seam and is retained as a frozen failure control.
Physical LargeUTF8 remains available for supported native StringDtype conversion and is accepted by the language-neutral core.
The narrow native guard runs only on fresh input, after the existing converter validates the entire current table and returns its native dtype facts.
Earlier invalid labels, duplicate labels and unfinished conversion families retain their current precedence.
A reused checkpoint does not revisit changed caller frames or apply this guard to supported physical LargeUTF8 owners.
The guard does not replace the old materializer's diagnostic as a final compatibility policy.

Both text key offset widths remain explicit unfinished frame-consumer paths.
Object, Decimal, categorical-extension and custom-hook inputs, text-time interpretation, supplied prediction rows, ranking, forecasting, link integration and fitted sessions remain outside this bounded substep and retain their Phase 4 obligations.
Typed timestamps retain the previously documented temporal and no-time boundaries.
No contract, generated artifact, version, lock, Java feature API or default routing change is part of this package.
