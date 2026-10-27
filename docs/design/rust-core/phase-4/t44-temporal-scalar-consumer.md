# T4.4 typed temporal scalar consumer

This development-seam substep extends the numeric consumer with typed timestamps on history tables of a static entity query.
Its user journey is `Schema.from_data → scalar PQL → materialize → inspect → scripted predict → merge`.
It admits the existing native datetime64/DatetimeTZ s/ms/us/ns conversion, numeric scalar keys and numeric/Boolean features, with typed timestamp features normalized in Rust.
Default routing does not change.

## Values and selection

The host supplies original Arrow counts, units, zones and validity through the existing constructor/materialization attachment paths.
`sdm-engine::frames` normalizes typed timestamps to UTC-naive nanoseconds with the existing shared instant arithmetic, then derives valid row maps, bounds and adjacency.
Native date parsing, custom conversion effects and text-time interpretation are not part of this extension.

Primary-key nulls and duplicates are identified before missing times are removed.
A first key occurrence with missing time therefore does not make a later duplicate valid.
Valid original row ordinals remain the graph identity; the checkpoint retains detached original Arrow owners and reconstructs derived indexes on resume.
The existing graph-drop notice now includes mutually exclusive missing-time counts after key causes.

Timed forward neighborhoods sort stably by destination and floor-second time, retaining source-row order on equal seconds under DF-1.
Each temporal hop keeps rows at or before the original example's floor-second anchor, then its latest N rows, without consuming the RS-3 stream.
Untimed hops retain the existing MT19937/Floyd behavior.
The same original example anchor is used at every hop.

`sdm-core::materialize` reads valid nanosecond bounds directly from the frame snapshot through its existing memo path, so an unspecified anchor is the latest relevant valid time without a provider request.
Validation uses original nanoseconds; the frame walk uses floor seconds, and temporal instance anchors return through the same seconds round trip as the frozen local sampler.
Timestamp feature values retain nanosecond instants independently of those cuts and emitted anchors.
No-time behavior and DT-3/H24 remain their separate phase acceptance obligation.

## Diagnostics and wire compatibility

The existing `FrameNoValidRows` Sentence/code renders one shared template with a core-selected `requirements` diagnostic value.
Static failures retain exactly `primary keys`; timed failures say `primary key and time column pairs`, matching the frozen reference.
The producer supplies that field through the existing open diagnostic map, without adding a required member to any serialized struct or a Sentence enum variant.
Contract 33, the frame manifest and checkpoint shapes remain unchanged.
No generated schema, lock, bridge baseline, Python facade or Java feature API is edited.

## Evidence and limits

Tests are authored before production commits and use public payloads recorded from frozen reference `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1`.
The recorded cases cover s/ms/us/ns, default/positive/negative anchors, subsecond source-row ties, null-time/duplicate precedence, all-null-time refusal, zero/all/two-hop selections and multiple model calls.
Python tests compare complete scripted model payloads, with timezone, cached mutation, warning-as-error and retry-after-failed-construction controls.
HostRun twins exercise the same cuts and feature times without a fetch, along with fresh-id checkpoint rebinding and the unchanged random-stream position of pure temporal hops.
These authored tests establish no passing execution result until exact code approvals and package verification complete.

Temporal target aggregation, timed entity queries and anchors at an entity's own time remain explicit unfinished development-seam paths.
Timestamp features on a graph with no time column retain their separate unfinished no-time acceptance boundary.
The frozen public route accepts constructor counts for years 1600 and 2263 at s/ms resolution, then refuses materialization with `INVALID_REQUEST` and native out-of-bounds timestamp wording.
Those counts remain lossless constructor inputs; their materialization remains an explicit unfinished development-seam family, rather than acquiring the shared provider's different `VALUE_OUT_OF_RANGE` policy.
Supplied prediction frames, text-time rules, extended native families, ranking, forecasting, link integration and fitted sessions remain later T4.4/Phase 4 work.
This substep does not claim full T4.4 completion or a behavior exception.
