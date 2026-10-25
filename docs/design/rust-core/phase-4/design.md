# Phase 4: frames, link prediction, tabular requests and sessions

## Scope and baseline

This is a planning draft based on provisional Phase 3 code freeze `21e6d40845f58e8e7f6cb683b3db203ee0e40cf2`, with contract version 29 and Python S0/S1/S2 routed to the core by default.
Phase 3 is still active; the owner requested isolated Phase 4 foundation work alongside its validation, on a separate branch cut from the landed Phase 3 head.
Reconcile final Phase 3 source and carried findings before dependent changes, preserve its frozen validation candidate, and rebase the Phase 4 branch onto `rust-core` after the owner's Phase 3 merge.
The format follows the current [handbook's design instructions](../orchestration/README.md#design-and-repeated-findings): an itemized plan with concrete ownership, interfaces, dependencies, tests and unresolved choices.
The much larger Phase 3 design and [historical preparation](../orchestration/later-phases-prep.md#phase-4-s3-frames-link-prediction-sessions-and-tabular) supply evidence, not a current schedule or process mandate.

- Deliver Python delivery slice 3, called S3 in [migration-plan.md sections 4 and 8](../migration-plan.md#4-delivery-slices-not-modules): in-memory frames and local CSV/Parquet files; frame ranking and forecasting; link prediction; tabular requests; and fitted sessions on every currently supported Python backend.
- Amazon S3 support and validation are not Phase 4 requirements.
  The owner explicitly directed this exclusion on 2026-10-25: "update the doc (if needed) so it doesn't have to support \"Amazon S3\"" (P4-OWNER-2026-10-25-NO-S3-1).
  Existing Python `sdm_connectors.read("s3://...")` behavior remains available without migration work; no Rust S3 client, public bucket access or S3-specific acceptance campaign is required.
- Enable all three owner-required deferred paths: `model`, `text_time_rules`, and `aggregation` for `LIST_DISTINCT` over a foreign key, including temporal link prediction (KO2-94, KO2-75, P2E-19).
- Complete the fallback behavior, sliced-page byte accounting and text-order decision carried as P2-T2.5-1/2/3, and the large-value behavior FX-3 (KO2-31).
- Cover `Schema.from_data`, hand-built frame tables, `PredictiveQuery(..., mapping)`, `materialize(indices=...)`, `materialize(predict=frame)`, direct `Request(context, predict, ...)`, one-shot prediction, fit, fitted prediction, close, result reading and merging, and connector reads feeding those journeys.
- Use [scope-python.yaml](../phase-0/scope-python.yaml) as the public-path inventory, including warehouse and semantic-view sessions, rather than assuming that S3 means only a frame backend.
  Its historical inventory has 121 slice-3 paths; T4.1 reconciles the actual manifest and current source with the owner's Amazon S3 exclusion rather than freezing this historical count as acceptance.
  Specifically, `connectors.read/csv/s3` and `connectors.read/parquet/s3` are excluded from Phase 4 delivery and acceptance under P4-OWNER-2026-10-25-NO-S3-1; do not claim them as newly migrated or tested paths.
- Java retains Phase 3's approved S0/S1/Databricks scope and receives only shared-contract compatibility changes and regression checks.
  Java frames, sessions, tabular, link prediction, ranking, forecasting, Snowflake, Lakebase and Spark require owner scope expansion.
- Release packaging, publishing, platform expansion, old-engine deletion, routing-seam deletion and distribution consolidation remain Phase 5 work.
  This phase creates no new Rust crate, Python distribution or Java artifact.

## Ownership and data flow

The public journey is native data → lossless Arrow plus explicit native facts → core schema/materialization → model-call bytes → host transport → core result → native presentation.
Local file reads remain in `sdm_connectors.reader`; the core never opens a file, database connection or network socket.
The existing optional S3 reader remains outside this phase's work and returns the same frame input when used by callers with access.
The same Arrow frame input is usable by Python and the pure-Rust test host, and later by another language without importing pandas semantics into a Rust crate.

| Owner | Existing modules to change | New modules proposed | Responsibility |
|---|---|---|---|
| `sdm-contract` | `src/ops.rs`, `data.rs`, `run.rs`, `types.rs`, `error.rs`, `schema_check.rs` | `src/frames.rs`, `src/session.rs` | Typed frame inputs, explicit request/session state and endpoints, errors, supported large-offset types; no algorithms |
| `sdm-schema` | `src/infer.rs`, `assemble.rs`, `held.rs`, `values.rs` | None initially | Infer frame schema, apply overrides and check links through the existing schema algorithms |
| `sdm-engine` | `src/sample.rs`, `random.rs`, `keys.rs`, `graph.rs`, `labels.rs`, `time/text.rs` | `src/frames.rs` | Arrow row selection and frame indexes, frame sampling and random-state transitions, link labels and text-time rules |
| `sdm-request` | `src/predict.rs`, `payload.rs`, `options.rs`, `ranking.rs` | `src/tabular.rs`, `src/session.rs` | Model-specific request validation, wire layout, session pin/scoring layouts, results and limits |
| `sdm-wire` | `src/payload.rs`, `response.rs`, `key.rs` | None initially | Model wire encoding/decoding, including tabular and link responses; no transport |
| `sdm-core` | `src/dispatch.rs`, `run.rs`, `compose.rs`, `describe.rs`, `materialize.rs`, `fallback.rs`, `subgraph.rs`, `predict.rs` | `src/frames.rs`, `src/prepare.rs`, `src/tabular.rs`, `src/session.rs` | Operation orchestration, model-neutral request preparation, source selection, held state, leakage prevention, session transitions and request composition |
| `sdm-data` | Existing dialect renderers only when a link-prediction request needs SQL support | None initially | Continue planning SQL and reading catalogs through `DataLayer`; no pandas or session rules |
| `sdm-host` and `sdm-py` | `sdm-host/src/admission.rs`, `runs.rs`, `codec.rs`; `bindings/sdm-py/src/lib.rs` | None initially | Arrow admission, message validation, buffer/run ownership and panic containment |
| `structured-data-models-client` / `sdm_client` | `schema.py`, `query.py`, `request.py`, `client.py`, `_routing.py`, `_rust/{schema,materialize,predict,spec,run,results,codec,errors}.py` | `_rust/frames.py`, `_rust/sessions.py` | Native conversion, object construction and lifetime, transport dispatch, exceptions and presentation |
| `structured-data-models-connectors` / `sdm_connectors` | `reader.py` and backend conversion helpers only where a public reproduction needs a fix | None initially | Local CSV/Parquet and warehouse I/O, driver conversion and connector errors |
| `structured-data-models-api` / `sdm_api` | Hand-written API source only if the existing model contract requires it | None planned | Existing model API types; regenerate generated output through its generator if required |
| `structured-data-models-pql` and `structured-data-models-relational-engine` | Reference code and tests remain available | None planned | Comparison baseline during migration; production core paths must not call either runtime |
| Java / `sdm-jni` | Existing contract/message/endpoint decoders and binding exhaustiveness as required | None planned | Preserve existing supported calls across the contract bump; no new public feature scope |

Paths in the Rust rows are relative to `rust/crates/<crate>/`, unless a binding path is given.
The crate dependency graph remains [the existing one](../../../../rust/README.md#dependency-diagram): bindings → `sdm-host` → `sdm-core` → domain crates → `sdm-contract`, with `sdm-data` composed through `DataLayer`.
Reuse Arrow compute/row kernels and the existing time library; port only the defined reference algorithms and registered random streams.
Do not promote [`sdm-testkit::memory`](../../../../rust/crates/sdm-testkit/src/memory.rs) into production: it is deliberately a limited, SQL-like test provider, with different sampling and only its documented filters.

## Contract and API boundaries

T4.1 owns the common contract change, provisionally version 30 if Phase 3 finishes at 29.
The following are the proposed contract responsibilities; T4.1 locks their serialized shapes and state versions together before dependent packages implement them.

1. **Frames enter as data, not as a Python provider.**
   Add one typed source selection to describe/materialize: an existing provider choice, or a frame-table manifest naming Arrow attachments and neutral table metadata.
   The manifest is an input-sized bytes attachment; data stays in Arrow attachments, with original row ordinals and column order preserved.
   `sdm-core::frames` owns a run-local table set and derives its capabilities; the caller does not supply arbitrary frame sampling policies.
   Existing `DataRequest` shapes, neutral catalog answers and expected-schema checks remain the shared vocabulary.
   A frame request is answered inside the core through `sdm-engine::frames`, while an external provider request retains the existing `DataLayer` planning and `FetchStep` path.
   Frame requests retain the same request validation and expected-schema admission checks; a frame-source distinction must not bypass the provider contract's key, filter, paging or completeness rules.
   Frame execution is an internal source branch in the existing run, not a second public run protocol, a new driver framework, or a SQL engine embedded for frames.
2. **Native conversion reports facts; the core decides behavior.**
   `_rust/frames.py` converts each pandas column once using a reviewed conversion table and existing codec scalars, preserving integer widths, decimals, nulls, zones, nested values and empty/all-null columns.
   Object columns are classified column-wise, with explicit diagnostics for values that have no lossless contract representation; they are never coerced wholesale through text or float64.
   Native index information travels only as typed input conversion facts, supplied before the run selects rows; the core owns filtering, sampling and ordering of the resulting subset.
   [`_context_within_limit`](../../../../packages/structured-data-models-client/src/sdm_client/materializers/tabular.py) runs `candidates.sample(...).sort_index()`: equal index labels can preserve sampled order, so sorting sampled rows by a precomputed full-frame permutation is insufficient.
   Candidate input facts are lossless typed labels where the closed set suffices, or explicit equality/comparison ranks and incomparable classes, null classifications, category ordering and MultiIndex level/code/row tuples where native comparison needs them.
   They describe the original index independently of any eventual sample, retaining row identity and input order; one global permutation is not the proposed representation.
   The exact sufficient representation and core algorithm remain a narrow T4.1 design obligation: trace the actual reference sorting call paths and prove their composition reproduces supported filtered/sampled subsets, including duplicate, mixed-label, null, categorical and MultiIndex behavior.
   That proof must account for subset-dependent monotonic shortcuts and tie behavior rather than inventing a stable tie rule, and must cover any comparison failure only on the subsets where the reference encounters it.
   Use existing Arrow/order primitives or a bounded defined algorithm established by that source analysis; do not build a generic pandas index/runtime emulator or an unbounded comparison framework.
   Validate input fact lengths, row/level/code bounds, encoding consistency and correspondence to the admitted frame snapshot without demanding a total order for labels the reference cannot compare; carry input-sized facts in attachments whose lifetime and mutation invalidation follow the frame input.
   Both Python and rust-host supply the same typed input facts without Python object handles; rust-host performs the actual subset sorting in the shared core, not by replaying a final host-sorted permutation.
   The [locked L0 rule](../migration-plan.md#6-testing) and [`Step`](../../../../rust/crates/sdm-contract/src/run.rs) permit only fetch/model/done steps, with no request for language sorting; do not add a semantic callback or disguise one as fetch, done or a second prepare call after selection.
   A pandas dtype name, object handle, arbitrary callback, `repr` or inferred PQL tree never crosses the contract.
   Core inference, overrides, task choice, leakage checks, sampling, limits and warnings do not move into this converter.
3. **State is an explicit value with a defined lifetime.**
   Extend the existing `SamplerState`/`SamplerMemo` pattern to carry the frame neighbor stream needed by RS-3, including continuation across unseeded calls and reset rules for seeded subgraphs.
   The host owns Arrow buffers and opaque versioned state between calls; the run borrows admitted data for its lifetime and returns replacement state, including the reference's state progression on failure.
   Data/schema identity and mutation invalidate cached frame indexes and incompatible sampler state; no process-global RNG, hidden core registry or host-object identity enters the memo.
   A request snapshots schema meaning as today, and changes to caller frames must follow the observed reference behavior rather than silently reusing stale converted data.
4. **Direct preparation is model-neutral; model binding happens when supplied.**
   Add a typed model-neutral prepare-request operation for context/predict Arrow halves, target, explicit task, held schema and options, preserving [`Request.__init__`](../../../../packages/structured-data-models-client/src/sdm_client/request.py)'s immediate generic checks and inspectable halves without a model argument or implicit default model.
   Its held state is explicitly distinguished and versioned separately from a model-bound materialized request; preserve the caller's direct task instead of inferring it from target cardinality.
   When `predict`, `fit` or explicitly model-bound `materialize` supplies a model, core preflight selects the model specification, checks capabilities/options and invokes the shared model-specific builders at the current [`SDMClient._prepare`](../../../../packages/structured-data-models-client/src/sdm_client/client.py) boundary.
   PQL materialization may derive a task by its existing query/target rules; that does not change the direct constructor's explicit-task contract.
   Preserve the public Python constructor and `materialize(..., model=...)` signatures, and permit reuse of model-neutral held state with a later supported model without mutating it into one model's state.
   The host does not select a Python adapter to decide validation or payload semantics, and generic constructor failures do not move to prediction time.
   Existing request state and held-payload compatibility remain accepted, including requests made before the core switch and inspection of their halves.
5. **Sessions are typed model calls and explicit session state.**
   Add fit, fitted-predict and close operations, and extend the current predict-only `Endpoint` with create-session, session-predict and delete-session calls carrying an explicit session id where required.
   The model-call contract must express the HTTP method, body presence and expected reply kind needed for DELETE and its successful empty reply; a successful close must not be parsed as a prediction response.
   The host maps those calls onto existing transport methods and returns raw bytes or structured failure facts, preserving configured retries, timeouts and endpoint-specific idempotence rules in [`core/transport.py::_build_retry` and `_build_session`](../../../../packages/structured-data-models-client/src/sdm_client/core/transport.py).
   Session creation retains connection-only retries, excluding read/status replay after a possible server-side effect; scoring retains its configured retry policy, and idempotent DELETE retains its retries and bounded cleanup timeout.
   The bridge adds no application-level retry loop, fit replay or independent lifecycle transition; the core does not schedule individual transport attempts.
   Versioned session state holds the returned id, model, pinned question/schema/task, selected outputs and identities needed for leakage checks, without retaining the uploaded context payload.
   The core validates later requests against that pinned contract and sends only scoring rows; the host owns `FittedModel` context-manager behavior and transport lifetime.
   While a step is suspended, the host keeps its run and referenced Arrow buffers alive; callbacks submit a reply only to that live run, under the binding's existing ownership/busy rules.
   Abandoning a pending call releases native run resources and permits no later callback into the disposed run; it does not invent a successful remote acknowledgment or automatically replay fit.
   A lost create reply yields no fitted handle without a confirmed id, and an unacknowledged delete leaves the handle retryable; remote expiry and transport cancellation retain the existing endpoint/host behavior.
6. **Buffer limits and error boundaries remain explicit.**
   Arrow admission preserves large-offset text/binary where FX-3 requires description, with an explicit closed-type extension rather than truncating offsets or converting through Python objects.
   Ordinary lossless large-to-small conversions still use the current admission path when values fit.
   Oversized values are blocked before an incompatible kernel or wire encoder reads them; FX-3's VARCHAR is described and then refused by name at materialize, while its BLOB is omitted without reading its contents.
   New errors join the existing closed code/kind/wording tables, with input-sized details attached and no host parsing of error text.
   Regenerate contract schemas/locks and public snapshots with their existing generators, update binding decoders together, and verify current Java calls still decode unchanged results.

The frame source branch is needed because the current fetch protocol sends host-provider requests to the host, which would otherwise put sampling and time rules back into Python.
The session additions are needed because [`Operation`](../../../../rust/crates/sdm-contract/src/ops.rs) has no session operations and [`Endpoint`](../../../../rust/crates/sdm-contract/src/run.rs) currently has only `Predict`.
These are bounded additions to existing contracts, not a new abstraction for every S3 feature.

## Work packages and dependencies

`T4.*` names units of work, not Cargo crates, distributions or installable packages.
Only the paths listed above and tests/contracts needed by each journey are implementation scope; tooling defects ride with the affected work only when a concrete failure requires them.
At most three packages run concurrently within eight vCPUs; full CI runs only nightly and at phase end.
Tests are written before implementation, and both independent Sol 6.1 High reviewers approve the same exact candidate before its package tests, lint or cheap checks run.
The owner's separate direction permits isolated foundation progress while Phase 3 validates; it does not approve a serialized contract bump, changed behavior, new live resources or a merge into `rust-core`.

### T4.1 – Shared input, state and session contracts

- Own the contract work above, binding/message compatibility and the mapping from the current S3 manifest to the following package responsibilities.
- Modify `sdm-contract`, `sdm-host`, `sdm-py`, Python codecs/stubs and the compatible Java/JNI decoders together; generate schemas and locks through the existing tooling.
- Specify attachment ownership, close acknowledgments, failed-call state progression, native index input facts, frame mutation invalidation and the large-offset admission boundary before freezing shapes.
- Resolve the RS-5 technical design obligation before freezing its input shape: show that the proposed facts and a concrete core sorting algorithm reproduce the reference on filtered/sampled subsets across every supported index family, including failures and duplicate-label ties.
  The representation above is a candidate, not a claim of completeness; if that proof finds a real unsupported behavior or necessary boundary exception, bring the concrete source reproduction and alternatives to the owner rather than adding a callback, approximation or refusal.
- Distinguish/version model-neutral prepared state from model-bound materialized state, preserving later model selection, immediate generic constructor checks and existing incompatible gathered-payload diagnostics.
- Write round-trip, unknown-version, attachment-kind, index-fact validation and core subset-order equivalence, empty-delete-reply and existing Java-call compatibility tests first.
- Dependency: final Phase 3 contract/source reconciliation; T4.2, T4.3, T4.5 and T4.6 may develop against this reviewed interface after it lands.

### T4.2 – Frame conversion, schema and large-offset admission

- Deliver `Schema.from_data`, mapping input and manually constructed frame tables through `_rust/frames.py`, `sdm-core::frames`, and existing `sdm-schema` inference/assembly/edit rules.
- Reuse `_rust/spec.py`'s held-schema serialization and `Column._source` facts; do not introduce a second schema representation or persistent copy that overrides live edits.
- Implement column-wise native conversion, the proven native index input facts and Arrow admission/lifetime changes, including FX-3 in `sdm-host::admission`, schema inference and the pre-materialize size guard.
- Exercise public constructors, snapshots, edits and reference pickles with mixed object columns, nullable/wide numeric keys, nested values, indexes with nonmonotonic labels, all-equal labels, mixed duplicates or mixed label types, MultiIndex, and caller frame mutation.
  Check filtering before sampling, the core's sampled-order preservation where the reference keeps equal labels, comparison failures on the affected subsets, malformed index facts, frame mutation and fact-buffer lifetime.
- Prove FX-3 using the actual DuckDB `arrow_large_buffer_size` path: unread oversized BLOB, described oversized VARCHAR, named materialize refusal, and adjacent fitting values.
  A reduced-capacity unit check supplements that public reproduction; it does not replace the real large-offset acceptance case.
- Dependency: T4.1; exports the Arrow frame input consumed by T4.4 and T4.6.

### T4.3 – Complete the reusable fallback

- Fix P2-T2.5-1 in `sdm-core::{compose,describe,materialize,fallback}`: a bounded `column_ranges` fallback failure follows describe's existing preview recovery where appropriate, and materialize accepts undeclared optional requests when the declared fallback can answer them.
  A provider that declares an operation and then fails still returns its error; it never silently falls back.
- Fix P2-T2.5-2 in `sdm-core::fallback`, `run` and `sdm-host::runs`: account for the logical Arrow page slice and retained data deliberately, including offsets, null buffers and nested children, without charging an unrelated shared parent buffer as transferred page bytes.
  Distinguish the transfer budget from actual retained memory and avoid unbounded retention as the price of correcting false refusals.
- Resolve P2-T2.5-3 explicitly: Arrow byte order is not PostgreSQL locale collation, so fallback cannot claim equivalence for a provider whose text ordering changes selected rows.
  Keep frame native ordering and external-provider collation separate; the owner choice below determines whether such a fallback remains restricted or gains an explicit supported ordering contract.
- Write public describe/materialize cases for undeclared `time_bounds`, `sample_entities`, `neighbors`, `lookup`, multi-page slices, exact/over limits, declared failures and text order affecting a cut.
- Dependency: T4.1 only for any required ordering/budget clarification; T4.4 reuses fallback kernels only where their measured semantics agree.

### T4.4 – Frame materialize and predict, including ranking and forecasting

- Deliver the frame journey end to end through `sdm-core::{frames,materialize,subgraph,predict}`, `sdm-engine::frames`, and the existing request, ranking, forecast and result modules.
- Port the actual rules from `rfm/backend/local/{graph_store,sampler}.py`, `csrc/neighbor_sampler.cpp` and `rfm/base/sampler.py`: sanitization, relationship-wide key representation, entity mapping, temporal/static hops, query-data traversal, row order and feature/label construction.
- Reuse existing Arrow filtering/take/grouping and time kernels; implement frame-specific selection policies once rather than duplicating the whole SQL sampler or copying the test-only memory provider.
- Implement RS-3 and RS-4 using the registered algorithms and explicit state; retain the already shipped RS-1/RS-2 behavior and verify no regression in SQL paths.
- Enable `TextTimes::ClientRules` using `sdm-engine::time::text`'s fixed-format reader and whole-column shape/day-order decisions, preserving existing approved out-of-format refusals.
  Remove both `text_time_rules` refusals in `sdm-core::materialize`, including timestamp features as well as time columns; no call delegates parsing to pandas.
- Test indices and supplied predict frames, zero/missing/repeated keys, null and duplicate time/key sanitization, static/temporal/keyless neighbors, tied cuts, composite/decimal relationships, seeded/unseeded call sequences and failure-state continuation.
- Include frame ranking and multi-step forecasting, UTC/named-zone/fold cases, DT-3's public frame/H24 proof, schema mutation and result inspection/merge through Python and rust-host.
- Dependencies: T4.2 and T4.3; preserve each path's observed ordering/null behavior, and obtain approval before accepting any proposed changed model inputs.

### T4.5 – Link prediction on frames and existing providers

- Lift P2E-19's `LIST_DISTINCT`-over-foreign-key refusal in `sdm-core::materialize` and carry the referenced entity identity through label construction, payload encoding and result decoding.
- Cover static and temporal link prediction across S1/S2 providers as well as frames, preserving time windows, distinctness, relationship key types and the difference between target ids and displayed keys.
- Modify `sdm-engine::labels/keys`, `sdm-request::payload/predict`, `sdm-wire::payload/response` and dialect renderers only where the existing typed requests cannot yet produce the required rows.
- Start with recorded reference requests and answers for `target.aggregation.list-distinct`, then cover nulls, duplicate links, empty neighborhoods, composite/decimal/text ids and time boundaries.
  Do not widen direct `Request(...)` to arbitrary multi-label targets: its current refusal is distinct from relational link prediction and needs a supported model wire shape before it can change.
- Dependencies: T4.1; existing SQL legs can proceed before T4.4, with frame integration following T4.4.

### T4.6 – Tabular preparation and one-shot prediction

- Move `materializers/tabular.py` and `adapters/tabular.py` into `sdm-core::tabular` and `sdm-request::tabular`, with the direct constructor's model-neutral data-dependent validation in `sdm-core::prepare`, reusing wire/result primitives.
- Support direct context/predict frames and single-table PQL via both indices and supplied predict rows, preserving constructor-time generic refusals and explicit task, PQL-only task derivation/overrides, later model/profile preflight, schema widening across both halves, identity mapping and empty scoring.
- Preserve holdout/leakage behavior, remove rows with unknown targets before sampling, implement the 10,000-context-row limit and warning at the appropriate materialization/model-preflight boundary, and implement RS-5's core-selected fixed-seed sample followed by core sorting of that subset using T4.1's proven input facts and algorithm.
- Test constructing a request without a model, inspecting it, and reusing it with later supported models; include unchanged immediate constructor failures and explicit binary/multiclass task preservation on the same target values.
- Lift `model` at the real model-selection/preflight boundary and test both `KumoTabular()` and `model='kumo-tabular'`; relational gathered payloads passed to the tabular model retain their existing diagnostic.
- Compare payloads and results with the reference for classification/regression, one/two/many classes, all-null versus typed predict columns, exact/wide numeric targets, context limits, filtering before sampling, indexes with equal labels, mixed duplicates, mixed types or nonmonotonic labels, MultiIndex, and missing/incompatible features.
- Dependencies: T4.1 and T4.2; its prepared-request/session layout feeds T4.7.

### T4.7 – Fit, fitted prediction and close

- Implement the typed lifecycle in `sdm-core::session` and request pin/scoring layout in `sdm-request::session`, with `_rust/sessions.py` translating objects and dispatching existing transport methods.
- Cover relational contexts materialized on all existing providers, semantic views and frames, plus tabular sessions; reuse one-shot request validation and response decoding.
- Preserve `client.py`'s observed behavior: context-only fit, holdout exclusion across temporal anchors, pinned-question checks, pinned dtype conversion, scoring-leak warnings, empty prediction, batched numbering, missing session ids and unsupported endpoints.
- Keep existing refusals for rewritten/multi-question requests unless the reference supports the exact journey; a session pins one context and must not silently collapse ranking or forecasting decomposition into it.
- Close becomes final only after successful deletion, remains retryable after a failed deletion, and is idempotent after success.
  Python context managers preserve an already active exception and client close handles outstanding fitted objects as today.
- Preserve the existing configured endpoint-specific transport policy with no extra bridge retry loop: scripted checks distinguish create's connection retries from excluded read/status replay, scoring retries and idempotent DELETE retries, including `max_retries=0`, ambiguous/lost create replies, transient delete failure and the cleanup timeout.
- Test create/predict/delete wire calls with a scripted endpoint, failed creation/deletion, repeated close, closed prediction, context-manager exceptions, concurrent separate handles, cancellation/late replies, native-buffer release, and state-version mismatch.
- Dependencies: T4.4, T4.5 where link sessions are reference-supported, and T4.6; warehouse-session live evidence is part of T4.8, not silently postponed to Phase 5.

### T4.8 – S3 integration, reference evidence and switch

- Complete all in-scope delivery-slice-3 manifest paths through local connector reads, result inspection/merge and reference-pickle compatibility, excluding the two Amazon S3 paths identified above; verify frame/file reads never invoke the old engine after routing to the core.
- Update `_routing.py` once for the complete S3 scope, retaining the test-only routing seam until Phase 5 and preserving S0/S1/S2 defaults.
- Remove the three enabled not-yet rules and their temporary refusal expectations/census bounds together; reconcile register entries with observed answers rather than merely deleting refusal tests.
- Run the slice and switch acceptance on the final reviewed candidate: Python and rust-host cases, existing Java regressions, generated/differential and refusal coverage, real-data/held-out evidence, memory and native-lifetime checks, plus the applicable live gates.
- Record bounded frame versus SQL comparisons on real AdventureWorks/RelBench data for ranking, forecasting and link labels, including the resolved ordering policy; performance claims use real data rather than generated accuracy tasks.
- Include local CSV/Parquet read → schema → materialize → predict, warehouse/view fit → fitted predict → close, tabular one-shot/session calls on an endpoint serving that model, and link calls on a supporting endpoint.
- Keep the existing slice/switch requirements for live L6, the Databricks matrix and applicable L7 timing; reuse Phase 3 evidence only when the accepted gate permits it and the relevant code/data identity is unchanged.
  No new live-resource permissions, gate relaxation or reuse policy is assumed by this draft.
- Dependencies: all prior packages and any owner decisions needed for proposed changed model inputs; phase-end full CI and the owner's merge decision remain separate from package checks.

The dependency spine is T4.1 → T4.2/T4.3 → T4.4, with T4.5 on the shared contract and T4.6 on frame conversion; those journeys converge at T4.7 and T4.8.
Run complete user journeys early on Python and rust-host so the final switch does not become the first integration of conversion, sampling, wire encoding or session state.

## Decisive risks and owner choices

1. **Make SQL/frame ordering explicit without changing the model's context accidentally.**
   F2 and ORD-2/ORD-3 explicitly defer this settlement to Phase 4.
   The historical register describes frame null times as retained, but the current reference's `local/graph_store.py::sanitize` builds a mask excluding null keys, duplicate keys and null times before traversal.
   T4.4 must reproduce the public journey and trace where that mask is applied before proposing final rules.
   Default and recommendation: keep each path's measured reference policies explicit in core-owned request/sampling rules, including order-sensitive float32 reductions.
   Record that compatibility settlement for F2 rather than making broad unification a prerequisite; propose unification to the owner only with a concrete measured benefit and examples of changed selected rows, labels or model inputs.
   Do not silently extend ORD-3's numerical allowance to arbitrary frame differences or copy its tolerance into a new comparator.
2. **Choose the text-order promise for external-provider fallback.**
   PostgreSQL locale collation and Arrow's byte order can choose different neighbors at a limit; Python-native frame ordering does not establish PostgreSQL equivalence.
   Recommendation: support frame ordering from its defined data/facts, and restrict external fallback when it cannot honor the provider's ordering, with a named refusal and exact trigger proposed to the owner if it removes reference support.
   Shipping a locale approximation or changing PostgreSQL pushdown order requires a separate owner decision.
3. **Text-time support has an existing boundary, not an invitation to emulate pandas.**
   Implement the accepted formats and client whole-column decisions under E66(h), DT-15 and DT-16, while reconciling PK-7's older wording against the current `core/dates.py` and recorded reference.
   Inputs inside those documented formats that change reading still need owner approval; documented out-of-format refusals already have it.
   Removing `text_time_rules` must demonstrate actual successful frame time and timestamp-feature predictions, not rename the refusal.
4. **Live tabular/session support is a resource prerequisite.**
   Phase 3's warm relational-v26 endpoint and Databricks serving transport are not evidence that tabular or session endpoints are available; Databricks serving explicitly has a one-shot sessions refusal today.
   Identify usable endpoints and permissions before T4.8's campaign and retain the existing unsupported-session diagnostics where the transport does not serve them.
   The historical suggestions to drop warehouse-session live tests or defer tabular were never authorization to shrink Phase 4.
5. **RS-5 index ordering has an explicit technical proof obligation.**
   T4.1 must establish sufficient input index facts and a concrete core algorithm before freezing that part of the contract; this planning draft does not claim the candidate representation already covers every supported index.
   T4.2 and T4.6 depend on that proof, including equal and mixed duplicate labels, mixed label types, nulls, categories, MultiIndex and filtering before sampling.
   This is an engineering obligation within the existing boundary, not an assumed owner exception; only a demonstrated need to change behavior or the locked boundary triggers a concrete owner proposal.

Concrete proposed user-visible changes, missing live-resource authorization and any change to locked acceptance or architectural boundaries need owner input.
Module names, use of existing Arrow kernels, package scheduling and serialized field choices are reviewable engineering decisions within the stated scope; the contract bump itself retains its existing owner-approval requirement.
No new owner decision is required to draft or independently review this plan.

## Exit conditions

- Every in-scope delivery-slice-3 inventory path, excluding the two owner-excluded Amazon S3 paths above, is linked to a passing public case or an already approved explicit refusal, with Python and rust-host agreeing on the same typed inputs and scripted/captured model replies.
- `model`, `text_time_rules` and foreign-key `LIST_DISTINCT` each have successful reference-equivalent end-to-end answers, their temporary refusal rules are gone, and their census/register expectations match measured behavior.
- P2-T2.5-1/2/3 and FX-3 each have a verified outcome or an owner-approved precise behavioral decision, with no silent transfer truncation, collation approximation or offset narrowing.
- Sessions pin context once, send only scoring rows thereafter, preserve question/schema/identity meaning, and release resources with the documented success/failure behavior.
- Default S3 calls perform core-owned semantics without importing the old relational engine or Python PQL runtime; native conversion, I/O, object behavior and presentation remain in their hosts.
- Package checks follow two exact-SHA independent approvals; nightly and phase-end full checks remain distinct, and both repository Python suites run separately as required by `CONTRIBUTING.md`.
- The final slice/switch gates and whole-phase review pass on the final reconciled candidate, with no unresolved P0/P1 and all other findings assigned or explicitly declined.
  Phase 3 completes independently, and only the owner authorizes the later phase merge into `rust-core`.
