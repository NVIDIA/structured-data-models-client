# Text frame domain consumer source checkpoint

This is unapproved source authoring on preserved Text tip `371cf6044d9467c411339d5341fa71a90dc9f47b`, whose integration base is `ab82735df3b1f324dc64470122679698fb428552`.
It is not rebased onto the current integration branch and has no candidate execution or validation evidence.
The new core modules remain unlinked until the complete coordinated composition.
The scope remains all 119 in-scope Phase 4 public paths, including all native families, Text, ranking, forecasting, LINK, default routing and memory/performance acceptance.

## Concrete source

`sdm-core/src/materialize/frame_values.rs` consumes actual graph, label, completed-task, request-plan, complete-reference, sampled-context and payload values through the existing domain kernels.
Graph preparation uses `PreparedGraph` and the retained detached `FrameRun` owners.
Labels use the existing `ResolvedTarget`, `entity_sample`, `LabelledContext` and `label_context` algorithms before the reached native prediction constructor.
Completed tasks apply context narrowing and the existing `BatchLayout` only after actual constructor success.
Plans use the existing request options policy, preserving the later caller-exclusion validity check after whole-task reference resolution.
Whole references finish before a batch can reset or draw the neighbor stream.
Contexts use `TaskRows` and `TaskBatch`, validate each actual per-batch native cast, and reuse `FrameRun::walk` and the shared `walk_resolved` sampler.
Native constructor, reference and Text failures retain their actual stage, ancestry, message, constructor diagnostic and graph-drop summary without a per-element core action loop.
The payload consumer calls the existing request encoder, retains its complete owned result until attachment publication, accumulates ordered encoded parts, emits whole-request notices and constructs the existing Materialized/RequestState value.
It accepts actual core-computed lag columns as domain input; retaining and supplying those columns through the fixed operation remains composition work.
Typed list labels preserve the existing Int, Float, Bool, Text and Decimal link-value families, including list order and decimal scale.

`sdm-core/src/frame_materialize/values.rs` explicitly reads and publishes the named Arrow and metadata attachments.
It binds native dtype-origin facts to the unchanged existing graph value fingerprint.
It publishes actual sampled raw columns, original source positions, example ownership, links and anchors rather than model JSON row trees.
Prediction construction and each selected Text feature now include a core-resolved FrameNativeColumnRef containing the exact original source Relation and column.
Sampled tables carry their actual source as well as their logical table name, and rebinding validates that source against the immutable graph.
Those source bindings participate in task/context identities.
The native facade must use the explicit source binding to find its retained dtype/value owner and must neither infer aliases nor reread a caller Schema after checkpoint reuse.
`sdm-core/src/frame_materialize/identity.rs` binds source/task/plan/reference/context meanings to actual ordered values without input ids, output ids, buffer addresses or Arrow chunk boundaries.
The identity algorithm extends the existing RowConverter-based fingerprint and is a coherence check rather than authentication.

`sdm-engine::frames::Snapshot::selected_rows` checks retained source positions without sampling again.
`selected_link_pairs` derives permissible row references from immutable graph rows and example ownership without reading a random stream.
Sampled-context rebinding verifies the retained raw values and relationship membership, while its context identity binds the exact ordered sampled links.
Relationship validation decodes only the selected foreign-key rows, rather than scanning an entire source column for every batch.

## Retained graph lifetime and complexity

The first source checkpoint rebuilt and fingerprinted the complete source at every fixed stage and twice per batch.
Source inspection established this as an acceptance blocker, and that path must never be linked.
The retained-owner repair replaces those repeated raw-table bindings with the existing actual PreparedGraph and Arc<Mutex<FrameRun>>.
`sdm-core/src/frame_materialize/retained.rs` owns the immutable graph/index allocations and the existing continuing neighbor stream directly.
The owner has no Python object, registry lookup, task rows, program counter or suspended workflow.
Its constructor is crate-private and is reached only after actual graph/native-source validation.
Mutable FrameRun/neighbors belong to one actual cached RelationalMaterializer/model engine, rather than a Schema or a source fingerprint.
Separate PredictiveQuery objects on the same Schema therefore require independent initial neighbor streams, even if immutable snapshot storage is shared.
Source identity is a coherence check, never an instruction to intern or merge mutable owners.

The fixed pure-core-call lease now waits for overlapping access to the same mutable neighbor owner, using a brief mutex/condition-variable admission gate.
It no longer invents a cross-HostRun RUN_BUSY refusal.
The existing per-Run busy contract remains separate.
The original C++ sample call retains the GIL, while the current native binding detaches the GIL during core calls, so waiting for pure Rust access does not hold Python execution hostage.
It is released on ordinary return, error, cancellation or unwind and does not roll back reached neighbor progression.
A FrameRun mutex poisoned during mutation causes existing RUN_POISONED rather than silent reuse of partially updated state.
The lease carries no stage and is not retained while the host performs a native conversion between calls.
The final strong native reference releases the graph/indexes and source rows.

PreparedFrameGraph declares a call-local typed graph-owner attachment with source identity and independently inspectable native facts.
Rebinding compares those immutable facts in O(schema), shares the same graph and FrameRun allocations, and does not fingerprint, read, detach or rebuild the source.
Fresh Graph construction still pays the existing one-time foreign detachment, RowConverter fingerprint and index/adjacency preparation cost.
An independently valid legacy raw checkpoint can initialize a new live owner once; that fallback is not repeated for every stage or batch.
After a live owner advances its neighbor state, an older raw checkpoint is not a valid substitute for that owner.
Native Python dtype/value owners remain a separate binding-local lifetime companion and never enter the Rust owner.
Original source/schema identity and mutation invalidation remain the existing materializer boundary's responsibility.

For N source rows and B batches, the repaired source removes the B-times full-source hash, string scan, detachment and graph reconstruction.
After one existing graph preparation, repeated graph binding is schema-sized, while reference lookups and source/relationship validation are selected-row-sized with the existing ordered-index lookup factors.
Context and request construction retain their actual output-sized work and payload memory.
The existing initial multi-chunk `one_batch` concatenation remains a one-time source-sized memory cost; this source inspection is not a measured zero-copy or performance claim.

Final shared composition must add the typed attachment alternative to the existing core/host attachment enum and kind, and export/import it through an exact frozen native class with no arbitrary Python constructor or coercion.
Only actual core Graph construction may produce that class.
The host must recognize this core-owned alternative before foreign Arrow admission, which remains confined to fresh raw graph inputs.
Pure HostRun uses the same Rust owner directly, without a registry or a second representation of keys/CSC indexes.
Existing Java scope receives only shared compatibility handling.

FrameRun now distinguishes its ordinary raw checkpoint capture from core-created retained ownership.
On the retained path, `keep` updates actual neighbor metadata without capturing or re-exporting the source tables.
The unlinked consumer calls a typed `Asker::keep_retained_frame` handoff that final shared composition must implement.
That handoff must clear an earlier raw checkpoint and retain/export the same live native owner alongside the small sampler memo, including on downstream failure and cancellation.
It must never overwrite reached neighbor state with a stale snapshot from an earlier stage or batch.
Ordinary raw/provider capture remains unchanged.

The root executed an immutable frozen-reference ownership witness against original public source `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1` on CPython 3.10 and 3.12.
Each cell contains fourteen case bundles and forty actual public materialize calls, with no observer, setup or network failures and no model calls.
The result receipt digests are `5ac982b76a9e85a598a9613640e37ff801c40bb5af11af4351ddf32f69549f5a` and `9abf91cae7c6f8965ff59e239c0f29ae7a928423e42b62cfeb6544b4c024bcfb` respectively.
Both cells agree on the following observed boundaries.

| Public boundary | Actual observation |
| --- | --- |
| Repeated unseeded calls on one query | One cached model/sampler/native RNG continues between calls. |
| Fresh query on the same Schema | A distinct model/sampler starts its own seed-123 stream. |
| Explicit seeded batches | Every reached batch resets before its draw. |
| Invalid native count dtype with a new seed | The seed reset survives failure, with no draw consumed. |
| Warning-as-error before sampling | No reset or draw occurs; the already constructed graph/model remains cached. |
| Text conversion failure after the second draw | Both reached draws remain consumed, matching a successful two-draw control. |
| Reentry between draws on the same query | The nested call advances/resets the outer call's continuing stream. |
| Reentry on a fresh query | Its independent stream leaves the outer stream unchanged. |
| Two warm threads under an explicit between-draw schedule | Both calls succeed and match the equivalent same-thread schedule. |
| Two cold initializers held at an explicit constructor barrier | Both independently construct and complete; the later cached assignment determines subsequent continuation. |

The witness records complete ordinary and lossless native payloads, selected neighbors and actual owner identities rather than treating the legacy NumPy global state as a native RNG checkpoint.
Its conversion reentry and cold scheduling barriers are explicit observer effects, and do not claim that an ordinary builtin cast naturally invokes a callback.
These are frozen reference observations, not candidate evidence.
They reject a whole-public-call lock, blanket cross-HostRun busy failure, source-hash RNG interning and invented singleflight cold construction.
The source lease ends before native conversions and native warning presentation, so public journeys sharing a materializer can interleave those calls and its continuing stream.

The asynchronous label consumer explicitly releases its graph access lease before awaiting shared label kernels.
Labels use call-local entity entropy and individually locked FrameRun access, rather than consuming the continuing neighbor RNG.
Source inspection found that shared `Labelling::read` still routes temporal aggregate reads through provider `fetch_rows`.
That kernel needs the existing local frame lookup/snapshot path at shared composition before temporal labels can run as a complete native frame journey.
Holding a graph gate across that foreign suspension would be invalid.
Static labels, frame time bounds and entity selection already use local frame branches.
The final composed path must also establish poison and cancellation behavior at each actual mutable FrameRun access; these source-owner tests alone do not establish HostRun behavior.

## Isolated native ownership source

The new, unconnected `_rust/frame_native.py` module holds the fresh converter's original native dtype owners in an immutable source/column mapping.
It resolves only explicit core-provided source Relations and columns, preserves dotted source names, and neither infers logical aliases nor reads a caller Schema on reuse.
The actual whole prediction constructor uses that retained dtype, while the actual selected Text conversion uses whole-column `fillna('').astype(str)` before Rust-owned normalization.
Successful results retain their actual native Series; failed conversions retain the exact exception without manufacturing Arrow values or starting diagnostic work early.
The companion stores no graph indexes, mutable RNG, full frame copy, callback, registry or workflow continuation.
Its lifetime must follow one cached materializer's exact core graph owner, including downstream warning/conversion failure.

Native exception capture preserves actual traceback, cause, context and suppression, builtin ancestry including KeyError and StructuredDataError, raw class lineage and the deepest native module.
Capture does not observe native error text or select an SDK code, wrapper, phrase match or sanitization detail.
Explicit native message/name reads are uncached and leave failures at those observation points untouched.
Raw qualified class lineage is descriptive metadata, not canonical class identity authority.
The final guard must establish any exact relational-error ancestry fact without importing or invoking the old engine on the new default path.
The source module does not yet choose the final core-selected text-observation interface, bind failure DTOs or perform warning presentation.
It is not a completed facade.

## Source tests and execution status

The new identity, range, native profile and conversion-order tests were written before their consumers.
The snapshot replay and link-alignment tests were written before the corresponding replay methods.
The retained-owner allocation-reuse, blocking-overlap, independent-query, final-reference release and unwind/poison tests were written before their corresponding owner changes.
Native observation tests were written before the new local owner module, covering exact exception identity, untouched StructuredDataError display, repeated message observations, native display failure, source-bound dtype identity and actual whole-column results.
These are source tests and have not been executed.
Only formatter WRITE on the explicitly owned handwritten files is permitted before two independent whole-code approvals.
No candidate import, test, lint, type check, build, generator, native probe or live call has run for this checkpoint.

## Exact composition dependencies and remaining implementation

The handwritten unlinked DTO now carries a completed-task identity, its actual prediction string encoding, a complete-reference identity and a fresh OS entropy fact at the label stage.
The graph's continuing neighbor checkpoint is not a source of fresh per-call entropy.
No shared operation, decoder, state version, generated schema, contract lock or routing file changes here.

The existing encoder returns only wire bytes, so the complete logical NativePayload attachment and RequestState coherence still require the same logical Value to be exposed before encoding.
Reparsing model JSON would lose the distinction between adjacent surrogate points and a supplementary scalar and is not a valid implementation of this requirement.
Earlier encoded calls currently bind their complete wire bytes, context identity, ordered range and summary to the plan; their native logical-value companion and its complete validity checks must be composed before this path is enabled.

The written native word consumer targets the single UnicodeText propagation through the existing `SampledColumn::Words` and wire `Value::Text` consumers.
Its `normalize_native` import requires linking and exposing the preserved native engine kernel.
Those APIs and the contract-native failure type are not available on this old base, and the modules cannot be enabled as they stand.
The new `Attachment::FrameGraph` alternative, `Asker::keep_retained_frame` implementation, sampler-state owner export and host/binding ownership wiring are also deliberately absent on this base.
The data-only RelationalOptions definition still requires the request-owned re-export described in the preserved facade composition delta.
The source now aligns mixed native string offset/byte representations after the actual constructor, retaining the ordinary matching-buffer fast path and preserving distinct surrogate points.
The existing key consumers still need their shared single UnicodeText logical meaning at final composition.

Retained lag columns, ranking/forecast decomposition, complete HostRun journey twins and the native retained-owner facade remain required before the complete Text package can be claimed.
Whole-task/per-batch native reference fallback, stopped constructor diagnostics, lossless StringData export and native kernel facts still require their concrete fixed facade composition.
HostRun no-rebuild, release, cancellation and concurrent-owner twins require the final fixed operation/typed attachment wiring and have not been executed or claimed by the core owner unit sources.
The final frame native error guard must preserve guarded versus unguarded warning contexts, actual native ancestry and the existing graph-drop summary.
Direct NativeFailure alone lacks the frame guard's KeyError/StructuredDataError/relational-error and native-class facts.
Native error text also needs its actual observation boundary: the original guard preserves StructuredDataError without reading its string, while TypeError signature detection consumes native text at its own point.
An early unconditional `str(error)` or a copied Python error-policy table is not a valid native facade implementation.
The existing unfinished-family guards and refusal census must be reconciled with actual admission at composition.

Rebase, serialized shared ownership, canonical generation, two independent exact-source approvals and candidate checks follow the actual approved preceding landing.
Exceptional native payload memory and performance measurements remain Phase 4 completion requirements.
