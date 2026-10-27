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

A fixed-call lease prevents overlapping operations on the same mutable neighbor owner using the existing RUN_BUSY code.
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

The actual public reentrant/native-warning/concurrent materialize behavior remains an open compatibility proof.
The source lease protects only one core call; it ends before native conversions and native warning presentation and does not serialize an entire public materialize journey.
Two public journeys sharing one materializer could still interleave those calls and its continuing neighbor stream.
Fresh call-specific leases do not by themselves establish the correct random-state authority.
No blanket public busy refusal, whole-call lock or new serialization policy is approved or implemented by this source checkpoint.

## Source tests and execution status

The new identity, range, native profile and conversion-order tests were written before their consumers.
The snapshot replay and link-alignment tests were written before the corresponding replay methods.
The retained-owner allocation-reuse, busy-state, final-reference release and unwind/poison tests were written before the owner implementation.
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
HostRun no-rebuild, release, cancellation and concurrent-owner twins require the final fixed operation/typed attachment wiring and have not been executed or claimed by the core owner unit sources.
The final frame native error guard must preserve guarded versus unguarded warning contexts, actual native ancestry and the existing graph-drop summary.
The existing unfinished-family guards and refusal census must be reconciled with actual admission at composition.

Rebase, serialized shared ownership, canonical generation, two independent exact-source approvals and candidate checks follow the actual approved preceding landing.
Exceptional native payload memory and performance measurements remain Phase 4 completion requirements.
