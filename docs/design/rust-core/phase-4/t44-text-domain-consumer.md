# Text frame domain consumer source checkpoint

This is unapproved source authoring on preserved Text tip `371cf6044d9467c411339d5341fa71a90dc9f47b`, whose integration base is `ab82735df3b1f324dc64470122679698fb428552`.
It is not rebased onto the current integration branch and has no candidate execution or validation evidence.
The new core modules remain unlinked until the complete coordinated composition.
The scope remains all 119 in-scope Phase 4 public paths, including all native families, Text, ranking, forecasting, LINK, default routing and memory/performance acceptance.

## Concrete source

`sdm-core/src/materialize/frame_values.rs` consumes actual graph, label, completed-task, request-plan, complete-reference and sampled-context values through the existing domain kernels.
Graph preparation uses `PreparedGraph` and the retained detached `FrameRun` owners.
Labels use the existing `ResolvedTarget`, `entity_sample`, `LabelledContext` and `label_context` algorithms before the reached native prediction constructor.
Completed tasks apply context narrowing and the existing `BatchLayout` only after actual constructor success.
Plans use the existing request options policy, preserving the later caller-exclusion validity check after whole-task reference resolution.
Whole references finish before a batch can reset or draw the neighbor stream.
Contexts use `TaskRows` and `TaskBatch`, validate each actual per-batch native cast, and reuse `FrameRun::walk` and the shared `walk_resolved` sampler.
Native constructor, reference and Text failures retain their actual stage, ancestry, message and constructor diagnostic without a per-element core action loop.

`sdm-core/src/frame_materialize/values.rs` explicitly reads and publishes the named Arrow and metadata attachments.
It binds native dtype-origin facts to the unchanged existing graph value fingerprint.
It publishes actual sampled raw columns, original source positions, example ownership, links and anchors rather than model JSON row trees.
`sdm-core/src/frame_materialize/identity.rs` binds source/task/plan/reference/context meanings to actual ordered values without input ids, output ids, buffer addresses or Arrow chunk boundaries.
The identity algorithm extends the existing RowConverter-based fingerprint and is a coherence check rather than authentication.

`sdm-engine::frames::Snapshot::selected_rows` checks retained source positions without sampling again.
`selected_link_pairs` derives permissible row references from immutable graph rows and example ownership without reading a random stream.
Sampled-context rebinding verifies the retained raw values and relationship membership, while its context identity binds the exact ordered sampled links.

## Source tests and execution status

The new identity, range, native profile and conversion-order tests were written before their consumers.
The snapshot replay and link-alignment tests were written before the corresponding replay methods.
These are source tests and have not been executed.
Only formatter WRITE on the explicitly owned handwritten files is permitted before two independent whole-code approvals.
No candidate import, test, lint, type check, build, generator, native probe or live call has run for this checkpoint.

## Exact composition dependencies and remaining implementation

The handwritten unlinked DTO now carries a completed-task identity, its actual prediction string encoding, a complete-reference identity and a fresh OS entropy fact at the label stage.
The graph's continuing neighbor checkpoint is not a source of fresh per-call entropy.
No shared operation, decoder, state version, generated schema, contract lock or routing file changes here.

The Payload entry, validation and ordered accumulation of earlier encoded parts, whole-request notices and final request assembly are still required.
The existing encoder returns only wire bytes, so the complete logical NativePayload attachment and RequestState coherence still require the same logical Value to be exposed before encoding.
Reparsing model JSON would lose the distinction between adjacent surrogate points and a supplementary scalar and is not a valid implementation of this requirement.

The written native word consumer targets the single UnicodeText propagation through the existing `SampledColumn::Words` and wire `Value::Text` consumers.
Its `normalize_native` import requires linking and exposing the preserved native engine kernel.
Those APIs and the contract-native failure type are not available on this old base, and the modules cannot be enabled as they stand.
The data-only RelationalOptions definition still requires the request-owned re-export described in the preserved facade composition delta.
Native string arrays with different physical encodings also need the common logical key/task-row bridge before their actual concatenation.

The label attachment currently reuses the existing scalar label-row representation.
Typed list labels, retained lag columns, ranking/forecast decomposition, complete HostRun journey twins and the native retained-owner facade remain required before the complete Text package can be claimed.
The final frame native error guard must preserve guarded versus unguarded warning contexts, actual native ancestry and the existing graph-drop summary.
The existing unfinished-family guards and refusal census must be reconciled with actual admission at composition.

Rebase, serialized shared ownership, canonical generation, two independent exact-source approvals and candidate checks follow the actual approved preceding landing.
Exceptional native payload memory and performance measurements remain Phase 4 completion requirements.
