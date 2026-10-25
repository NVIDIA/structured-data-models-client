# T4.4 first numeric frame consumer

**Proposed boundary, awaiting coordinator review.**
Base `6f29cd29e970e5c57c55339e14445044abf48d66`; defaults and contract 31 remain unchanged.

**Journey.**
Admitted numeric/Boolean frames and scalar numeric keys: `Schema.from_data → PREDICT users.age FOR EACH users.user_id → materialize → inspect → scripted predict → merge`.
Frozen reference `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1` completed this journey, including two call envelopes merged in input order.
Static target `COUNT`/`SUM` retain their missing-time-range refusals; windowed aggregation follows temporal admission.

**API and ownership.**
Keep `ProviderChoice::Frames`; dispatch `Materialize` directly to the core frame consumer, without `DataLayer`, a second provider or semantic callback.
Propose operation-specific `MaterializeFrames { tables: Vec<FrameInput> }`, reusing ordered `FrameInput { source, rows }` bindings and existing Arrow admission, not describe's Table/Schema policies.
At the reference's first graph-build position, detach current values; constructor snapshots are too early.
The core owns sanitization, row maps, CSC adjacency, selection and labels.
Extend the sampler-state channel with a versioned opaque `FrameCheckpoint`: detached Arrow buffers, frozen graph facts and MT19937 words/position.
Python retains it on this query's existing relational materializer and rebinds buffers with fresh attachment ids on every run, without interpreting semantics.
Pure derived adjacency may be rebuilt from that immutable snapshot; native frames are not reread.
No persistent attachment ids or process-global query state exist.
A fresh query takes a fresh snapshot and neighbor seed 123.

**Lifetime and failure.**
Column replacement after first materialization leaves the cached query unchanged; a fresh query reads the replacement.
Publish the snapshot after graph construction and preserve consumed state on both done and error.
Unknown-entity failure consumes no neighbor draw; the 15-table cap fails after sampling and leaves state advanced.
A seed resets before each subgraph/batch; `None` continues across calls and batches.
Release run-owned views on every exit; preserve only returned checkpoint owners under existing close acknowledgments.

**Reuse and ordering.**
Reuse Arrow take/filter/grouping, labels/keys, payload/results and RS-1; preserve SQL traversal and provider fallback.
Frame policies belong in `sdm-engine`; orchestration belongs in `sdm-core`.
Extract the existing RS-2 MT primitive once, retaining SQL coefficient vectors; review shared ownership before implementation.
RS-3 keeps modulo-biased Floyd draws only for over-cap static hops; RS-4 keeps legacy permutation order and approved OS entropy for `None`.
Preserve native adjacency order: this cell uses NumPy 2.5.3 integer argsort; optional Torch uses its own argsort and stable lexsort chain.
Recorded cases prove neither every NumPy dispatch nor Torch equivalence; those obligations remain before widening the consumer or switching defaults.

**Review gate and first tests.**
Tests first: scalar inspection/prediction, multi-call merge, stream continuation/reset, failure before draws, snapshot reuse and aggregation refusals, with an old-engine tripwire.
Frozen observations also reach the compiled sampler and public failure after sampling.
Review checkpoint shape/MT ownership before production edits; add HostRun twins for rebinding, failed state, versions and buffer release with the consumer.

## Consumed interface amendment

The coordinator accepted the first-consumer direction with these concrete conditions on 2026-10-25.
Implementation advances the working contract through its real generator; contract 31 describes the baseline only.
`MaterializeFrames` remains an operation-specific bytes manifest of Arrow bindings, selected by the existing Frames identity.
The existing `SamplerState` gains an optional versioned `FrameCheckpoint` naming a bytes manifest, whose typed members name metadata bytes and separately owned Arrow tables.
Metadata contains frozen logical graph facts, a source/snapshot fingerprint and MT words/position; it contains no Arrow buffers, attachment ids used as identity, or CSC graph dump.
Each call rebinds checkpoint tables and metadata under fresh ids, while the snapshot identity is computed from logical schema/source and detached row values.
The core publishes the checkpoint immediately after graph construction and updates it as draws occur, before any later error; an unknown-key failure therefore retains the built snapshot without advancing its neighbor RNG.
Done, error and cancellation preserve only explicitly transferred checkpoint owners; run teardown and existing close acknowledgments release all other admitted and exported owners.
Python holds opaque checkpoint values on the existing query materializer and never rereads mutable caller frames after that graph was built.
HostRun twins will exercise rebinding with different ids, unknown state versions, pre/post-draw failures, cancellation and lifetime release, alongside existing Java decoding compatibility.
The shared MT primitive moves once into `sdm-engine::random`; the existing SQLite coefficient adapter and RS-2 vectors continue to use that implementation.
Temporal constructor files remain owned by their author; this consumer uses separate materialize orchestration and does not rewrite describe.
