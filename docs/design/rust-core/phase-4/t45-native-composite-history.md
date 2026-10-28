# T4.5: native composite history identity

This is the bounded implementation plan for the recorded native composite-history failure after the StringDtype prerequisite source checkpoint.
The unchanged full public LINK receipt succeeds on the original client and fails in the candidate because a native frame source calls the SQL-only `Graph::new` constructor.
Keep that SQL constructor and its reserved-name behavior unchanged.
The target is an already admitted multi-column identity on nonentity history with outgoing scalar foreign keys.
Incoming references to a compound destination and compound entity roles retain their existing refusals and positions.

## Reference and fixed read schedule

The pinned original client first completes every table's declared native casts.
`Graph.from_data` then creates every physical local table, including its shallow native frame owner and fixed inferred column types, before assigning any caller key/time/stype metadata.
The materializer visits tables in schema order and reads one table's current primary key at its setter position.
That key definition is fixed before its native fold callbacks run.
The original setter checks component presence and current engine time conflicts, approximate component types and the derived name before rendering values.
For each reached component in declared order, `encode_frame` checks actual native missingness, then performs the native map, rendering/escaping and `astype(str)` operation.
Later components are not observed after an earlier native failure.
The native joined result is assigned to the local shallow frame with pandas' actual index alignment.
Only then does the caller read that same table's current time and per-column stype metadata and advance to the next table.
Relationships are read after the table loop, followed by the existing graph validation and all-table sanitation boundary.

Do not freeze `held_schema(schema)` before the new native fold observations.
Do not reread a table's primary key after its callback and replace the key that was actually assigned.
Keep already constructed physical table owners and types separate from later logical metadata edits.
Native callback changes to a later table, the current table's later metadata or the relationship list must remain observable at their original positions.
Retained graph reuse performs none of these source or metadata reads again.

## Sufficient useful products

Use the existing full ordered `SourceDescription.primary_key` vector as the authoritative row-identity declaration.
The topology used by scalar traversal is an operational projection, never evidence that a compound table is keyless.
Introduce an explicit native graph product containing that topology and its complete row-identity definitions.
Both raw frame graph preparation and complete native graph preparation construct that product through the same native policy; neither calls the SQL constructor.

At the native primary-key assignment boundary, a closed key-definition value binds table/source identity, ordered component names, fixed cast-source types and the derived name selected and validated in Rust.
Its type facts describe the actual constructed native columns, independently of caller-edited logical dtype names.
In particular, an original numerical source edited to logical string is still approximate, while a native categorical source's inferred string type is not inferred from its exported floating buffer alone.
Reuse existing inference and validation kernels where their actual observations are sufficient, adding a narrow neutral source fact only where that distinction cannot already be represented.
Do not add a library-origin tag, an accept/reject field or an opaque workflow token.

The converted row-identity product is actual typed string data bound to that key definition and complete original source-row order, using the existing explicit string encoding and Arrow ownership conventions.
It is independently useful key data, not a serialized instruction or suspended callback.
The native host must observe the real render/map/astype effects that a raw Arrow tuple cannot reproduce.
Rust retains key selection, type/name/collision/null policy and the identity consumer.
Keep the encoding algorithm shared where the operation is purely mathematical, but do not replace an actually reached native `replace`, cast, addition or alignment effect with a mathematical approximation.
The bounded old-only probe and read-only architecture advice resolve this precise codec boundary before production implementation.
No native callback is added to `RunStep`, and no generic interpreter, per-cell program, crate or provider is introduced.

Raw Rust frame callers provide the corresponding complete typed identity or construct the same standard identity through shared fixed encoders for their admitted native scalar domain.
They consume the same native graph and Snapshot rules as Python.
Every new field has explicit attachment discovery and typed rebinding; arbitrary JSON integer traversal is forbidden.
The current unpublished contract version remains coherent and published locks remain untouched until canonical authoring by the coordinator.

## Concrete representation and admission boundaries

`FrameTableColumnFacts` gains a closed neutral value representation beside the existing original physical declaration.
The alternatives are typed physical scalar storage, enumerated values, or actual observed object scalar content.
Object inference is observed at the constructed table boundary, using the original prefix and missingness schedule; the Rust reader maps those observations to its existing constructed dtype vocabulary.
`FrameTableMeaning` retains those fixed ordered constructed columns alongside its existing cast-time declarations.
The input carries the original reached native container-empty observation before any header or dtype work.
Do not add an early native row-count read; complete Arrow source and identity owners later establish cardinality directly.
It carries no future metadata position or callback.

`FrameKeyMeaning` binds one table's captured ordered primary key to that immutable constructed table.
Its optional `FrameCompositeDefinition` records the complete ordered constructed component types, selected derived name while the associated constructed-table meaning retains any pre-existing column under that name.
Rust selects it at the reached key assignment, validates component presence, approximate types and maximum derived-name length, and emits no fold for scalar or absent keys.
A derived-prefix collision is not an early refusal on the native route.
`FrameKeyNulls` validates a reached component's actual `isna().any()` observation without storing an execution position.
`FrameRowIdentity` binds that definition to one complete aligned typed string column with explicit encoding and original row order.
Its completion validates shape/null representation and the native primary-key setter's dtype support.
The setter assigns ID meaning to the derived column.
The later per-table metadata product then validates the current caller stype of any pre-existing derived column before advancing to the next table; overwriting its assigned ID with numerical or text meaning fails there.
The successful integer/text collision twins with explicit ID metadata prove this distinction.
The fixed initial stype is overwritten by later caller metadata, but its native inference observations remain observable and must still occur before the next physical column.

`MaterializeFrames` and detached `FrameCheckpointManifest` carry explicit row-identity bindings separately from ordinary source-table bindings.
`FrameCheckpointMetadata` retains the existing string-encoding facts needed to reconstruct the same snapshot.
The detached manifest binds the complete immutable definitions to their separately owned arrays, and the metadata source fingerprint binds both.
Attachment discovery, Python take/rebind, core checkpoint publication and plain HostRun rebinding enumerate these typed resources explicitly.
The source fingerprint includes actual identity values and their complete definitions and encodings, independently of call-local attachment ids.
Retained graph owners continue to retain the same Snapshot directly.

The engine's new native graph product contains the existing scalar traversal topology and full ordered source-key definitions.
Its native constructor does not call or weaken SQL `Graph::new`.
Snapshot consumes separately supplied identity columns for compound tables, while scalar keys keep their existing readers.
A raw frame host can produce the same useful identity through the shared fixed standard encoder for its admitted primitive domain; its strings are another producer's observation, not a verifier for native callback output.
Raw preparation and complete native preparation share this graph and sanitation consumer.

Per-table metadata is captured after that table's actual fold and before advancing to the next table.
Only relationships are serialized after the metadata loop; the existing relationship serializer can be reused without rereading earlier table declarations.
The captured original primary key replaces any callback-mutated key in that table's final graph declaration, matching the original setter's already selected key.
Native typed export follows these graph declarations and excludes a newly created internal derived column from feature projection.
A pre-existing source column under that name retains the original native admission and post-fold behavior instead of being mistaken for a newly added feature.

A callback can legally move the caller's primary key to another unique key while the already selected native compound assignment is in progress.
The successful original control keeps the captured native identity and releases the old component's caller key marking, so that component can be a categorical feature afterwards.
Native graph meaning/link checks therefore use the existing folded operational key projection, while full ordered source identities remain unchanged in the native graph, source fingerprint, Snapshot and checkpoint.
This projection never makes a table keyless and never replaces the authoritative SourceDescription used for sanitation.
Preserve current component meanings instead of calling the caller-schema key marker on the captured compound vector again.
SQL and the initial caller-schema checks keep their existing key marking.

## Resolved question lifetime through native metadata effects

The existing retained question already owns the validated tree, canonical query, filter, initial source description, declared types and query facts before native export.
Ordinary `Questions` must retain that same useful owner and explicitly rebind it into `Population`.
Each decomposition child supplies its own ordinary question owner.
Keep the exact raw query in that owner and require equality at consumption; equal query facts alone cannot identify it.
An absent owner keeps raw hosts' existing reached query resolution, while a present wrong-kind or wrong-query owner is a protocol failure with no reparse fallback.
This adds no stage, callback, cursor or second query representation.

Population consumes the retained validation for query meaning and final native graph metadata for actual time bounds and topology.
Do not reparse the query against metadata edited by the fold, and do not restore the old time column from its validation snapshot.
With an explicit anchor, the original reaches seed-generator construction and then its sample-target required time-column lookup before any entity or neighbor draw.
The missing final time metadata can then fail with native `KeyError(table)` after successful sanitation and graph notices.
The reached iteration and explicit anchor/context validations precede that lookup.
Reuse those existing guards before selecting the missing-time failure, and retain their native ValueError causes at this frame task boundary.
Preserve that selected native cause at the existing frame error presentation boundary instead of returning an unrelated PQL or internal-error diagnostic.
This closes the reached same-table time mutation in the recorded original journey.
At that same native task-classification boundary, a retained LIST_DISTINCT query whose final graph no longer has its unique registered target foreign key keeps the original unsupported-multilabel NotImplementedError presentation before label fetches or sampling.
Reuse the existing unambiguous target-role kernel; leave SQL and late payload admission unchanged.

## Snapshot and lifetime invariants

Snapshot sanitation consumes the complete identity, preserving the first original occurrence before missing-time removal.
A duplicate after an earlier missing-time row cannot replace that row.
Null components are refused at the earlier original encoding boundary; the lower typed reader still validates and safely handles malformed or missing identity data at its own boundary.
Keep disjoint null/duplicate/time drop counts, valid-row bounds, all-invalid keyed refusal, original source positions and stable adjacency order.
An encoded native identity may not be reconstructed by joining unescaped values, hashing to a scalar or assuming raw typed tuple equality matches native rendering.

History payloads keep their original component features and existing synthetic `__node_id` primary key.
An internal identity must not leak as a new payload column or erase its component columns.
Full metadata, identity data, string encodings and source rows participate in immutable source identity.
Raw checkpoints export and rebind all sufficient typed identity data, while retained checkpoints reuse the same actual owner and indexes.
Restore validates source/definition/value coherence and preserves the original neighbor checkpoint without another draw or source read.
Contradictory definition/source bindings, row cardinality, encoding, attachment types or checkpoint metadata are protocol failures rather than alternative cache identities.
The actual native converted strings are authoritative; the core never attempts to replay arbitrary native rendering and compare them with a second mathematical encoding.

## Readers that change together

| Producer or reader | Obligation |
| --- | --- |
| Native source preparation and table/key products | Preserve all-casts/physical-table/table-metadata order and actual component callbacks. |
| Contract products and attachment discovery | Bind complete definitions and real Arrow identity values, including raw checkpoint companions. |
| Native graph preparation and restore | Retain full descriptions and construct native topology without SQL reserved-name policy. |
| Snapshot sanitation, indexes and adjacency | Consume full identity with stable row masks, counts, bounds and source positions. |
| Retained graph and source fingerprint | Bind actual immutable identity data and preserve owner reuse under fresh ids. |
| Context publish/rebind, feature selection and payload | Preserve component columns, synthetic history IDs, sample order and unchanged entity-role limits. |
| Python and pure Rust hosts | Produce and consume the same useful values and malformed/lifetime boundaries. |

## Failing-first evidence and execution

The existing complete frozen composite-history public and plain HostRun LINK consumers are already committed and failing before this repair.
Add boundary consumers before production for arity/order, Boolean/integer/text/Unicode components, separators/backslashes, null-before-render, duplicate/time precedence, empty/all-invalid histories, exact approximate-type/name/collision behavior, incoming/entity limits, native error chains, native callback mutations and two-call owner/RNG continuity.
Preserve the frozen full payload/result/warning expectations.
The UInt64 HostRun test's independently discovered run-local owner collision is repaired only by a test-local explicit typed rebind, preserving the full pipeline and raw schema/query/index inputs.

The private bounded original-only producer checks frozen client/engine source and C++ identities, blocks network use and requires the ordinary public control to succeed before its remaining cases.
Its first two partial receipts are failed setup evidence, not composite compatibility evidence.
The corrected third source uses the explicit scalar relationships of the existing successful frozen fixture and a two-hop neighborhood so historical RHS entities can actually be reached.
Every executed probe source and receipt remains immutable.
No candidate SDK import, build, test, checker, generator or endpoint is executed by this source author.
The coordinator composes the complete source and canonical artifacts, obtains two fresh independent whole CODE approvals and then runs the required public, HostRun, checkpoint and quality consumers under the existing resource guard.

The narrow follow-up collision receipt contains seven successful original-route observations with frozen provenance.
Its ordinary, integer-with-ID and text-with-ID controls succeed through prediction, while ordinary integer/text/category/float metadata collisions fold before their primary-key stype refusal.
The earlier 34-case receipt and every executed producer remain immutable.

The narrow explicit-anchor precedence witness confirms that context-anchor entity mismatch and zero iterations each precede missing final time metadata, with their exact native ValueError causes.
The absent-anchor default-time lookup is a distinct earlier read; do not attribute the explicit-anchor witness to it.

## Native physical-constructor observations

The additional original-only constructor probe proves that physical table construction itself reaches native timestamp parsing before the first key fold.
A string subclass receives an actual `__str__` callback inside the native parser, even though its initial inferred semantic type is overwritten by later caller metadata.
The constructor-effects and constructor-stack receipts are immutable successful original-only observations; they are not candidate parity evidence.
Preserve native timestamp, multi-value and distinct observations in their original per-column short-circuit order, with Rust selecting the existing semantic predicates from sufficient neutral samples and counts.
Do not suppress those operations merely because the final caller stype is already known.
Native failures keep the original constructed-column RuntimeError wrapper and their real cause at this boundary.
The actual native source sample resets its index before each original inference read; this must not freeze later source or metadata observations ahead of callbacks.
A focused fixed constructor profile may own useful immutable dtype/name facts and native statistical observations; it cannot retain a callback, issue a generic instruction stream or change RunStep.
