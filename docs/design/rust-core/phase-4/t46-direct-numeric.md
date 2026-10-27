# T4.6 direct numeric preparation and prediction

This bounded development route implements `Request(context, predict, ...)`, public inspection and one KumoTabular prediction over NumPy or pandas masked numeric and Boolean columns.
Both constructor paths and `client.predict/tabular/frames` must be selected explicitly through `SDM_CORE_ROUTES`.
Default routes are unchanged.
PQL preparation, sampling, index sorting, sessions, custom native protocols, nested/object/Decimal/temporal storage and successful unencodable column labels remain unfinished in this route.

Python retains the original DataFrame, schema, Task and mapping objects exposed by the public API.
It observes native argument forms, Index lookup ordinals and duplicate ordinals, performs primitive Arrow conversion, copies mappings at the existing native boundary, sends bytes and presents diagnostics/results.
These observations carry no validation decision.
Errors that need native label or task presentation obtain it only after the core selects the corresponding sentence.
Rust owns constructor guards, inferred or supplied schema interpretation, model policy, dtype widening, wire precision, response attribution and empty results.
Unknown task is refused before header lookup or Arrow conversion.
Native lookup faults remain facts until the core reaches the corresponding source guard.
Related-table arguments carry only ordered native frame forms; their unused native names stay local until the core selects a frame-name or whole-related diagnostic.
Native name sorting and display occur at that selected presentation boundary, preserving earlier frame errors, unencodable diagnostic names and mixed-key comparison failures without adding successful related-table support.

Working contract 34 adds one closed `PrepareRequest` family with explicit arguments, headers, schema, model, outputs, capabilities, task and capacity stages.
Each returned state contains useful typed facts and an explicit version, without native handles, callbacks or a workflow interpreter.
The constructor result is model neutral and is used to build the public objects, without retaining a second schema snapshot on Request.
It is distinct from a per-call model-bound value.
Accepted capability facts remain in the bound call, including the independently consumed capability kind, anchor, relationship truth and table count.
A fixed Overlap stage verifies the pre-warning task, actual row count and schema shape against that bound call, with header/Arrow coherence and the shared model policy.
It computes one-shot overlap advice before the adapter captures its frame/schema references.
An Envelope stage then captures those current references, checks schema consistency, resolves wire columns and delivers text-as-categorical advice before row/inference encoding.
The returned versioned TabularEnvelope contains the actual core-resolved columns, wire widths and target, with no native handle.
The primary key is a later observation from the same captured native table, after text advice; an in-place edit changes that call without replacing its earlier column plan.
A fixed EncodeBody preparation consumes that plan and the explicitly later observations over those same captured native frames.
It returns an independently useful complete tabular envelope, with optional metadata and inference absent, plus bound-call, schema, table-shape and owned-payload facts.
Only after both row encoders succeed does Python draw the native request identifier and reach the inference truthiness/read/conversion.
Native decimal-integer display failures therefore follow earlier row failures and retain their original exception.
Final `Predict.direct` verifies that the actual encoded payload agrees with its typed schema, shapes, kind, model and outputs, then adds metadata/inference in Rust and sends the same encoded rows.
No row re-encoding, earlier-policy replay, partial JSON fragment, native callback or opaque resume state is introduced.
Integer-cell coherence roundtrips through the same shared integer codec that encodes the rows, accepting its canonical lossless int64 text and rejecting alternate or incompatible spellings.
In-place warning-side edits are visible, replacement Request frame/schema references are not recaptured after text advice, and the later task kind/inference reads remain fresh.
The frozen source permits context growth and task replacement during text advice, so earlier model policy is not rerun against those later effects.
Empty scoring retains its separate input with no later context or inference observations.
The existing HostRun admission and detachment path owns those buffers.
Header ordinals, positional Arrow names, physical types, field counts and row counts must agree before value semantics.

Direct and gathered prediction use the same model `Spec` table and extracted model, output, capability, task and capacity functions.
The direct body reuses numeric JSON primitives and the existing answer decoder/result tables.
A direct-only held-schema read retains the wire dtype through the existing `Dtype` enum, because schema edits otherwise group numeric widths into language kinds.
The existing `HeldColumn` gains native `untyped` and `stype_declared` facts, defaulting to false for older internal inputs.
An untyped placeholder yields to scoring storage, while an explicitly assigned meaning survives widening.
No second held-schema representation or dtype policy table is introduced.

Flat finite primitive inference values travel in an input-sized attachment.
Integer digits remain exact until Rust applies the final float64 JSON precision rule, after row encoding.
The body retains supplied row order, large integer text, valid masked NaNs/nulls, declared dtype widening, current task/schema/options and one-shot leakage advice.
Boolean participates at the bottom of the native numeric widening order, preserving categorical numeric values across Boolean, signed/unsigned integer and floating halves.
Calls reread the public state, including after mutation or an earlier failure.
Capacity is checked before the empty-scoring shortcut.
Native floating storage states whether missingness comes from NaN or a validity mask; Rust decides the conditional nullable-integer cast failure from those facts and values.
An error-only presenter obtains the selected pandas cast exception wording at the actual error boundary, without host classification.
The transport and presentation functions are shared with existing core prediction.
Direct response attribution recognizes the closed primitive list/key forms emitted by numeric/Boolean/string widening, before restoring caller keys.
Direct and gathered answers share one layout implementation, with gathered admission unchanged.
Numeric class restoration uses the existing wire integer and floating text grammar, including supported Unicode decimal digits and single underscores, while preserving malformed-value pass-through and probability-key collision protection.

Tests were authored and pushed before production implementation.
Public tests compare constructor inspection, exact scripted bodies, results, warnings, native negative precedence, repeated mutations, precision, supplied schema flags and wire widths.
HostRun tests cover the same preparation/prediction journey, explicit versions/phases and header/data coherence.
A binding ownership test mutates and releases producer arrays after Run start but before the first poll.
Frozen reference probes used source `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1`, a readonly runtime, blocked sockets, disabled bytecode and numerical threads set to one.
Canonical schema generation is authoring, not validation credit.
The first two source reviews requested changes, and their candidate/evidence remain preserved.
Supplemental frozen public cases establish cast storage distinctions, exact ID refusals, warning/error order and the two different warning-side reference boundaries.
Candidate execution requires two renewed independent reviews of the exact final source; this document makes no passing-check claim.

The overlap stage widens only declared key columns, keeping their original scoring ordinals and native storage facts.
A nullable cast failure in an unrelated feature therefore follows overlap advice instead of suppressing it.
Encoded numeric key components use the existing canonical decimal identity for comparison, including signed zero, while encoded text keeps its spelling.
The body consumes each complete column across all Arrow batches before the next column, retaining native competing-error order before scoring and inference.
Shared answer coercion and table layout select attribution explicitly: gathered envelopes use entity-reference columns, and direct envelopes start with response IDs as written before applying their declared primary key.
Ordinary feature names cannot activate gathered attribution.

A fixed decoded-response boundary owns the encoded request payload, normalized model response and typed decoded row buffers.
The shared model wire decoder runs once before attribution, while shared declared-target coercion remains after every response row is attributed and completeness is checked.
The actual server response identifier and the request payload's metadata identifier remain distinct facts; the server is not required to echo the request identifier.
Rust derives expected keys, raw-ID membership and prior response-row occurrences once, and observes their bounds, duplicate and completeness failures at the source guard.
An explicit versioned scope and source ordinals bind compact row values and native observations to the decoded buffers.
Final layout independently derives the rows again from those shared buffers and verifies every attributed row and observation before returning the result.

A direct response ID is read as string JSON first, including the default JSON reader's nonfinite numbers, under the reference's 1024-character limit.
This inner reader retains JSON root/member kinds, exact integer digits and Unicode code points, and does not inherit the outer model-byte decoder's encoding, depth or finite-number restrictions.
A JSON success, including a non-list value, never invokes native literal conversion.
Only a reached `NativeLiteralText` is passed to Python's actual `ast.literal_eval`, exactly once under active warning filters and handlers.
Python reports literal-built builtin values or native exception ancestry, without deciding singleton selection, whether a conversion failure is caught, ID membership or attribution.
Integer facts use hexadecimal digits, floats use bits and Unicode text uses code points, so ignored values acquire no eager decimal display or UTF-8 refusal.
Rust selects the singleton member before the bridge calls native `str` on that original local member.
Native objects and exceptions remain local to the bridge; no native handle or callback crosses the contract.
A selected uncaught failure rethrows the original native exception.
The core then chooses raw-ID fallback, key comparison and restoration before another row can reach native conversion.
This naturally preserves syntax warnings, promotion to a caught SyntaxError and caller warning-handler effects without implementing a Python compiler or a second warning policy.

Consecutive ordinary JSON rows use a fused pure core path.
After a native conversion boundary, fixed-size row slices bound repeated serialization, with no whole-response replay or growing seen-set copy per row.
Owned shared buffers and compact observations retain expected O(N) work and memory across the fixed data pipeline.
The same builtin/failure facts are useful inputs to a pure Rust HostRun host.
No language step, warning acknowledgment protocol, generic workflow interpreter, literal evaluator in Rust, Unicode-name data dependency or new workspace crate is added.

Direct model-contract errors retain the generated decoder's selected item ordinal and reached row-index diagnostic.
The same shared decoder keeps gathered width admission unchanged, while its explicit direct mode preserves exact canonical row-integer digits through bounds checking before result-table narrowing.
Finite whole floats use standard native-compatible conversion, including negative-zero normalization; arbitrary integer tokens acquire no fixed i64 or i128 bound before direct attribution.
Global decoding still precedes all native row-literal effects, and later bounds remain deferred until their own row.
An error-only native presenter renders the core-selected bounds integer using maintained exact conversion at that reached presentation boundary, preserving an active decimal digit-limit failure after earlier warning effects.
The existing common value formatter supplies quoted values for selected generated-contract diagnostics; it does not validate or reinterpret model rows.
