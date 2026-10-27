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

Working contract 34 adds one closed `PrepareRequest` family with explicit arguments, headers, schema, model, outputs, capabilities, task and capacity stages.
Each returned state contains useful typed facts and an explicit version, without native handles, callbacks or a workflow interpreter.
A constructor descriptor remains model neutral and is distinct from a per-call model-bound value.
`Predict.direct` consumes a fresh typed call input and fresh detached Arrow halves, rather than reusing a gathered payload or rerunning constructor guards.
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
Calls reread the public state, including after mutation or an earlier failure.
Capacity is checked before the empty-scoring shortcut.
The transport and presentation functions are shared with existing core prediction.

Tests were authored and pushed before production implementation.
Public tests compare constructor inspection, exact scripted bodies, results, warnings, native negative precedence, repeated mutations, precision, supplied schema flags and wire widths.
HostRun tests cover the same preparation/prediction journey, explicit versions/phases and header/data coherence.
A binding ownership test mutates and releases producer arrays after Run start but before the first poll.
Frozen reference probes used source `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1`, a readonly runtime, blocked sockets, disabled bytecode and numerical threads set to one.
Canonical schema generation is authoring, not validation credit.
Candidate execution requires two independent reviews of the exact final source; this document makes no passing-check claim.
