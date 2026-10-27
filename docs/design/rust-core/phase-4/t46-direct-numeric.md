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
The constructor result is model neutral and is used to build the public objects, without retaining a second schema snapshot on Request.
It is distinct from a per-call model-bound value.
Accepted capability facts remain in the bound call, including the independently consumed capability kind, anchor, relationship truth and table count.
A fixed Overlap stage verifies the pre-warning task, actual row count and schema shape against that bound call, with header/Arrow coherence and the shared model policy.
It computes one-shot overlap advice before the adapter captures its frame/schema references.
An Envelope stage then captures those current references, checks schema consistency, resolves wire columns and delivers text-as-categorical advice before row/inference encoding.
The returned versioned TabularEnvelope contains the actual core-resolved columns, wire widths, primary key and target, with no native handle.
`Predict.direct` consumes that plan and explicitly later body observations over those same captured native frames.
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
Calls reread the public state, including after mutation or an earlier failure.
Capacity is checked before the empty-scoring shortcut.
Native floating storage states whether missingness comes from NaN or a validity mask; Rust decides the conditional nullable-integer cast failure from those facts and values.
An error-only presenter obtains the selected pandas cast exception wording at the actual error boundary, without host classification.
The transport and presentation functions are shared with existing core prediction.
Direct response attribution recognizes the closed primitive list/key forms emitted by numeric/Boolean/string widening, before restoring caller keys.
Direct and gathered answers share one layout implementation, with gathered admission unchanged.

Tests were authored and pushed before production implementation.
Public tests compare constructor inspection, exact scripted bodies, results, warnings, native negative precedence, repeated mutations, precision, supplied schema flags and wire widths.
HostRun tests cover the same preparation/prediction journey, explicit versions/phases and header/data coherence.
A binding ownership test mutates and releases producer arrays after Run start but before the first poll.
Frozen reference probes used source `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1`, a readonly runtime, blocked sockets, disabled bytecode and numerical threads set to one.
Canonical schema generation is authoring, not validation credit.
The first two source reviews requested changes, and their candidate/evidence remain preserved.
Supplemental frozen public cases establish cast storage distinctions, exact ID refusals, warning/error order and the two different warning-side reference boundaries.
Candidate execution requires two renewed independent reviews of the exact final source; this document makes no passing-check claim.
