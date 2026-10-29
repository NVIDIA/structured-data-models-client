# T4.6 materialization implementation

The next source vertical connects single-table PQL materialization to core-owned context selection and the existing model-neutral Request constructor.
It preserves the full T4.6 scope in [the phase design](design.md), including supplied scoring rows, indices, task overrides, context sampling and native index order.
Direct numeric prediction is an existing foundation, not completion of this package.

## Development and acceptance run concurrently

The owner directed that long-running checks continue in the background and do not block independent source implementation.
The frame/link candidate remains immutable while its existing Python 3.10, remote CI and provider acceptance finish.
This tabular source work uses a separate worktree based on that candidate, with no edits to its source, runtime, build outputs or evidence.
Landing still requires the applicable checks and acceptance results.
There is one background frame/link acceptance stream and one active tabular source package, with at most one Phase 4 local heavy job.
Reviews of the complete new source precede its compiler, test, lint and package execution.
An already passing check is repeated only if the new source changes its applicability or a concrete failure requires it.

## Source ownership

Python supplies the original native frame's consumed value facts and retains the original objects for public Request construction.
Core chooses the scored and labelled candidate rows, context limit, fixed seed, selected-subset ordering, task kind and notices.
Python applies returned row ordinals without selecting a second context or evaluating a second task policy.
The shared engine owns row selection and the existing random and sorting kernels.
Core composes those operations and the existing request preparation; the wire and response paths remain shared with direct prediction.
No Python provider for sampling, new crate, SQL engine, language callback or generic preparation interpreter is introduced.

## Reference boundaries

The indices route consumes the supplied indices once and preserves their order and repetitions when gathering every matching entity row.
It refuses missing entities before observing the target's context mask.
Scoring rows lose the target column, and all their original row positions are excluded from labelled context before any context sample.
The supplied-predict route retains the supplied scoring frame and removes matching entity values from context before filtering unknown targets.
Missing entity columns retain the existing no-exclusion behavior at this boundary.
The complete labelled candidates retain their original order when their count is within the requested limit.
Only a strict reduction samples with the legacy seed-42 RandomState and sorts the sampled subset by its native index semantics.
A full-frame sorted permutation cannot replace that subset operation.
The automatic-cap notice is emitted before sampling and index sorting, retaining caller effects and warning-to-error behavior.
Task derivation observes the resulting context after that notice and selection, before the schema snapshot and Request constructor.
Native custom sorting overrides retain only the owner's approved narrow Python-route exception.
Ordinary built-in index families remain migration obligations rather than inheriting that exception.

## Source and validation sequence

1. Write public old/core materialization controls and a Rust consumer over the same input facts before implementing the connected selection path.
2. Reuse the existing frame-value, key, random and index-kernel vocabulary where its native equality and order semantics actually agree with the tabular source.
3. Preserve stage-local reads and native failures; do not eagerly encode ignored features or silently reuse a stale native snapshot across advice.
4. Connect the selected rows to the existing Request constructor and direct prediction path, with explicit routing until the remaining T4.6 families are complete.
5. Obtain two independent whole-source reviews, then run uncovered checks using the existing warm target and heavy guard.

Expected controls cover repeated and missing entities, unknown targets, leakage, empty scoring, supplied scoring rows absent from context, uncapped order, explicit and automatic caps, exact seed-42 rows, duplicate index ties and selected-subset comparison failures.
They compare public Request frames, task, query, schema and encoded calls with the frozen Python reference.
Rust-host controls must consume the input facts and execute the actual core selection and ordering; a precomputed final native permutation is not proof of this boundary.

T4.7 sessions and T4.8 activation remain follow-ups to this shared preparation foundation.

## First connected source slice

The first implementation admits ordinary frames with primitive int64 entity storage and an initially strictly increasing RangeIndex.
Native getters can change the actual candidate index after that initial admission.
For reduced contexts only, Python captures the complete actual built-in int64 index labels after the native reads, candidate take and cap advice, before Rust draws any sample.
Rust validates the full label extent and uniqueness, draws the seed-42 sample prefix, and sorts only that subset by those labels.
This unique integer ordering needs no unstable sorter profile, duplicate-label tie rule or precomputed native permutation.
The shared engine returns missing requested ordinals before expanding scored rows, and preserves every duplicate source occurrence and requested repetition for this primitive key family.
Core binds the context selector to the existing model limit; Python retains only native conversion, original-value error presentation, frame takes and the existing cap notice presentation.
An already captured candidate frame survives cap advice, and the selector supplies positions within that frame rather than rereading the caller's potentially edited source afterward.
An uncapped population returns no replacement take, preserving the existing candidate frame and avoiding another copy.
It consumes no native index labels and runs no sampling or sorting.
Missing required label facts and inconsistent extents retain the invalid-message protocol refusal rather than falling back to source-row ordering.
Complete duplicate labels describe an unfinished native sorting family, so Core selects the existing H17 materialization refusal and Python presents that selected development boundary through RouteError.
The Rust host exercises the same public typed selection product exported through the composition root and host layer.
No new serialized message or opaque continuation is introduced by these owned primitive values.

The next source increment extracts the original task boundary into shared task-meaning and lazy numeric-prefix products.
Its handwritten source is complete but remains unexecuted, with contract36/catalog schema authoring pending before validation.
Broader entity equality, duplicate or mixed index labels, other index families, custom native protocols, counts outside native row widths, final default activation and complete T4.6 acceptance remain open.
The current source and controls have not been executed; the frozen frame/link background run provides no validation credit for this new source.

The first source reviews found that supplied-row entity reads and target missingness must retain the reference's intermediate native populations, even when final row values agree.
The successor observes the supplied entity Series first and completes a useful owned scoring-key set in Rust.
An empty set selects the original source owner without reading source entity values.
A nonempty set returns a population of retained original source positions, including a full take when no entity matches.
Python performs that native holdout take before observing target missingness.
The retained population checks the mask extent, returns local labelled positions, and maps those positions back to original source ordinals for the existing context selector.
Candidate values are captured before context-cap validation and advice, preserving that separate native boundary.
Public source-read controls cover actual target operand identity, present and no-match holdouts, empty scoring and scoring-accessor edits to source keys.
The next controls also retain full old/core contexts, request bodies and results when an empty-scoring getter changes the source index or a target getter changes the held-out index before sampling.

## Shared task meaning

The opt-in materialization path captures the actual final context target Series before target-stype or stated-task observations.
Rust owns the two-task vocabulary, supported target meanings, stated membership interpretation, two-whole inference, selected refusals and classification notice.
Python retains the original stated object and returns it unchanged on acceptance, including str subclasses.
It performs actual native stype membership, stated isinstance/hash/equality, and selected diagnostic repr only at their reached boundaries.
The stype refusal precedes stated-task operations, and native display faults are presented outside an internal CoreError handler.
ModelLacksTask is reused with exactly the old task/supported public details and no extra model detail.
A finite task presentation preserves the original ordinary raise, including its suppression flag and active caller context, for only these selected task refusals.
Categorical and explicit shortcuts read no target distinct values.
Only absent numerical tasks reach the captured target's actual dropna, unique, len and lazy iterator.
The prefix product consumes reached bool/Integral, Decimal finite/round/equality/truth or completed Real float/failure-ancestry facts and stops after the first nonwhole result.
An actual reported length of two does not imply two iterator members; native exhaustion is observed rather than fabricated, including zero, one or more than two yielded members.
The owned prefix retains its consumed extent and complete conjunction meaning in constant space, with malformed phase/order/extent/completion facts refused.
No integer width limit or Decimal digit encoding is introduced into these native observations.
Native failures outside OverflowError, TypeError and ValueError on Real conversion propagate as their original owners.
The generator-expression versus returned-truth StopIteration boundaries remain distinct.
Two-whole classification emits the finite shared first-party notice before schema snapshot, Request construction and query recording.
Public original/Core materialize-to-predict controls retain full contexts, tasks, schema, encoded bodies, results and native error graphs, with actual native protocol/effect order controls.
Typed Rust Host consumers execute the same task products and pin vocabulary/refusal metadata, cardinality shortcuts, exact exhaustion, short circuit and malformed facts.
No new crate, native object registry, callback RunStep, generic interpreter or second task-policy table is introduced.
The bounded contract36 preparation additions describe completed domain data rather than a serialized native workflow.
The contract36 source change and finite catalog additions require genuine canonical generation under the normal source-authoring controls; no old schema is claimed to describe that new source.
The preexisting broader entity/index families, custom sorter handling, default activation, sessions and complete T4.6 acceptance remain open.


## Task objects through construction, calls and responses

The seven-task model-neutral constructor vocabulary is separate from the two-task PQL materialization rule.
For non-literal native task objects, the constructor observes the actual ordered tuple membership after the frame-form guards and retains the accepted original object in Request.task.kind.
A constant target reaches the original first classification membership, lazy regression inequality and second classification membership before the selected notice and later Task metadata reads.
Per-call default-output lookup, capability equality/truth observations and fresh model membership use shared Core vocabulary with the original native objects.
Runtime task admission detaches string storage before native membership and releases that operand as the membership expression completes.
If that membership replaced the request task, the replaced owner's finalizer therefore runs before later frame reads, and refusal presentation reads the current task afterward.
Equality truth owners remain with the native Capabilities value until their original ordered shape consumption and preflight lifetime boundary.
A reached callback can replace the task with non-string data; absence of string storage is represented separately from its actual comparison facts so the normal model/task refusal still runs first.
If such non-string data is nevertheless admitted and later needs unsupported wire storage, the existing selected unfinished direct boundary remains explicit rather than inventing a spelling.
Empty scoring has no wire-storage requirement and constructs a real empty result without a task payload, display or model call.

The adapter completes wire widths before native per-column task meanings.
Each ordered column declaration is bound to one retained width-plan owner and exact ordinal.
A Text target does not read task classification, while a non-Text target observes the actual Task.is_classification property at the original position.
Each non-Text column first observes its fresh native inequality against the current request task target.
That reached fact is distinct from the earlier target retained for the wire task, so replacement of the task can select a later column without rewriting that captured target.
The second native column lookup and any reached declaration flag are observed afterward, so a task callback can change a later column's metadata before that later read.
Only the complete ordered column product emits text advice and supplies later body encoding.
Missing, foreign, repeated, out-of-order and unsolicited native facts retain strict protocol refusals.

The response retains the original task owner captured for the payload instead of rereading Request after transport.
The shared row-attribution validator first returns the complete actual attributed rows, including the existing missing-row and native-ID refusals and rejection of duplicate scoring indices hidden behind distinct response ordinals.
Only then does Python observe str on that captured owner once, including on regression.
For an exact builtin string, the shared restoration policy reads lossless Unicode storage.
For a returned string subclass, Python performs the reached containment and truth operation using the existing Core-owned classification vocabulary, and supplies a closed native meaning fact alongside that storage.
The temporary display owner is released after containment, before later restoration work.
Invalid responses and the empty-scoring shortcut do not observe task display.
Public original/Core controls retain full requests, encoded calls, results and native effects, including noncanonical accepted task spelling, later-column mutation, non-string task mutation, comparison truth failure, changed response display and failure on the second actual response conversion.
Rust Host controls consume the same typed column, constant-task, completed-row and response-text products.

All of this source remains unexecuted pending the exact source-authoring review pair, genuine canonical contract36/schema and bridge generation, fresh completed-source reviews and required validation.
No generated schema or prior runtime is claimed to represent these new inputs.


### Native preflight lifetime and late direct restoration

The native required-capabilities value retains actual task-comparison truth owners through the context row read and shared capacity admission.
A capacity refusal retains that owner in its native traceback; successful preflight releases it before the later live adapter reads.
The bound count remains an earlier capacity observation, while the overlap product checks its later headers against actual attachments independently.
An in-place frame edit at successful preflight release therefore remains visible without being mistaken for a forged earlier count.

A direct call retains its genuine native payload task container and captured kind owner independently of immutable Core bytes.
The optional private payload argument follows the existing shared transport and adapter-error boundary.
Byte-capable transports receive the exact original bytes without an added decode of the final inference body.
Decoded transports receive the actual retained dictionary, refreshed from those final bytes before transport while preserving the original native kind owner.
Transport replacements and edits to that dictionary remain visible at normalization.
No current Request task is substituted for that normalization owner.

Complete row attribution and its population checks still precede task display and native containment.
The temporary display owner is released before the lazy actual target dtype read.
The closed response-task meaning returns Core-owned classification meaning and numeric-prefix vocabulary.
A separate closed target declaration records missing/builtin text or actually reached native false-truth/false-prefix shortcuts.
Rust validates required versus unsolicited facts and owns builtin dtype restoration, prediction conversion and probability key policy.
Sent bytes and completed row facts are never replaced by this later declaration.

Native dtype truth and startswith failures remain native, and false native shortcuts leave answers unchanged.
Positive custom dtype overrides consume a complete, scope-bound native target-row product after complete attribution and the reached truth/prefix observations.
The native host applies the existing as_declared protocol per prediction and then per probability key, retaining actual key-display and exception ordering with the same held dtype owner.
Rust validates complete response scope, row and key ordinals, unchanged probability values, native storage and collision fallback, then lays out those completed builtin values without recasting them.
Ordinary builtin target dtypes retain their Rust kernels.
Representation-only native value codecs are shared by direct and relational responses, with the existing relational entry points preserved.
The original task local is released when the genuine native payload takes ownership, and temporary dtype/task-container locals are released at native normalization completion before final layout.
This does not authorize a blanket conversion-timing exception or complete the broader entity, index, sorter, default-route, session or full Phase4 obligations.
The public old/Core controls and Host malformed/ordered controls are source-first controls until the canonical generation, whole-source review and execution gates complete.


### R74 official connected response-owner plan

R73 remains frozen at 6239e2e80331cbb96c2b883972ad964ecc48a592.
This successor closes the actual decoded-response owner boundary, rather than treating an equal serialized value as an ownership substitute.
The plan extends existing direct response scope, native target-row, native output and presentation-lineage products.
It introduces no process-global registry, serialized Python handle, reusable native callback dispatcher, generic effect interpreter or native emulator.

#### Actual generated ownership and decode boundary

One call-local native owner retains the actual decoded fallback response and constructs its actual generated PredictionResponse exactly once after transport succeeds and before any native ID attribution effect.
The byte-capable transport continues receiving original request bytes, and immutable response bytes remain independent admission and attribution authority.
The parsed owner is never serialized into a replacement response.
PredictionItem.prediction retains its actual source object; probabilities and quantiles are new generated normalized maps, scores and embeddings are generated numeric tuples, and metadata/explanation/ranking entries are generated shallow copies whose nested owners stay shared.
Response metadata is a generated shallow copy and the public PredictionResult constructor makes its existing additional shallow copy.
Native construction reuses that existing generated codec instead of copying its field-conversion, copy, key-string and finite-number rules into another host table.

Generated construction belongs to normalization, outside the shared transport exception mapping.
A successful decode is never replayed during restoration or error presentation.
A Core-selected malformed response associates the once-only actual generated exception and native context with the existing response-contract error boundary, rather than constructing a replacement cause or reporting a transport failure.
If actual native generated conversion fails after an immutable-byte admission, a closed response-failure observation selects the same Core response-contract wording while retaining the actual cause; uncaught native exceptions preserve their original propagation.
No row ID effect, target display or target conversion runs before this decode boundary completes.

#### Scope, per-item facts and Core policy

All native sources and results bind to the existing complete ResponseScope and consecutive response ordinal, with field roles from the existing native output vocabulary.
These ordinals describe one already admitted response population, not reusable object identities or object handles.
The immutable response buffers, complete attribution rows and Core entity/row layout are never edited to represent a later mutation.
Missing, duplicate, reordered, wrong-scope and unsolicited source/restoration/field facts remain protocol refusals.
Legitimate later mutations are described by later facts and are not rejected by comparing mutable values to an earlier byte snapshot.

The native loop reads each actual generated item's prediction immediately before its reached conversion, then its probabilities at the original later point.
An earlier item keeps its selected actual source or conversion-result owner, allowing a later row or probability callback to mutate that same owner.
A closed prediction-source product distinguishes lossless builtin storage from a retained native owner whose storage is absent; it never invents a JSON value or serializes an opaque owner.
Core selects passthrough, its existing builtin dtype kernel or a specifically reached native conversion, and returns closed source-versus-converted field selection facts over the same ordinal.
Ordinary builtin dtype policy and conversion stay Rust-owned; custom dtype/value protocols use the existing actual native as_declared operations at the reached point.
Native completed results remain actual retained owners even when they have no builtin storage, with Core-selected field association carrying no Python object across the contract.

The existing probability rebuild, collision fallback, keyed != given and dataclasses.replace sequence remains in its original native order.
The native host observes actual key conversion/display/hash/equality effects and retains the original generated probability map whenever the original rule leaves it unchanged or collision fallback selects it.
Core owns probability conservation, collision/admission policy and field selection over closed observations; the host retains and presents the selected owners.
Sharing the existing loop must preserve its task/dtype/per-item local lifetime as well as its operations, rather than releasing loop locals at a newly extracted helper return.
Final value facts are captured after the complete loop and its source-equivalent cleanup, while selected actual owners remain retained for presentation.

#### Presentation, cleanup and absence advice

Core first validates and lays out the normalized row/field plan with explicit ownership lineage, without issuing final absence advice from the earlier byte snapshot.
The host constructs the public result once using Core-selected row identities and field roles, retaining the actually selected prediction, probability, quantile, tuple and shallow-copy products.
Core-computed ordinary conversions use their completed builtin products; passthrough and native conversion selections reuse actual owners, preserving shared aliases and post-return mutation.
No legacy adapter normalization result is returned wholesale and no business admission or row-layout rule moves back into Python.

Raw transport response, parsed response and other normalization-only owners remain alive through the original normalization-result construction, then are explicitly released at that source-equivalent exit before numbering, merge and absence advice.
Task, dtype and per-item locals are released in original order, and the payload-plan owner ends at its corresponding native plan boundary.
Only selected public result owners survive those cleanup points.
This permits unused/detached owner finalizers to mutate selected aliases before absence is observed without retaining an extra owner until predict returns.

The host next observes actual normalized output presence using the existing Core-owned field vocabulary and native comparison order, bound to the same final response/row associations.
A closed final output-policy stage validates those observations and Core selects absence notices.
Advice handlers may mutate retained public owners again; the same already constructed result is returned afterward, preserving those changes and later external mutations.
The final stage does not serialize or reconstruct those owners after advice.

#### Required controls and acceptance

Source-first public old/Core controls cover mutable prediction identity, task/dtype mutation before conversion, later-row/key mutation of an earlier owner, and mutation after client.predict returns.
They cover generated probability/quantile copies, tuple conversions, response/item metadata and explanation/ranking shallow aliases, top-level replacement independence, absent/present changes and warning-handler mutation.
They also cover generated decode errors without duplicate construction or transport reclassification, and raw/parsed/task/dtype/per-item cleanup effects on selected values and advice.
Host controls reject malformed source/field/ordinal/scope/presence populations independently and prove actual valid late source changes remain admitted.
All controls keep the all-Core routes selected and require strict results, requests and error parity.
There is no custom conversion-timing exception or reduction of the original broader Phase4 obligations.
Root still owns two independent whole handwritten reviews, canonical generation, an actual generated commit, a fresh final whole pair and execution validation after the connected source boundary is complete.
