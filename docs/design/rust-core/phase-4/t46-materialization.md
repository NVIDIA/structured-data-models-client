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
