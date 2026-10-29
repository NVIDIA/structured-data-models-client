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

The first implementation admits ordinary frames with primitive int64 entity storage and a strictly increasing RangeIndex.
Original source ordinals are order-isomorphic to those unique labels, so sorting only the selected ordinals preserves the reference sample-then-sort result without choosing an unproved SIMD profile or changing duplicate-label ties.
The shared engine returns missing requested ordinals before expanding scored rows, and preserves every duplicate source occurrence and requested repetition for this primitive key family.
Core binds the context selector to the existing model limit; Python retains only native conversion, original-value error presentation, frame takes and the existing cap notice presentation.
An already captured candidate frame survives cap advice, and the selector supplies positions within that frame rather than rereading the caller's potentially edited source afterward.
An uncapped population returns no replacement take, preserving the existing candidate frame and avoiding another copy.
The Rust host exercises the same public typed selection product exported through the composition root and host layer.
No new serialized message or opaque continuation is introduced by these owned primitive values.

Task derivation still uses the existing implementation in this slice and remains an explicit T4.6 extraction obligation.
Broader entity equality, other index families, custom native protocols, counts outside native row widths, final default activation and complete T4.6 acceptance remain open.
The current source and controls have not been executed; the frozen frame/link background run provides no validation credit for this new source.

The first source reviews found that supplied-row entity reads and target missingness must retain the reference's intermediate native populations, even when final row values agree.
The successor observes the supplied entity Series first and completes a useful owned scoring-key set in Rust.
An empty set selects the original source owner without reading source entity values.
A nonempty set returns a population of retained original source positions, including a full take when no entity matches.
Python performs that native holdout take before observing target missingness.
The retained population checks the mask extent, returns local labelled positions, and maps those positions back to original source ordinals for the existing context selector.
Candidate values are captured before context-cap validation and advice, preserving that separate native boundary.
Public source-read controls cover actual target operand identity, present and no-match holdouts, empty scoring and scoring-accessor edits to source keys.
