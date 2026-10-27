# T4.6 RS-5 shared consumer implementation plan

This plan starts from landed kernel integration `f7215a525eb8778a3c5ecec1d3058c79de2676e7` and preserves the isolated, unlinked `fce7fd257` code-admission helper checkpoint.
The Direct contract-34 work remains under its author's ownership while its current review findings are fixed.
This document changes no contract, shared source, route or generated file and claims no candidate execution.
Root composes the following changes only after the Direct range is settled.

## Implement one original-row input model

Introduce an engine-owned `OriginalRowFacts` value admitted together with one frame snapshot.
Its logical row space is the original frame's `0..source_rows`, before scored-row removal, target-missing filtering or sampling.
The value owns references to Arrow attachments and a closed `IndexLayout`, not native objects or a precomputed sort permutation.
The descriptor belongs in the existing frame-manifest contract when root serializes that contract change; the decoded data and algorithms belong in `sdm-engine`.
Keep all input-sized values outside descriptor JSON.

| Layout | Concrete Arrow representation | Required admission |
| --- | --- | --- |
| Primitive | Original typed values plus native missing representation where validity alone is insufficient. | Exact row count, physical type/width and missing-mask coherence; retain NaN bits and signed zeros. |
| Structural object | Dense union tags/offsets and typed scalar children, with recursive tuple-list offsets. | Type-code membership, offset/child bounds and recursion/length coherence; read union slice offsets correctly. |
| Categorical | Full dictionary attachment, signed code attachment, ordered flag and dictionary layout. | Reuse `IndexCodes` for -1/bounds/row checks; retain unused dictionary entries and original code width. |
| MultiIndex | Ordered full level descriptors and one original-row signed code attachment per level. | Validate each level independently, then matching row counts and level/code correspondence; retain unused levels and categorical level identity. |
| Interval | Typed left/right endpoint attachments plus closure and validity facts. | Matching endpoint type/row counts and the source's missing representation. |
| Period/temporal | Typed ordinals or temporal values with unit, zone/frequency and semantic backing family. | Retain original units and missing sentinels; i64 temporal storage must not activate numeric SIMD. |
| Sparse | Logical length, typed nonfill values, nonfill positions and typed fill value. | Position count/order/bounds, value/fill compatibility and original logical length; expand only the selected view if useful. |

The validated observer union is a useful concrete starting encoding, not an already complete object-type contract.
Its arbitrary-width signed integer bytes, float bits, bool/null distinctions and recursive tuple structure are ordinary typed data.
Extend that same structure for complex components, exact Decimal facts and date/temporal values through the existing scalar/Decimal owners rather than creating a parallel numeric policy.
For Unicode labels that cannot enter Arrow UTF-8 losslessly, carry Unicode code points in an Arrow integer-list child, reusing the existing native builtin Unicode fact meaning.
An unencodable row-index string does not become a new refusal merely because the first observer used `pa.string()`.
Categorical/MultiIndex dictionary children can use the same structural layout, avoiding separate mixed-level conversion rules.

Admission must reject malformed internal attachment facts through the existing protocol error path, without turning accepted ordinary native labels into unsupported-family errors.
Bind frame rows and index facts in the same call-local admitted snapshot and retain their attachments together.
Any caching key must include index values, dictionary identity/order, missing facts, structure and the associated frame snapshot.
Never identify a snapshot by a Python object address or reuse a full-frame sort permutation after mutation.

## Select explicit source algorithms

Keep a closed, source-proven producer profile distinct from the input's physical width and index layout.
The Python converter reports actual runtime/build conversion facts before selection; core validates them against the reviewed profile inventory.
The pure Rust host supplies the identical profile and Arrow facts in its independently constructed input.
This is internal producer identity, not a user option or host-selected sorting policy.
An unmapped ordinary producer is an unresolved migration obligation, not authorization for a new public refusal, blanket extension fallback or silent scalar substitution.
Keep its current public support while resolving the source algorithm and required-cell coverage before routing activation.

| Consumer seam | Concrete implementation work |
| --- | --- |
| Scalar NumPy numeric/code/temporal path | Reuse `sample::scalar_argsort` where its exact source and comparison semantics apply. |
| NumPy 2.5.3 Intel7 four/eight lanes with GNU14 fallback | Reuse landed `index_sort::simd_argsort` with its explicit `Avx2Gnu14` or `Avx512Gnu14` profile. |
| NumPy 2.2.6 Intel5 four/eight lanes with GCC10 producer | Add a separately named source profile; prove shared source components before reuse and preserve the older entry/network/fallback behavior where it differs. |
| Ordinary object/Python-string backing | Add a fallible source-defined scalar argsort comparator over the closed typed values, preserving comparison visitation and the actual unsupported-operand failure when reached. |
| MultiIndex and interval paths | Implement the source-defined stable lexsort over admitted level codes or endpoint values, retaining equal-key input order. |
| Arrow-backed sorting | Use an exact-equivalent library kernel if established, otherwise one bounded Arrow25 algorithm consumer from its actual source; do not assume NumPy or Rust unstable tie semantics match. |

The Intel5 implementation is not authorized to call the Intel7 public entry and ignore its sorted shortcut.
The exact source files and compiler producer differ, and the new CP310 controls observe the missing-partition seam where the difference can be public.
Likewise, the existing GNU14 forced-fallback proof is not proof of the GCC10 producer fallback.
Review source equality for shared GNU routines or implement the exact separately proven fallback before activating that profile.
Other required producer/platform cells retain their existing proof obligations; the descriptor must not claim their equivalence from this Linux cell.

The ordinary object comparator uses typed builtin semantics, including exact integer/float comparisons beyond 2**53 and 2**63, lexicographic tuple comparison, Unicode code points and the actual missing classification.
Do not coerce integers to f64 or text to obtain a total order.
The existing infallible `PartialOrd` scalar helper cannot by itself reproduce a reached int/string TypeError.
A Rust-internal fallible comparator is an engine function over data, not a native comparison callback.
Keep custom native comparator/equality effects as a separate unresolved boundary; only user-defined index sorting overrides currently have an approved legacy-routing exception.

## Implement the selected-subset engine function

Add one `context_rows` consumer whose inputs are admitted frame data, admitted original-row index facts, scored entity facts, target identity and the validated context-size argument.
It returns original row ordinals in final context order plus independently useful selected-frame/task facts needed by the existing model-neutral preparation path.
It must not request host sorting or accept a host-returned sorted subset.

1. Resolve the scoring rows and preserve their requested entity order through the existing frame/key policy.
   For indices materialization, exclude every scored source position and every missing-target row.
   For predict-frame materialization, reuse the shared entity equality/key policy to exclude matching scored entities before missing-target filtering.
2. Keep surviving candidate ordinals in original row order.
   Reuse the shared model limit and error table instead of duplicating the 10,000-row policy in the converter or engine.
   If candidates fit the wanted size, return that sequence immediately without normalizing levels, comparing mixed labels or consuming an argsort.
3. Preserve automatic-cap advice before any reached sorting failure.
   If the current run interface cannot deliver that existing notice before sampling/sorting, return a useful filtered-candidate snapshot and validated cap decision at that fixed boundary, then continue shared core selection after normal warning presentation.
   That boundary precedes selection, contains no selected subset, and introduces neither a native-domain scheduler nor a sorting callback or second preparation after selection.
   A warning promoted to an exception must precede an otherwise reached mixed-label TypeError.
4. Reuse `Mt19937::seeded(42).permutation(candidate_count)` and take its first `wanted` positions after verifying the source-equivalent `RandomState.choice(..., replace=False)` composition for the registered cells.
   Preserve the drawn candidate order exactly; do not use the modulo-biased Floyd sampler, PCG64 or another seed stream.
5. Build the index view only for those selected original row ordinals.
   Apply MultiIndex level normalization at the source's location, retaining its caught incomparable-level behavior and dictionary/code remapping.
   Evaluate the actual family's selected-index monotonic rule, including its empty/singleton/missing details, rather than a universal generic missing rule.
   On the shortcut, return the sampled sequence unchanged.
6. For an ordinary single-level source, partition selected missing labels in their existing order and run the proven argsort over only the selected nonmissing backing values.
   Remap its local indices through nonmissing-selected positions and then selected original row ordinals, appending missing positions as the source does.
   Dispatch categorical values by actual code width, temporal values by their temporal backing and Arrow/interval/MultiIndex paths by their distinct source algorithms.
   Keep the native kernel's own sorted shortcut separate from the earlier selected-index shortcut.
7. Apply the final original-row sequence once to frame data, carrying the selected target into shared task derivation and preserving the ordered selected index structure for public presentation.
   An internal kernel permutation never becomes a host policy input or a final externally supplied reference order.

This function must preserve the distinction between sample-then-sort and selecting from a full-frame permutation.
All-equal selected labels can retain sampled order, incomparable labels can be irrelevant within the cap and unused category dictionaries can change a sampled kernel's width.
Those are algorithm inputs and source rules, not exceptional fallback cases.

## Compose with the settled Direct range

The root-owned contract change extends existing frame materialization with the original-row descriptor and typed context result after the working Direct34 shapes are settled.
It must not extend `Step` beyond fetch/model/done or create a new crate, SDK artifact or generic workflow protocol.
The Python bridge adds only lossless original-index conversion, attachment ownership and public object reconstruction around those existing useful values.
It neither filters candidates, chooses the context cap, derives the task, sorts selected rows nor supplies comparison results after core selection.

`sdm-core::frame_materialize` selects the tabular branch and composes the new engine consumer with the existing PQL validation/model specification.
The selected context and scoring halves then enter the same model-neutral preparation and later model-bound envelope/body path that Direct34 supplies.
Reuse its model policy, target/feature validation, held-schema ownership, warnings, wire encoders, response attribution and public object construction.
Do not call the legacy Python TabularMaterializer or add another tabular adapter to decide those policies.
Direct `Request(context, predict, ...)` continues to preserve supplied context order and its existing over-cap refusal; the new sampler is reached only by the PQL tabular materialization routes.

Python may present the core's selected original-row sequence with ordinary native row-taking to retain exact public pandas index type, names, unused categories and MultiIndex identity.
That operation applies a core decision; it does not compute a sort.
The shared result must also carry enough typed selected data/index structure for a pure Rust HostRun host to inspect the same context and produce the same model rows without pandas.
Keep native presentation-only name objects local where they do not affect shared sorting, while any consumed comparison meaning remains typed input data.

Freshness follows the actual source boundaries.
The boolean-filtered candidate snapshot, selected context and later Request/schema observations must not be collapsed into one constructor-time conversion.
A mutation between materialize calls creates fresh original-row facts, and source mutations or warning effects retain their observed relationship to already copied candidates and later schema/task reads.
Reuse the settled Direct warning/constructor boundaries instead of replaying already consumed guards against later mutations.

## Ownership and immediate implementation order

| Owner | Concrete files and deliverable |
| --- | --- |
| RS-5 engine author | Link/adapt `sdm-engine/src/index_codes.rs` after review; add `index_facts` admission/views, the structural union reader and selected-subset `context_rows` engine consumer. |
| Existing kernel owner | Source-profile extension in `sdm-engine/src/index_sort.rs` and its existing `gnu` module for Intel5/GCC10 only after equivalence evidence, preserving the landed Intel7 behavior. |
| Root with Direct author | Serialize `sdm-contract/src/frames.rs` and existing materialization/preparation shapes, existing admission/attachment validation and schema generation after Direct34 settles. |
| Existing shared value/Decimal owners | Supply exact closed scalar comparison/conversion primitives needed by object labels; avoid a second bigint/Decimal/temporal policy. |
| Root-authorized bridge/core composition | Extend `_rust/frames.py`, `_rust/frame_materialize.py` and core frame materialization while reusing settled `_rust/prepare.py` and `sdm-core::prepare` boundaries. |

First implement and independently review the original-row reader and engine views against the already captured typed artifacts, without routing activation.
Next implement the complete ordinary selected-subset consumer, including fallible mixed comparisons, level normalization and profile-specific kernels, while retaining the exact row-identity chain.
Then compose the two public PQL routes through the shared Direct preparation/body path and run their pure Rust HostRun twins early.
Use the actual original, repaired and CP310 profile observations as frozen expected data; do not make their source receipts count as candidate checks.
The final code range requires two independent whole-code approvals before root grants candidate checks, followed by the relevant public Python/HostRun tests, formatting/lint and required package suites.
No helper, one producer profile, callback-free numeric path or successful wire capture alone closes RS-5 or the full Phase4 public-path inventory.
