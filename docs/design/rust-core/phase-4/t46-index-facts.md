# T4.6: original index input facts

This is source preparation for RS-5, not a finished implementation or a contract freeze.
The bounded frozen-reference public matrix is prepared for root review and serialized execution.
No behavior result, candidate test result, platform equivalence or RS-5 completion is claimed here.
The only source helper prepared in this worktree is `sdm-engine/src/index_codes.rs`, which is not yet included in the crate's module tree.
Shared contract, core, schema, frame admission, value and Python bridge files remain with the Direct workstream until root serializes composition.

## Source rule

The frozen client at `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1` consumes index ordering only when tabular candidates exceed the wanted context size.
`materializers/tabular.py::indices` excludes scored rows and missing targets before calling `_context_within_limit`.
The predict-frame route excludes matching scored entities and missing targets before that same call.
Candidates within the limit retain their existing row order.
Over-limit candidates run `sample(n=wanted, random_state=42).sort_index()`.
Request context inspection and encoded wire rows expose the resulting exact row order.
Filtering, sampling, subset sorting, task derivation and warning policy therefore belong to shared core behavior.

For the inspected pandas 3.0.6 installation, `core/sorting.py::get_indexer_indexer` first calls `_sort_levels_monotonic` and then checks the selected index's monotonicity.
A monotonic selected subset retains its sampled order, including duplicate ties.
Otherwise the single-level path calls `nargsort`, whose ordinary-array branch partitions missing labels, argsorts only the nonmissing values and appends the original missing positions.
Extension arrays consume their own source-defined sorting representation, not a universal nullable-array implementation.
`arrays/base.py::argsort` uses `_values_for_argsort` and its original missing mask.
Categorical storage is signed codes plus the entire category dictionary, including unused categories.
Category count can change physical code width and therefore the ordinary NumPy kernel family even when observed label values are unchanged.

MultiIndex sorting first normalizes level dictionaries with `indexes/multi.py::_sort_levels_monotonic`.
That method preserves outward row values and catches a level's `TypeError` instead of always rejecting mixed levels.
The later nonmonotonic MultiIndex path uses `_get_codes_for_sorting` and `lexsort_indexer(..., codes_given=True)`.
Missing code -1 becomes the level's last rank for that selected subset, then `np.lexsort` produces the row indexer.
This behavior cannot be replaced by sorting a tuple of all labels with an ordinary object comparator.

The ascending default interval path calls `np.lexsort((right, left))` in `arrays/interval.py::argsort`.
Arrow-backed arrays call `pc.array_sort_indices` in `arrays/arrow/array.py::argsort`, even when the public request's default sort kind is quicksort.
Those are distinct producer paths and require their own source/runtime identity and exact public observations.
A sampled RangeIndex can become a different backing index, so original class identity alone does not select the eventual sorting kernel.

The scalar NumPy kernel is landed at `ed375509578bfea61dda703b36a6299f3f46ab8c`.
The separate four/eight-lane SIMD and GNU fallback source candidate remains a separate reviewed workstream until its required checks pass and root composes it.
The exact installed NumPy 2.5.3 source/native identity established earlier does not prove equivalence for older required producer cells or other platforms.

## Finite built-in family inventory

| Original native family | Facts needed before selection | Shared consumer obligation |
| --- | --- | --- |
| Signed/unsigned integers at every width, nullable integers and booleans | Original physical width, exact values, validity and original row alignment. | Selected-subset monotonic check, missing partition and the proven backing kernel. |
| Floating values, nullable floats and complex values | Exact physical widths and value bits, missing classification, signed zeros, infinities and complex components. | Preserve source comparisons and missing partition without casting all labels to float64. |
| Range labels | Exact label values and row alignment. | Derive the selected backing path from the facts and actual supported source rule. |
| Python/Arrow strings and ordinary text/binary objects | Exact text/binary values, null classification and source conversion family. | Dispatch the actual ordinary-object or Arrow algorithm; do not silently substitute tie semantics. |
| Ordinary numeric, Decimal, date and tuple objects | Closed typed values, equality/comparison meaning and missing classification. | Reproduce accepted comparisons and actual subset-dependent failures; never stringify or refuse the entire object family. |
| Categorical | Whole dictionary identity/order, ordered flag, original signed code width and original-row code attachment. | Retain unused categories and missing code -1 through selection, then sort observed nonmissing codes with the correct kernel. |
| Datetime, zoned datetime, timedelta and period | Original temporal values, unit/zone/frequency, physical backing family and missing facts. | Preserve temporal comparison and its scalar backing path rather than treating all i64 storage as numeric SIMD. |
| Interval | Typed left/right endpoint attachments, closure and original validity. | Apply the proven endpoint lexsort path to the selected subset. |
| MultiIndex | Full level dictionaries with nested categorical identity where present, original signed per-level code width, original-row code attachments and missing code -1. | Normalize levels by the defined source rule, check selected-subset monotonicity, then apply stable level-code lexsort. |
| Arrow primitive/string/decimal extensions | Arrow values/validity and exact producer family. | Preserve the actual Arrow sorting algorithm and missing behavior. |
| Sparse and other ordinary built-ins | Lossless native conversion facts and evidence of the actual consumed source path. | Record real acceptance before deciding the minimum additional typed representation. |

The inventory covers every ordinary built-in family as an obligation, not a claim that every representation is already sufficient.
The first public matrix includes primitive and nullable widths, complex labels, strings/bytes, mixed objects, categorical width boundaries, unused categories, temporal/interval labels, MultiIndex variants, Arrow extensions and sparse labels.
Controls cover all-equal/all-missing labels, unsampled incomparable labels, mutation between public calls and both automatic-cap public routes with full 10,000-row captures.
Construction failures, SDK refusals and observer conversion failures must be reported separately before extending this design.

## Minimum boundary and composition

Native conversion supplies typed facts aligned with the original admitted frame snapshot before the core chooses rows.
Input-sized labels, dictionaries, category/level codes, missing masks and compound components remain Arrow attachments.
Small metadata describes neutral structure, temporal meaning and a closed reviewed producer profile.
No pandas dtype string, object handle, callback, native-domain scheduler, row-shaped JSON or host-computed full-frame sort permutation crosses this boundary.
The producer profile must be derived from actual source/build identity and shared by Python and pure Rust HostRun fixtures; it is not a user-selected sorting policy.
Its serialized vocabulary remains unfrozen pending complete source/runtime coverage.

Core frame admission validates fact lengths against the admitted row count, dictionary correspondence, code ranges, missing sentinels, physical widths and attachment lifetime.
Identity belongs to the entire frame/index snapshot, including unused dictionary levels and mutation state.
A previous conversion or sorted permutation cannot survive a native source/index mutation unless the public behavior establishes that exact reuse.
After normal core filtering and seed-42 selection, the engine derives the selected label/code view, applies the selected-subset monotonic shortcut, dispatches the proven sorting path and returns a row indexer for shared row application.
Core/request composition must preserve inspectable Request context identity and the exact model-wire row order.
Both hosts must exercise the same full path with typed inputs rather than replaying a final reference permutation.

The isolated `IndexCodes` helper validates signed 8/16/32/64-bit Arrow code attachments before subset selection.
It retains the physical width and dictionary length, accepts only -1 or an in-bounds dictionary position, rejects Arrow-null alternate encodings, and projects precisely the core-selected row sequence.
It preserves attachment ownership without copying, leaves repeated selected rows intact and checks original row bounds.
It does not enforce a pandas dtype system, choose an ordering algorithm, normalize dictionary values, infer category order, sample rows or create a new ordinary-input refusal.
Its local validation errors describe malformed host fact attachments and still require the existing shared protocol error mapping during composition.
The unit-test source covers unused category thresholds, all-missing/empty dictionaries, invalid unselected codes, row/snapshot mismatches and selected row identity.
It has not been executed, and these helper tests cannot replace a public Python/pure-Rust HostRun twin.

## Explicit unresolved boundaries

The owner's approved exception keeps only tabular calls with user-defined index sorting overrides on the existing Python route.
It preserves custom execution, side effects, errors and timing.
It does not approve a blanket extension fallback, stable tie substitution, eager custom conversion, native sorting callback, second preparation stage or new ordinary-family refusal.

Custom object comparison/equality methods are distinct from user-defined index sorting overrides.
Whether they introduce executable effects on this public journey needs a separate real witness.
This workstream cannot infer a legacy failure, broaden the approved exception or invoke those effects early as conversion.
For ordinary mixed built-in values, the finite typed comparison representation must preserve actual subset-dependent comparisons and failures without requiring a false total order.
No generic host runtime emulator or unbounded pairwise comparison table is proposed.

Remaining work includes the reviewed matrix execution, any observer repairs as separately hashed packets, required producer-cell identities, the minimal remaining ordinary-value facts, composition with the shared engine/core/bridge owners and both public-host end-to-end legs.
All Phase 4 public paths remain in scope, including holdout, task/advice and mutation effects around ordering.
Neither this helper nor one producer's reference observations establish RS-5 completion.
