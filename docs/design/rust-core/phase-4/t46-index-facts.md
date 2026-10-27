# T4.6: original index input facts

This is source preparation for RS-5, not a finished implementation or a contract freeze.
The root executed the bounded frozen-reference public matrix after inspecting its immutable READY packet.
The observations below are reference input/output evidence, not a candidate test result, platform equivalence or RS-5 completion.
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
Construction failures, SDK refusals and observer conversion failures are reported separately below.

## Actual frozen-reference observations

The CP312 reference process used pandas 3.0.6, NumPy 2.5.3 and Arrow 25.0.1 with the frozen SDK sources and recorded source/native identities.
The immutable result receipt SHA256 is `7dc2b05a227f49ce5176c0acf2f01f1fea9a4157f81082a3f25edf45eb9fa655`.
The 64 planned cases produced 62 successful public cases and 63 logical payload captures, because the built-in mutation case made two successful calls.
Every successful capture matched the inspectable Request context row IDs exactly.
Both automatic-cap routes produced full 10,000-row captures and passed the independent filter-before-seed-42-sample assertions.
These captures preserve the actual dictionaries received by the scripted transport; they are not raw HTTP request-byte captures.

Native float16 Index construction raised `NotImplementedError('float16 indexes are not supported')` before an SDK input existed.
This is a native construction limit for that exact cell, not a new SDK refusal or proof about another producer cell.
The over-cap ordinary int/string object case raised the real materialization TypeError, `'<' not supported between instances of 'str' and 'int'`, from sorting the selected subset.
The same ordinary mixed-label family succeeded within the cap, where no sorting was consumed.
That pair prohibits unconditional mixed-family admission refusal or eager comparison merely because labels are mixed.

Recursive receipt inspection found ten nested Arrow observer failures across six cases despite successful SDK payloads in five of those cases.
The affected facts were mixed int/float/bool context labels, mixed int/string input labels, tuple input/context labels, mixed MultiIndex level input/context dictionaries, sparse input/context labels and the within-cap mixed input/context labels.
An inferred homogeneous Arrow array also does not prove preservation of mixed builtin scalar kinds where conversion silently succeeds.
The original matrix and receipt remain immutable.
A new targeted observer packet proposes rerunning only those six public cases and checking their original SDK outcomes, row identities, features and every logical payload field.

The repair's ordinary mixed values use a dense Arrow union with explicit scalar kinds, integer bytes, float bits and recursive tuple-list children.
Its roundtrip checks retain bool/int/float distinction, signed zero, NaN bits, null-sentinel identity, arbitrary signed integer magnitude, text/binary values and nested tuple shape.
Mixed MultiIndex levels use structural dictionary attachments independently of their original signed row-code attachments.
Sparse facts use typed stored values, original nonfill positions, a typed fill value and the original logical length, with exact dense dtype/bit reconstruction checked only by the observer.
No host comparison ranks, final sort permutation, row-shaped JSON or runtime callback is introduced.
Observer conversion and IPC roundtrip failures remain separate from SDK results at every nesting level.
The first repair attempt failed before producing a final receipt because its SDK signature comparison incorrectly required identical generated request IDs.
Static comparison of its actual first payload and the original found exactly one differing leaf, `metadata.request_id`.
Frozen `adapters/tabular.py:71-72,272` generates that field from `uuid.uuid4()` for each payload.
The failed packet and partial files remain immutable.
The actual r2 packet preserved complete baseline/current SDK signatures, actual payload files and hashes, and every field difference before any assertion.
Only that exact source-proven request-ID leaf is classified separately; every other payload field must match.
Root's actual r2 receipt is `aea402c20199c121ce164cef06ccb5254309d8dd5b7687d1d3f8ea0bee5348cc`.
All six SDK signatures matched, and the five captured payloads differed only at that source-proven generated request ID.
Tuple input/context and the separate codec control still failed the prewrite reader assertion, so r2 alone did not close all observer gaps.
The precise failure was line161 before writing, not the later IPC assertion at line172; no tuple/control Arrow files existed in r2.
The source diagnosis found that PyArrow25 union type-code/offset accessors expose unsliced buffers, while the recursive reader used nonzero child-union slices.
The new r3 reader applies the union array's actual offset to both buffers.
Root's actual conversion-only r3 receipt is `c8c2fd68716acbf3d081a180ed6416a1a89787e3a3c876110c0b5e342e5be9c6`.
It verified exact kinds/bits for source input40, actual public context30 and codec control22 before writing, after IPC and on zero/nonzero/empty slices, with no SDK/native imports or observer failures.
Together, actual r2 and r3 close the original ten observed attachment gaps for these finite source/public identities.
They remain reference/representation evidence, not candidate execution or RS-5 completion.

## Required CP310 producer evidence

The existing CP310 runtime contains pandas 2.3.3, NumPy 2.2.6 and Arrow 25.0.1.
NumPy's recorded revision is `2b686f659642080e2fc708719385de6e8be0955f`.
The installed native producer SHA256 is `c634eaf05367af02b815cf3d2020fbeee9b47426af1fe26cf3189e4798932dc5`.
Its bytes match the native member in the independently acquired official CP310 manylinux wheel.
The official wheel SHA256 is `fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915`, and the exact NumPy 2.2.6 sdist SHA256 is `e29554e2bef54a90aa5cc07da6ce955accb83f21ab5de01a62c8478897b264fd`.
This was source/artifact inspection only, without runtime package imports or behavior execution.

The exact 2.2.6 source bundles Intel x86-simd-sort 5.0.x, while the inspected 2.5.3 source bundles 7.0.x.
The 2.2.6 build selects AVX2 and AVX512_SKX argsort targets, and its installed symbols confirm those six numeric wrapper families.
The inspected ELF compiler note identifies GCC 10.2.1, so the existing GCC 14 producer fallback proof cannot be reused without source/runtime equivalence evidence.
The bundled argsort and key/value network files differ substantially; in particular, the 2.2.6 entry lacks the 2.5.3 `std::is_sorted` early return.
Pandas partitions missing labels after its selected-index monotonic check, so an index containing missing labels can still send an already sorted nonmissing backing array to the kernel.
That creates a concrete necessary public control for the distinct producer profiles rather than an assumption that all duplicate permutations match.

The installed pandas 2.3.3 source retains the relevant selected-subset monotonic, MultiIndex level-normalization, Arrow argsort and interval lexsort dispatch seams.
This is source-rule evidence, not yet a claim of identical public outcomes.
The CP310 and CP312 Arrow 25.0.1 `libarrow_compute.so.2500` files are byte-identical, with SHA256 `847cd8c8aaa9c77ee5984891589ba13e80933bd8668b06d3bf3c06872f260e7e`.
Their Python/native wrapper identities are still recorded separately, and matching compute bytes do not prove every input-conversion rule agrees.
Only a bounded additional frozen-reference matrix over the distinct or unobserved producer seams is warranted; the original 64 cases and full captures need not be rerun wholesale.

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
