# T4.6 portable SIMD index source checkpoint

This checkpoint provides the complete four/eight-lane source algorithm and an explicit GNU 14 depth fallback in `sdm-engine::index_sort`.
It has no production consumer, runtime detector, serialized profile, native callback, new dependency or tabular routing activation.
The candidate source and tests have not been executed.
Two independent whole-source reviews must approve the same immutable candidate before root grants candidate checks.
The [preceding checkpoint](t46-index-kernels.md) records the exact NumPy source/archive/native identity and the full register, network and partition specification.

## Source boundary

`simd_argsort` accepts only the six primitive types in the pinned source's ascending SIMD wrappers: signed/unsigned 32/64-bit integers and float32/float64.
The sealed `SimdKey` trait supplies each type's real maximum/minimum sentinel without coercing keys through floating point.
`Avx2Gnu14` describes four lanes, reversed right partition groups and the explicit GNU fallback.
`Avx512Gnu14` describes eight lanes, increasing right partition groups and the same explicit GNU fallback.
These names describe source operations and do not establish that a particular native runtime is bound to either profile.
They are not a dispatcher over NumPy versions, CPUs or standard libraries.

The caller must remove missing and incomparable values, including NaNs, before this kernel and prove its producer's source/profile binding.
The source's original-array sorted shortcut is retained independently of the caller's later selected-subset monotonic shortcut.
The source entrypoint converts only the length through `log2`, truncates before doubling and decrements depth once per recursive edge.
Floating key values never cross a width-changing conversion.
The source's NaN/descending branches are outside these ascending nonmissing profiles and are not silently substituted with another algorithm.

The register operations preserve their exact source masks, shuffles and current-lane equality selection.
The masked network loads real integer maxima or floating positive infinity into inactive key lanes and `usize::MAX` into inactive index lanes, and writes only the requested index range.
The base case preserves the source's vector-count halving, pair reversals, full merge stages and per-register cleaners through 256 elements.
The larger-range path preserves the fixed scalar remainder, first/last saved groups, four-vector unrolling, space-based load direction and both store profiles.
AVX2 stores each complete rearranged vector at both cursors, including its reversed right group; AVX512 writes its two increasing compressed groups.
The recursion uses the source pivot positions and comparison-based minimum/maximum pruning.

Two temporary-value simplifications preserve all comparison classes on this non-NaN domain.
For floating zero, Intel min/max may choose a different zero sign, but every subsequent operation uses only numeric `<`, `>=` or `==`, all of which treat both signs equally.
Indices follow the source's current-lane equality masks, and final payload values remain in the original immutable input.
Likewise, the key-only pivot network can be replaced by insertion ordering of the four/eight sampled keys: its chosen median has the same numeric comparison class, including equal zeros, and it carries no row index into a tie decision.
Neither simplification makes the index network stable or introduces an index tiebreak.

## Explicit GNU fallback

The depth check precedes the 256-element network base case.
An exhausted range always uses the explicit GNU policy, including ranges shorter than 256; it never delegates to Rust's sort or the existing scalar NumPy heap.
The policy uses the median of positions 1, midpoint and last, preserves the GNU median tie branches, partitions around a pivot moved to position zero, recurses right and iterates left, and finishes with insertion ordering after its threshold-16 introsort.
Its exhausted internal introsort builds and drains a heap with GNU's right-child tie choice and leaf-then-rise adjustment.
This is a source-defined index algorithm over the original typed keys, rather than a C++ dependency or vendored implementation.

GNU's guarded first-16 insertion plus unguarded insertion tail is expressed as one guarded insertion pass.
For ranges over 16, introsort places a minimum in the first block, so the tail never crosses the beginning.
For ranges at most 16, GNU already uses guarded insertion over the whole range.
The bound check therefore changes neither comparisons of key values that affect movement nor the resulting equal-key permutation.

The read-only producer audit establishes that the pinned ELF reports `GCC: (GNU) 14.2.1 20250110 (Red Hat 14.2.1-11)`.
It also preserves every actual AVX2 local `argsort_` specialization, each public wrapper's call sites, and the compiled GNU introsort/heap specializations.
The signed-64 entry at ELF `0x445f70` tests depth at `0x445f9d`, before its length comparison at `0x445fb1`, and its zero-depth branch invokes the actual compiled GNU introsort at `0x411870` followed by threshold-16 insertion.
This is binary corroboration, not proof that a specific public input naturally reaches exhausted depth.

The audit also pins the official [GCC 14.2.0 algorithm header](https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-14.2.0/libstdc%2B%2B-v3/include/bits/stl_algo.h) and [heap header](https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-14.2.0/libstdc%2B%2B-v3/include/bits/stl_heap.h) as comparison sources.
The exact Red Hat producer headers have not been acquired, and those official comparison files are not claimed to be the producer's installed headers.
The policy's exact producer binding remains unestablished until the outstanding binary/source review and candidate comparison with independently acquired actual-fallback controls are complete.
No fallback-unreachability proof is asserted.

## Tests and evidence

Tests-first commit `62b657f4d` retains all 129 independently recorded public native AVX2 permutations among the preceding acquisition's 184 controls.
It selects only the six dtypes that the pinned source dispatches through SIMD and retains exact little-endian input bytes and full output permutations.
Those controls cover empty/singleton inputs, duplicate ties, network thresholds, maximum sentinels, unsigned extremes, zero/subnormal/infinity values and partition-sized organ-pipe inputs.
No expected permutation is calculated with a second copy of the proposed Rust algorithm.
The tests additionally assert that typed input bytes stay unchanged.

The authored eight-lane checks assert ordering and permutation preservation only, and are explicitly not AVX512 native receipts.
The authored depth-zero check pins small-range GNU insertion ordering before the network base case.
The authored GNU heap and introsort checks exercise forced internal heap paths and even/odd heap edges as invariants; they do not claim independently acquired native tie permutations.

Root inspected and executed an immutable private acquisition packet containing 123 nonempty controls against the actual wheel's six local `argsort_` specializations with depth forced to zero.
It pins the ELF hash and function offsets, verifies the primitive-pointer/count ABI against source signatures and real call sites, derives load bias only from the same mapped ELF, and confines execution to a bounded separate process with guarded arrays.
It imports no candidate or SDK, compiles no C++, and writes complete input/output bytes and permutations to a new receipt.
The acquisition completed all 123 calls successfully, preserving guard regions and key bytes and checking complete ordered permutations.
Its receipt hashes to `b3fa744af2117b91ff7a76b7e6ec13c7ea16d3fb5b6c0977fea93f1809889947` and the exact expected permutations are retained separately in `tests/data/t46_simd_fallback_controls.json`.
The authored depth-zero tests apply those native expected answers to both source profiles, whose fallback policy is identical.
Those tests have not run, and the shared fallback checks do not establish native AVX512 network or partition behavior.
The forced branch witness remains distinct from naturally reachable public NumPy fallback and cross-producer coverage.

## Remaining integration and profile inventory

| Obligation | Current evidence and remaining work |
| --- | --- |
| Pinned NumPy 2.5.3 AVX2 source | Exact archive/native identity, 129 independent public native controls and full source authored; source reviews and candidate checks pending. |
| Pinned producer fallback | Exact compiler comment, compiled branch evidence and 123 acquired actual-wheel depth-zero controls; exact producer headers unacquired, candidate comparison and whole binary/source binding review pending. |
| Natural fallback reachability | No unreachable claim and no natural public-input witness; complete fallback is implemented independently of reachability. |
| NumPy 2.5.3 AVX512 | Eight-lane source authored and invariants written; no actual AVX512 native acquisition or producer binding. |
| NumPy 1.23.5 floor | Prior scalar source evidence only; required-cell native identity and public integration remain separate. |
| NumPy 1.26.4, 2.1.3 and 2.3.4 | Actual compiled dispatch, bundled source, lane/store differences and producer fallback policy still need acquisition and binding. |
| Newest, ARM, macOS and Windows | Required-cell source/build identities and applicable scalar/SIMD/standard-library policies remain unproved. |
| Filter/sample/sort context journey | Closed native index facts, filtering before seed-42 sampling, selected-subset monotonic decisions, missing placement, payload rows and Python/rust-host integration remain with T4.1/T4.2/T4.6. |

The full Phase 4 inventory remains 119 required public paths.
This algorithm source checkpoint does not complete RS-5, activate an unsupported profile or extend the owner's custom-index override exception to ordinary built-in indexes.
