# T4.6 index kernel source checkpoint

This checkpoint extends the existing scalar index algorithm to typed values and specifies the distinct SIMD algorithm that ordinary tabular indexes still need.
It does not activate tabular sorting, choose a runtime profile, change context order, or establish complete RS-5 acceptance.
The new Rust source and tests have not been executed at this checkpoint.
Two independent exact-source code approvals are required before candidate checks.

## Scalar kernel

`sdm-engine::sample::introsort_argsort(&[i64])` retains its existing entrypoint and delegates to `scalar_argsort<T: PartialOrd>`.
The median-of-three partition, insertion threshold, depth accounting, stack order and heap fallback are the existing algorithm with typed comparisons and borrowed pivot values.
No integer is converted through floating point and no language callback is accepted.
Callers must partition missing and incomparable values before this kernel, including floating NaNs, and establish that their actual source uses the scalar algorithm.
Positive and negative zero compare equal while original values and their bits remain in the caller's unchanged input.
The kernel is not an alternative implementation of SIMD dispatch.

The first tests precede the implementation and retain exact native scalar duplicate permutations at lengths 30, 65 and 257.
An independent subsequent native acquisition records 184 scalar controls and 184 native AVX2 controls over identical inputs.
The scalar controls include signed and unsigned widths, Boolean values, finite floating extrema, signed zero, subnormals, infinities, padding-sensitive maximum keys, lengths around 16 and 256, sorted/equal arrays and an organ-pipe input.
Their 184 exact scalar permutations are retained in `sdm-engine/tests/data/t46_scalar_kernel_controls.json` and consumed without adding a crate dependency.
The acquisition checks that every native answer is a permutation and leaves input bytes unchanged.
It imports no SDM package and supplies no candidate-execution credit.
Across these inputs, 105 of the 184 AVX2 answers differ from scalar answers, all among the six 32/64-bit integer and floating families.
Those differences are evidence against replacing a SIMD route with this scalar kernel.

## Exact source identity

The source is the official NumPy 2.5.3 source archive with SHA-256 `df2d5874ff183595a4ba404edd04f6bd9b5505c1d7708573f6a6c17489a67563` and recorded revision `dd88c0c19b54ad9ed3533224221285bf0873249a`.
The observed CPython 3.12 native module hashes to `ceec5ba13ab88318632307fdd2df74f317fa293ce77be559c4298ddd1f4306b0` and matches the official wheel.
The complete source archive pins its bundled Intel code even though its metadata does not preserve an Intel gitlink commit.
Source paths below are relative to `numpy/_core/src/npysort/` in that archive.

| Source | SHA-256 |
| --- | --- |
| `quicksort.hpp` | `ee55b06cdc82556f5bac021a605419561203def7fd4074b5039c5f9ca665801e` |
| `heapsort.hpp` | `d77c64ddccd4b259a8675599d771d3ea7303a30731dbf2d0a255aa3371d7a22e` |
| `x86_simd_argsort.dispatch.cpp` | `49b2c9f36c82129ae7f41477719b1431914228013a3425c43b40611fba573339` |
| `x86-simd-sort/src/xss-common-argsort.h` | `58f720e91e3ae3ef82cc2099adcd11448b01e3299b0b92689e71330d004d704f` |
| `x86-simd-sort/src/xss-network-keyvaluesort.hpp` | `412bbdba713503fcb8b5d235fb5cd3492c76f776c9a3a8c27e1b91edfc29f0bd` |
| `x86-simd-sort/src/xss-reg-networks.hpp` | `b00982b00d35aedf44950706ef8da539e29b75cd11cd93059dfb0d66dc81c2b2` |
| `x86-simd-sort/src/avx2-emu-funcs.hpp` | `1073c6abd870794b36b8e208b7988cc50da3ae8cc157f4d392b6b7f5197fa2fb` |
| `x86-simd-sort/src/xss-common-includes.h` | `02bf51c63b3f87009234b81a5487db6e38719f309cd0f36c1f3dc31b5a4aa185` |

## Portable SIMD index-network specification

Keys stay immutable and the algorithm mutates only an index permutation initially equal to `0..n`.
The six NumPy wrappers are signed/unsigned 32/64-bit integers and float32/float64, with no descending dispatch in this source.
For the observed x86 build, AVX2 uses four index lanes and AVX512 uses eight, including half-width 32-bit key vectors paired with the 64-bit index vectors.
The caller's selected-subset monotonic shortcut and missing-value partition remain separate from these kernels.
The SIMD entrypoint also has its own original-array sorted shortcut and a floating-NaN fallback; neither can be silently removed or used to justify a different source family's order.

### Lane comparisons

A register operation keeps keys and indices together through the source's masks and permutations.
For each lane, the min/max result's numeric equality with the original first key selects the original first index; equality therefore retains that lane's own index, without comparing index values.
This is not a stable global sort because subsequent lane shuffles and vector reversals reorder equal keys.
Original input values are never rewritten, including signed-zero bits.
Any portable simplification of temporary floating min/max bits must first prove that it preserves every later comparison and index selection for the admitted non-NaN domain.

Let `R(b)` reverse lanes within each consecutive block of `b` lanes and `S(b)` exchange the two halves of each such block.
Each comparison mask selects maxima in its set bits and minima in its clear bits, while indices follow the equality rule above.

| Register width | Full sorting network, in order | Bitonic half-cleaner, in order |
| --- | --- | --- |
| 4 | `R(2)/0xA`, `R(4)/0xC`, `S(2)/0xA` | `S(4)/0xC`, `S(2)/0xA` |
| 8 | `R(2)/0xAA`, `R(4)/0xCC`, `S(2)/0xAA`, `R(8)/0xF0`, `S(4)/0xCC`, `S(2)/0xAA` | `S(8)/0xF0`, `S(4)/0xCC`, `S(2)/0xAA` |

These are the key/index overloads in `xss-reg-networks.hpp:171-249` and `:335-397`, not the nearby key-only routines.

### The network base case

The quicksort base case has 256 elements, despite its stale source comment naming 64.
Start with `256 / lanes` vectors, then repeatedly halve their number while `2*n <= vectors*lanes`, stopping at one vector.
Load the first half of the chosen vectors without masks and load the remaining vectors with their actual low-lane masks.
Inactive key lanes hold the key type's maximum sentinel, which is positive infinity for floating types and the actual finite maximum for integer types.
Inactive index lanes hold the maximum unsigned `size_t`, which is 64-bit on the supported hosts.
An integer padding sentinel must not be replaced by a fictitious value greater than the integer maximum, and no extra index tiebreak may be introduced when real keys equal that sentinel.
Masked gathers never read an inactive index and masked stores write only the original `n` index positions.

Sort each vector with the exact register network, then merge consecutive blocks of 2, 4, 8 and more vectors until the whole vector set is merged.
For a two-vector merge, reverse the upper vector, perform lane-wise compare/exchange with the lower vector, then reverse the upper vector back.
For larger merges, compare vector `i` with reversed vector `vectors-1-i`, restoring the upper vector's direction after comparison.
Clean the vector groups at successively halved strides, then apply the register half-cleaner to every vector.
These rules are `xss-network-keyvaluesort.hpp:126-280` and `:355-368`.

### Partitioning beyond 256

The pivot is the median selected from four or eight regularly spaced values addressed through the current permutation, using the source's lane ordering and median position.
Partition masks send keys strictly less than the pivot left and keys greater than or equal to the pivot right.
Before vector loading, process the fixed scalar remainder from the left, swapping a right-going index with the decremented right cursor and advancing the left cursor only for a left-going index.
The ordinary vector partition saves its first and last vectors before overwriting either end, and processes those saved vectors after all middle vectors.
The unrolled partition uses groups of four vectors, with its own remainder and four saved vectors from each end.
Its middle-loop load direction compares available overwrite space on both ends, rather than alternating ends or preserving original global row order.
Every loaded group is emitted in its original `ii` order, followed by the saved left group and then the saved right group.
Smallest/largest key reductions control whether either recursive side is omitted when all its keys equal the pivot.

The AVX2 and AVX512 group stores differ beyond lane count.
AVX512 compress-stores each mask group in increasing lane order.
The AVX2 lookup generator in `avx2-emu-funcs.hpp` appends left-going lanes from low to high and right-going lanes from high to low, then writes that complete rearranged vector at both store cursors.
A portable AVX2 step must preserve the reversed right-group order and the source's cursor updates and overwrite schedule.
Replacing both versions by the same stable two-way partition changes duplicate-key permutations.
The relevant paths are `xss-common-argsort.h:82-175`, `:181-295` and `:297-428`.

### Depth fallback and unresolved profile work

The source tests its remaining depth before its 256-element base case, so an exhausted recursive call uses `std::sort` even if that subproblem would otherwise fit a network.
Its initial counter uses twice the integer conversion of `log2(n)` and decrements once on each recursive edge.
The comparator passed to `std::sort` compares key values only, without an index tiebreak.
Rust's standard unstable sort and the NumPy scalar heap fallback are not equivalent replacements for that C++ library call.
A complete portable SIMD implementation therefore still needs the actual producer's C++ standard-library sort semantics, or a valid proof that the fallback cannot be reached over its declared input domain.
No such fallback-unreachability proof is claimed here.
Likewise, the current acquisition executes AVX2 and scalar paths but does not execute an AVX512 CPU path or establish equivalence across other NumPy versions and platform toolchains.
This checkpoint deliberately does not provide a partial SIMD dispatcher that silently uses the scalar algorithm for those unresolved paths.

## Remaining integration proof

1. Review and run this exact scalar candidate, including the existing temporal vectors and the 184 new native scalar controls.
2. Preserve the native AVX2 receipt as the expected answers for a source-faithful four-lane network and partition implementation, including all maximum-sentinel and unsigned controls.
3. Establish the depth fallback and required producer profiles before activating a whole SIMD route.
4. Cover the required NumPy/runtime cells and connect closed index input facts to filtered sampling, selected-subset monotonic decisions and final encoded context rows through both Python and rust-host.
5. Keep only the owner's approved user-defined sorting overrides on their existing route; ordinary built-in indexes retain their measured reference ordering.

The public frame/index representation, ordinary-provider behavior and final routing switch remain owned by their existing packages.
