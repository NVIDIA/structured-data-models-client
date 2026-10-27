# Native Text profiles

The verified categorical frame consumer does not yet admit the whole Text semantic family.
This follow-up keeps normalization in Rust while representing the native string kernel selected at the actual normalization boundary.
The versioned casing kernel and its data are foundations for the required staged consumer; their presence alone does not complete the public Text journey.
The default SDK route remains unchanged.

## Actual public observations

The full frozen profile and boundary receipts are in the client tests as t44_native_text_profile_*_reference.json and t44_native_text_boundaries_*_reference.json.
They record complete public request bodies, results, warnings, native casts, argument mutations and source/native identities.
The pandas 2 cell's astype(str) produced a NumPy-backed result and Python full lowercase regardless of the tested source string storage and target storage option.
The pandas 3 cell's effective target storage selected Python full lowercase or Arrow simple lowercase; disabling string inference also selected the NumPy/Python path.
Source storage and pandas major version are therefore insufficient kernel facts.

The frozen Unicode examples distinguish final sigma and dotted-I expansion, and prove that the 300-character cut precedes lowercase.
For example, full lowercase turns ΟΣ into ος and İ into i followed by U+0307, whereas the simple kernel produces οσ and i.
The cut counts Unicode scalars, including multibyte text, rather than UTF-8 bytes or the expanded lowercase output.
Null and empty values produce no words.
Existing punctuation replacement, HTML-break replacement, whitespace splitting and the 50-word limit are shared with the provider kernel.

A HOPS_IGNORED handler changing target storage affects the current body.
A later context-answer notice changing storage affects the next cached materialization and leaves the current body unchanged.
A graph-drop handler can mutate the caller's indices before native task construction.
A HOPS_IGNORED handler can mutate the caller's neighbor list and inference config before those values are consumed.
The sampling-error control reaches the native sampler and never reaches Text normalization.
Two batches have separate sample-then-normalize sequences.
These observations require useful graph, task, sampled-context and payload boundaries rather than an early cached profile or a language-behavior callback.

## Versioned Rust kernel

words::normalize retains the accepted provider behavior.
words::normalize_with_profile shares its bounded post-lowercase pipeline and uses explicit simple or full-root casing over pinned data.
The data profiles are Unicode 13.0, 14.0, 15.0, 15.1, 16.0 and 17.0.
The full-root algorithm applies unconditional full lowercase and the standard Final_Sigma context rule over the original clipped input.
It skips Case_Ignorable characters before testing Cased context and computes the suffix once, so the bounded pass is linear in the input length.
Locale-specific Turkish, Azeri and Lithuanian rules are excluded by root lowercase semantics.
The generator rejects any unknown language-independent context rule rather than silently omitting it.

scripts/rust_core/generate_text_unicode.py consumes exact pinned official UnicodeData, SpecialCasing and DerivedCoreProperties inputs.
It derives mappings and case-property ranges, deduplicates equal tables and emits one simple base table with exact historical differences.
All declared databases have the same 29-character Python whitespace predicate, which the generator verifies before emitting one shared predicate.
This avoids selecting historical behavior through the Rust toolchain's current Unicode tables.
The generator records input URLs and hashes, generator/helper hashes, output hashes and the Unicode-3.0 license in rust/crates/sdm-engine/data/CASING-PROVENANCE.json.
The generated data and license are writer-owned and must never be edited manually.
The first-party Rust algorithm contains no vendored third-party implementation code.

Regenerate with pinned local inputs:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/rust_core/generate_text_unicode.py --inputs /path/to/pinned-unicode-inputs --write
python3 scripts/rust_core/rust_notices.py --write
```

## Portable kernel identity

The actual frozen Arrow 25 library's official utf8proc metadata reports Unicode 17, although its stock source dependency pin would suggest another database.
Package numbers cannot identify the actual native casing table.
The portable identity receipts record a fixed public Arrow utf8_lower operation over U+0130, U+03A3, U+2C2F, U+1C89 and U+A7CE.
No user column or value participates in this observation.
The complete ordinary and batched public bodies, results, warnings and sampler traces match their unobserved baselines exactly, and the earlier sampling failure produces zero observations.
The options before and after the fixed observation are equal in both cells.

Rust resolves the raw identity output against generated mapping classes.
The complete simple maps for Unicode 14, 15 and 15.1 are identical and deliberately share one representative class.
The generator proves that two distinct declared complete simple tables never collide on the fixed identity.
It also verifies that equal complete tables have equal signatures.
Five sentinels do not certify an arbitrary modified kernel over every scalar; this is a bounded native identity within the declared data profiles.
An unknown-signature public policy is not established here, and no new refusal or guessed-version fallback is introduced.

## Useful stages and retained state

The proposed graph operation constructs or rebinds the actual immutable frame snapshot and emits graph-drop notices before the caller's indices are converted for its task.
The task operation produces complete task rows, labels, anchors, narrowing and batch meaning before pre-sampling option notices.
The sampled-context operation consumes the late native neighbor/config observations, performs existing lookup and graph walks, and returns actual selected typed tables, links and row maps.
The payload operation consumes those useful context values plus the normalization-time native kernel facts, performs Rust normalization and reuses the existing request encoder.
Each is a fixed operation that returns a useful domain artifact through Done.
There is no new Step, provider Fetch for native behavior, opaque suspended program or normalized host column.

Input-sized rows, labels, anchors, selections and links remain typed attachments.
Core-owned metadata and attachment manifests are bound to the graph's logical fingerprint, independent of call-local attachment ids.
The cached graph retains its original source rows and schema meanings while the existing neighbor stream remains authoritative across success, error and cancellation.
A task or sampled context cannot silently rebind to another logical graph.
The native kernel is observed per reached payload boundary and is not permanent checkpoint semantics.
The staged protocol and facade wiring remain pending the serialized internal contract update.

Native warning exceptions also have a source boundary.
Graph-drop notices are outside the original engine-error guard and propagate their warning exception directly.
HOPS_IGNORED is inside that guard, so warning-as-error becomes the exact INTERNAL_ERROR wrapper and consumes no sampler draws.
The staged consumer must preserve that explicit presentation context while retaining the registered P3E-2 code prefix and provider behavior.
The frozen retry proves the successfully built graph survives that failure and later caller frame mutations do not replace its retained rows.

Text identifying keys, exact native Arrow LargeUTF8 behavior, and the other unfinished frame families remain required Phase 4 work.
Their native conversion and failure-order obligations are not inferred from the casing kernel or these nonkey Text receipts.
