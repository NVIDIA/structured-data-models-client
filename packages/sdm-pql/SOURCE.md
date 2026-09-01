# `sdm-pql`: provenance

This distribution ships a **verbatim copy** of the `predictive_query` package
from the Predictive Query Language repository. Not a fork, not a subset: the
same bytes, under the same import name.

## Baseline

| | |
|---|---|
| Upstream repository | `gitlab-master.nvidia.com/kumo/predictive-query-language` |
| Distribution | `predictive-query-language` |
| Import package | `predictive_query` |
| Version | `0.1.0` |
| Revision | `7ca9cb6673f2f3591000fa1416bf6ebeb4c9cf18` |
| License | Apache-2.0 |

## Changes made against that baseline

**None.** Every file is byte-identical to upstream, and
`VENDOR_MANIFEST.json` records a SHA-256 per file so that stays true.
`tests/test_vendored_tree.py` asserts it.

The import name is deliberately *not* renamed to `sdm_pql`. Renaming it would
make the eventual swap a source change in every consumer; leaving it means the
swap is one line of dependency metadata.

## Why this is vendored at all

Upstream is first-party NVIDIA code under a compatible license, but it has no
publish job yet — nothing consumes it from an index. Vendoring is the bridge.

## Replacing it with the real dependency

When `predictive-query-language` publishes:

1. Delete `packages/sdm-pql/` entirely.
2. Add `predictive-query-language==<version>` to `sdm-client`'s dependencies.
3. Delete this file and the manifest test.

No import changes anywhere, because the import name never differed.

**Do not install both.** Two distributions providing `predictive_query` will
clobber each other. `sdm-client` depends on exactly one of them.

## Rules that keep this honest

A repo-wide identifier rename matches any text, including text inside a
vendored file. That has already happened in this family of repos: a rename
rewrote two identifiers inside a vendored minified bundle, turning a "verbatim"
artifact into a modified one while the license still described it as
upstream's bytes. Six bytes in a 3.3 MB file, and no test caught it.

So:

1. `VENDOR_MANIFEST.json` pins a SHA-256 per file, asserted in CI.
2. A test asserts the tree carries no identifier this repo introduced —
   `sdm_api`, `sdm_client`, `sdm_relational_engine`, `sdm-client`. It is *not*
   a ban on the word `kumo`: upstream legitimately contains NVIDIA copyright
   and historical references. The assertion is "unchanged from upstream", not
   "contains no forbidden word".
3. Any repo-wide rename **excludes this path by configuration before it runs**.
   The manifest verifies afterwards; exclusion is what prevents the damage.
4. Bumping the vendored version means re-recording the manifest **and**
   re-reading upstream's own dependency set, because a version bump moves
   transitive versions too.
