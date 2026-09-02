# Vendored assets

## mermaid.min.js

- Project: https://github.com/mermaid-js/mermaid
- Version: 11.15.0 (browser IIFE bundle, `dist/mermaid.min.js`)
- Source: https://cdn.jsdelivr.net/npm/mermaid@11.15.0/dist/mermaid.min.js
- License: MIT (see `MERMAID_LICENSE`)
- SHA-256: `70137e77bb273bb2ef972b86e8b0400cca8be53cb25bfc45911a186dc98665de`

Bundled so that `Graph.visualize()` renders entirely offline: no system
executables (graphviz `dot`), no CDN fetch, and no calls to the mermaid.ink
web service are needed to display a graph in a notebook. This matters for
restricted notebook environments (Snowflake, Databricks) whose content
security policies or egress rules block external scripts and services.

The bundle is upstream bytes, unmodified. The SHA-256 above is the digest of
the file as published, and `test_vendored_mermaid.py` re-checks it. That test
exists because a repo-wide identifier rename once rewrote two strings inside
this file, which silently turned a vendored artifact into a modified one and
invalidated the third-party attribution built from upstream's manifest.

To upgrade: replace `mermaid.min.js` and `MERMAID_LICENSE` from the same
jsdelivr path at the new version, update the version and SHA-256 here, re-run
the visualization tests, and re-enumerate the bundled components in the root
`LICENSE` from the new version's package manifest.
