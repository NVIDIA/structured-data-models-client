<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 1. Schema visualization

**Status: design only. Support both Python and Java; notebook integration is Python-specific.**
Restore diagram generation and export on the maintained `Schema`, without loading data or changing sampling.
Phase 6 boundary review: [decision and evidence](../../../phase-6/boundary-review.md).

```text
Python Schema / Java Schema
           ↓ existing schema metadata
Python / Java: Mermaid text + shared HTML/renderer assets
           ↓
Python / Java: standalone HTML, file export
Python only: inline notebook display
```

| Layer | Responsibility |
|---|---|
| Python / Java formatter | Mermaid text from existing schema metadata; safe identifiers/labels, composite keys and column visibility |
| Rust | Existing schema meaning and validation only; no new rendering operation or retained display state |
| Python / Java | File writes, HTML resource loading, optional image-export HTTP calls and errors |
| Python only | Jupyter/Databricks notebook display and terminal fallback |

## Proposed user API

- Both: `Schema.to_mermaid()` / `toMermaid()` and `to_html()` / `toHtml()`.
- Python: `schema.visualize(path=None, show_columns=True, height=540)` for the old display/export workflow.
- Java: explicit file export; return HTML for embedding instead of guessing a notebook/browser environment.

Keep HTML self-contained using bundled Mermaid; restore license notices and asset-security checks.
Use one version of the renderer/template for both distributions, not independently maintained copies.
Preserve `.mmd`, `.html`, `.png` and `.svg` export; PNG/SVG uses an explicitly documented network renderer as the old client did.
Only explicit image-export requests may send schema text to that service; HTML/source generation stays offline.
No model endpoint, warehouse query or browser process is needed to generate Mermaid/HTML.

## Acceptance

Shared fixtures must verify equivalent diagram content in both languages, including empty schemas, quoted/unusual names and composite keys.
Each formatter must be deterministic; incidental whitespace need not be identical.
Test that untrusted names cannot inject script into HTML, offline HTML contains its renderer, and export dispatch/errors work.
Python notebook display gets a host-specific test; both languages get file-export tests.
Record notebook display as intentionally Python-specific in the existing parity manifest.
