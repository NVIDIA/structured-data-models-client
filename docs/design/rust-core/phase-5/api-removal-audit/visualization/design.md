<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 1. Schema visualization

**Status: implemented, independently reviewed and merged into `rust-core-phase-6`.**
Restore diagram generation and export on the maintained `Schema`, without loading data or changing sampling.
Independent boundary review is incorporated below; this document is the implementation contract.

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

## Implementation and review gate

First finalize the public signatures, result shapes, ownership and error behavior here; obtain coordinator contract review before implementation.
Add focused public-API and cross-language tests first where practical, then implement the reviewed behavior.
Use existing parity/snapshot tooling; do not hand-edit generated snapshots or introduce migration-only machinery.
Finish with independent code review, address findings, and run affected tests before merging into `rust-core-phase-6`.

Short formatting implementations in each language are acceptable; parity covers diagram meaning and escaping, not identical whitespace or notebook APIs.

## Reviewed contract

Python: `Schema.to_mermaid(show_columns: bool = True) -> str`, `to_html(show_columns: bool = True) -> str`, and `visualize(path: Path | str | io.BytesIO | None = None, show_columns: bool = True, height: int = 540) -> None`.
Java: `toMermaid()` / `toMermaid(boolean)`, `toHtml()` / `toHtml(boolean)`, and `visualize(Path)` / `visualize(Path, boolean)`; Java does not guess a display environment.
Both format existing metadata only, return fresh immutable strings, and neither mutate nor retain the schema.
Tables/columns/relationships retain their schema order, with generated safe diagram identifiers and escaped display labels; every composite-key component is marked once.
When `show_columns` is false, retain key, foreign-key and time columns; empty schemas still return valid Mermaid source.
Python requires an actual boolean and positive integer height, rejecting boolean heights; Java has static boolean types and no unused height option.
Unsupported/missing suffixes and invalid Python options raise `ValueError`/`TypeError`; Java uses `IllegalArgumentException`.
Local filesystem failures use native I/O exceptions; image-export HTTP failures use `RuntimeError` in Python and `IOException` in Java, with sensitive URL-bearing causes redacted.
Only explicit PNG/SVG or Python BytesIO export calls `https://mermaid.ink` with a bounded 30-second timeout; export failure does not silently return HTML or print a URL containing schema data.
Offline HTML contains the pinned renderer and strict security configuration, with source encoded as inert text and a sandboxed iframe for Python notebook display.
The single canonical renderer asset/template lives in Python package resources and is also included by Maven resource configuration, with source identity, SHA-256 and third-party notices checked in tests.

## Implementation notes

Host formatters emit flowchart table boxes, with explicit PK/FK/TIME markers and directed foreign-key edges labeled “many to one”.
Generated node identifiers and numeric entities keep unusual names inside labels, including script terminators and Mermaid syntax.
The shared strict-security Mermaid template uses HTML labels because the pinned renderer otherwise displays edge entities literally.
Python restores both IPython and Streamlit display, including restricted Streamlit fallback to source.
The shared hostile-name/composite-key fixture renders in Chromium for both Java and Python without network requests or injected script execution.
Renderer-browser tests are optional locally; the ordinary contract suite covers source/export/security behavior without a browser.
