<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Typed settings and explicit explanation summaries

Restore convenient configuration and optional generated prose without moving host transport into Rust.

```text
Python / Java InferenceConfig → plain options → existing Rust validation → model request

Model explanation → ExplanationView → read existing prose locally
                              ↓ explicit summarize call only
                      configured text model → summary string
```

| API | Responsibility |
|---|---|
| Python `InferenceConfig(...)`; Java `InferenceConfig.builder()` | Immutable named fields; `to_dict()` / `toMap()` emits only supplied options. |
| Rust inference options | Existing defaults, valid values and task-specific refusal; unchanged. |
| Python / Java `ExplanationSummarizer` | One explicit HTTP call; caller names exact endpoint, model and optional key. |
| `summarize(view, query, prediction)` | Return existing prose without a call; otherwise return generated text without mutating the view. |

The seven fields are `num_estimators`, `column_shuffle`, `category_shuffle`, `hop_shuffle`, `class_shuffle`, `target_transforms`, and `output_type`.
Unset fields remain absent; no duplicate host defaults or old run-mode override is introduced.
Existing mapping configuration remains supported.

The summary request uses the chat-completions message shape, with a fixed instruction to describe evidence without claiming causality and a JSON message containing query, prediction and explanation.
This explicitly sends the explanation's raw values to the configured endpoint.
No environment fallback, automatic prediction-time call, retry, or default remote endpoint is added.
Bad input, transport failure and malformed/empty response raise structured errors without including credentials or response bodies.

Tests exercise both public host APIs, typed/mapping equivalence through Rust, copied inputs and invalid task options.
Local HTTP servers check request/auth details, successful prose, existing-prose no-call behavior, timeout/status errors and malformed responses.
Real text-model validation is separate and requires an existing authorized endpoint; current relational/tabular endpoint configuration does not establish one.

## Usage

```python
config = InferenceConfig(num_estimators=2, output_type='mean')
request = query.materialize(indices=[1], model=KumoRelational(), inference_config=config)
# The supplied-example equivalent is schema.materialize_examples(..., inference=config).
summary = ExplanationSummarizer(endpoint, model, api_key=key).summarize(
    result.explanation_view, query_text, result.to_pandas().to_dict('records')
)
```

Java uses `MaterializeOptions.builder().inferenceConfig(config)` or `ExamplesOptions.builder().inferenceConfig(config)`.
`new ExplanationSummarizer(endpoint, model)` needs no credentials for an unauthenticated endpoint; optional key and timeout constructors are available.
No remote text endpoint was configured in the available model validation configuration, so live prose quality is not claimed.
The HTTP contract is verified locally with synthetic data.

## Verification

Production native contract 65: 22 new Python tests and six Java tests pass.
The neighboring Python supplied-example suite also passes (15 local tests; nine credentialed live cases skipped).
Ruff and targeted mypy pass; the coordinator regenerates the combined API ledger/snapshots after merging all three efforts.
Compared with the old implicit optional fallback, generation is now explicitly requested, does not use ambient credentials, and reports provider failures as errors rather than misleading summary text.
