# TFM API validation plan

This plan lists the minimum validations needed to keep
`scripts/generate_sdm_api.py` at the repository root, the generated `sdm_api/generated/tfm_api.py`
models, and the handwritten client adapters aligned with the canonical OpenAPI spec
from the `structured-data-api` repo at `../structured-data-api/nim-sd.openapi.yaml`.

## Why this exists

The client intentionally uses a small first-party generator instead of a full
OpenAPI client generator. Endpoint metadata, enum constants, and schema names are
derived from the spec, but `PredictionItem` and `PredictionResponse` are still
hand-rendered by `_response_model_lines()`.

That leaves a few drift surfaces that tests must pin down:

- response model fields vs `PredictionItem` / `PredictionResponse` schemas
- parser behavior vs documented response examples
- `sdm_relational_engine/client/rfm.py` row mapping vs parsed response fields
- `sdm_relational_engine/rfm/payload.py` request envelope vs `PredictionRequest`
- generated output constants vs `OutputSpec.fields.items.enum`

## Matching policy

Use strict field parity for generated response models:

- `PredictionItem` dataclass fields must equal spec `properties`.
- `PredictionResponse` dataclass fields must equal spec `properties`.
- Unknown `PredictionItem` wire fields may still be ignored at parse time because
  the spec allows `additionalProperties: true`.

This gives a clear failure when the spec adds, removes, or renames a modeled
response field and the client has not been updated.

## Must-have validations

### 1. Response schema parity

Add a spec-driven test in `test/client/test_tfm_codegen.py` that loads
`nim-sd.openapi.yaml` and asserts:

- `dataclasses.fields(PredictionItem)` equals
  `components.schemas.PredictionItem.properties`
- `dataclasses.fields(PredictionResponse)` equals
  `components.schemas.PredictionResponse.properties`
- each schema's `required` fields are represented by the dataclass
- the prediction operation still returns `PredictionResponse`

Keep the local checkout skip behavior for developer machines, but make sure CI
runs this with the canonical spec available.

### 2. Parse documented response examples

Replace or extend the handwritten parser fixture with examples from:

`paths./v1/predictions.post.responses.200.content.application/json.examples`

For each example:

- run `PredictionResponse.from_dict(example["value"])`
- assert required top-level fields are present
- assert key coercions still hold, such as stringified item ids, float
  probability values, tuple embeddings/scores, and dict explanations/metadata

This catches documented wire JSON that the client can no longer parse.

### 3. Adapter field coverage

Add a focused test for `_prediction_item_to_row()` using a `PredictionItem` with
every optional field populated. Assert the legacy row shape is intentional:

- `id` maps to `ENTITY`, including integer coercion when possible
- `probabilities` maps to `{name}_PROB`
- `quantiles` maps to `q_{name}`
- `scores`, `rankings`, `embeddings`, and `explanation` are preserved
- `metadata` is explicitly documented as parsed but not mapped

Also assert that any future parsed `PredictionItem` field is either mapped here
or listed as intentionally unmapped.

### 4. Request envelope parity

Add request-side coverage for `sdm_relational_engine/rfm/payload.py`, which is handwritten and
currently outside the response-model checks:

- compare the keys emitted by `predict_request_to_json()` / `_base_payload()` to
  `PredictionRequest.properties`
- assert all `PredictionRequest.required` keys are emitted
- parse the spec's request examples for Kumo Tabular and Kumo Relational and assert their
  envelope shape matches the client's expected top-level shape

This is the largest remaining blind spot once response parsing is pinned down.

### 5. Output enum parity

Assert `TFM_OUTPUT_FIELD_VALUES` exactly matches
`OutputSpec.fields.items.enum`.

`explanation` is already present in the canonical spec, so
`PROPOSAL_OUTPUT_FIELDS = ('explanation',)` should be removed rather than kept as
a permanent generator override. If any temporary override is reintroduced later,
it should have an explicit allowlist and a test that fails once the spec no
longer needs it.

## Useful generator guard

After the tests above exist, add a lightweight `--validate-contract` mode to
`scripts/generate_tfm_api.py` that runs the same spec comparisons without
rewriting files. It should fail with a readable diff for:

- response dataclass fields vs spec properties
- output enum constants vs spec enum
- the prediction operation response schema not being `PredictionResponse`

This is useful for local codegen and CI, but it should reuse the same comparison
logic as the tests.

## Not required for the first pass

Defer these until the core drift checks are in place:

- JSON Schema validation of every example
- property-based fuzzing for `from_dict`
- moving response field definitions into a separate shared table
