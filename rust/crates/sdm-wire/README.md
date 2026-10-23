# sdm-wire

The model API's JSON, as typed Rust.
Modules: `json` (reads JSON keeping every number as written and every object in order), `payload` (`StatePayload`, `Task`, `PayloadSchema`, `InferenceConfig` and the other request types), `response` (`PredictionResponse`, `PredictionItem`, `Refused`).
`schema_json()` derives the payload's JSON Schema, which `sdm-contract`'s lock holds.
Depends on `sdm-contract` for the column types and on serde and schemars.
Must not hold: how a payload is laid out from sampled data (that is `sdm-request`), HTTP, or any choice about what to send.
