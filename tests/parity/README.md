# Public client parity

This harness compares installed Python and Java clients against frozen Python `43e55f4f5` using the same data, options and model version.
Local checks never certify live prediction parity.

| File | Purpose |
|---|---|
| `manifest.json` | Required scenarios, input paths, API mappings and any explicitly approved deferrals |
| `fixtures/shop.json` | Ordered rows, primary keys and relationships shared by both runners |
| `observation.schema.json` | JSONL record contract; one record per case/connector/model/language |
| `python_runner.py` | Public Python calls, including source verification and cleanup |
| `routes.py` | Legacy-policy traps around setup, calls, cleanup and worker threads |
| `compare.py` | Frozen-reference comparison and generated evidence matrix |

Run all applicable Python cells by omitting filters, or repeat `--case`, `--connector` and `--model` to select a scope:

```sh
BASELINE_PYTHON tests/parity/python_runner.py --phase baseline --commit 43e55f4f58924384c3cb52f67c8ed2e73ba45fa1 --mode live --config PRIVATE_CONFIG --out baseline.jsonl
CANDIDATE_PYTHON tests/parity/python_runner.py --phase candidate --commit CANDIDATE_COMMIT --provenance WHEEL_PROVENANCE.json --mode live --config PRIVATE_CONFIG --out python.jsonl
```

The Java runner emits the same schema with `language=java`; concatenate its records with `python.jsonl` before comparison.
The frozen Python observation is reused for Java, rather than comparing Java only to the new Python implementation.

```sh
python tests/parity/compare.py --baseline baseline.jsonl --candidate candidates.jsonl --candidate-commit CANDIDATE_COMMIT --out report.json --matrix matrix.md
```

`--local` compares local materialization evidence only.
Comparison filters also accept repeated `--case`, `--connector`, `--model` and `--language`.
The exit status reports the requested scope; `goal_complete` remains false when any applicable cell is missing or only locally checked.
No credentials, absent test, skipped journey or unobserved child process can create a passing cell.

Private configuration has two independent sections:

```json
{
  "sources": {
    "postgres": {
      "connect": {},
      "source": {"schema": "fixture_schema"},
      "tables": {"users": "users", "events": "events", "items": "items"},
      "snapshot": "fixture-version"
    }
  },
  "endpoints": {
    "relational": {
      "kind": "live",
      "version": "immutable-model-version",
      "transport": "nim",
      "client": {"url": "https://model.example"}
    }
  }
}
```

`connect` and `source` are keyword arguments for the existing public driver and source constructors.
Databricks serving uses `transport=databricks` and its public client constructor arguments.
Semantic sources additionally name `view`; S3 readers use a `paths` mapping.
Remote fixtures are provisioned by the caller; the runner reads them and verifies values against the shared fixture before making predictions.
It creates and cleans up only local temporary fixtures and fitted sessions.

Tabular warehouse input uses `read_table` / `Readers.readTable` followed by a one-table frame schema.
`Schema.from_source` / `Schema.fromSource` remains the Relational path.

The manifest expresses expected coverage, not evidence of success.
Raw request payloads remain in observations; any accepted representation difference needs a narrowly tested semantic comparison, never ignored rows or columns.
Candidate records must identify the requested commit and one consistent installed package/native artifact per language.

API-only behavior gates are mapped in `manifest.json` under `api_contracts`, separately from the prediction matrix.
`python tests/parity/api_contracts.py --language python --gate local --run --out api-report.json` runs those existing public tests; missing or skipped coverage stays incomplete.
The registry pins the exported API snapshot so API additions require an explicit coverage update.
Candidate provenance must match the installed wheel, native library, source commit and clean-build sidecar.

The current manifest requires exact numeric equality.
Any future model tolerance must name prediction columns explicitly while keeping identities, timestamps and inputs exact.

SQL fixture inspection uses public `read_table(query=...)` with explicitly quoted table identifiers so case-sensitive fixture names survive unchanged.
The separate reader API gate retains ordinary `table=` coverage; source-backed schema construction still receives bare table names.

Six memory-only supplemental cells verify requested prediction/quantile/probability outputs, two-call batching and fitted holdout removal from instance and related rows.
They retain public result items, metadata and explanations alongside actual HTTP request bodies and response IDs.
Only generated request nonces, response IDs and session handles with proven request metadata, response grouping and actual session correlations become per-call labels for comparison.
All other metadata and entity IDs remain exact.
Session scoring uses the existing graph identity proof only with the actual captured fit schema, verified unchanged from the original request.

New scenarios may set `input_mode: raw-reader` to pass public reader tables directly into frame schemas in both languages.
This bypasses fixture casts, sorting and Java Arrow reconstruction, and uses the read tables for Relational too.
Targeted provider cases additionally set `schema_input: source` to exercise warehouse materialization; their inspection reads remain unchanged.
Snapshot verification may inspect sorted copies, but must never replace the model input.
Existing scenarios retain their prepared-input behavior and historical evidence.
The raw-reader mode is being validated under [the type parity plan](../../docs/design/rust-core/phase-4/type-parity-plan.md); its presence alone is not passing live evidence.

`expected_by_model` records verified model-specific refusals, independently of candidate-only `expected_by_language` improvements.
Supported predictions and endpoint-stage refusals both require live endpoint identity.
Typed fixture declarations describe data before source creation; they never repair reader output.
