# Public client parity

This harness validates installed Python and Java clients against the maintained contract and compares their behavior on the same data, options and model version.
Local checks never certify live prediction parity.

| File | Purpose |
|---|---|
| `manifest.json` | Required scenarios, input paths, API mappings and any explicitly approved deferrals |
| `fixtures/shop.json` | Ordered rows, primary keys and relationships shared by both runners |
| `observation.schema.json` | JSONL record contract; one record per case/connector/model/language |
| `python_runner.py` | Public Python calls, including source verification and cleanup |
| `routes.py` | Legacy-policy traps around setup, calls, cleanup and worker threads |
| `compare.py` | Current-contract validation, cross-language comparison and generated evidence matrix |

## Credential-free contract checks

`manifest.json` is also the maintained API/capability ledger.
Public symbols map to API groups; those groups name required test methods, and model/connector scenarios link back to the groups.
Type records reuse the existing Arrow vectors and explicit schema tests.
Support declarations and local checks do not replace the live evidence matrix below.

```sh
python scripts/check_java_api.py --base origin/rust-core
python scripts/snapshot_java_api.py --out java/target/public-api.json
python tests/parity/ledger.py --base origin/rust-core --java-inventory java/target/public-api.json
python tests/parity/api_contracts.py --run --out python-api-results.json
python tests/parity/api_contracts.py --gate native --run --out python-type-results.json
python tests/parity/api_contracts.py --language rust --gate native --run --out rust-contract-results.json
```

Use an installed candidate Python client with its test extras, JDK/Maven and the pinned Rust toolchain.
Java behavior tests use the existing `tests/parity/check_junit.py --run` command and its fresh Surefire reports; local Postgres is required for its database cases.
CI reuses its built wheel/native artifacts and existing Python, Java and Cargo jobs.
The separate compatibility job needs no native compilation, endpoint or private credentials.

For an API or capability change, update the symbol mapping, required test methods and applicable scenario requirements together.
Record a `contract_changes` entry with the target merge-base commit, changed record ID and reason.
An absent language implementation needs a specific `api_exceptions` reason; it cannot silently disappear.
Do not mark a capability passed just because its tests exist.

## Live evidence

Run all applicable Python cells by omitting filters, or repeat `--case`, `--connector` and `--model` to select a scope:

```sh
CANDIDATE_PYTHON tests/parity/python_runner.py --phase candidate --commit CANDIDATE_COMMIT --provenance WHEEL_PROVENANCE.json --mode live --config PRIVATE_CONFIG --out python.jsonl
```

The Java runner emits the same schema with `language=java`; concatenate its records with `python.jsonl` before comparison.
Python records must pass artifact, route, lifecycle and outcome checks; Java records must also agree with their actual Python peers.
Explicit manifest differences in refusal behavior are reported as language differences, never agreement.

```sh
python tests/parity/compare.py --comparison current-clients --candidate candidates.jsonl --candidate-commit CANDIDATE_COMMIT --out report.json --matrix matrix.md
```

Historical old-client comparison is an effort 3 investigation, using the preserved reference and recipes.
The optional `--comparison historical --baseline baseline.jsonl` mode keeps that distinction explicit; routine CI installs only current clients.

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
Raw-reader coverage is tracked in the [verification tracker](../../docs/design/rust-core/phase-5/verification.md) and the compatibility ledger; the mode’s presence alone is not passing live evidence.

`expected_by_model` records verified model-specific refusals, independently of candidate-only `expected_by_language` improvements.
Supported predictions and endpoint-stage refusals both require live endpoint identity.
Typed fixture declarations describe data before source creation; they never repair reader output.

When the target branch has no compatibility ledger or Java package yet, the checks report that introduction explicitly.
Current inventory and behavior requirements still apply; historical ledger/API comparisons begin once that baseline exists.
A missing or invalid Git reference remains an error.
