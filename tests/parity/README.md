# Current client correctness and compatibility

This harness validates installed Python and Java clients against the maintained contract and compares their behavior on the same data, options and model version.
Local checks never certify live prediction parity.

| File | Purpose |
|---|---|
| `manifest.json` | Required scenarios, input paths, API mappings and any explicitly approved deferrals |
| `fixtures/shop.json` | Ordered rows, primary keys and relationships shared by both runners |
| `observation.schema.json` | JSONL record contract; one record per case/connector/model/language |
| `python_runner.py` | Public Python calls, including source verification and cleanup |
| `compare.py` | Current-contract validation, cross-language comparison and generated evidence matrix |

The shop rows are stored once in `fixtures/shop.json`.
Variants contain ordered additional user columns with literal values and optional types, or uppercase identifiers for Snowflake semantic views.
Python and Java expand the same data before provisioning, reading, verification and hashing.
Fixture hashes describe the expanded logical data, including nulls, numeric types, row order and column order; changes to the base also change each affected variant's hash.

## Credential-free contract checks

`manifest.json` is also the maintained API/capability ledger.
Public symbols map to API groups; those groups name required test methods, and model/connector scenarios link back to the groups.
Type records reuse the existing Arrow vectors and explicit schema tests.
Support declarations and local checks do not replace the live evidence matrix below.

```sh
python scripts/check_java_api.py --base origin/main
python scripts/snapshot_java_api.py --out java/target/public-api.json
python tests/parity/ledger.py --base origin/main --java-inventory java/target/public-api.json
python tests/parity/api_contracts.py --run --out python-api-results.json
python tests/parity/api_contracts.py --gate native --run --out python-type-results.json
python tests/parity/api_contracts.py --language rust --gate native --run --out rust-contract-results.json
```

Use the actual PR target for `--base`; `origin/main` above is the ordinary main-target example.
Use an installed candidate Python client with its test extras, JDK/Maven and the pinned Rust toolchain.
Java behavior tests use the existing `tests/parity/check_junit.py --run` command and its fresh Surefire reports; local Postgres is required for its database cases.
CI reuses its built wheel/native artifacts and existing Python, Java and Cargo jobs.
The separate compatibility job needs no native compilation, endpoint or private credentials.

For an API or capability change, update the symbol mapping, required test methods and applicable scenario requirements together.
Record a `contract_changes` entry with the target merge-base commit, changed record ID and reason.
An absent language implementation needs a specific `api_exceptions` reason; it cannot silently disappear.
Do not mark a capability passed just because its tests exist.

## Live evidence

### Required hosted profile

`parity-live.yml` runs after successful full CI on `main`, or by manual dispatch on an explicitly approved environment branch.
It reuses that commit's production wheel and JNI artifacts.
The broad connector matrix checks read and materialization behavior; it does not run a model for every cell.
Separate CPU smoke checks cover pretrained classification/regression gradients in both languages and Java session behavior.
The final gate requires both sets of results and identical artifact receipts.

| Resource | Hosted setup |
|---|---|
| PostgreSQL | Disposable local service and generated tables |
| S3 readers | Local Moto server, generated files and dummy credentials |
| Snowflake / Databricks | Dedicated protected API credentials; a unique schema created and removed per shard |
| Models | Public pretrained weight download, CPU only; no model endpoint credential |
| Real S3 / Lakebase / giant-value stress | Optional, outside the required hosted profile |

`hosted_profile.py` declares every connector's execution policy and rejects unclassified additions.
`hosted_runner.py` provisions fixtures and invokes both public clients.
`hosted_aggregate.py` checks complete results, cleanup receipts and the separate CPU smoke evidence.
Local or emulated results never certify real AWS or Lakebase behavior.
Ordinary PR CI remains credential-free; the warehouse workflow has no weekly schedule.

Warehouse setup uses the protected `parity-live` secrets `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `SNOWFLAKE_URL`, `SNOWFLAKE_USERNAME` and `SNOWFLAKE_ACCESS_TOKEN`.
Environment variables select writable test namespaces: `PARITY_DATABRICKS_CATALOG`, optional `PARITY_DATABRICKS_WAREHOUSE_ID`, `PARITY_SNOWFLAKE_DATABASE` and `PARITY_SNOWFLAKE_WAREHOUSE`.
Databricks Spark uses serverless compute.
Credentials and private driver logs are never uploaded as evidence.

The commands below remain available for explicitly configured full live-model validation.

Run all applicable Python cells by omitting filters, or repeat `--case`, `--connector` and `--model` to select a scope:

```sh
CANDIDATE_PYTHON tests/parity/python_runner.py --phase candidate --commit CANDIDATE_COMMIT --provenance WHEEL_PROVENANCE.json --mode live --config PRIVATE_CONFIG --out python.jsonl
```

The Java runner emits the same schema with `language=java`; concatenate its records with `python.jsonl` before comparison.
Python records must pass artifact, native identity, lifecycle and outcome checks; Java records must also agree with their actual Python peers.
Explicit manifest differences in refusal behavior are reported as language differences, never agreement.

```sh
python tests/parity/compare.py --comparison current-clients --candidate candidates.jsonl --candidate-commit CANDIDATE_COMMIT --out report.json --matrix matrix.md
```

Only current clients are executed and compared.
No frozen old-client checkout, historical equivalence waiver or baseline execution mode is required.
Independent expected outcomes remain necessary: agreement between Python and Java alone does not prove correctness.

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
The installed API gate checks generated-file integrity and public wire behavior without requiring a sibling source repository.
The two canonical-source provenance tests remain in the maintainer suite and explicitly skip when the pinned external contract checkout is unavailable; an installed API pass does not prove those source comparisons.
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
Raw-reader coverage is tracked in the [validation evidence](../../docs/development/validation.md) and the compatibility ledger; the mode’s presence alone is not passing live evidence.

`expected_by_model` records verified model-specific refusals, independently of candidate-only `expected_by_language` improvements.
Supported predictions and endpoint-stage refusals both require live endpoint identity.
Typed fixture declarations describe data before source creation; they never repair reader output.

When the target branch has no compatibility ledger or Java package yet, the checks report that introduction explicitly.
Current inventory and behavior requirements still apply; historical ledger/API comparisons begin once that baseline exists.
A missing or invalid Git reference remains an error.

## Executing the live API registry

This optional full live-model registry remains available for targeted/manual acceptance.
The required hosted workflow uses the materialization profile and bounded CPU smoke described above; it does not claim to execute this entire registry.
Both workflows share one repository-wide concurrency group, while the required profile permits two connector shards at a time.
They reuse production wheel, JAR and JNI artifacts from successful full CI for the exact candidate commit.
The matrix loads the exact JAR; focused Java API and giant tests compile the exact clean source commit and load the verified packaged JNI, recording source-equivalent execution.
Set the same `ci_run_id` dispatch input to pin identical bytes across reruns.
For the optional registry, use the installed candidate Python interpreter, its pretrained model runtime, the candidate JNI library and Maven:

```sh
CANDIDATE_PYTHON tests/parity/live_api.py \
  --config PRIVATE_CONFIG --commit CANDIDATE_COMMIT \
  --out EVIDENCE/live-api.json --private-dir PRIVATE_DIAGNOSTICS \
  --provenance WHEEL_PROVENANCE.json --java-provenance JAVA_PROVENANCE.json \
  --java-jar CANDIDATE.jar --java-native CANDIDATE_JNI_LIBRARY
```

`--preflight` checks configuration presence only; it never reports successful execution.
All source authentication, endpoint capabilities, pretrained model availability and actual assertions must still pass during execution.
The runner reuses `api_contracts.py --gate live` and `check_junit.py --gate live`, including fresh Java reports and exact Python parameterized invocation accounting.
Every live Java record requires one invocation by default; a parameterized method must declare its positive `invocations` count.
Missing methods, deselection, skipped tests, duplicate results and failed invocations fail acceptance.

The ordinary source/endpoint config is the default for `SDM_LABELS_WAREHOUSE_CONFIG`, `SDM_END_TIME_CONFIG`, `SDM_SQL_COLUMN_CONFIG` and `SDM_EXAMPLES_CONFIG`.
Existing environment variables can point these at separate fixture configurations.
Endpoint client URLs supply `SDM_RELATIONAL_URL` and `SDM_TABULAR_LIVE_URL` when unset.
The private config may additionally contain `live_api.environment`, with overrides for those variables plus `SDM_TEST_S3_READER_URI`, `SDM_GRADIENT_URL` and `SDM_RUN_LOCAL_MODEL`.
Set the last to `1` only with the actual pretrained CPU runtime available.
The private configuration must also supply `live_api.model_versions.pretrained` and `live_api.model_versions.gradients`, identifying the actual pretrained and gradient models.
Endpoint entries must supply their immutable `version` and `transport`.
The receipt retains these identities without endpoint URLs.
Execution verifies the installed Python wheel through the existing provenance checker and validates Java JAR/JNI hashes, contract version and clean candidate checkout before running Maven.
A gradient-capable endpoint and the pretrained Python runtime are separate requirements from ordinary model prediction endpoints.
The current fixture tests also require their declared source namespaces and values; configuration presence does not certify fixture compatibility.

Raw runner logs and detailed failure reports stay under the private diagnostics directory.
The public summary contains registry targets, execution status and the candidate commit, without private connection values or failure messages.
Keep private diagnostics outside any uploaded evidence directory.

Giant-value acceptance moved to the manually dispatched `parity-stress.yml` workflow.
It retains sequential Python/Java execution, explicit model identity and the 24 GiB available-RAM preflight.
Ordinary live parity no longer requires giant-value resources.
Publication requires successful ordinary-live and stress workflows on the exact release commit.
Live workflows do not run automatically on pull requests; manual dispatch requires a trusted branch allowed by the protected environment.
Dedicated credentials and branch policies are deployment settings, not committed secrets; see [recorded acceptance](../../docs/development/validation.md) for the tested configuration.

Private HTTP bridges retain `transport: private-bridge` rather than being relabeled as direct model endpoints.
For each such endpoint, `live_api.bridge_routes` must declare the matching `url`, `entity` and `version`, plus upstream `transport: databricks` and serving `endpoint`.
The sanitized receipt records a route digest and explicitly labels its basis `configured-route`.
This validates configuration consistency, not live model attestation.
Keep the separate observed process-route and downstream-version preflight evidence privately with the acceptance receipt.

The model identity receipt includes every configured primary endpoint, including specialized link or forecast models.
Each `endpoints_by_case` override must match exactly one recorded primary endpoint; the receipt retains the case-to-route mapping.
