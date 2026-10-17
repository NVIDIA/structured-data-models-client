# T2.1a: the S1 judging path

Lane f (moved from lane A), on the phase head 5a33dd7d (T2.0c's merge of `main`); the test author's and the implementation commits were cherry-picked onto it in order.
The acceptance tests are the test author's (`T2.1a tests:` commits, 7290ef3d to 8fa90a57); each implementation commit removes the PENDING marks of the tests it makes pass.

## What changed

- **The case format's database axis** (choice 2): `databases`, a non-empty list of distinct names of `duckdb`, `sqlite` and `postgres`, only beside `exclusion`, a decision row `KO<p>-<n>` (the schema's `dependentRequired`; the Rust loader's own check).
  `input.materialize` gains `candidates` (entity keys read as indices are) and `top_k` (a JSON integer written as one).
  `run_python.case_databases` and `sdm-testkit`'s `Case::databases` give a case's databases in order.
  The single-field vectors are in `contract/tests/malformed-cases.json`, which `s0_cases.rs` runs; that test now sets each change's value as the file writes it, since `serde_json::Value` loses an integer past 2**64.
- **The contract loader and the type map** (`contract/runners/databases.py`): a case's DuckDB tables loaded through Arrow into SQLite (a new file) or Postgres (a DSN's current schema), each column as the map says, keys and `NOT NULL` carried as DuckDB declares them, rows in DuckDB's scan order.
  The map (`TYPE_MAP`, the same as `contract/type-map-vectors.json`'s `map`) uses an engine's own type wherever it has one and names the stored form otherwise (`sqlite_value`, `postgres_value`); a value the target cannot hold exactly is refused by name, never rounded.
  `canonical_decimal_text` is implemented in the loader and in `sdm-data` (`canonical_decimal_text(value, scale)`, the text a SQLite key binder binds a decimal key as), each held to the vector.
- **The scripted endpoint** numbers its calls from zero and fails a chosen one (`serve(fail_call=)`, `answer(..., call=, fail_call=)`; Rust `Endpoint::new(fail_call)` and `answer_call`).
- **The temporary-approval rule**: `register.py`'s `TEMPORARY` names KO2-45 and KO2-50, and `ratchet.py check --locked` (which `gate.sh --locked` passes) fails naming each approved entry citing one.
- **The staging rule**: `run_python.check_legs` (and `run_python.py legs`), over plain-data markers: `sdm_data::PROVIDERS`, reported as `sdm_client._native.PROVIDERS`, and `sdm_testkit::PROVIDERS`, printed by `sdm-contract-run --providers`.
- **Runner and comparator on three databases**: `run_python.py --database` runs a case on a loaded SQLite file or a new Postgres database (its public schema), through the driver capture (ADBC's SQLite and psycopg, each under its driver's package), keyed `<case>@<database>`; `compare.py --database` compares that database's lines, with the reference's line there as the expectation.
- **CI**: `contract-python (<cell>, <database>)`, nine jobs; the SQLite and Postgres ones run the reference side, the core legs where a provider is reported, the staging rule and the comparator; one shared install step brings the SQLite and Postgres drivers; the Postgres jobs and `rust-test` start the pinned service and require it.
- **The refusal census** (T2.0d, folded in): `contract/tools/refusal_census.py` (`sites`, `check`, `run`), its probe list `contract/refusal-census.json`, and its first DuckDB run, `evidence/census/duckdb.json`.

## Probes

- Every S1 case's tables load on SQLite and on a local postgres:17 through the loader.
- The reference (python-ref, 6298913c's environment) ran all 52 S1 cases on SQLite and on Postgres, every capture complete; compared with `--database` against the candidate's old client, what differs is the pre-merge client against the new reference (main's `inference_config` and sampling), which T2.0c's merge removes.
- With a random Postgres schema per case, every `from_source` differed between two runs (`SELECT current_schema()`); each case now gets a database of its own, its tables in `public`.
- The census, run on the phase head after T2.0c with a clean tree: 71 sites, 39 probes reaching their site on python-core, 23 sites where the core refuses what the reference answers (all of E92's class).
  T2.0c's exact decimals lifted the decimal static target the `target_type` probe first used; its probe now sums a UBIGINT column past int64.
  Not reached, each said in its probe: sites the query validator shields on both routes, answers the scripted endpoint never gives, provider declarations DuckDB's provider makes, and host checks a case input cannot reach.
  Two probes show the core answering `PROVIDER_ERROR` where the reference fails with `INTERNAL_ERROR` (a text time column with a value outside the formats, and one with day and month order open).

## Deviations and what is left

- The reference lines `<case>@sqlite` and `<case>@postgres` are recorded by the recorder (test-author role) into `contract/reference/6298913c...jsonl` with this tooling; `tests/test_t21a_reference_lines.py` stays PENDING, and the SQLite and Postgres CI jobs fail, until those lines land.
- Moved to T2.1c by the owner's lean plan, not built here: `coverage.py --gate S1` (G12, G16, G17), the L3 wiring over three databases, the `coverage-s1`, `l3` and `ci` step tools and `--ci-run`, `gate.sh` running the three databases, the common-input comparison (H24), C6 and C18.
- `contract/tools/type_map_record.py` writes the decimal entry of `map` only; the full map is the loader's `TYPE_MAP`, which a loader test holds equal to the file.
- The floor cell's SQLite and Postgres jobs take T2.0c's pins (pandas 2.2.2, NumPy 1.23.5).
