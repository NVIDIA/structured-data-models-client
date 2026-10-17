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
- The reference (python-ref, 6298913c's environment) ran all 52 S1 cases on SQLite and on Postgres, every capture complete.
  Compared with `--database` against the candidate's old client, the SQLite lines differ in sampling: the reference holds main's `f0f324ce` (seeding SQLite's entity sample) and `d1590a17`, which the phase branch's old route lacks (T2.0c merged `main` at 327fda03, before them), so python-old samples SQLite unseeded; T2.0c's merge does not remove it.
  From fix 4 python-old runs on DuckDB only, and the SQLite and Postgres jobs hold python-ref's committed lines (the orchestrator's decision).
- With a random Postgres schema per case, every `from_source` differed between two runs (`SELECT current_schema()`); each case now gets a database of its own, its tables in `public`.
- The census, run on the phase head after T2.0c with a clean tree: 71 sites, each with a probe written for it (its intended site), both routes' outcomes recorded, 23 sites whose probe the core refuses where the reference answers (all of E92's class).
  Which site produced an outcome is not claimed: reachability stays unmeasured until a real-data run or instrumentation needs it (the orchestrator's stop rule at fix 2, after attribution from outcomes failed review twice).
  T2.0c's exact decimals lifted the decimal static target the `target_type` probe first used; its probe now sums a UBIGINT column past int64.
  Where the probe's author knows the public calls do not get to the site that way, its `why` says so: the query validator shields it on both routes, the scripted endpoint never gives the answer, DuckDB's provider makes the declaration, or a case input cannot carry the host's input.
  Two probes show the core answering `PROVIDER_ERROR` where the reference fails with `INTERNAL_ERROR` (a text time column with a value outside the formats, and one with day and month order open).

## Deviations and what is left

- The reference lines `<case>@sqlite` and `<case>@postgres` for all 86 cases are recorded by the recorder (test-author role) into `contract/reference/6298913c...jsonl` (391d3e04, PENDING marks removed there); fix 4's query logs and named Postgres types need them re-recorded, which the recorder does.
- The SQLite and Postgres `contract-python` jobs exist in the reference cell only until T2.2b and T2.3b bring the core legs and the other cells; the locked set's patterns for the other cells wait for them.
- Moved to T2.1c by the owner's lean plan, not built here: `coverage.py --gate S1` (G12, G16, G17), the L3 wiring over three databases, the `coverage-s1`, `l3` and `ci` step tools and `--ci-run`, `gate.sh` running the three databases, the common-input comparison (H24), C6 and C18.
- `contract/tools/type_map_record.py` writes the decimal entry of `map` only; the full map is the loader's `TYPE_MAP`, which a loader test holds equal to the file.
- The floor cell's SQLite and Postgres jobs take T2.0c's pins (pandas 2.2.2, NumPy 1.23.5).

## Fix rounds

- Fix 1 (round 1, both reviewers): a DuckDB line needs its query log; the loader keeps a NULL list's neighbours, an empty map, a list of dates, nested Postgres composites, and refuses by name a NaN SQLite would store as NULL, a type Arrow does not carry whole (TIMETZ, BIT), a time past what it writes and text with a NUL on Postgres; the census's reach needs a refusal only its site gives; the Rust loader takes `top_k` of any size; an exclusion must leave a database out; `PROVIDERS` is in the native stub.
  The temporary-approval rule holds the slice gate's locked run only (a Phase 2 half or later with `--wheel-run`, as `final_gate.sh` runs it; orchestrator's decision), so Phase 1's halves and the gate's own controls run as before.
  The S0 cases run on SQLite and Postgres too, and the recorder records all of them (orchestrator's decision).
- rust-host's marker (`sdm-contract-run --providers`) is read by no CI step yet: rust-host runs no SQLite or Postgres pass before T2.2b and T2.3b, and the `ci` step that reads it is T2.1c's.
- Fix 2 (round 2, Codex): the census records each probe against its intended site and claims no reach (stop rule); the loader reads lost types from DuckDB's type tree, so a field named `BIT` is a name, and refuses a NUL inside a map's or union's text on Postgres before COPY.
- Fix 3 (round 2, Claude): `phase_end.sh`'s locked gate run, the slice gate before the one-shot held-out run, passes `--wheel-run` with its CI run, as `final_gate.sh` does, so both hold temporary register entries; a hand run of `--locked 2a` without it, the gate's own controls among them, does not.
- Fix 4 (round 3, Codex; the orchestrator's decisions): every SQLite and Postgres line carries a query log, the driver's own statement log (each statement the ADBC or psycopg cursor executes and each relation ADBC binds, whatever object called it, for one case), held to the capture as DuckDB's log is, so the lines validate against the checkpoint's schema, which stays as it was; a Postgres result type the server numbers (the loader's composite types) is recorded by its name, and composite names are digests of their path, collision-free and the same on every run; python-old runs on DuckDB only, and the SQLite and Postgres jobs hold python-ref's committed lines to the cases (the reference side), with python-core's legs against them once the core reports a provider.
  Probe: the reference recorded all 86 cases on SQLite and Postgres with the new logs: every line valid against the checkpoint's schema and complete; two consecutive Postgres runs compare equal; no server-assigned number or generated database name in any line.
  The SQLite and Postgres jobs run in the reference cell only (the orchestrator's decision), and a table name holding a double quote loads (the loader reads its types from a relation over the quoted name).
