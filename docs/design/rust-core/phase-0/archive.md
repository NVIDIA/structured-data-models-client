# T0.2 archive snapshot and scan results

This records how the archive of the first attempt was built, what the scans found, and what was pushed.

## The branch

The branch is pushed to GitHub as `archive/rust-core-attempt-1` at commit `3887762`.
It holds that one commit, with no parent and no earlier history.
It was built in its own worktree of the GitHub clone, so the main checkout was not touched.
The commit is authored by the owner, carries a DCO sign-off, and has no co-author trailer.
The commit message names the source commit, says the branch is a read-only source to port from and never merged, and lists the removals and redactions.

## How it was built

The source is commit `ab85a0567dcaaebcc9062673a63bedefe2bdee71` (tag `pm/2026-10-07/all-public-cases-pass`) in the GitLab clone `~/code/sdm-client-wt/rc-integration-draft`.
The tree was exported with `git archive ab85a056 | tar -x` into a scratch directory.
`.github/` (3 files) and `.gitlab-ci.yml` were deleted from the export.
A worktree was created with `git worktree add --orphan -b archive/rust-core-attempt-1`, the export was copied in with `rsync -a`, and everything was staged with `git add -A -f` so the snapshot's own `.gitignore` could not drop a tracked file.
The staged index was compared with `git ls-tree -r ab85a056` minus the removed paths, and every path, mode and blob id matched.

## Size

The snapshot holds 7,078 files (the source commit has 7,082).
The files total about 217 MB (223 MB on disk), and no file is over 50 MB.

## Redactions

No scan found a real credential.
The internal sandbox catalog name was replaced with `example-catalog`, and the workspace name with "the Databricks validation workspace", in 45 files.
The stored hashes in those files (recordings, transcripts and reference policies) are now stale, because they were computed over the unredacted text.
Every other scan hit below was accepted by the owner as listed (decision 16 in [decisions.md](decisions.md)).

## Scan 1: hygiene check

The snapshot's own `scripts/check_repo_hygiene.py` was run inside the worktree; its `internal` patterns are identical to those on GitHub `main`.
`internal` reported 1 hit: `tests/rust_core/freeze/assembly/investigations/p97-acceptance-assembly/checks.py:146` builds a path template for the package directories (the repository name followed by `-{package}`), which the bare-repository-name pattern matches.
That is a path template for the public package directories, not a leak; recommendation: accept.
`retired` reported no hits.
The `internal` check only reads shipped file types and skips its allowlist, so the deny-list scan below repeats its patterns over every file.

## Scan 2: credentials (gitleaks 8, `gitleaks dir`, values redacted in the report)

`generic-api-key`: 5 hits in 3 files.
- `tests/parity/acceptance/accept_scrub.py` (1 hit): a planted fake AWS secret built by string concatenation, used to test that recordings get scrubbed. The same file plants fake Databricks, Snowflake and Postgres hosts, tokens and passwords, all marked `planted` and split across concatenations. Recommendation: accept.
- `tests/rust_core/freeze/date-reference-policy.json` and `tests/rust_core/freeze/semantic-public-reference-policy.json` (4 hits): SHA-256 hashes of source files keyed by path, which the rule mistakes for keys. Recommendation: accept.

## Scan 3: deny-list grep

Every tracked text file was scanned (binary files skipped), with case-insensitive patterns.

| Pattern | Hits | Files | Example paths | Recommendation |
|---|---|---|---|---|
| Internal sandbox catalog name (the Databricks catalog the parity runs used) | 1,978 | 46 | `tests/parity/dialects/databricks.json`, `tests/parity/requests/databricks/*/calls.json`, `tests/parity/sampling/databricks/*`, `tests/parity/transcripts/databricks/*`, `scripts/parity/_parity/live.py`, `tests/rust_core/schema/_views.py` | Redacted to `example-catalog` (see Redactions); the stored hashes are stale |
| Same name in prose ("the ... notebooks", "Phase 9.5's ... jobs") | 5 | 4 | `scripts/parity/_parity/scenarios.py`, `tests/parity/acceptance/accept_matrix.py`, `tests/rust_core/switch/CONTRACT.md` | Accept as internal |
| Old GitLab host, old GitLab group, internal bug tracker, wiki, artifact mirror, cluster and sibling-repository names | 9 | 1 | `scripts/check_repo_hygiene.py` (the pattern list itself) | Accept; they define the check |
| `nemotron` | 7 | 4 | `scripts/check_repo_hygiene.py`, `packages/structured-data-models-client/tests/test_local_notebooks.py`, `tests/rust_core/switch/base.json`, `packages/structured-data-models-relational-engine/tests/test_vendored_mermaid.py` | Accept; a retired interface name (`a retired predict helper`) and test ids, not a Lakebase instance |
| Lakebase instance names | 1 | 1 | `tests/parity/acceptance/accept_scrub.py` (a planted fake `instance-<uuid>` host) | Accept |
| Databricks workspace hosts (`*.cloud.databricks.com`, `*.azuredatabricks.net`) | 6 | 2 | `tests/parity/acceptance/accept_scrub.py`, `packages/structured-data-models-connectors/tests/test_databricks_helpers.py` | Accept; all are planted or `adb-1...` placeholders |
| Warehouse ids | 1 | 1 | `packages/structured-data-models-connectors/README.md` (`0123456789abcdef`) | Accept; placeholder |
| `snowflakecomputing` | 7 | 4 | `packages/structured-data-models-relational-engine/tests/client/test_snowflake_serving.py`, `tests/parity/acceptance/accept_scrub.py` | Accept; `acct` style placeholders |
| Serving endpoint names | 2 | 2 | `packages/structured-data-models-client/tests/test_databricks_serving.py` | Accept; placeholder `x` |
| Email addresses, real domains | 15 | 4 | `MAINTAINERS.md` (6 maintainers), `SECURITY.md` (product security contact), `LICENSE` and `packages/structured-data-models-relational-engine/LICENSE` (third-party copyright notices) | Accept; the same maintainer and security addresses are already on GitHub `main` |
| Email addresses, placeholder domains (`example.com`, `acme.com` and similar) | about 50 | 8 | `packages/structured-data-models-relational-engine/tests/test_telemetry.py`, `tests/rust_core/switch/base.json`, `packages/structured-data-models-client/tests/test_transport_hardening.py` | Accept |
| Email-shaped strings in vendored Mermaid | 142 | 2 | `packages/structured-data-models-relational-engine/src/sdm_relational_engine/rfm/assets/mermaid.min.js`, `MERMAID_COMPONENTS.json` | Accept; third-party package metadata |
| `nvidia.com` hosts | 3 | 1 | `SECURITY.md` (`www.nvidia.com`) | Accept; public |
| Local harness paths (`~/.local/share/sdm-dbx/...`) | 17 | 16 | `tests/rust_core/*/CONTRACT.md`, `tests/rust_core/sampling/driver/accept_driver_ledger.py`, `scripts/parity/coverage.py` | Accept as internal; no username, only a default path overridable by `SDM_RC_HARNESS` |
| S3 buckets | 11 | 5 | `packages/structured-data-models-connectors/tests/test_s3.py`, `qa/probe_cloud.py` | Accept; placeholders or names chosen not to exist |
| Private IPs (`10.x`) | 16 | 4 | `rust/crates/schema/src/engine/result.rs`, `tests/fixes/test_fix_pg_read.py` | Accept; test values |
| Slack links, internal tickets, `/Users/<name>` or `/home/<name>` paths, AWS account ids, workspace org ids (`?o=`) | 0 | 0 | | None |

## Outcome

No file needed removing.
The owner chose to redact the internal catalog and workspace names (45 files) and to accept every other hit as listed (decision 16).
The commit message was reworded before pushing.
