# T3.20f: complete warehouse reference recordings

This package supersedes the unlanded T3.20d capture candidate.
T3.20e fixed fetch-time failure recording and full per-case validation before import.
The interrupted candidate contained 633 unfetched SQL reads in 275 cases and is retained only as diagnostic history.

## Inputs and provenance

The 41 pending case inputs are restored byte-for-byte from the prior T3.20d candidate.
The existing Databricks backtick write-guard exclusion also covers the odd-column case containing that character; no new exclusion category is introduced.
The reference remains the clean frozen client at `43e55f4f58924384c3cb52f67c8ed2e73ba45fa1`.
All imported captures retain their original recording/reference identities and versions.

| Leg | Cases | Recording source | Evidence |
| --- | ---: | --- | --- |
| Databricks SQL | 300 | `8a1c0142e13d5ac6a511888ee1c42377cc919ce8` | Fresh campaign using the repaired writer; every case passed full validation before import |
| Databricks Spark | 300 | `792c507229e7652d34d28c84e436d2a44a2f803e` | Existing genuine raw recording revalidated against the exact current inputs |
| Snowflake additions | 9 | `85172c3654202f2b02d0d2dcaf82b6d15ee568a4` | Existing genuine raw recording revalidated against the exact current inputs |
| Lakebase additions | 9 | `85172c3654202f2b02d0d2dcaf82b6d15ee568a4` | Existing genuine raw recording revalidated against the exact current inputs |

The fresh SQL campaign recorded all 300 cases with zero campaign failures.
The reused 318 recordings passed the full locked raw-recording validator with zero pending inputs.
Canonical imports of those reused recordings are byte-identical to their prior imported files.
The four recording cohorts are stored under `contract/captures/r2/`, leaving existing base captures and their manifests intact.
No raw recording, Arrow result, expectation or provenance field was edited by hand.

## Review and checks

The case-policy regression was committed before the restored exclusion entry.
Two independent Sol High reviews of the final candidate precede lint and package tests.
These reference captures do not constitute final-candidate live core evidence; that remains part of T3.10.
