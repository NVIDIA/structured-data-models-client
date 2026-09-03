# Security

NVIDIA is dedicated to the security and trust of our software products and
services, including all source code repositories managed through our
organization.

## Reporting a vulnerability

Please do **not** report security vulnerabilities through GitHub. If a potential
security issue is inadvertently reported via a public issue or pull request,
NVIDIA maintainers may limit public discussion and redirect the reporter to the
appropriate private disclosure channels.

To report a potential vulnerability:

- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- Email: <psirt@nvidia.com>, optionally encrypted with the
  [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key)

A report is most useful when it includes the affected package and version, the
type of vulnerability, the minimal steps or script that reproduce it, and what
an attacker gains.

NVIDIA runs a public bug bounty and vulnerability disclosure programme on
[Intigriti](https://app.intigriti.com/programs/nvidia/nvidiapublicbugbounty/detail),
and acknowledges externally reported issues addressed under its coordinated
disclosure policy. See the
[PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/) page.

## Supported versions

The client is distributed as five packages — `sdm-client`, `sdm-api`,
`sdm-pql`, `sdm-connectors` and `sdm-relational-engine` — released together.
Only the latest released version of each receives security fixes; there are no
long-term-support branches. Fixes ship in a new release rather than as patches
to older ones.

## Scope notes

The following are intentional design decisions rather than vulnerabilities.
Please read them before filing.

- **Authentication is deployment-owned.** The Universal TFM API contract defines
  no authentication. `api_key` exists only for deployments that front a NIM with
  their own authenticating gateway; it is sent as `X-API-Key` and is refused
  over plaintext `http://` to a non-localhost host.
- **The endpoint you configure is trusted with your data.** Prediction context,
  including the raw cell values of the tables in your schema, is sent to the URL
  you pass to `SDMClient`.
- **Two paths reach a third party, and both are opt-in.** The `[explain]` extra
  posts row data to an OpenAI-compatible endpoint you configure. `visualize()`
  renders locally from a bundled copy of mermaid and reaches nothing; only
  exporting to PNG or SVG sends the graph *structure* — table, column and
  relationship names, never cell values — to the public `mermaid.ink` service.
- **Data-source arguments are trusted input.** Connection settings, table names,
  `query=`, `path=` and `storage_options=` are executed or resolved as written,
  with the credentials you supply, as is a `ColumnSpec(expr=...)` you write
  yourself. Do not pass attacker-controlled values into them. Entity
  identifiers are the exception: no sampler interpolates them into SQL text.
  SQLite, Snowflake and Databricks bind them as query parameters; DuckDB
  registers them as a temporary relation the query joins against. Either way
  they may come from untrusted input.
