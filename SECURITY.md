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

## Verifying a release

Every file this project publishes to PyPI carries a
[PEP 740](https://peps.python.org/pep-0740/) attestation. There is no NVIDIA
signing key to fetch: the signature is issued to the short-lived identity of
this repository's release workflow and recorded in Sigstore's transparency
log.

```bash
pip install pypi-attestations
pip download --no-deps structured-data-models-client==1.0.0 -d /tmp/verify
python -m pypi_attestations verify pypi \
  --repository https://github.com/NVIDIA/structured-data-models-client \
  /tmp/verify/*.whl
```

`--repository` is not optional: without it the command refuses to verify a
file published from a repository at all, and with the wrong one it fails. The
same command verifies any of the five packages; name the file you want to
check.

A passing check tells you these exact bytes were published to PyPI by an
authorized Trusted Publisher of this repository, and have not been altered
since. That is integrity and origin. It is not build provenance: it does not
establish how the bytes were produced, and it does not tell you the release is
free of defects or vulnerabilities.

## Supported versions

The client is distributed as five packages: `structured-data-models-client`, `structured-data-models-api`,
`structured-data-models-pql`, `structured-data-models-connectors` and `structured-data-models-relational-engine`. They are released
together.
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
  exporting to PNG or SVG sends the graph *structure* (table, column and
  relationship names, never cell values) to the public `mermaid.ink` service.
- **Data-source arguments are trusted input.** Connection settings, table names,
  `query=`, `path=` and `storage_options=` are executed or resolved as written,
  with the credentials you supply, as is a `ColumnSpec(expr=...)` you write
  yourself. Do not pass attacker-controlled values into them. Entity
  identifiers are the exception, and may come from untrusted input: no sampler
  ever interpolates one into SQL text. Depending on the backend and on which
  query is being built, they travel either as bound parameters or as a
  temporary table the query joins against, and both paths keep the value out
  of the statement.
