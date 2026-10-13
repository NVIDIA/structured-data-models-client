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

These distributions are published to PyPI by **Kitmaker**, the team that owns
NVIDIA's PyPI organization, from artifacts attached to the corresponding
GitHub release in this repository. Kitmaker publishes to
[pypi.org](https://pypi.org/) and `pypi.nvidia.com` together.

What that gives you is NVIDIA's ownership of the project on PyPI: the
distribution lives under the NVIDIA organization, which is the same assurance
that covers NVIDIA's other published packages. What it does not give you is a
per-file signature traceable to this repository. The uploader is Kitmaker, so no PEP 740
attestation names this repository's release workflow. Whether PyPI holds a
Kitmaker attestation instead is Kitmaker's to say; what this repository can
tell you is that it does not sign these files itself.

You can compare a downloaded file against the asset of the same name on the
corresponding GitHub release. The release workflow checks this itself: it will
not finish unless PyPI holds the same bytes it built and attached, and it never
replaces an asset it has already uploaded. So the two should agree. It is still
not proof of how either was built, and a release asset can be replaced by hand
afterwards, so a mismatch is not by itself evidence of tampering on PyPI and a
match is not evidence that this repository produced them.

```bash
pip download --no-deps structured-data-models-client==1.0.0 -d /tmp/verify
# then compare that file's digest against the asset of the same name at
# https://github.com/NVIDIA/structured-data-models-client/releases/tag/v1.0.0
```

None of this tells you the release is free of defects or vulnerabilities. It
establishes origin, and nothing more.

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
