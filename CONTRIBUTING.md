# Contributing to structured-data-models-client

Thanks for your interest in contributing. This project is licensed under the
[Apache License 2.0](./LICENSE), and we welcome external contributions.

## Developer Certificate of Origin (DCO)

All contributions must be signed off to certify that you have the right to
submit them under the project's open-source license, per the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO 1.1).

Sign off each commit by adding a `Signed-off-by` line with your real name and
email (matching the commit author):

```bash
git commit --signoff
```

which appends:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Pull requests whose commits are not signed off cannot be merged.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install \
  -e './packages/structured-data-models-api[test]' \
  -e './packages/structured-data-models-pql[test]' \
  -e './packages/structured-data-models-client[test]' \
  -e './packages/structured-data-models-connectors[sqlite,duckdb,postgres,databricks,test]'
.venv/bin/python -m pytest \
  packages/structured-data-models-api/tests packages/structured-data-models-pql/tests \
  packages/structured-data-models-client/tests packages/structured-data-models-connectors/tests tests -q
```

`tests/` at the root is the cross-package suite: that the five distributions
agree on a version and a licence, that the recorded public surface and wire
payloads still match, and that the claims the configuration files make about
themselves are true.

The relational engine is a native build and is installed and tested on its
own, from its own directory:

```bash
.venv/bin/pip install -e './packages/structured-data-models-relational-engine[test]'
cd packages/structured-data-models-relational-engine && ../../.venv/bin/python -m pytest tests -q
```

Lint and format with `ruff`; the formatter owns line length. Type-check with
`mypy`, which is blocking in CI:

```bash
.venv/bin/mypy packages/structured-data-models-api/src packages/structured-data-models-pql/src \
               packages/structured-data-models-client/src packages/structured-data-models-connectors/src
```

Changing anything a caller can observe, such as an exported name, signature,
or wire request shape, requires recording a new snapshot in the same commit.
This makes the change visible in review:

```bash
.venv/bin/python scripts/snapshot_public_api.py --write
.venv/bin/python scripts/snapshot_wire_payloads.py --write
```

## Testing against something live

The two commands above need nothing running. These need an endpoint, and are
deselected by default:

```bash
# a NIM you are running
SDM_TABULAR_URL=http://localhost:8000 \
SDM_RELATIONAL_URL=http://localhost:8001 \
  .venv/bin/python -m pytest packages/structured-data-models-client/tests \
    -m live_nim -q

# real warehouses, with credentials in the environment
.venv/bin/python -m pytest \
  packages/structured-data-models-connectors/tests -m live_warehouse -q
```

## Third-party code

A new third-party *dependency* is declared in the owning package's
`pyproject.toml`, and resolved by the package manager at install time;
[`NOTICE`](./NOTICE) does not cover it, and deliberately says so.

Third-party code *vendored into the tree* is different: it ships with this
repository, so it must be recorded in [`NOTICE`](./NOTICE) with its license and
source, its license text included, and its bytes pinned by digest and asserted
in CI. Today `mermaid.min.js` is the only such tree. Adding a second is a
decision to discuss on an issue first, not a detail of a pull request.

## Full text of the DCO

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
