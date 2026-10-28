# Working in this repository

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first: it has the development setup,
the test commands, lint, type-checking and the snapshot rule, and
[`README.md`](README.md) for what the packages are. This file is only what is
easy to get wrong here, and most of it has been got wrong at least once.

## What you install is not what you import

| Distribution | Module |
|---|---|
| `structured-data-models-client` | `sdm_client` |
| `structured-data-models-api` | `sdm_api` |
| `structured-data-models-pql` | `predictive_query` |
| `structured-data-models-connectors` | `sdm_connectors` |
| `structured-data-models-relational-engine` | `sdm_relational_engine` |

Package directories under `packages/` carry the distribution name. Nothing
derives one name from the other: read the import root off `src/`.

## Strings that look like names and are not

A rename that sweeps the tree will change these, and nothing will fail:

- `DATABRICKS_PRODUCT`, kept twice because the connectors never import the
  client: `sdm_client/core/databricks_user_agent.py` and
  `sdm_connectors/_databricks_partner.py`, held to the same value by a test.
  `SAMPLING_QUERY_TAG` sits beside the second. They are what a warehouse sees,
  so changing them splits Databricks partner telemetry and query history at
  that version.
- `APPLICATION` in `sdm_connectors/backends/snowflake.py`, `nvidia/kumo-sdm`.
  It is the application name Snowflake attributes this client's sessions to,
  so changing it splits that attribution at that version.
- `_term('sdm-relational-engine')` in `rfm/graph_fingerprint.py`. It is hashed.
  Changing it gives identical graphs different names, which the module says
  only `FINGERPRINT_VERSION` may do.

## Run the two suites separately

The engine and the other four packages cannot be collected in one pytest run;
together they fail at import. `CONTRIBUTING.md` gives both commands. Run both.

## skills/ is executable

[`tests/test_skills.py`](tests/test_skills.py) reads the tables out of
`skills/` and puts them to the client, and with `-m live_nim` it runs every
complete example and compares what it printed to the block pasted beneath it.
A `**Verified**` block is a claim that an endpoint produced exactly that. Do
not write one by hand, and do not edit the numbers in one.

An example is complete when every name it reads was bound by itself or by an
earlier complete example. One that is not is a fragment: it is checked as code,
executed nowhere, and may not show output.

## Generated files

Regenerate these; never edit them:

- `packages/structured-data-models-api/src/sdm_api/generated/` -- regenerate
  with `scripts/generate_sdm_api.py`, which also writes `PROVENANCE.json`. A
  change to the generator itself changes the recorded hash.
- The third-party attribution in the two `LICENSE` files, the root one and the
  engine's -- `scripts/mermaid_notices.py --write`, checked with `--check`. It
  renders `rfm/assets/MERMAID_COMPONENTS.json`, which `--refresh` regenerates;
  neither touches `NOTICE`.
- `rfm/assets/relbench_registry.json` -- `scripts/pin_relbench.py`. It pins
  source commits and table digests.
- The ANTLR output in `predictive_query/grammar/`: the `.py`, `.tokens` and
  `.interp` files. Edit `PQLGrammar.g4` and regenerate. The `README.md` and
  `__init__.py` there are hand-written.
- `tests/golden/` -- the exported surface and the recorded wire payloads.

## Gates that fail on prose

`scripts/check_repo_hygiene.py all` runs in CI. It refuses a shipped file that
names internal infrastructure: the internal git host, a bug tracker id, a
security-program id, the internal wiki, or an internal sibling repository. The
patterns are in that script, and this file is written around them rather than
listed as an exception. A comment citing an internal ticket is a build
failure, not a style note.

Every first-party source file carries an SPDX header.

## Commits

Sign off every commit (`git commit --signoff`); see
[`CONTRIBUTING.md`](CONTRIBUTING.md). A pull request whose commits are not
signed off cannot be merged.
