# What is public, and what changing it costs

This page is the compatibility contract. It exists because Hyrum's law is not a
joke: with enough users, every observable behaviour is depended on by someone,
so a library that has not said what it promises has promised everything.

## What is public

Public is what a distribution's top-level `__all__` names, and nothing else:

```python
import sdm_client
sdm_client.__all__
```

Concretely, the public surface is:

| Package | Public |
|---|---|
| `sdm_client` | Everything in `sdm_client.__all__` — the client, `Schema`, `Request`, `PredictiveQuery`, the model handles, `PredictionResult`, the exception types and `SDMWarning` |
| `sdm_api` | Everything in `sdm_api.__all__` — the wire vocabulary |
| `predictive_query` | Everything in `predictive_query.__all__` — the parser, the AST, the validators, the profiles |
| `sdm_connectors` | Everything in `sdm_connectors.__all__` |
| `sdm_relational_engine` | Reached through `sdm_client`. Import it directly at your own risk |

Not public, and free to change in any release:

- Any name beginning with `_`, at any depth.
- Any submodule not re-exported from the top level — `sdm_client.core.transport`,
  `sdm_client.adapters`, `sdm_client.materializers`, `sdm_client.wire` and their
  contents are implementation, however stable they look.
- The exact text of an exception or warning message. Branch on the exception
  **type** and on `StructuredDataError.code`, never on the message. Codes are
  public; prose is not.
- The order of rows in a `PredictionResult`. Join on the entity key when the
  model has one, otherwise on `row_index`. Row order is not a promise.

What *is* promised beyond the names: the exception hierarchy under
`StructuredDataError`, the `code` values it carries, the `SDMWarning` category,
and the columns `PredictionResult.to_pandas()` produces for a given set of
requested output fields.

## What a version number means

[Semantic versioning](https://semver.org), applied to the public surface above
and to all five distributions together — they share one version and are released
at the same time, so `sdm-client==1.2.0` always means `sdm-api==1.2.0`.

| Bump | Means |
|---|---|
| **Major** | Something public was removed or changed shape. Read the changelog before upgrading. |
| **Minor** | Something was added. Existing code keeps working. A deprecation may begin here; nothing is removed. |
| **Patch** | A defect was fixed, with no intended change to the public surface. |

A fix that changes an observable behaviour — a wrong answer becoming a right
one, a request that used to be sent now being refused — is a minor bump and is
called out in the changelog, even though the API did not change shape.

## When this starts

**At the first published release.** Nothing in this repository has been
published yet — no version of any of the five distributions exists on PyPI —
so 1.0.0 is where the promises below begin, not something they already cover.

That distinction is load-bearing rather than pedantic. The release this
document ships in also renames two names out of `predictive_query.__all__`
(`GraphDefinition` and `Edge`, now `SchemaDefinition` and `Relationship`) with
no deprecation period, which the rule below would otherwise forbid. There is no
period to serve: nobody can be holding the old names, and shipping aliases that
warn from the first release would be deprecating something that was never
available. Everything after 1.0.0 gets the full treatment.

## Deprecation

Nothing public is removed without a release that warns about it first.

1. In the release where it is deprecated, the name keeps working and raises
   `DeprecationWarning` on use, naming the replacement and the version it will
   be removed in.
2. It keeps working for **at least one minor release**, and removal only
   happens in a major one.
3. The changelog entry names the old thing, the new thing, and the upgrade.

```python
import warnings

warnings.simplefilter('always', DeprecationWarning)   # see them in a notebook
```

To fail your own build on them, so an upgrade cannot surprise you later:

```bash
python -W error::DeprecationWarning -m pytest
```

## How this is enforced

Not by review. `tests/golden/public_api.json` records every exported name and
its signature, and `tests/test_release_contract.py` compares the two on every
run — so changing the public surface requires re-recording the snapshot in the
same commit, which is what makes it visible in review. The wire payloads of
canonical requests are recorded the same way, in
`tests/golden/wire_payloads.json`, because what the endpoint receives is part
of the contract too.
