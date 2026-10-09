# Releasing

Five distributions, one version, published together. A release is a tag; the
tag is the only manual step.

The tag has to sit on a commit that is on `main` and whose CI run passed.
The publish workflow checks both before it builds anything, so a tag on an
unmerged or failing commit stops there rather than reaching the index.

That check lives in the workflow, and a tag runs the workflow as it was at
the tagged commit, so it cannot defend against a tag placed on a commit
older than the check. Two repository settings do, and both are configured
outside this tree: a tag ruleset limiting who may create `v*`, and required
reviewers on the `pypi` environment, which hold the upload itself.

## Before tagging

```bash
# Everything that needs nothing running.
python -m pytest packages/structured-data-models-api/tests packages/structured-data-models-pql/tests \
                 packages/structured-data-models-client/tests packages/structured-data-models-connectors/tests tests \
  -q -m 'not live_nim and not live_warehouse'

# The native engine, from its own directory.
cd packages/structured-data-models-relational-engine && python -m pytest tests -q && cd -

ruff check . && ruff format --check .
mypy packages/structured-data-models-api/src packages/structured-data-models-pql/src \
     packages/structured-data-models-client/src packages/structured-data-models-connectors/src \
     packages/structured-data-models-relational-engine/src
mypy packages/structured-data-models-connectors/tests/typing/check_sources.py
python scripts/check_repo_hygiene.py all
python scripts/mermaid_notices.py --check
```

Then, against running NIMs, because a release should not be the first time
these run:

```bash
SDM_TABULAR_URL=... SDM_RELATIONAL_URL=... \
  python -m pytest packages/structured-data-models-client/tests -m live_nim -q
```

## Bumping the version

One literal per distribution, in `_version.py`. All five must match, and
`tests/test_release_contract.py` fails if they do not.

```bash
for f in packages/*/src/*/_version.py; do
  sed -i '' "s/__version__ = '.*'/__version__ = '1.2.0'/" "$f"
done
```

Internal pins (`structured-data-models-api==1.0.0` in `structured-data-models-client`'s dependencies) move with them.

Update `CHANGELOG.md`: the entries for the release go under its own heading
with today's date, so the tag carries a changelog that names the version it
publishes.

## Tagging

```bash
git tag v1.2.0 && git push origin v1.2.0
```

That is the whole release. `.github/workflows/publish.yml` builds the four
pure-Python distributions plus the engine's wheel matrix through `cibuildwheel`
(cp310 to cp313, manylinux_2_28 x86-64 and aarch64, each smoke-tested, as set in
the engine's `[tool.cibuildwheel]`), and then refuses to
continue unless **all five** are present and agree with the tag. That check is
not ceremony: `structured-data-models-client[relational]` pins the engine by exact version, so a
release missing it publishes an extra nobody can install. It then generates the
SBOM, and takes the release's notes from the changelog's entry for that
version, refusing a tag that has none: notes that read like a diff are fixable
afterwards and a distribution on PyPI is not, so both happen before anything is
uploaded. It uploads through PyPI Trusted Publishing with PEP 740 attestations,
then creates the GitHub Release for the tag with those notes and attaches the
SBOM to it, so the dependency inventory outlives the ninety days an Actions
artifact gets.

**There is no stored PyPI token.** Publishing authenticates with the
workflow's OIDC identity, which PyPI exchanges for a token minted on the spot
and good for about fifteen minutes. That is why `publish` is the only job
holding `id-token: write`. If you find yourself wanting to add a token to the
repository's secrets, something is wrong with the trusted-publisher
configuration instead.

## After

The `verify` job has already checked every distribution this workflow built
against this repository's identity, so there is nothing to confirm by hand. It runs beside
the job that creates the GitHub Release rather than before it: an upload
cannot be undone, and a release whose SBOM never got attached because a cache
was cold is worse than one whose verification has to be re-run. A red `verify`
on a green `publish` means the files are on PyPI and something about their
attestations needs looking at, not that the release did not happen.

It checks the artifacts the build produced against the provenance PyPI
serves, which is the stronger direction: for each file this workflow built, it
answers whether PyPI attested those exact bytes. It does not compare the
index's file list against the build, so it would not notice an upload that
published all of them faithfully and added one more. SECURITY.md's "Verifying
a release" section is the same command written for a consumer, who starts from
the file they downloaded instead.

## Yanking

A release that is wrong is yanked, not deleted. Deleting it breaks anyone who
pinned it, and yanking leaves it installable for them while keeping resolvers
away from it. Yank on PyPI, then release the fix as a patch.
