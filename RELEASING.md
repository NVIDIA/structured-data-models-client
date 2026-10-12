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
uploaded. It attaches every distribution and the SBOM to the GitHub Release for the
tag, and then asks Kitmaker's Portal to publish from those asset URLs.

**Publishing goes through Kitmaker, not straight to PyPI.** Kitmaker owns
NVIDIA's PyPI organization and this project's OSRB record commits to releasing
through it. Kitmaker does not accept an upload; it fetches each artifact from a
URL and publishes it itself, to pypi.org and pypi.nvidia.com together. It only
accepts an Artifactory path or a GitHub release asset, and it refuses private
GitHub assets, so **a release cannot publish while the repository is
internal**.

One Portal project exists per distribution, because Kitmaker matches the
project name against the distribution name parsed from each filename. Their
ids are in `publish.yml`, and the job checks each id still carries the name it
expects before sending anything, since a mismatch would publish one
distribution's files under another's name. Every project is dry-run with
`upload: false` before the real submission; Kitmaker batches are sequential and
not atomic, and a published file cannot be recalled.

**There is a stored credential.** `KITMAKER_API_TOKEN` is a long-lived bearer
token in this repository's secrets, held only by the `kitmaker` job behind the
protected `pypi` environment. The previous Trusted Publishing setup avoided a
stored token; this is the cost of publishing through Kitmaker.

**There are no PEP 740 attestations.** They are issued to whoever uploads, and
that is Kitmaker. SECURITY.md tells consumers what they can verify instead.

## After

Check the five projects on pypi.org and confirm each shows NVIDIA organization
ownership. Portal reporting that a release completed is not the same as the
package being installable, so install it the way a user would.

## Yanking

A release that is wrong is yanked, not deleted. Deleting it breaks anyone who
pinned it, and yanking leaves it installable for them while keeping resolvers
away from it. Yank on PyPI, then release the fix as a patch.
