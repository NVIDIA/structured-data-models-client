# Releasing

Five distributions, one version, published together. A release is a tag; the
tag is the only manual step.

## Before tagging

```bash
# Everything that needs nothing running.
python -m pytest packages/sdm-api/tests packages/sdm-pql/tests \
                 packages/sdm-client/tests packages/sdm-connectors/tests tests \
  -q -m 'not live_nim and not live_warehouse'

# The native engine, from its own directory.
cd packages/sdm-relational-engine && python -m pytest tests -q && cd -

ruff check . && ruff format --check .
mypy packages/sdm-api/src packages/sdm-pql/src packages/sdm-client/src packages/sdm-connectors/src
python scripts/check_repo_hygiene.py all
```

Then, against running NIMs — a release should not be the first time these run:

```bash
SDM_TABULAR_URL=... SDM_RELATIONAL_URL=... \
  python -m pytest packages/sdm-client/tests -m live_nim -q
```

## Bumping the version

One literal per distribution, in `_version.py`. All five must match, and
`tests/test_release_contract.py` fails if they do not.

```bash
for f in packages/*/src/*/_version.py; do
  sed -i '' "s/__version__ = '.*'/__version__ = '1.2.0'/" "$f"
done
```

Internal pins (`sdm-api==1.0.0` in `sdm-client`'s dependencies) move with them.

Update `CHANGELOG.md`: move `Unreleased` entries under the new version with
today's date.

## Tagging

```bash
git tag v1.2.0 && git push origin v1.2.0
```

That is the whole release. `.github/workflows/publish.yml` builds the four
pure-Python distributions plus the engine's wheel matrix through `cibuildwheel`
(cp310–cp313, manylinux x86-64, each smoke-imported), and then refuses to
continue unless **all five** are present and agree with the tag. That check is
not ceremony: `sdm-client[relational]` pins the engine by exact version, so a
release missing it publishes an extra nobody can install. It then generates the
SBOM and uploads through PyPI Trusted Publishing with PEP 740 attestations.

**There is no PyPI token.** Publishing authenticates with the workflow's OIDC
identity, which is why the `publish` job is the only one holding `id-token:
write` and runs nothing but the upload. If you find yourself wanting to add a
token, something is wrong with the trusted-publisher configuration instead.

## After

```bash
pip download --no-deps sdm-client==1.2.0 -d /tmp/verify
python -m pypi_attestations verify pypi /tmp/verify/*.whl
```

## Yanking

A release that is wrong is yanked, not deleted — deleting it breaks anyone who
pinned it, and yanking leaves it installable for them while keeping resolvers
away from it. Yank on PyPI, then release the fix as a patch.
