# Releasing

Three Python distributions, one version, published together. A release is a tag, then
an approval, then a publish. The tag builds everything. The `pypi` environment
has required reviewers, so the job that creates the GitHub Release and attaches
those distributions waits for one. Publishing them to PyPI is a third step,
run by hand, because the service that owns those projects is not reachable
from a public CI runner.

The tag has to sit on a commit that is on `main` and whose CI run passed.
The publish workflow checks both before it builds anything, so a tag on an
unmerged or failing commit stops there rather than reaching the index.

That check lives in the workflow, and a tag runs the workflow as it was at
the tagged commit, so it cannot defend against a tag placed on a commit
older than the check. Two repository settings do, and both are configured
outside this tree: a tag ruleset limiting who may create `v*`, and required
reviewers on the `pypi` environment, which hold the artifacts back from being
published on the Releases page at all.

## Before tagging

```bash
# Everything that needs nothing running.
python -m pytest packages/structured-data-models-api/tests \
                 packages/structured-data-models-client/tests packages/structured-data-models-connectors/tests tests \
  -q -m 'not live_nim and not live_warehouse'

# Shared core and bindings (see CONTRIBUTING.md for native test setup).
cargo test --manifest-path rust/Cargo.toml --workspace --locked
python -m pytest rust/bindings/sdm-py/tests -q --native-build test-ops
python tests/compatibility/check_junit.py --run

ruff check . && ruff format --check .
mypy packages/structured-data-models-api/src \
     packages/structured-data-models-client/src packages/structured-data-models-connectors/src
mypy packages/structured-data-models-connectors/tests/typing/check_sources.py
python scripts/check_repo_hygiene.py all
python scripts/rust_core/rust_notices.py --check
```

Then, against running NIMs, because a release should not be the first time
these run:

```bash
SDM_TABULAR_URL=... SDM_RELATIONAL_URL=... \
  python -m pytest packages/structured-data-models-client/tests -m live_nim -q
```

## Java

The Java library is built and tested in CI, but it is not released yet.
The release workflow attaches no JAR or POM, and nothing is published to a Maven repository.
Keep `java/pom.xml` at the same version as the rest, so the library is ready when it is released.

## Pre-releases

A tag such as `v2.0.0rc1`, `v2.0.0b1` or `v2.0.0a1` runs the same workflow with the same checks.
The version is the PEP 440 spelling, as in `_version.py`; Rust's `Cargo.toml` writes the same version in SemVer (`2.0.0-rc.1`).
The GitHub Release is marked as a pre-release.
`CHANGELOG.md` needs a `## 2.0.0rc1` heading, since the release notes are taken from it.
Tagging `v2.0.0rc1` is on hold until the owner approves it.

## Bumping the version

The version is written in several places, and all of them move together:

| Where | Spelling for `2.0.0rc1` |
| --- | --- |
| `packages/*/src/*/_version.py` (one per distribution) | `2.0.0rc1` |
| `rust/Cargo.toml`, `[workspace.package] version`, which maturin stamps on the client wheel | `2.0.0-rc.1` |
| `rust/Cargo.lock`, the workspace crates' entries (`cargo update --workspace` rewrites them) | `2.0.0-rc.1` |
| `java/pom.xml`, `<version>` | `2.0.0rc1` |
| `packages/structured-data-models-client/pyproject.toml`, the pins on `structured-data-models-api` and the connectors extras | `2.0.0rc1` |
| `tests/golden/public_api.json`, rewritten by `scripts/snapshot_public_api.py --write` | `2.0.0rc1` |

`tests/test_release_contract.py` fails if `_version.py`, the Rust workspace, `java/pom.xml` or a first-party pin disagree.
Search for the old version afterwards to catch anything else, such as the coordinates in `java/README.md`.

Update `CHANGELOG.md`: the entries for the release go under its own heading
with the date you are tagging on, not the date you wrote them, so the tag
carries a changelog that names the version it
publishes.

## Tagging

```bash
git tag v1.2.0 && git push origin v1.2.0
```

That starts the release workflow.
It requires successful CI and live validation for the tagged commit, then builds the API and connector distributions.
The client wheels come from that commit's successful CI run: Linux x86-64/aarch64, macOS x86-64/arm64 and Windows AMD64.
Each client wheel uses CPython's stable ABI from Python 3.10 onward.
The workflow verifies each wheel's source commit, clean-tree record, production features and digest before assembling the release.
Expired or missing CI artifacts require rerunning CI; publication does not substitute an untested build.

All three distributions must be present at the tagged version.
The workflow also generates an SBOM from the exact wheels, reads the version's changelog entry and attaches distributions, build identities and the SBOM to the GitHub Release.
**A green workflow means the artifacts are built and public, not that anything is on PyPI.**

**Publishing goes through Kitmaker, not straight to PyPI.** Kitmaker owns
NVIDIA's PyPI organization and this project's OSRB record commits to releasing
through it. Kitmaker does not accept an upload; it fetches each artifact from a
URL and publishes it itself, to pypi.org and pypi.nvidia.com together. It only
accepts an Artifactory path or a GitHub release asset, and it refuses private
GitHub assets, so **a release cannot publish while the repository is
internal**.

**The publish is run by a person, from inside NVIDIA's network.** Kitmaker's
Portal is not on the public internet, so a GitHub-hosted runner cannot resolve
it, and no workflow here can do this step. The repository holds no Portal
address, project id or publishing credential for the same reason: none of them
belongs in a public repository, and nothing in CI has a use for them. The
procedure and the script live with the release owner, outside this repository.

What that procedure has to do, so that a replacement is held to it: one Portal
project exists per distribution, because Kitmaker matches the project name
against the distribution name parsed from each filename, and each id is checked
against the name it should carry before anything is sent, so a renamed or
reassigned id fails early rather than being trusted. Every project that has
something left to publish is validated and dry-run to a terminal state before
*any* of them publishes, because the three distributions pin each other by exact
version and an incomplete release can leave the client uninstallable. Kitmaker batches are sequential and not atomic, and a published file
cannot be recalled.

The GitHub Release is created behind the protected `pypi` environment, because
creating it is what makes the distributions public. It cannot be a draft
instead, because a draft's asset URLs are private and Kitmaker could not fetch
them. Declining the approval means nothing was created at all.

**Nothing in CI can publish to an index.** There is no stored publishing
credential and no environment secret; the most the `release` job can do is
attach files to a Release. The credential Kitmaker needs is a long-lived bearer
token, which is why it is held by the person running the publish rather than by
a workflow on a public repository.

**This workflow does not sign what it ships.** PEP 740 attestations go to
whoever uploads, and that is Kitmaker, so nothing here produces one naming this
repository. SECURITY.md tells consumers what they can check instead.

### If a run fails partway

**Use "Re-run failed jobs", never "Re-run all jobs."** These builds are not
byte reproducible, so rebuilding produces a different wheel and sdist for the
same version. The workflow will not clobber an already-attached asset, so a
full re-run would leave the release holding the original bytes while `dist/`
holds new ones. The publish compares each attached asset's digest against what
the run built and refuses before submitting anything, since Kitmaker fetches
the asset rather than the local file and a published file cannot be recalled.
"Re-run failed jobs" reuses the original `distributions` artifact, which is
what you want.

### If the publish fails partway

The summary the publish writes is the record to read before doing anything
else. It names each of the three projects. For any whose upload was requested it says
what **PyPI** holds for this version, not what Portal reported: Portal's batches
are not atomic, so a failure can follow files that already published, and a
refusal is not proof that nothing did. Only the index knows. The other states
describe what this run did, since the index has nothing to say about a project
it was never offered.

| The summary says | What it means |
| --- | --- |
| `published` | Every file is on PyPI with the bytes this run built. Nothing to do. |
| `already published before this run` | It was there before this run started. |
| `not reached` | No upload was requested for it. A dry run may still have run. |
| `not on PyPI` | The index reports nothing for this version. Usually means nothing landed, but see the caveat below. |
| `PARTLY on PyPI, missing [...]` | Some files landed. A re-run finishes the rest. |
| `on PyPI with UNEXPECTED bytes for [...]` | Needs a person. |
| `on PyPI with files this run did not build` | Needs a person. |
| `UNKNOWN, could not ask PyPI` | The index could not be reached, or did not answer sensibly. Look by hand. |

A re-run is safe for the first five states and needs a person for the last
three. **The caveat: PyPI's JSON endpoint is cached**, so a project published
in the last few minutes can still read as `not on PyPI`. That state is what the
index said, not proof that nothing landed. If the summary surprises you, check
the project page on pypi.org before re-running, or you may submit the same
files twice. Confirm every expected file, not just that the version exists: a
batch that stopped halfway leaves only some of them.

Before each project is submitted, PyPI is asked what it already holds at that
version. A project fully published with the artifact's bytes is skipped, and one
that is part-published is reduced to the files still missing, since PyPI takes
files into a version one at a time. A re-run after a partial publish therefore
finishes what did not land.

**If the build artifact is gone**, "Re-run failed jobs" stops at
`actions/download-artifact`, since an expired artifact cannot be downloaded.
"Re-run all jobs" builds fresh, non-reproducible files, finds the earlier
assets still attached, leaves them alone and reports each as already attached.
That run goes green with the release holding one set of bytes and `dist/`
another, so the workflow is not what catches this. The publish is, because it
compares every attached asset's digest against the artifact before it submits
anything. This happens once the artifact's retention runs out, and after any
change to the workflow that needs a fresh build.

Recovery depends on the index: if nothing for this version is on PyPI, delete
the release assets and run again; if anything is, the version is spent, so
release a patch.

Two states it refuses to guess at, both needing a hand:

- **PyPI holds different bytes than the artifact.** Someone published this
  version from other artifacts. PyPI will not replace a file, so do not
  re-run; work out what was published first, then yank and release a patch.
- **PyPI holds files this run did not build.** Same reasoning: the version on
  PyPI is not the one in `dist/`.

## After

Check the three projects on pypi.org and confirm each shows NVIDIA organization
ownership. Portal reporting that a release completed is not the same as the
package being installable, so install it the way a user would.

For a final release, then mark the GitHub Release Latest; leave an alpha, beta
or release candidate unmarked. The workflow leaves every release unmarked on
purpose, because until the publish has worked it would advertise a version the
index does not have. One exception is worth knowing: when there is no earlier
release to hold the position, GitHub reports the newest non-draft release as
the latest one regardless, so the first release reads as Latest from the
moment it is created.

## Yanking

A release that is wrong is yanked, not deleted. Deleting it breaks anyone who
pinned it, and yanking leaves it installable for them while keeping resolvers
away from it. Yank on PyPI, then release the fix as a patch.
