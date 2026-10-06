# Governance

## Who decides

The client is maintained by NVIDIA. Maintainers are listed in
[MAINTAINERS.md](MAINTAINERS.md) and hold final say on what is merged, on the
public API, and on release timing.

This is a single-vendor project rather than a foundation-governed one. Decisions
are made in the open on issues and pull requests, but there is no voting
process, no technical steering committee, and no path to commit access outside
NVIDIA today. Saying so plainly is better than implying a shared governance
model that does not exist.

## How a change gets in

1. Open an issue describing the problem before writing a large change, so the
   approach can be agreed. Small fixes can go straight to a pull request.
2. Open a pull request against `main`, signed off under the DCO (see
   [CONTRIBUTING.md](CONTRIBUTING.md)).
3. At least one maintainer reviews. CI must pass.
4. A maintainer merges. Contributors do not merge their own changes.

## What gets accepted

Changes are judged on whether they make the client more correct, clearer, or
better tested, and on whether they can be maintained afterwards.

The client deliberately stays thin: it builds a request, sends it to a NIM, and
turns the response into a DataFrame. Model behaviour lives in the NIM, not
here. Changes that move inference logic into the client, or that couple the
client to one deployment's conventions, are unlikely to be accepted regardless
of quality.

The wire contract is shared with the NIM. A change to the request or response
shape cannot be made in this repository alone and needs the server side to move
with it. `structured-data-models-api` is generated from that contract and is not edited by hand.

## Releases

The five packages are versioned and released together: a given version of one
is only tested against the same version of the others. Releases are cut by
maintainers; there is no fixed cadence.

Breaking changes are avoided within a major version. Where one is unavoidable,
it is called out in [CHANGELOG.md](CHANGELOG.md) under a heading that says so,
with the migration in the same entry.

Publishing is automated. A version tag on a commit that is on `main` and
whose CI passed builds every distribution and uploads it through PyPI
Trusted Publishing, with no stored token. [RELEASING.md](RELEASING.md)
describes what a maintainer does; `.github/workflows/publish.yml` is what
actually runs. No version has reached an index yet.

## Changing this document

Governance changes are made by maintainers, as a pull request like any other, so
the history of how the project was run stays readable.
