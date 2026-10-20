# Phase 2 tracker: T2.0i

The design branch owns `tracker.md`; this file holds T2.0i's section until it folds in.

## T2.0i

T2.0i brings `main`'s two new commits into Phase 2: `origin/main` moves from `327fda03` to `f2ad8d61`, `0aee420a` (the relational engine built for macOS, Windows and Python 3.14) and `f2ad8d61` (the README's logo and badges).
The owner approved it in KO2-81 and approved the new reference commit `43e55f4f` (`rust-core-reference` with `f2ad8d61` merged) in KO2-84, which lands first as T2.1u.

### The merge

`git merge --no-ff origin/main` merged with no conflict, as `git merge-tree` had said, first at `1d85bce3` and again at the end of the landing chain (T2.1u, T2.14a, T2.21, T2.11) where T2.0i lands.
Neither commit changes what the client does on Linux.

### What the merge needed besides

| Path | Change | Why |
|---|---|---|
| `.github/workflows/ci.yml`, `runtime-install` | Downloads `engine-wheels-linux-x86_64` | `main` renamed the engine's artifacts to `engine-wheels-<name>` with its new rows; CI7's job still asked for `engine-wheels-x86_64`, which a clean textual merge leaves broken |
| `.github/workflows/ci.yml`, `native-wheel-aarch64` | `MANYLINUX` is the ghcr.io copy of PyPA's `manylinux2014_aarch64:2026.09.14-1`, same digest `d4c40df2...` | The owner's go for the ARM build image copy (2026-10-20); every other image CI pulls is a ghcr.io copy (E131) |
| `tests/test_ci_workflow.py` | One test: every engine artifact a job downloads is one `engine-wheels` uploads, and the aarch64 image is on ghcr.io | One test per tooling fix |
| `uv.lock` | `uv lock`: the lock's `requires-python` moves to `<3.15` and the 3.14 wheels are added; no package version moves | The engine's `requires-python` moved in `0aee420a` |

`main`'s macOS and Windows `engine-wheels` rows come in unchanged: the engine depends on the api, pql and connectors wheels, all pure Python, and not on the client's native wheel, so the rows install on those runners.

### The reference

The recorder recorded python-ref at `43e55f4f` for every case on all three databases at `1d85bce3` (`T2.0i tests: python-ref at 43e55f4f`, taken unedited); all 652 lines equal their `6298913c` lines under `ratchet.semantic`, so no answer changed and nothing is registered.
The environment digest is unchanged.
`contract/reference/43e55f4f58924384c3cb52f67c8ed2e73ba45fa1.jsonl` replaces the `6298913c` file, and `contract/reference-env.txt` and every vector file and test citing the old commit name the new one.
T2.10 moved T2.18's 9 deferred filter-role lines into the reference before T2.0i; with the other cases the chain adds (T2.10's, T2.21's, T2.11's), the recorder records them at `43e55f4f` too, so the new file carries every case and database line the old one did.
`tests/test_contract_reference.py` holds the environment file's reference to the `43e55f4f` prefix.
No `bridge_changes.json` entry: neither commit touches a frozen file.

### Open

- The recorder's lines at `43e55f4f` for T2.11's new cases; the 57 the chain added through T2.21 (T2.10's and T2.21's cases, T2.18's deferred lines among them) are recorded, each equal to its `6298913c` line under `ratchet.semantic`.
- Packages after T2.0i record their new reference lines at `43e55f4f` directly.
