# Support matrix

G1 to G14 were decided by the owner as recommended below; see [decisions.md](decisions.md), rows 20 to 26.

Task T0.4 of [Phase 0](design.md).
This records what today's client supports, per distribution and extra, and proposes the native wheel coverage for the Rust core.
Every gap between the two is listed at the end and decided by the owner ([decisions.md](decisions.md) rows 20 to 26).
Plan decision 3 makes dropping a platform or version Python supports today a regression unless the owner decides it explicitly.

## Sources

- `packages/*/pyproject.toml` and `packages/structured-data-models-relational-engine/setup.py` (dependencies, Python bounds, classifiers, `[tool.cibuildwheel]`).
- `.github/workflows/publish.yml` (what a release builds) and `.github/workflows/ci.yml` (what CI tests).
- `README.md`, `docs/get-started/installation.md`, `docs/get-started/prerequisites.md` and `CHANGELOG.md`.
- `examples/databricks/*.ipynb`, which print their environment after installing.
- The first attempt's Databricks job outputs, `~/.local/share/sdm-dbx/result-review-fix.json` and `result-typed.json`, for the versions each runtime had after installing the client; those runs used the pre-rename package names and were not re-run in Phase 0.

## Distributions today

| Distribution | Kind | Python | Wheels published | Runtime dependencies |
|---|---|---|---|---|
| `structured-data-models-client` | Pure Python | `>=3.10`; classifiers 3.10 to 3.14; "OS Independent" | `py3-none-any` and an sdist | `-api` and `-pql` pinned, `pandas>=1.5`, `numpy>=1.23`, `requests>=2.28`, `urllib3>=1.26`; no pyarrow |
| `structured-data-models-api` | Pure Python | `>=3.10`; classifiers 3.10 to 3.14 | `py3-none-any` and an sdist | None |
| `structured-data-models-pql` | Pure Python | `>=3.10`; classifiers 3.10 to 3.14 | `py3-none-any` and an sdist | `antlr4-python3-runtime==4.13.2`, `numpy`, `pandas`, `pydantic`, all unbounded except ANTLR |
| `structured-data-models-connectors` | Pure Python | `>=3.10`; classifiers 3.10 to 3.14; "OS Independent" | `py3-none-any` and an sdist | `pandas>=2.0`, `pyarrow` (unbounded) |
| `structured-data-models-relational-engine` | Native (pybind11, scikit-build-core, CMake) | `>=3.10,<3.14` | cp310 to cp313, `manylinux_2_28` x86_64 and aarch64 only; no musllinux, macOS, Windows or sdist | `pyarrow>=10.0.1`, `pandas`, `numpy`, `pydantic>=2.7`, `python-dateutil`, `requests>=2.28.2`, `urllib3`, `typing_extensions>=4.5.0`, `rich>=9.0.0`, `jinja2`, `tabulate`, the four siblings pinned |

The client wheel's sdist comes from `python -m build` in `publish.yml`, which builds both by default.
The engine's aarch64 wheels are built under QEMU emulation on an x86_64 runner, and each wheel is smoke-tested by importing its compiled sampler.

## Extras of the client today

| Extra | Brings | What it adds to the constraints |
|---|---|---|
| (none) | The client, PQL, wire types | pandas 1.5 or later, no pyarrow, any platform |
| `relational` | The engine | CPython 3.10 to 3.13, Linux glibc 2.28 or later, x86_64 or aarch64, `pyarrow>=10.0.1` |
| `sqlite` | Connectors with `adbc-driver-sqlite` | pandas 2.0 or later, pyarrow |
| `duckdb` | Connectors with `duckdb>=1.5.0` | pandas 2.0 or later, pyarrow |
| `postgres` | Connectors with `psycopg[binary]>=3.1` | pandas 2.0 or later, pyarrow; the binary wheel aborts on Databricks aarch64 FIPS hosts |
| `snowflake` | Connectors with `snowflake-connector-python` | pandas 2.0 or later, pyarrow |
| `databricks` | Connectors with `databricks-sql-connector>=4.1.3`, `pyyaml>=5.4`, `databricks-sdk>=0.89,<1.0` | pandas 2.0 or later, pyarrow |
| `lakebase` | Connectors with plain `psycopg>=3.1` and `databricks-sdk>=0.89,<1.0` | pandas 2.0 or later, pyarrow, the system's libpq |
| `s3` | Connectors with `s3fs>=2023.12.0` | pandas 2.0 or later, pyarrow |
| `databricks-serving` | `databricks-sdk>=0.68.0,<1.0` | None beyond the SDK |
| `all` | `relational` and every backend extra except `databricks-serving` | The `relational` constraints; must not be installed on Databricks aarch64 (it brings `psycopg[binary]`) |

The engine has its own extras (`snowflake-serving`, `explain`, `relbench`, and backend extras, its `snowflake` one pinning `numpy<2.0`); the client's extras do not reach them.

## Effective ranges today

| Install | Python | pandas | pyarrow | Platforms |
|---|---|---|---|---|
| Base client (tabular model, PQL over frames) | 3.10 or later | 1.5 or later | Not required; used if present for `pd.ArrowDtype` columns | Anywhere CPython and pandas install |
| Any backend extra | 3.10 or later | 2.0 or later | Any | Anywhere the driver installs |
| `relational` (every `Schema.from_source` and relational `materialize` path) | 3.10 to 3.13 | 2.0 or later (through the connectors) | 10.0.1 or later | Linux, glibc 2.28 or later, x86_64 or aarch64 |

`Schema.from_source` and relational materialization import `sdm_relational_engine`, so every in-place path and every `kumo-relational` path is Linux-only today, while `kumo-tabular` runs on any laptop (README, "Requirements").

## What CI proves today

- `ci.yml` runs the pure packages' tests on Linux x86_64 only, for CPython 3.10, 3.11, 3.12, 3.13 and 3.14, with whatever pandas and pyarrow pip resolves.
- The engine is tested on CPython 3.12 only, Linux x86_64.
- No CI job runs on macOS, Windows, aarch64 hardware, musl, an old pandas (1.5) or an old pyarrow, so "OS Independent" and `pandas>=1.5` are declared rather than tested.

## Databricks runtime cells

These are the 20 cells of the job matrix ([databricks-jobs.md](databricks-jobs.md)), collapsed by runtime.
"Shipped" is what the runtime has before installing; "after install" is what the first attempt's jobs printed after `%pip install --only-binary=:all: structured-data-models-client[relational,...]` with NumPy pinned.

| Runtime | Architectures in the matrix | Python | pandas shipped | pandas after install | pyarrow shipped | pyarrow after install | NumPy |
|---|---|---|---|---|---|---|---|
| DBR 14.3 LTS | x86_64, aarch64 | 3.10.12 | Hypothesis: below 2.0 (the install upgraded it) | 2.3.3 | 8 (setup.py, CHANGELOG) | 25.0.1 | 1.23.5 |
| DBR 15.4 LTS | x86_64, aarch64 | 3.11.11 | Hypothesis: below 2.0 | 2.3.3 or 3.0.6 | Hypothesis: 14.0.1 (kept) | 14.0.1 | 1.23.5 (1.26.4 in runs without the pin) |
| DBR 16.4 LTS | x86_64, aarch64 | 3.12.3 | Hypothesis: below 2.0 | 3.0.6 | Hypothesis: 15.0.2 (kept) | 15.0.2 | 1.26.4 |
| DBR 17.3 LTS | x86_64, aarch64 | 3.12.3 | Hypothesis: 2.2.3 (kept) | 2.2.3 | Hypothesis: 19.0.1 (kept) | 19.0.1 | 2.1.3 |
| Serverless env 1 | x86_64 observed | 3.10.12 | Hypothesis: below 2.0 | 2.3.3 | 8 (setup.py) | 25.0.1 | 1.23.5 |
| Serverless env 2 | x86_64, aarch64 observed | 3.11.10 | Hypothesis: below 2.0 | 2.3.3 or 3.0.6 | Hypothesis: 14.0.1 (kept) | 14.0.1 | 1.23.5 (1.26.4 without the pin) |
| Serverless env 3 | x86_64, aarch64 observed | 3.12.3 | Hypothesis: below 2.0 | 3.0.6 | Hypothesis: 15.0.2 (kept) | 15.0.2 | 1.26.4 |
| Serverless env 4 | x86_64, aarch64 observed | 3.12.3 | Hypothesis: 2.2.3 (kept) | 2.2.3 | Hypothesis: 19.0.1 (kept) | 19.0.1 | 2.1.3 |
| Serverless env 5 | x86_64, aarch64 observed | 3.12.3 | Hypothesis: 2.2.3 (kept) | 2.2.3 | Hypothesis: 21.0.0 (kept) | 21.0.0 | 2.1.3 |
| Serverless env 6 | x86_64, aarch64 observed | 3.12.3 | Unknown, 2.0 or later | 2.3.3 | Hypothesis: 21.0.0 (kept) | 21.0.0 | 2.3.4 |

The "after install" columns and NumPy are observations.
No pre-install output is recorded for these cells, so every shipped version marked "Hypothesis" is inferred from post-install resolution alone and is not an observation.
"Hypothesis: below 2.0" rests on the version after install being higher while pip only upgrades an unmet requirement (`pandas>=2.0` from the connectors); the exact shipped version is not recorded here.
"Hypothesis: ... (kept)" rests on the version after install satisfying every requirement, so pip probably left the shipped one in place; since the transitive dependency set was not frozen, another requirement may have changed it.
The pyarrow 8 entries come from documentation (setup.py, CHANGELOG), not from a job's output.
A job that prints versions before `%pip install` turns these hypotheses into observations.
Every observed cell installed the engine's `manylinux_2_28` wheel, so every runtime above has glibc 2.28 or later.
The Spark route needs DBR 17.3 LTS or serverless env 4 or later (`DataFrame.toArrow()`); older runtimes use a SQL warehouse.
By the hypotheses above, a tabular-only install (no extras) on DBR 14.3 to 16.4 or serverless env 1 to 3 keeps pandas below 2.0 and, on 14.3 and env 1, pyarrow 8, so pandas 1.x and pyarrow 8 are probably live configurations for `kumo-tabular` users there.

Unknown, and not claimed by the docs: DBR 13.3 LTS and older (the docs start at 14.3), non-LTS runtimes, ML and GPU runtimes, DBR 18 and later, serverless env 7 and later, and the exact architecture of serverless env 1 on aarch64.

## Other environments

- Snowflake notebooks: the warehouse skill shows passing a notebook's Snowpark `session` to `SnowflakeSource`, but which notebook runtime users run (the warehouse runtime installs only from Snowflake's Anaconda channel; the container runtime takes pip wheels) is not recorded; unknown.
- Python 3.10 reaches upstream end of life this month (October 2026), but DBR 14.3 LTS and serverless env 1 still run it, so it stays in scope.

## Proposed native wheel coverage

The Rust core ships inside the `structured-data-models-client` wheel (plan section 7), built with maturin and PyO3 as `abi3` wheels.

| Item | Proposal |
|---|---|
| Interpreter | CPython only, `abi3-py310` (tag `cp310-abi3`): one wheel per platform for 3.10, 3.11, 3.12, 3.13, 3.14 and later, each tested in L8 |
| Linux | `manylinux2014` (glibc 2.17) for x86_64 and aarch64, falling back to `manylinux_2_28` only if the PQL C++ build needs it (G6; the PQL C++ condition superseded by Phase 1 decision E18 (owner, 2026-10-08)) |
| Linux musl | `musllinux_1_2` for x86_64 and aarch64, if the PQL team's static libraries build there (G5; the PQL condition superseded by Phase 1 decision E18 (owner, 2026-10-08)) |
| macOS | x86_64 (macOS 10.12 or later, Rust's floor) and arm64 (macOS 11.0 or later) (G4) |
| Windows | `win_amd64` (G2) |
| Not built | 32-bit Windows, Windows on Arm, Linux i686, armv7, ppc64le, s390x, PyPy, GraalPy, free-threaded CPython (G3, G7, G8) |
| sdist | None, as for the engine today, so a platform without a wheel fails at resolution (G12) |
| pyarrow | A new client dependency, `pyarrow>=10.0.1`; Arrow crosses by PyCapsule from pyarrow 14 and by `_export_to_c` below it (G9; superseded by Phase 1 decision E16 (owner, 2026-10-08): `pyarrow>=14.0.1`, PyCapsule only, on the epic branch) |
| pandas | `pandas>=1.5` unchanged, with pandas 1.5 added to L5's conversion tests (G10; superseded by Phase 1 decision E16 (owner, 2026-10-08): `pandas>=2.2` on the epic branch, L5 on pandas 2 and 3) |
| NumPy | Unchanged; the core does not use it (superseded by Phase 1 decision E16 (owner, 2026-10-08): `numpy>=1.26` on the epic branch) |
| OpenSSL | None linked into the core; HTTP stays in the host, so the Databricks aarch64 FIPS abort cannot recur through the core |
| Build and test | Native Arm runners instead of QEMU for aarch64; each wheel installed and loaded in a clean container per platform (L8) |
| Java JAR | Native libraries for linux-x86_64 and linux-aarch64 (same glibc floor as the Python wheel), macOS x86_64 and arm64, and Windows x86_64, on JDK 17 and 21; the Windows target is requested by the first Java consumer (G13 amendment pending owner review) |

Each platform added here must also be in the PQL asks ([pql-asks.md](pql-asks.md)); superseded by Phase 1 decision E18 (owner, 2026-10-08): the asks are retired, and no PQL C++ conditions a platform.

What the proposal gains over today: `kumo-relational` and `Schema.from_source` on Python 3.14 and later, and on macOS, Windows and musl if those are built; none of these is a decision, but each needs L8 coverage before it is claimed.

## Gaps and owner decisions

Each gap compares the proposal with today's support; each needs an owner decision before Phase 1a builds its first wheel.
The PQL C++ and PQL-ask conditions in G2, G4, G5, G6 and G12 are superseded by Phase 1 decision E18 (owner, 2026-10-08): no PQL C++ is linked.

**G1. The base client stops being pure Python.**
Today `pip install structured-data-models-client` installs anywhere CPython runs, from a `py3-none-any` wheel.
With the core inside the client wheel, it installs only where a wheel is published, and that includes `kumo-tabular` users.
Recommendation: accept, with the coverage above, since keeping a pure-Python fallback would keep the old engine past Phase 5.

**G2. Windows.**
Today the base client and the connectors install on Windows; the engine does not.
Recommendation: build `win_amd64`, and add Windows (MSVC) to the PQL asks; if the PQL team cannot build it, the owner decides between waiting and dropping Windows.

**G3. Other architectures and 32-bit platforms.**
Today the base client installs on 32-bit Windows, Windows on Arm, Linux i686, armv7, ppc64le and s390x.
Recommendation: drop them, and say so in the installation guide and changelog; there is no evidence of users there.

**G4. Minimum macOS version.**
Today the base client has no macOS floor.
Recommendation: macOS 10.12 on x86_64 and 11.0 on arm64, unless the PQL C++ build needs newer.

**G5. musl Linux (Alpine).**
Today the base client installs on musl; the engine deliberately does not (`skip = "*-musllinux_*"`).
Recommendation: build `musllinux_1_2` for x86_64 and aarch64 if the PQL static libraries build there, and add musl to the PQL asks; otherwise drop musl for the whole client.

**G6. glibc older than 2.28.**
Today the base client runs on any glibc, for example Amazon Linux 2 (glibc 2.26) and RHEL 7 (2.17); the engine needs 2.28.
Recommendation: build `manylinux2014` (glibc 2.17), which Rust supports, so the base client loses no Linux it has today; use `manylinux_2_28` only if the PQL C++ toolchain forces it, and then the owner accepts dropping older glibc.

**G7. Interpreters other than CPython.**
Today the pure distributions may install on PyPy or GraalPy; `abi3` wheels cover CPython only.
Recommendation: drop them explicitly; nothing records use, and CI has never run them.

**G8. Free-threaded CPython (3.13t, 3.14t).**
Today the pure client installs on free-threaded builds; `abi3` wheels cannot be installed there.
Recommendation: drop for now and revisit when a free-threaded stable ABI exists; building version-specific `cp313t` and `cp314t` wheels is the alternative.

**G9. pyarrow becomes a dependency of the base client.**
Today the base client needs no pyarrow, and on DBR 14.3 LTS and serverless env 1 a tabular-only install keeps the shipped pyarrow 8.
Recommendation: require `pyarrow>=10.0.1`, the engine's floor today, so one floor covers every path; on those two runtimes the install then upgrades pyarrow, as `[relational]` already does (observed working, to 25.0.1).
The alternative is to support pyarrow 8 through `_export_to_c` and test it in L5, which keeps tabular users on those runtimes from upgrading.
G9 and G10 are superseded by Phase 1 decision E16 (owner, 2026-10-08): `pyarrow>=14.0.1`, `pandas>=2.2` and `numpy>=1.26` on the epic branch, PyCapsule only; `main` keeps today's floors until the epic merges.

**G10. pandas 1.5.**
The client declares `pandas>=1.5`, and pandas below 2.0 is, by hypothesis, what a tabular-only install keeps on DBR 14.3 to 16.4 and serverless env 1 to 3, but the plan's L5 tests only pandas 2 and 3.
Recommendation: keep `pandas>=1.5` and add pandas 1.5 to L5 as a third major; raising the base floor to 2.0 (the connectors' floor) would be a regression for those users.

**G11. The `relational` extra and the engine distribution after Phase 5.**
Today `[relational]` pins `structured-data-models-relational-engine` by exact version, and the docs tell every Databricks user to install it.
Recommendation: keep `relational` as an extra name (empty, or carrying whatever relational paths still need) so documented install lines keep working, and stop publishing the engine distribution from the release that deletes the old engine.

**G12. Source distribution.**
Today the pure distributions publish an sdist, and the engine publishes none.
Recommendation: publish no sdist for the client once it carries the core, so pip fails at resolution on an unsupported platform instead of trying to compile Rust and the PQL C++; distributors such as conda-forge build from the tagged source.

**G13. Java native platforms.**
Java has no support today, so this is not a regression, but the Java consumer's environment decides it (plan section 10).
Original recommendation: Linux x86_64 and aarch64 first, since Databricks is where Java must be proven, plus macOS for development; confirm with the Java consumer before Java packaging.
The first Java consumer has now stated a requirement for Windows x86_64 as well as Linux x86_64. Add Windows x86_64 to the Java packaging and L8 load-test matrix before claiming support; this amendment awaits owner review. The Maven artifact layout remains a Java-phase decision.

**G14. Unrecorded environments.**
Snowflake notebook runtimes and Databricks runtimes outside the matrix (13.3 LTS and older, ML and GPU runtimes, DBR 18 and later) are unknown, not supported or unsupported.
Recommendation: claim only what the job matrix and L8 prove, and ask the owner whether any user runs the client in a Snowflake warehouse-runtime notebook, where a native wheel from PyPI cannot be installed.
