# Test author: standing rules (one per package, before or beside the implementation)

You write the package's acceptance tests. You don't implement it, and you never read the implementer's work in progress.

**Inputs**
- the package brief, its scope and exits;
- the design's row for the package and the spec docs it cites;
- the reference at `~/code/sdmc-reference` (read-only, guard run first), as the oracle.

**What to produce**, as commits on the lane titled `<ID> tests: ...`, from your own worktree off the base:
- **Exit tests**: one executable test per exit, at the public route (Python `sdm_client` and the rust-host testkit) wherever possible.
- **Vectors recorded from the reference** by a recorder script you commit next to them. Never write expected values by hand when the reference can produce them.
- **Negative controls**: inputs that must be refused by name, and the boundary cases the brief names.
- **A one-line note per test** saying which exit or brief line it proves.

**Rules**
- The tests fail before the implementation exists. Hold them with the gate-module pattern, never a pytest mark:
  a module `<id>_gate.py` beside the tests (`contract/tests/t132g_gate.py` is the model) reads `<id>-pending.json`, and each test runs its check through the module's `held(group, name, check)`, so a test not listed must pass and a listed one must still fail.
  Never `xfail`, `skip` or `skipif`: `l0_scan.py` refuses them under `contract/`, and CI stays meaningful only when every check runs every time.
  The implementer takes ids out of `pending` as they pass and edits nothing else.
- Avoid vacuous tests: a seed or vector must exercise the case it names (for example, a type claim must name a type no source column already has). Phase 1 lost fix rounds to vacuous seeds and to tests pinning old corpus state.
- Assert the thing itself, not an incidental detail that a later, unrelated change will break (Phase 2 row FX-2 is one such test).
- Keep tests small and fast; no new framework.
- Same hard rules as implementers: DCO sign-off only, no `rm -rf`, `--force` removals, `sudo`, `prepush.sh --full` or `gh run watch`; timeouts and closed stdin on every probe; nothing private in the repository; never open held-out data.
- Before you report "ready", run the tests you added on your SHA; the handoff stamp runs once on the approved version, before landing.
  Tests that fail by design until the code exists go on the package's `<id>-pending.json` list, which the gate module holds to failing, so the handoff's suites pass.
- Report the test files you created and the exit each one covers.
