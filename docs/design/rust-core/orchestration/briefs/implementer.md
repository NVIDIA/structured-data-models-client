# Implementer: standing rules (appended to every package brief)

The package brief above gives the scope, the exits, the design row and the decision rows. These rules apply to every package.

**Workflow**
- Work and push only from your lane worktree, to `rust-core-phase-N-lane-<x>`. Never switch branches there; `land.py` needs it on the lane. To work ahead, stack the next package in its own worktree off the approved SHA.
- Every commit subject starts with the package ID: `<ID>: ...`, `<ID> fix <n>: ...`, `<ID> tracker: ...`, `<ID> docs: ...`. Cite other packages only in the body (`Refs: ...`). The first commit records the previous package's landing in the tracker.
- The acceptance tests come from a separate test author (`<ID> tests: ...` commits). Make them pass without editing them. If one looks wrong, report it; the test author fixes it. Your own unit tests are welcome but aren't acceptance evidence.
- While iterating, run only your package's tests. Before each push: `scripts/rust_core/prepush.sh` (the cheap tier; the hook blocks `--full`).
- Before you report "ready": `scripts/rust_core/handoff.py run --package <ID> --python <your lane environment's python>` (and `--previous <last round's SHA>` in a fix round, `--case <id>` for each case the brief lists). It runs your cases against the recorded old-client answers, your new tests on the previous SHA, every suite that reads a touched file, and lint, and stamps the SHA only when all pass. Review rounds refuse a SHA without the stamp.
- A fix commit (`<ID> fix <n>: ...`) that changes product code carries `Rule: <the old client's rule it ports, one sentence>`; the commit-msg hook and `land.py` refuse it otherwise.
- Push review-ready states, report the SHA at once and end your turn. Don't watch CI; the orchestrator sends you any CI failure as a fix item.
- Final report: final SHA, each exit with its evidence, deviations, decision rows added, and anything needing the owner.

**Judging paths and guards**
- The judging paths are listed in `scripts/rust_core/judging-paths.txt`. After the phase's first package, `land.py` refuses a package changing one unless it is the tooling train (`land.py --stack --train`, first in its stack). If you need a judging change, report it: the orchestrator moves it into the train.
- Any guard or gate change must also pass in its post-landing state: the phase branch at the landing, judged against itself.
- Guards are frozen at bug fixes. No new frameworks or harness layers; extend the existing runners, vectors and maps.
- A new case-format key lands with accept and refuse vectors in `contract/tests/malformed-cases.json`, run by both loaders.
- A contract version bump also updates `rust/bindings/sdm-py/tests/messages.py` until Phase 2 row FX-6 derives it.

**Behavior**
- Every outcome you measure to differ from the reference gets a register entry, marked proposed, before review. Never approve your own entry.
- A refusal of input the reference accepts gets a proposed register entry and a diagnostic test. If the work would diverge from the product in a way the plan doesn't cover, stop and report.
- The core emits codes and structured fields; each host renders the wording. No host-language knowledge or SQL dialect in the core.
- When the core concludes something from a provider's answer (absence, existence, resolution), the provider must establish it the way the source resolves names. Probe it with views, other schemas and attached databases.
- Avoid what reviews kept catching: process-global hooks in production code, checkers that accept empty or vacuous evidence, diagnostic checks presented as gates, proxies that change driver return types, and silently guessing where the reference does something else.

**Probes before the first push** (record what each found in the tracker or commit text)
- Where a route changes: an old-route against core-route differential on about 50 generated DuckDB sources (varied types, keys, nulls, sizes), every difference fixed or listed.
- For every reader of outside JSON: thousands of nested levels and a very wide object, each refused by name or read, never a crash or `RecursionError`.
- For every new comparator or gate rule: a seeded fault showing the rule fails when its evidence is missing, empty or retyped, not only when it differs.

**Shared docs and IDs**
- New register IDs and decision rows come only from your lane's ID block. When it runs out, ask the orchestrator.
- Write only in your lane's sub-table of `decisions.md` and your package's section of `as-built.md` and `tracker.md`; add register entries under your lane's banner. Never append at the end of a shared file.
- A decision row is at most 1,000 characters; evidence goes in as-built. Fix-round history goes once, briefly, in the tracker.
- In docs, give each claim the mechanism that enforces it.

**Hard rules**
- DCO sign-off only (`git commit -s`), no `Co-Authored-By`. No pull requests, no landing commits (`land.py` makes those). Never `git config`.
- Keep credentials, hostnames, account and workspace names and internal infrastructure names out of everything you write. The repository is public.
- The command hook (`scripts/rust_core/orchestration/guard_cmd.py`) blocks `rm -r` outside your scratch, `--force` and `-D` removals, `--no-verify`, `sudo`, `git config`, `git stash`, `gh run watch` and `git fetch --prune`, naming the alternative. `git worktree prune` is the only cleanup allowed. Never touch another lane's worktrees. If something seems to need approval, report it instead.
- Run every probe and subprocess under a timeout with stdin closed: `perl -e 'alarm shift; exec @ARGV' 600 <cmd> </dev/null`, with `GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0`.
- Run in your lane's environment: `scripts/rust_core/cells.sh /mnt/localssd/cells/<lane>` and `SDM_LANE_ENV` (the runners refuse the shared `.venv`). Keep `CARGO_TARGET_DIR` at `/mnt/localssd/target/<you>`; the shared sccache holds the builds.
- Heavy runs go through `$SDM_ORCH_STATE/bin/heavy`, which queues them until the machine can take them.
- Never open held-out data.
