# Setting up a machine to orchestrate a phase

The owner does the steps marked **(owner)**, since they need logins or approvals. The orchestrator checks everything else with prompt 1 in [prompts.md](prompts.md).
Where a step needs a private name (a profile, a workspace, a volume, a database instance), ask the owner. Those names never go in this repository.

## Accounts and access (owner)

1. **GitHub CLI**: `gh auth login`, then `gh auth refresh -s write:packages,read:packages`. `gh auth status` should list `repo`, `workflow` and the package scopes.
2. **SSH key for pushes**, added to GitHub and authorized for the organisation's SSO (GitHub, Settings, SSH keys, "Configure SSO"). Without that, pushes fail with "has enabled or enforced SAML SSO". Check: `ssh -T git@github.com`.
3. **Databricks CLI profile** for the test workspace, only if the phase has live work (Phase 3 especially). Tell the orchestrator the profile name, the authorized volume and the database instance it may use; it keeps them out of the repo.
4. **Codex CLI** logged in (`codex login`), and **herdr** for the Codex panes. Plan on one Codex pane per lane.
5. **Claude Code** with the repository's `AGENTS.md` (loaded through `CLAUDE.md`), in a permission mode that lets background agents run without approval prompts for ordinary commands.

## Tools

| Tool | Version | Check |
|---|---|---|
| Rust | 1.97.0 with rustfmt and clippy (pinned by `rust/rust-toolchain.toml`; rustup installs it on first use) | `cd rust && cargo --version` |
| Python | 3.10 to 3.14 through uv (the reference uses 3.12) | `uv python list` |
| uv | 0.11 or newer | `uv --version` |
| Docker | Docker Desktop or equivalent, for the wheel-smoke and Postgres checks | `docker info` |
| maturin | 1.15.0, run through uv as `CONTRIBUTING.md` shows | `uv run --no-project --with maturin==1.15.0 maturin --version` |
| gh, jq, perl, git | recent | `gh --version` |

## Checkouts

```bash
mkdir -p ~/code && cd ~/code
git clone git@github.com:NVIDIA/structured-data-models-client.git structured-data-models-client
cd structured-data-models-client
git fetch origin rust-core rust-core-reference
git worktree add ~/code/sdmc-rust-core rust-core
git worktree add ~/code/sdmc-reference rust-core-reference
```

Set up the reference environment as [phase-0/reference-environment.md](../phase-0/reference-environment.md) describes, then run its guard:

```bash
~/code/sdmc-reference/.venv/bin/python ~/code/sdmc-reference/scripts/reference_guard.py
```

Set up the development environment in `~/code/sdmc-rust-core` as `CONTRIBUTING.md` describes (note its `RUSTUP_TOOLCHAIN=1.97.0` for the native module), then build once (`cd rust && cargo build --workspace`) so the first agent doesn't pay for a cold build.

## Orchestrator state

The helpers in `scripts/rust_core/orchestration/` keep their state outside the repository:

```bash
export SDM_ORCH_STATE=~/.local/share/sdm-orchestration   # land queue, ignore list, logs, briefs, review reports
export SDM_PHASE_BRANCH=rust-core-phase-2
mkdir -p "$SDM_ORCH_STATE"
```

See [the helpers' README](../../../../scripts/rust_core/orchestration/README.md) for the other variables.

## Keep the machine awake

Agents run unattended for hours, and on a sleeping laptop they lose about an hour each. On macOS, run `caffeinate -dimsu` in a terminal for the length of the phase and keep the machine on power. On Linux, use `systemd-inhibit`.

## Disk

Each lane worktree with its Cargo target takes 5 to 15 GB, and one phase can use well over 100 GB of scratch space. Start with at least 200 GB free. `clean_tmp.sh` (dry run by default) lists what is safe to remove.
