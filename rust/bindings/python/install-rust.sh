#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs, in a build container that has no Rust, the release that
# rust/rust-toolchain.toml pins, so every build uses the same compiler.
# Usage: install-rust.sh RUST_DIR
set -eu
toolchain=$(sed -n 's/^channel = "\(.*\)"$/\1/p' "$1/rust-toolchain.toml")
test -n "$toolchain"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal --default-toolchain "$toolchain"
