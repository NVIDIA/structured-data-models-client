#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pulls an image CI runs, pinned by digest, with retries (decision E131).
#
#     pull_image.sh REGISTRY/NAME[:TAG]@sha256:DIGEST
#
# A registry that refuses a pull for its request rate often takes it a
# minute later, and wheel-smoke's two lanes start at the same moment, so a
# failed pull is tried again after 20, 60 and 120 seconds (PULL_DELAYS, in
# seconds, overrides them) before the step fails. A refusal for a data
# quota ("Data limit exceeded", ECR Public's monthly allowance for
# anonymous pulls from the runner's address) fails at once: no wait
# here clears it. An image not pinned by digest is refused. For an image on ghcr.io it first logs in with
# GITHUB_TOKEN, when the step has one, as GITHUB_ACTOR; every other
# registry is pulled anonymously, as before.

set -euo pipefail

image=${1:?usage: pull_image.sh REGISTRY/NAME@sha256:DIGEST}
if ! [[ $image =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "pull_image.sh: $image is not pinned by digest" >&2
  exit 2
fi
if [[ $image == ghcr.io/* && -n ${GITHUB_TOKEN:-} ]]; then
  printf '%s\n' "$GITHUB_TOKEN" |
    docker login ghcr.io --username "${GITHUB_ACTOR:-github-actions}" --password-stdin
fi
read -r -a delays <<<"${PULL_DELAYS-20 60 120}"
attempt=1
errors=$(mktemp)
trap 'rm -f "$errors"' EXIT
until docker pull --quiet "$image" 2>"$errors"; do
  cat "$errors" >&2
  if grep -qi 'data limit exceeded' "$errors"; then
    echo "pull_image.sh: $image: the registry's data quota for this address is spent; no retry clears it" >&2
    exit 1
  fi
  if ((attempt > ${#delays[@]})); then
    echo "pull_image.sh: $image: no pull in $attempt attempts" >&2
    exit 1
  fi
  delay=${delays[attempt - 1]}
  echo "pull_image.sh: attempt $attempt failed; again in ${delay}s" >&2
  sleep "$delay"
  attempt=$((attempt + 1))
done
