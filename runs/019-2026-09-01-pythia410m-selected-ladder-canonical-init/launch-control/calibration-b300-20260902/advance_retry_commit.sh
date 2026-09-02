#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/sparsity-spillover
bundle=/workspace/run019-h200-retry.bundle
commit=71f9c4709c4ca091215b9e1baf219557666145ed

cd "$repo"
test -z "$(git status --porcelain --untracked-files=no)"
echo '3bcbb322be5cca8e14be90fff5cd84b5e9fbfa939e26adc73a8f57fcb6a21517  /workspace/run019-h200-retry.bundle' | sha256sum -c -
git fetch "$bundle" refs/heads/main
git checkout --detach "$commit"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain --untracked-files=no)"
