#!/usr/bin/env bash
# Mirror the files in budgie-sync-manifest.txt into a budgie-ml-service checkout,
# verify them there, and commit. Pushing to budgie's main triggers its CI and deploy.
# Usage: scripts/sync-budgie-ml-service.sh <budgie-ml-service-path> [--push]
set -euo pipefail

source_repo="$(cd "$(dirname "$0")/.." && pwd)"
target_repo="${1:?usage: $0 <budgie-ml-service-path> [--push]}"
push="${2:-}"
manifest="$source_repo/scripts/budgie-sync-manifest.txt"

mapfile -t files < <(grep -Ev '^\s*(#|$)' "$manifest")

# Uncommitted source edits would reach production without ever being reviewed.
if [[ -n "$(git -C "$source_repo" status --porcelain -- "${files[@]}")" ]]; then
  echo "Commit the synced files in Odin-ml first:" >&2
  git -C "$source_repo" status --short -- "${files[@]}" >&2
  exit 1
fi
if [[ -n "$(git -C "$target_repo" status --porcelain)" ]]; then
  echo "budgie-ml-service has uncommitted changes; refusing to sync." >&2
  exit 1
fi
if [[ "$(git -C "$target_repo" branch --show-current)" != "main" ]]; then
  echo "budgie-ml-service must be on main." >&2
  exit 1
fi

for file in "${files[@]}"; do
  install -D -m 644 "$source_repo/$file" "$target_repo/$file"
done

cd "$target_repo"
# Odin-ml and budgie group first-party imports differently; let budgie's config decide.
uv run ruff check --fix --select I "${files[@]}"
uv run ruff format "${files[@]}"

if [[ -z "$(git status --porcelain)" ]]; then
  echo "budgie-ml-service is already in sync."
  exit 0
fi

uv run ruff check .
uv run ruff format --check .
uv run mypy app
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q

source_sha="$(git -C "$source_repo" rev-parse --short HEAD)"
source_subject="$(git -C "$source_repo" log -1 --format=%s)"
git add -- "${files[@]}"
git commit -q -F - <<EOF
chore: sync from Odin-ml@$source_sha

Source: $source_subject
EOF
git log -1 --stat --format='%h %s'

if [[ "$push" == "--push" ]]; then
  git push origin main
fi
