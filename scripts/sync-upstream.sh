#!/bin/bash
# Sync this fork (Chedrian07/exo-10gbe-wsl) with upstream exo-explore/exo.
#
# This fork is a clean commit series on top of upstream, so we REBASE the fork
# onto new upstream rather than merge (keeps history linear, conflicts surface
# per-commit). Every fork change in an upstream file is tagged with a
#   FORK(exo-10gbe-wsl): ...
# comment, and FORK.md documents each divergence + how to resolve it.
#
# Usage: ./scripts/sync-upstream.sh
set -euo pipefail

UPSTREAM="${UPSTREAM:-upstream}"
BRANCH="${BRANCH:-main}"
UPSTREAM_URL="https://github.com/exo-explore/exo.git"

if ! git remote get-url "$UPSTREAM" >/dev/null 2>&1; then
  echo "Adding remote '$UPSTREAM' -> $UPSTREAM_URL"
  git remote add "$UPSTREAM" "$UPSTREAM_URL"
fi

echo "Fetching $UPSTREAM ..."
git fetch "$UPSTREAM" --quiet

merge_base="$(git merge-base HEAD "$UPSTREAM/$BRANCH")"
upstream_head="$(git rev-parse "$UPSTREAM/$BRANCH")"

if [ "$merge_base" = "$upstream_head" ]; then
  echo "Already up to date — $UPSTREAM/$BRANCH has nothing new. Nothing to sync."
  exit 0
fi

echo
echo "New upstream commits ($merge_base..$UPSTREAM/$BRANCH):"
git log --oneline "$merge_base..$UPSTREAM/$BRANCH"

echo
echo "Files BOTH the fork and upstream changed since the merge base (= likely conflicts):"
comm -12 \
  <(git diff --name-only "$merge_base" HEAD | sort) \
  <(git diff --name-only "$merge_base" "$UPSTREAM/$BRANCH" | sort) \
  | sed 's/^/  /' || true

cat <<'NOTE'

To sync (recommended: rebase):
  git rebase upstream/main
  # On conflict, grep the file for FORK(exo-10gbe-wsl) and consult FORK.md.
  # uv.lock conflicts: don't hand-merge — `git checkout --theirs uv.lock && uv lock`
  # Then run the pre-commit checks:
  #   uv run basedpyright && uv run ruff check && nix fmt && uv run pytest
  # Finally: git push --force-with-lease

NOTE

read -r -p "Run 'git rebase $UPSTREAM/$BRANCH' now? [y/N] " ans
if [ "${ans:-N}" = "y" ] || [ "${ans:-N}" = "Y" ]; then
  git rebase "$UPSTREAM/$BRANCH"
else
  echo "Skipped. Run the rebase yourself when ready."
fi
