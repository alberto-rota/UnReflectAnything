#!/usr/bin/env bash
set -euo pipefail

# --- User knobs ---
REPO_DIR="${REPO_DIR:-$HOME/UnReflectAnything}"
WS_DIR="${HOME:-/anvme/workspace/v120bb18-unreflectanything}"
SNAP_ROOT="${SNAP_ROOT:-$HOME/snapshots}"     # snapshots live here
SBATCH_FILE="${1:-train_a100_40_asap.sbatch}"   # pass sbatch filename or defaults
EXCLUDES=(
  ".git"
  ".venv"
  "__pycache__"
  ".mypy_cache"
  ".pytest_cache"
  ".ruff_cache"
  "runs"
  "wandb"
  "datasets"       # keep using central datasets
  "results"        # keep using central results
  ".ipynb_checkpoints"
  "demos"
  "sandboxes"
)

mkdir -p "$SNAP_ROOT"

# Timestamp + commit id
ts="$(date +%Y%m%d_%H%M%S)"
pushd "$REPO_DIR" >/dev/null

# Try to get git metadata (works even with uncommitted changes)
commit="no-git"
dirty=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commit="$(git rev-parse --short HEAD)"
  if ! git diff --quiet || ! git diff --cached --quiet; then
    dirty="-dirty"
  fi
fi

snap_dir="${SNAP_ROOT}/${ts}_${commit}${dirty}"
mkdir -p "$snap_dir"

# rsync working tree to snapshot, respecting excludes
rsync -a --delete \
  $(printf -- '--exclude=%q ' "${EXCLUDES[@]}") \
  "$REPO_DIR/." "$snap_dir/"

# Metadata for reproducibility
{
  echo "SNAPSHOT_TIME: $ts"
  echo "COMMIT: $commit$dirty"
  echo "HOST: $(hostname)"
  echo "USER: $USER"
  echo "PYTHON: $(command -v python || true)"
  echo
  echo "=== git status ==="
  git status || true
  echo
  echo "=== git diff (unstaged) ==="
  git diff || true
  echo
  echo "=== git diff (staged) ==="
  git diff --cached || true
} > "$snap_dir/SNAPSHOT_METADATA.txt" 2>&1 || true

# Freeze Python environment (best-effort)
if [ -f "$REPO_DIR/.venv/bin/pip" ]; then
  "$REPO_DIR/.venv/bin/pip" freeze > "$snap_dir/requirements-freeze.txt" || true
else
  pip freeze > "$snap_dir/requirements-freeze.txt" 2>/dev/null || true
fi

popd >/dev/null

echo "📦 Snapshot created at: $snap_dir"
echo "📝 Metadata: $snap_dir/SNAPSHOT_METADATA.txt"

# Submit the job, exporting SNAPSHOT_DIR so the sbatch script can cd there
sbatch --export=ALL,SNAPSHOT_DIR="$snap_dir" "$SBATCH_FILE"
