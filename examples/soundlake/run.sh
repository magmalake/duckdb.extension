#!/usr/bin/env bash
# Generate the dataset, check the kernels against numpy, run the tour.
#
#     ./run.sh            everything except the benchmark
#     ./run.sh bench      everything, including the benchmark
#
# The extension and the Mojo bridge must already be built: `pixi run build` in
# the repository root. Python comes from a throwaway venv, because duckdb has
# to be pinned to the submodule's version and that is not a pin anyone wants in
# their own environment.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"
EXTENSION="$ROOT/build/release/extension/mlake/mlake.duckdb_extension"
BRIDGE="$ROOT/mojo/build/libmlake_bridge$(uname -s | grep -qi darwin && echo .dylib || echo .so)"
for built in "$EXTENSION" "$BRIDGE"; do
	[ -f "$built" ] || { echo "missing $built — run 'pixi run build' in $ROOT" >&2; exit 1; }
done
export MLAKE_BRIDGE="$BRIDGE"

VENV="${TMPDIR:-/tmp}/soundlake-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet -r requirements.txt

[ -f data/recordings.parquet ] || "$VENV/bin/python" generate.py

echo
echo "── checking every feature against numpy ──────────────────────────────"
"$VENV/bin/python" verify.py

echo
echo "── the tour ─────────────────────────────────────────────────────────"
"$ROOT/build/release/duckdb" -unsigned < tour.sql

if [ "${1:-}" = "bench" ]; then
	echo
	echo "── against the alternatives ─────────────────────────────────────────"
	"$VENV/bin/python" -W ignore bench.py
fi
