#!/usr/bin/env bash
# Generate the Iceberg fixture, then run the SQL tests against it.
#
# The fixture needs PyIceberg and the extension needs the Mojo bridge; both are
# built by `pixi run test`, which calls this. Run it directly only if you have
# already built both.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${TMPDIR:-/tmp}/duckdb-extension-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26' 'pyiceberg[sql-sqlite]>=0.8,<1'

MLAKE_FIXTURE="$("$VENV/bin/python" test/make_fixture.py test/fixtures)"
export MLAKE_FIXTURE
echo "fixture: $MLAKE_FIXTURE"

# The bridge is dlopened, and the path baked in at build time points at the
# working tree — but say so explicitly so a stale build cannot pick up a
# different one.
MLAKE_BRIDGE="$PWD/mojo/build/libmlake_bridge$(uname -s | grep -qi darwin && echo .dylib || echo .so)"
export MLAKE_BRIDGE
[ -f "$MLAKE_BRIDGE" ] || { echo "no bridge at $MLAKE_BRIDGE — run 'pixi run bridge'" >&2; exit 1; }

# The C ABI on its own first. A failure in the SQL tests could be either side
# of the boundary; this one can only be the Mojo side.
"$VENV/bin/python" test/bridge_check.py "$MLAKE_BRIDGE" "$MLAKE_FIXTURE"

./build/release/test/unittest "$@" 'test/sql/*'
