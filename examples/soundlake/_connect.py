"""Open a DuckDB connection with the locally built extension loaded.

The extension is built from the DuckDB submodule at v1.4.1, so the `duckdb`
package in the venv has to be that version too: an extension is compiled
against one version's internals and refuses to load into another. requirements
pins it, and this checks it, because the error DuckDB gives otherwise names a
platform string rather than the cause.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"


def _built(name: str, hint: str) -> Path:
    path = ROOT / name
    if not path.exists():
        sys.exit(f"{path} is missing — run `pixi run build` in {ROOT}, then {hint}")
    return path


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    extension = _built(
        "build/release/extension/mlake/mlake.duckdb_extension",
        "re-run this script",
    )
    bridge = _built(
        "mojo/build/libmlake_bridge" + (".dylib" if sys.platform == "darwin" else ".so"),
        "re-run this script",
    )
    # The extension dlopens the bridge; the path baked in at build time already
    # points here, but saying so explicitly means a stale build cannot quietly
    # pick up a different one.
    os.environ.setdefault("MLAKE_BRIDGE", str(bridge))

    con = duckdb.connect(database, config={"allow_unsigned_extensions": "true"})
    try:
        con.execute(f"LOAD '{extension}'")
    except duckdb.Error as e:
        sys.exit(
            f"could not load the extension: {e}\n"
            f"duckdb {duckdb.__version__} is installed; this extension is built "
            f"for v1.4.1. `pip install 'duckdb==1.4.1'`."
        )
    return con


def require_data() -> Path:
    if not (DATA / "recordings.parquet").exists():
        sys.exit(f"no dataset in {DATA} — run `python generate.py` first")
    return DATA
