"""What the extension is actually worth, against the ways you would do this
without one.

Four ways to get the same column, over the same clips, producing the same
number to twelve decimal places:

    mojo            the extension's scalar function
    py-arrow        a DuckDB Python UDF handed whole pyarrow arrays
    py-native       a DuckDB Python UDF handed one blob at a time
    fetch+numpy     pull every blob into Python, loop there

and, separately, the extension's table function over the files themselves,
which is a different input and so a different row rather than a competitor.

Two features, because they have different shapes. `rms_db` is a pass over the
samples: nearly all of its cost is decoding the container and crossing the
language boundary. `centroid_hz` runs sixty-odd fast Fourier transforms per
clip, so it is dominated by arithmetic numpy is extremely good at. Reporting
only the first would flatter the extension; reporting only the second would
hide what it is for.

Every configuration is also run at `threads=1`. The single-threaded row is the
one that compares implementations; the default-threads row is what the query
actually costs, and the gap between them is the part that is about DuckDB
parallelising a function it can call from every worker.
"""

from __future__ import annotations

import argparse

import sys
import time

import duckdb
import numpy as np
import pyarrow as pa
from duckdb.sqltypes import BLOB, DOUBLE

import reference
from _connect import connect, require_data

FEATURES = {
    "rms_db": (reference.rms_db, "one pass over the samples"),
    "centroid_hz": (reference.centroid_hz, "sixty-odd FFTs per clip"),
}


def _python_feature(name: str):
    """The numpy implementation, as a function of one WAV blob."""
    fn, _ = FEATURES[name]

    def compute(clip) -> float | None:
        try:
            samples, rate = reference.decode(bytes(clip))
        except Exception:
            return None
        return fn(samples, rate) if name == "centroid_hz" else fn(samples)

    return compute


def _vectorised(compute):
    """A one-argument callable, because DuckDB reads the UDF's arity from its
    signature and a default argument would count as a second parameter."""

    def over_chunk(column) -> pa.Array:
        # A whole chunk at a time, which is the fair version: the per-call
        # overhead is paid once for two thousand clips rather than once each.
        # Every blob still becomes a Python object on the way in, and that is
        # the cost the extension does not pay.
        return pa.array([compute(v) for v in column.to_pylist()], type=pa.float64())

    return over_chunk


def register_udfs(con: duckdb.DuckDBPyConnection) -> None:
    for name in FEATURES:
        compute = _python_feature(name)
        con.create_function(f"py_native_{name}", compute, [BLOB], DOUBLE, type="native")
        con.create_function(
            f"py_arrow_{name}", _vectorised(compute), [BLOB], DOUBLE, type="arrow"
        )


def timed(fn, repeat: int) -> tuple[float, object]:
    """Best of `repeat`, because the thing being measured is the work and not
    whatever else the machine was doing."""
    best, answer = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        answer = fn()
        best = min(best, time.perf_counter() - start)
    return best, answer


def run_sql(con, sql: str):
    rows = con.execute(sql).fetchone()
    return (rows[0], round(rows[1], 9))


def fetch_and_loop(con, name: str):
    """What a data scientist writes first: get the blobs, loop over them."""
    compute = _python_feature(name)
    blobs = con.execute("SELECT clip FROM recordings").fetch_arrow_table().column("clip")
    values = [compute(v) for v in blobs.to_pylist()]
    live = [v for v in values if v is not None]
    return (len(live), round(float(np.mean(live)), 9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-slow", action="store_true", help="drop py-native, which is the slow one")
    args = ap.parse_args()

    data = require_data()
    con = connect()
    con.execute(f"CREATE VIEW recordings AS SELECT * FROM read_parquet('{data / 'recordings.parquet'}')")
    register_udfs(con)

    clips, total_bytes = con.execute(
        "SELECT count(*), sum(octet_length(clip)) FROM recordings"
    ).fetchone()
    threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    print(f"{clips} clips, {total_bytes / 1e6:.1f} MB of 16-bit PCM, {threads} threads available")
    print()

    for name, (_, shape) in FEATURES.items():
        print(f"── {name} — {shape} " + "─" * (44 - len(name) - len(shape)))
        contenders = [
            ("mojo", lambda n=name: run_sql(con, f"SELECT count({n}), avg({n}) FROM (SELECT mlake_{n}(clip) AS {n} FROM recordings)")),
            ("py-arrow", lambda n=name: run_sql(con, f"SELECT count(v), avg(v) FROM (SELECT py_arrow_{n}(clip) AS v FROM recordings)")),
            ("fetch+numpy", lambda n=name: fetch_and_loop(con, n)),
        ]
        if not args.skip_slow:
            contenders.insert(
                2,
                ("py-native", lambda n=name: run_sql(con, f"SELECT count(v), avg(v) FROM (SELECT py_native_{n}(clip) AS v FROM recordings)")),
            )

        results = []
        for threads_setting in (1, 0):
            con.execute(f"SET threads = {threads_setting}" if threads_setting else f"SET threads = {threads}")
            for label, fn in contenders:
                seconds, answer = timed(fn, args.repeat)
                results.append((label, threads_setting or int(threads), seconds, answer))

        baseline = min(s for label, t, s, _ in results if label == "mojo" and t == 1)
        answers = {a for _, _, _, a in results}
        for label, t, seconds, answer in results:
            rate = clips / seconds
            note = "" if len(answers) == 1 else f"   -> {answer}"
            print(
                f"  {label:<12} {t:>2} thread{'s' if t != 1 else ' '}  "
                f"{seconds * 1000:8.1f} ms  {rate:8.0f} clips/s  "
                f"{baseline / seconds:6.2f}x{note}"
            )
        if len(answers) != 1:
            print(f"  !! the implementations disagree: {answers}")
        print()

    # A different input — files rather than a column — so it is reported and
    # not ranked. It is what a lakehouse query actually does, and it includes
    # reading 74 MB off disk.
    con.execute(f"SET threads = {threads}")
    seconds, answer = timed(
        lambda: run_sql(
            con,
            "SELECT count(centroid_hz), avg(centroid_hz) FROM mlake_audio_features("
            f"'{data / 'clips' / '*.wav'}')",
        ),
        args.repeat,
    )
    print(f"── from the files themselves " + "─" * 26)
    print(
        f"  mlake_audio_features  {threads} threads  {seconds * 1000:8.1f} ms  "
        f"{clips / seconds:8.0f} clips/s   (reads {total_bytes / 1e6:.0f} MB and computes every feature)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
