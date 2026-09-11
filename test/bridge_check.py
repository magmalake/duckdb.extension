"""Exercise the Mojo bridge's C ABI directly, before any C++ is involved.

The extension is the eventual consumer, but a failure there could be either
side. This drives the same entry points from ctypes and checks them with
pyarrow, so a bug in the bridge is found as a bridge bug.
"""

import ctypes
import math
import sys
from pathlib import Path

import pyarrow as pa

SPLIT_SIZE = 4096
"""Small on purpose. The fixture is a few tens of kilobytes, so at the
production default its files would never divide and the disjointness check
below would be testing one ticket per file — which a broken splitter passes.
"""


class Bridge:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        p, i = ctypes.c_void_p, ctypes.c_int64
        lib.mlake_plan.restype, lib.mlake_plan.argtypes = i, [p, i, i, p]
        lib.mlake_plan_count.restype, lib.mlake_plan_count.argtypes = i, [i]
        lib.mlake_plan_ticket.restype, lib.mlake_plan_ticket.argtypes = i, [i, i, p]
        lib.mlake_plan_free.argtypes = [i]
        lib.mlake_schema.restype, lib.mlake_schema.argtypes = i, [p, i, p, p]
        lib.mlake_read_split.restype = i
        lib.mlake_read_split.argtypes = [p, i, p, i, i, p, p]
        lib.mlake_free.argtypes = [i]
        lib.mlake_audio_batch.restype = i
        lib.mlake_audio_batch.argtypes = [i, p, p, i, p, p, p]
        lib.mlake_audio_plan.restype, lib.mlake_audio_plan.argtypes = i, [p, i, i, p]
        lib.mlake_audio_schema.restype, lib.mlake_audio_schema.argtypes = i, [p, p]
        lib.mlake_audio_read.restype, lib.mlake_audio_read.argtypes = i, [p, i, p, p]
        self.lib = lib

    def _err(self):
        return (ctypes.c_int64 * 2)()

    def _check(self, err, what):
        if err[0]:
            msg = ctypes.string_at(err[0]).decode()
            self.lib.mlake_free(err[0])
            raise RuntimeError(f"{what}: {msg}")
        raise RuntimeError(f"{what}: failed with no message")

    def plan(self, table_dir, split_size=SPLIT_SIZE):
        b = table_dir.encode()
        err = self._err()
        h = self.lib.mlake_plan(b, len(b), split_size, err)
        if not h:
            self._check(err, "plan")
        n = self.lib.mlake_plan_count(h)
        out = []
        for i in range(n):
            ln = ctypes.c_int64()
            ptr = self.lib.mlake_plan_ticket(h, i, ctypes.byref(ln))
            out.append(ctypes.string_at(ptr, ln.value).decode())
        self.lib.mlake_plan_free(h)
        return out

    def schema(self, table_dir):
        b = table_dir.encode()
        err = self._err()
        # The caller owns the struct, so pyarrow's importer frees it: nine
        # words is an ArrowSchema, and _import_from_c releases what it takes.
        slot = (ctypes.c_int64 * 9)()
        if not self.lib.mlake_schema(b, len(b), ctypes.byref(slot), err):
            self._check(err, "schema")
        return pa.Schema._import_from_c(ctypes.addressof(slot))

    def read(self, table_dir, ticket, split_size=SPLIT_SIZE):
        d, t = table_dir.encode(), ticket.encode()
        err = self._err()
        slot = (ctypes.c_int64 * 5)()
        if not self.lib.mlake_read_split(d, len(d), t, len(t), split_size, ctypes.byref(slot), err):
            self._check(err, "read_split")
        return pa.RecordBatchReader._import_from_c(ctypes.addressof(slot)).read_all()

    # ── audio ───────────────────────────────────────────────────────────────

    def audio_batch(self, kind, clips):
        """One feature for a list of WAV blobs, the way DuckDB calls it: a
        whole vector in one call, described in place by address and length."""
        n = len(clips)
        buffers = [ctypes.create_string_buffer(c, len(c)) for c in clips]
        ptrs = (ctypes.c_int64 * n)(*[ctypes.addressof(b) for b in buffers])
        lens = (ctypes.c_int64 * n)(*[len(c) for c in clips])
        values = (ctypes.c_double * n)()
        valid = (ctypes.c_uint8 * n)()
        err = self._err()
        if not self.lib.mlake_audio_batch(
            kind, ptrs, lens, n, values, valid, err
        ):
            self._check(err, "audio_batch")
        return [values[i] if valid[i] else None for i in range(n)]

    def audio_plan(self, pattern, batch_rows=0):
        b = pattern.encode()
        err = self._err()
        h = self.lib.mlake_audio_plan(b, len(b), batch_rows, err)
        if not h:
            self._check(err, "audio_plan")
        out = []
        for i in range(self.lib.mlake_plan_count(h)):
            ln = ctypes.c_int64()
            ptr = self.lib.mlake_plan_ticket(h, i, ctypes.byref(ln))
            out.append(ctypes.string_at(ptr, ln.value).decode())
        self.lib.mlake_plan_free(h)
        return out

    def audio_schema(self):
        err = self._err()
        slot = (ctypes.c_int64 * 9)()
        if not self.lib.mlake_audio_schema(ctypes.byref(slot), err):
            self._check(err, "audio_schema")
        return pa.Schema._import_from_c(ctypes.addressof(slot))

    def audio_read(self, ticket):
        t = ticket.encode()
        err = self._err()
        slot = (ctypes.c_int64 * 5)()
        if not self.lib.mlake_audio_read(t, len(t), ctypes.byref(slot), err):
            self._check(err, "audio_read")
        return pa.RecordBatchReader._import_from_c(ctypes.addressof(slot)).read_all()


RMS_DB, PEAK_DB, CENTROID_HZ, ZCR, DURATION_S = range(5)
"""The feature selectors, in the order mojo/src/bridge.mojo declares them."""


def check_audio(b, clip_dir):
    """The audio half of the ABI, against clips with known answers.

    Checked here and not only in SQL because a wrong number could come from the
    kernel, from the vector gather in the C++, or from the two lists of feature
    constants having drifted apart. Driving the ABI straight from ctypes
    removes the last two from the picture.
    """
    clips = Path(clip_dir)
    tone = (clips / "tone-1000.wav").read_bytes()
    bright = (clips / "tone-4000.wav").read_bytes()
    quiet = (clips / "silence.wav").read_bytes()
    junk = (clips / "broken.wav").read_bytes()

    # A half-scale sine: RMS is amplitude over root two, peak is the amplitude,
    # and it crosses zero twice per cycle.
    batch = [tone, bright, quiet, junk]
    rms = b.audio_batch(RMS_DB, batch)
    assert math.isclose(rms[0], -9.0309, abs_tol=0.01), rms
    assert math.isclose(rms[1], -15.0515, abs_tol=0.01), rms
    assert math.isclose(rms[2], -120.0, abs_tol=1e-9), rms
    # The bad clip is a null in the middle of a batch, and the good rows around
    # it are unaffected — the compaction the C++ does has to put every answer
    # back on the row it came from.
    assert rms[3] is None, rms

    centroid = b.audio_batch(CENTROID_HZ, batch)
    assert math.isclose(centroid[0], 1000.0, abs_tol=15.0), centroid
    assert math.isclose(centroid[1], 4000.0, abs_tol=15.0), centroid
    assert centroid[2] is None, "silence has no centroid"

    zcr = b.audio_batch(ZCR, [tone, bright])
    assert math.isclose(zcr[0], 0.125, abs_tol=0.001), zcr
    assert math.isclose(zcr[1], 0.5, abs_tol=0.001), zcr

    assert b.audio_batch(DURATION_S, [tone])[0] == 1.0
    assert b.audio_batch(PEAK_DB, [quiet])[0] == -120.0
    print(f"audio: {len(batch)} clips measured through the scalar ABI")

    # The empty batch is what the schema is derived from, so it has to work.
    schema = b.audio_schema()
    assert schema.names[0] == "path" and schema.names[-1] == "error", schema
    print(f"audio schema: {len(schema)} columns — {', '.join(schema.names)}")

    # Two clips per ticket: with six matching files that is three tickets, so
    # the batching is exercised rather than assumed.
    tickets = b.audio_plan(str(clips / "*.wav"), batch_rows=2)
    assert len(tickets) == 3, tickets
    table = pa.concat_tables([b.audio_read(t) for t in tickets])
    assert table.num_rows == 6, table.num_rows
    assert table.schema == schema, f"{table.schema}\n{schema}"

    # Same numbers from the file path as from the blob. They run the same
    # kernel, so a difference means one of the two ways in is wrong.
    by_path = {
        Path(p).name: v
        for p, v in zip(table.column("path").to_pylist(), table.column("rms_db").to_pylist())
    }
    assert math.isclose(by_path["tone-1000.wav"], rms[0], abs_tol=1e-12), by_path
    assert by_path["broken.wav"] is None
    errors = dict(zip(table.column("path").to_pylist(), table.column("error").to_pylist()))
    assert any(e and "RIFF" in e for e in errors.values()), errors
    print(f"audio: {table.num_rows} rows across {len(tickets)} tickets, 1 with an error")


def main(lib_path, table_dir, clip_dir=None):
    b = Bridge(lib_path)

    schema = b.schema(table_dir)
    print(f"schema: {len(schema)} columns — {', '.join(schema.names)}")

    tickets = b.plan(table_dir)
    print(f"plan: {len(tickets)} tickets")
    assert tickets, "a non-empty table should plan at least one ticket"
    # The fixture is three data files, one of which has room to divide, so a
    # working splitter produces more tickets than files.
    assert len(tickets) > 3, f"split planning produced only {len(tickets)} tickets"

    # Every ticket names the same snapshot: a plan is one consistent read.
    snapshots = {t.split("|", 1)[0] for t in tickets}
    assert len(snapshots) == 1, f"plan spans snapshots {snapshots}"

    parts = [b.read(table_dir, t) for t in tickets]
    for part in parts:
        assert part.schema == schema, (
            f"a split's schema differs from the table's:\n{part.schema}\n{schema}"
        )
    table = pa.concat_tables(parts)
    print(f"read: {table.num_rows} rows across {len(parts)} splits")

    # The union of the tickets is the table, with nothing repeated and nothing
    # lost. If splitting is ever wrong this is what catches it.
    ids = table.column("id").to_pylist()
    assert len(ids) == len(set(ids)), "a row was read by two splits"
    print(f"ok: {len(set(ids))} distinct ids, no overlap between splits")

    if clip_dir:
        check_audio(b, clip_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
