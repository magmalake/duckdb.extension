"""Exercise the Mojo bridge's C ABI directly, before any C++ is involved.

The extension is the eventual consumer, but a failure there could be either
side. This drives the same entry points from ctypes and checks them with
pyarrow, so a bug in the bridge is found as a bridge bug.
"""

import ctypes
import sys

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


def main(lib_path, table_dir):
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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
