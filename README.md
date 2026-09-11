# duckdb.extension

A DuckDB extension that puts Mojo inside the database: reading Apache Iceberg
tables through [magmalake](https://magmalake.org)'s Mojo read stack, and
computing things SQL cannot express over the data it finds there.

```sql
LOAD mlake;
SELECT region, count(*), sum(amount)
FROM mlake_scan('warehouse/db/taxi')
GROUP BY region;
```

No server, no socket, no IPC frames: the Mojo side hands DuckDB an Arrow
`ArrowArrayStream` and DuckDB reads the buffers where they already are.

## The seam

[`flight.mojo`](https://github.com/magmalake/flight.mojo) already serves these
tables over Arrow Flight, and a DuckDB client could use that. It would also
encode every batch to IPC, push it through a socket and decode it again — to
move data between two libraries in one address space. Flight is the right
answer across a network and the wrong one across a function call.

The Arrow C Data Interface is the answer for the second case. Both sides agree
on a struct of pointers, so what crosses is an address.

```
  DuckDB                     ArrowArrayStream                 Mojo
  ──────                     ────────────────                 ────
  mlake_scan(…)   ──plan──▶                        iceberg.mojo  scan planning
                  ◀─tickets─                       parquet.mojo  decoding
  thread 1  ──ticket──▶  get_next() ──▶ batch      arrow-mlake   C Data Interface
  thread 2  ──ticket──▶  get_next() ──▶ batch
```

## Units of work

Planning returns **tickets**, one per scan task, each naming a snapshot, a data
file and a byte range within it. A file that records its row-group offsets
divides into several, so one large file is not one thread.

Each ticket is read as its own `ArrowArrayStream`. Two properties follow, and
both are what this design is for:

| | |
|---|---|
| **Bounded memory** | A thread holds one split, not one table. |
| **Parallelism** | Threads take tickets from a shared cursor and never touch each other's rows. |

The union of the tickets is the table, with nothing repeated and nothing lost —
asserted in `test/sql/mlake.test` at a split size small enough to make the
fixture actually divide.

`split_size` is a named parameter because the right value depends on the data:

```sql
SELECT count(*) FROM mlake_scan('warehouse/db/taxi', split_size => 4096);
```

## Snapshot isolation

Every ticket carries the snapshot it was planned against, and reading one
re-plans at that snapshot rather than at whatever is current. A commit landing
mid-query cannot change what the query returns.

## Computing in Mojo, not just reading

The seam works in the other direction too: a scalar function runs a Mojo kernel
on the bytes of a DuckDB vector, in place. DuckDB evaluates a scalar function
over a whole vector, so the bridge takes a whole vector — 2048 blobs described
by address and length, one call — rather than putting a language boundary in
the inner loop of every query.

**[`examples/soundlake`](examples/soundlake)** is the worked example: five
audio features over a `BLOB` column, a table function over a directory of
recordings, and what both cost.

## Building

Two halves, and the Makefile builds both.

```sh
pixi run build     # the Mojo bridge, then the extension and a duckdb binary
pixi run test      # generates the fixtures, then runs the SQL tests
./build/release/duckdb -c "SELECT * FROM mlake_scan('…') LIMIT 5"
cd examples/soundlake && ./run.sh bench
```

The Mojo kernels have their own tests, which need neither DuckDB nor a fixture:

```sh
pixi run --manifest-path mojo/pixi.toml mojo run -I src tests/dsp_test.mojo
```

The C++ half is a normal DuckDB out-of-tree extension: `duckdb/` and
`extension-ci-tools/` are submodules, so `git clone --recursive`. The built
extension is named `mlake` — `LOAD mlake`, `mlake_scan(…)` — which is what a
DuckDB user sees; the repository carries magmalake's `.extension` suffix the
way its Mojo tins carry `.mojo`.

The Mojo half lives in `mojo/` with its own `pixi.toml`, because it needs the
`max-nightly` channel and a compiler pin this build has no reason to inherit.
It builds `libmlake_bridge`, which the extension **dlopens** rather than links:
a `.duckdb_extension` built here still loads on a machine where the bridge
lives elsewhere. `MLAKE_BRIDGE` overrides the path baked in at build time.

## The C ABI

`src/include/mlake_bridge.hpp` is the contract; `mojo/src/bridge.mojo`
implements it. Both sides pass strings as (pointer, length) and errors as a
two-word out-parameter — there is no last-error slot because the Mojo side has
no globals to put one in.

That constraint shaped the API. Mojo 1.0 has no placement-init for a type with
a destructor and no module-level globals, so there is no way to keep an open
scan alive between two calls from C. What crosses instead is a plan, and every
read is a fresh scan of one ticket — which turned out to be the shape that
gives DuckDB a unit of work per thread anyway.

## What is not here

**Projection pushdown.** `mlake_scan` reads every column and lets DuckDB
discard the rest. The Mojo scan can project by name, so the fix is to carry
DuckDB's column ids into `mlake_read_split`; until it is wired, claiming the
capability would mean returning the wrong columns rather than slow ones.

**Filter pushdown.** Iceberg planning already prunes partitions and row groups
using the table's own residual, so a `WHERE` clause reaches the scan only as
Iceberg metadata, never as a DuckDB filter.

**Writes.** Read path only.

**A lazy stream.** Every batch of a split is exported before DuckDB sees the
first one. The split bounds it, so the cost is a split rather than a table, but
a stream that read inside `get_next` would be better — and needs a scan that
can be resumed from inside a C callback.

## Licence

MIT. DuckDB and the Iceberg stack keep their own.
