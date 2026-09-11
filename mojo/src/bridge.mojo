"""The C ABI a DuckDB extension calls: read an Iceberg table, and measure a
recording.

Everything here is `@export`ed and takes or returns integers, because the
other side of this boundary is C++ and the only types both languages agree on
without ceremony are addresses and lengths.

Two halves that share their plumbing. The Iceberg half hands DuckDB rows it
read from somewhere else; the audio half hands DuckDB numbers it computed from
rows DuckDB already had. Both plan into tickets, both return Arrow, and the
audio half reuses the Iceberg half's plan handle unchanged, because a plan is a
list of strings either way.

## Why the API is stateless

The natural design would hand C++ an open scan and let it pull batches. This
Mojo cannot: `alloc` has no placement-init for a type with a destructor, and
there are no module-level globals, so there is no way to keep a `BatchReader`
alive between two calls from C.

What survives the boundary instead is a **plan** — a list of tickets, each a
string naming a snapshot, a file and a byte range — and every read is a fresh
scan of exactly one ticket. That is the same shape `flight.mojo` serves over
the wire, and it turns out to be the right one here too: DuckDB gets a unit of
work per thread, and the memory held is one split rather than one table.

## Errors

A call that can fail takes `err`, the address of two words. On failure it
returns 0 and writes a message pointer and its length there; on success both
stay 0. Free the message with `mlake_free`. No global error slot, for the same
reason there is no global scan: there are no globals.

## Ownership

Every address this returns is freed by a named call and by nothing else:
`mlake_plan_free` for a plan, `mlake_free` for a string, and for an
`ArrowArrayStream` the consumer's own `release` callback, which is what the C
Data Interface already requires.
"""

from std.memory.alloc import unsafe_alloc

from arrow_mlake.arrow import ArrayArena, ArrayData, ArrowType, AT_STRUCT
from arrow_mlake.carrow import ExportedArray, export_c
from arrow_mlake.carrow_import import release_c_array
from arrow_mlake.carrow_stream import export_stream_of
from iceberg.catalog.filesystem import find_latest_metadata
from iceberg.io import FileIO
from iceberg.metadata import TableMetadata
from iceberg.nested import empty_tree
from iceberg.read import ScanOptions
from iceberg.scan import TableScan

from clips import features_batch, find_clips
from dsp import peak_db, rms_db, spectral_centroid, zero_crossing_rate
from wav import parse_wav

comptime DEFAULT_SPLIT_SIZE = 16 * 1024 * 1024
"""Target bytes per ticket when the caller does not choose.

The unit of work without splitting is the whole data file, so a single large
file is a single DuckDB thread no matter how many are free. 16 MiB is small
enough that an ordinary file divides and large enough that a split is still a
sequential read.

Callers override it — `mlake_scan(path, split_size => 4096)` — and the tests
do, because the fixture is a few tens of kilobytes and would never divide at
the default, leaving a broken splitter to pass unnoticed.
"""


def _split_size(requested: Int) -> Int:
    return DEFAULT_SPLIT_SIZE if requested <= 0 else requested


# ── strings across the boundary ─────────────────────────────────────────────
def _in(ptr: Int, length: Int) -> String:
    """Copy a (pointer, length) pair from C into a Mojo string.

    A copy because the caller owns those bytes and may free them the moment
    this returns.
    """
    var bytes = List[UInt8]()
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ptr)
    for i in range(length):
        bytes.append(p[unsafe_offset=i])
    return String(unsafe_from_utf8=bytes)


def _out(s: String) -> Int:
    """Copy a Mojo string to a NUL-terminated buffer C frees with `mlake_free`.

    NUL-terminated *and* length-reported: C++ wants the length, and the
    terminator costs one byte and makes the pointer printable in a debugger.
    """
    var bytes = s.as_bytes()
    var buf = unsafe_alloc[UInt8](len(bytes) + 1)
    for i in range(len(bytes)):
        buf[unsafe_offset=i] = bytes[i]
    buf[unsafe_offset=len(bytes)] = 0
    return Int(buf)


def _fail(err: Int, message: String) -> Int:
    """Park a message on the caller's two words and return the failure value.
    """
    if err != 0:
        var slot = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=err)
        slot[unsafe_offset=0] = _out(message)
        slot[unsafe_offset=1] = message.byte_length()
    return 0


def _move_into(src: Int, dst: Int, words: Int, free_shell: Bool):
    """Move a C struct into the caller's own storage.

    The caller hands us the address of an `ArrowSchema` or `ArrowArrayStream`
    it already owns — DuckDB keeps both inside its own wrappers — so returning
    a heap address would leave an allocation only this library can free.
    Moving the words leaves the caller owning exactly one thing, which its own
    `release` callback frees.

    `free_shell` is the difference between the two structs and is not
    cosmetic. A stream is its own allocation and freeing it is right. An
    exported schema sits *inside* the block its `private_data` points at, so
    its address is not an allocation at all — freeing it is a bad free, and
    the block is freed later by the release callback the caller now holds.
    Zeroing instead is what keeps the handover a move: one releasable copy,
    not two.
    """
    var s = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=src)
    var d = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(words):
        d[unsafe_offset=i] = s[unsafe_offset=i]
    if free_shell:
        s.unsafe_free()
    else:
        for i in range(words):
            s[unsafe_offset=i] = 0


@export("mlake_free")
def mlake_free(ptr: Int) abi("C") -> None:
    """Free one string this library returned. Not for plans or streams."""
    if ptr != 0:
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ptr).unsafe_free()


# ── opening a table ─────────────────────────────────────────────────────────
def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _open(table_dir: String) raises -> TableScan:
    """A scan over whatever metadata is current in `table_dir`."""
    var io = FileIO.local()
    var meta_path = find_latest_metadata(io, table_dir)
    var metadata = TableMetadata.parse(_read_text(meta_path))
    return TableScan(metadata^, io^)


# ── the plan ────────────────────────────────────────────────────────────────
# Layout, all 8-byte words: [n, (ptr, len) * n]. A plain array rather than a
# Mojo collection because it has to stay valid between two calls from C, and
# nothing with a destructor can.


@export("mlake_plan")
def mlake_plan(
    dir_ptr: Int, dir_len: Int, split_size: Int, err: Int
) abi("C") -> Int:
    """Plan a scan; returns a handle whose tickets are the units of work.

    Planning reads manifests, not data files: partitions are pruned, delete
    files attached and a residual computed per task before anything is opened.
    """
    try:
        var table_dir = _in(dir_ptr, dir_len)
        var scan = _open(table_dir).with_split_size(_split_size(split_size))
        if not scan.has_any_snapshot():
            # An empty table is not an error. Zero tickets and a schema is a
            # complete answer, and DuckDB will select zero rows from it.
            var empty = unsafe_alloc[Int](1)
            empty[unsafe_offset=0] = 0
            return Int(empty)

        var snapshot_id = scan.snapshot().snapshot_id
        var tasks = scan.plan_files()
        var words = unsafe_alloc[Int](1 + 2 * len(tasks))
        words[unsafe_offset=0] = len(tasks)
        for i in range(len(tasks)):
            ref t = tasks[i]
            var ticket = (
                String(snapshot_id)
                + String("|")
                + String(t.start)
                + String("|")
                + String(t.length)
                + String("|")
                + t.data_file.file_path
            )
            words[unsafe_offset=1 + 2 * i] = _out(ticket)
            words[unsafe_offset=2 + 2 * i] = ticket.byte_length()
        return Int(words)
    except e:
        return _fail(err, String(e))


@export("mlake_plan_count")
def mlake_plan_count(plan: Int) abi("C") -> Int:
    if plan == 0:
        return 0
    return Pointer[Int, MutUntrackedOrigin](unsafe_from_address=plan)[]


@export("mlake_plan_ticket")
def mlake_plan_ticket(plan: Int, i: Int, len_out: Int) abi("C") -> Int:
    """The `i`th ticket. The plan owns it; do not free it separately."""
    if plan == 0:
        return 0
    var words = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=plan)
    if i < 0 or i >= words[unsafe_offset=0]:
        return 0
    if len_out != 0:
        Pointer[Int, MutUntrackedOrigin](unsafe_from_address=len_out)[] = words[
            unsafe_offset=2 + 2 * i
        ]
    return words[unsafe_offset=1 + 2 * i]


@export("mlake_plan_free")
def mlake_plan_free(plan: Int) abi("C") -> None:
    if plan == 0:
        return
    var words = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=plan)
    for i in range(words[unsafe_offset=0]):
        mlake_free(words[unsafe_offset=1 + 2 * i])
    words.unsafe_free()


# ── the schema ──────────────────────────────────────────────────────────────
@export("mlake_schema")
def mlake_schema(
    dir_ptr: Int, dir_len: Int, out_ptr: Int, err: Int
) abi("C") -> Int:
    """Fill the caller's `ArrowSchema` with the table's schema; 1 on success.

    Built from a zero-row array of the table's type rather than from a batch,
    so a table with no rows — or one whose every file was pruned — still binds
    to the right columns instead of failing or guessing.
    """
    try:
        var table_dir = _in(dir_ptr, dir_len)
        var scan = _open(table_dir)
        var schema = scan.schema()
        var arena = ArrayArena()
        var root = empty_tree(
            arena, schema.store, schema.root, String("row"), 0, False
        )
        var exported = export_c(arena, root)
        var pair = exported.into_raw()
        # The array is a zero-row placeholder; only the schema was wanted.
        release_c_array(pair[0])
        _move_into(pair[1], out_ptr, 9, False)  # nine words per ArrowSchema
        return 1
    except e:
        return _fail(err, String(e))


# ── reading one ticket ──────────────────────────────────────────────────────
def _decode_ticket(ticket: String) raises -> Tuple[Int64, Int64, Int64, String]:
    """Split on the first three bars; the rest is the path, which may contain
    one of its own."""
    var bytes = ticket.as_bytes()
    var cuts = List[Int]()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(124):  # "|"
            cuts.append(i)
            if len(cuts) == 3:
                break
    if len(cuts) < 3:
        raise Error(
            "mlake: malformed ticket, expected"
            " '<snapshot>|<start>|<length>|<path>'"
        )
    var snap = Int64(atol(String(unsafe_from_utf8=bytes[: cuts[0]])))
    var start = Int64(
        atol(String(unsafe_from_utf8=bytes[cuts[0] + 1 : cuts[1]]))
    )
    var length = Int64(
        atol(String(unsafe_from_utf8=bytes[cuts[1] + 1 : cuts[2]]))
    )
    var path = String(unsafe_from_utf8=bytes[cuts[2] + 1 :])
    return (snap, start, length, path^)


@export("mlake_read_split")
def mlake_read_split(
    dir_ptr: Int,
    dir_len: Int,
    ticket_ptr: Int,
    ticket_len: Int,
    split_size: Int,
    out_ptr: Int,
    err: Int,
) abi("C") -> Int:
    """Fill the caller's `ArrowArrayStream` with one ticket's rows; 1 on
    success.

    Planned at the ticket's own snapshot rather than at whatever is current,
    so a query that started before a commit keeps reading the table it was
    planned against. The rest of the plan still applies to this one file — its
    residual, its delete files — which is what makes the union of the tickets
    equal the table.
    """
    try:
        var table_dir = _in(dir_ptr, dir_len)
        var decoded = _decode_ticket(_in(ticket_ptr, ticket_len))
        var paths = List[String]()
        paths.append(decoded[3])
        var starts = List[Int64]()
        starts.append(decoded[1])

        var batches = (
            _open(table_dir)
            .use_snapshot(decoded[0])
            .with_split_size(_split_size(split_size))
            .to_batches_for_splits(paths^, starts^, ScanOptions())
        )

        var exported = List[ExportedArray]()
        for bi in range(len(batches)):
            ref b = batches[bi]
            # A record batch is a struct array with one child per column; the
            # scan hands back the columns loose, so the struct is added here.
            var row = ArrayData(ArrowType(AT_STRUCT), String("row"))
            row.nullable = False
            row.null_count = 0
            row.length = b.num_rows
            row.children = b.roots.copy()
            var root = b.arena.add(row^)
            exported.append(export_c(b.arena, root))

        if len(exported) == 0:
            # A split whose rows were all deleted or filtered away. The stream
            # still has to carry the schema, so it gets one empty batch rather
            # than none: a stream with no schema is not a stream.
            var scan = _open(table_dir).use_snapshot(decoded[0])
            var schema = scan.schema()
            var arena = ArrayArena()
            var root = empty_tree(
                arena, schema.store, schema.root, String("row"), 0, False
            )
            exported.append(export_c(arena, root))

        _move_into(export_stream_of(exported^), out_ptr, 5, True)  # five words
        return 1
    except e:
        return _fail(err, String(e))


# ══ audio features ══════════════════════════════════════════════════════════
#
# The second thing this bridge does, and the one that is not about Iceberg at
# all: compute a number from a recording. Two ways in, because the samples
# arrive two ways.
#
# `mlake_audio_batch` is the scalar path. DuckDB evaluates a scalar function
# over a whole vector at a time, so this takes a whole vector at a time — 2048
# blobs in, 2048 doubles out, one call. Per-row entry points would put a
# language boundary in the inner loop of every query.
#
# `mlake_audio_plan` / `_schema` / `_read` is the table path, over files on
# disk rather than blobs in a column, and it reuses the plan handle and the
# ticket machinery above unchanged: a plan is a list of strings either way.


comptime FEATURE_RMS_DB = 0
comptime FEATURE_PEAK_DB = 1
comptime FEATURE_CENTROID_HZ = 2
comptime FEATURE_ZCR = 3
comptime FEATURE_DURATION_S = 4
"""Which number to compute. Mirrored in src/include/mlake_bridge.hpp, which
registers one SQL function per value; a mismatch between the two lists is the
only way to get the wrong feature, so both name their constants."""


def _feature_of(kind: Int, base: Int, length: Int) raises -> Float64:
    var clip = parse_wav(base, length)
    var samples = Span(clip.samples)
    if kind == FEATURE_RMS_DB:
        return rms_db(samples)
    if kind == FEATURE_PEAK_DB:
        return peak_db(samples)
    if kind == FEATURE_CENTROID_HZ:
        var c = spectral_centroid(samples, clip.sample_rate)
        if c < 0.0:
            raise Error("wav: no energy in this clip")
        return c
    if kind == FEATURE_ZCR:
        return zero_crossing_rate(samples)
    if kind == FEATURE_DURATION_S:
        return clip.duration_s()
    raise Error("mlake_audio: unknown feature ", kind)


@export("mlake_audio_batch")
def mlake_audio_batch(
    kind: Int,
    ptrs: Int,
    lens: Int,
    count: Int,
    out_values: Int,
    out_valid: Int,
    err: Int,
) abi("C") -> Int:
    """Compute one feature for each of `count` clips; 1 on success.

    `ptrs` and `lens` are two arrays of `count` words describing the blobs in
    place — nothing is copied, because a DuckDB vector already holds the bytes
    and copying 2048 of them to look at their headers would cost more than the
    transform does.

    A clip that cannot be decoded is **not** an error. It writes 0 to
    `out_valid[i]` and the query sees a NULL, the same answer `try_cast` gives
    for a string that is not a number. One bad file in a directory should cost
    one row, not the query — and `mlake_audio_features` exists to say which row
    and why. Only an unrecognised `kind`, which is a bug in the extension
    rather than in the data, fails the whole call.
    """
    try:
        var p = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=ptrs)
        var l = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=lens)
        var v = Pointer[Float64, MutUntrackedOrigin](
            unsafe_from_address=out_values
        )
        var ok = Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=out_valid
        )
        if kind < FEATURE_RMS_DB or kind > FEATURE_DURATION_S:
            raise Error("mlake_audio: unknown feature ", kind)
        for i in range(count):
            try:
                v[unsafe_offset=i] = _feature_of(
                    kind, p[unsafe_offset=i], l[unsafe_offset=i]
                )
                ok[unsafe_offset=i] = 1
            except:
                v[unsafe_offset=i] = 0.0
                ok[unsafe_offset=i] = 0
        return 1
    except e:
        return _fail(err, String(e))


comptime DEFAULT_BATCH_ROWS = 64
"""Clips per ticket when the caller does not choose.

The unit of work is a ticket, so this is also how finely the files divide
between DuckDB's threads. Small enough that eight threads have something to do
on a modest directory, large enough that the Arrow batch a ticket produces is
worth building.
"""


@export("mlake_audio_plan")
def mlake_audio_plan(
    pattern_ptr: Int, pattern_len: Int, batch_rows: Int, err: Int
) abi("C") -> Int:
    """Expand a pattern into tickets of clips; returns a plan handle.

    A ticket is its own newline-separated list of paths rather than a range
    into a glob that would have to be expanded again to be read. Re-expanding
    would mean every thread walking the directory, and worse, it would mean a
    file created mid-query changing what the later tickets refer to.

    Read the handle with `mlake_plan_count` / `mlake_plan_ticket` and free it
    with `mlake_plan_free`, exactly like an Iceberg plan: both are a list of
    strings and there is no reason for two of them.
    """
    try:
        var paths = find_clips(_in(pattern_ptr, pattern_len))
        var per = DEFAULT_BATCH_ROWS if batch_rows <= 0 else batch_rows
        var tickets = (len(paths) + per - 1) // per
        var words = unsafe_alloc[Int](1 + 2 * tickets)
        words[unsafe_offset=0] = tickets
        for t in range(tickets):
            var ticket = String()
            for i in range(t * per, min((t + 1) * per, len(paths))):
                if i > t * per:
                    ticket += String("\n")
                ticket += paths[i]
            words[unsafe_offset=1 + 2 * t] = _out(ticket)
            words[unsafe_offset=2 + 2 * t] = ticket.byte_length()
        return Int(words)
    except e:
        return _fail(err, String(e))


@export("mlake_audio_schema")
def mlake_audio_schema(out_ptr: Int, err: Int) abi("C") -> Int:
    """Fill the caller's `ArrowSchema` with the feature table's schema.

    Built from a zero-clip batch, so the schema is by construction the one the
    rows will have rather than a second declaration of it that could drift.
    """
    try:
        var built = features_batch(List[String]())
        var exported = export_c(built[0], built[1])
        var pair = exported.into_raw()
        release_c_array(pair[0])  # only the schema was wanted
        _move_into(pair[1], out_ptr, 9, False)  # nine words per ArrowSchema
        return 1
    except e:
        return _fail(err, String(e))


@export("mlake_audio_read")
def mlake_audio_read(
    ticket_ptr: Int, ticket_len: Int, out_ptr: Int, err: Int
) abi("C") -> Int:
    """Fill the caller's `ArrowArrayStream` with one ticket's rows; 1 on
    success.

    Decoding happens here and not at plan time, which is what makes the work
    parallel: planning reads the directory, and every thread then opens its own
    files and runs its own transforms with nothing shared between them.
    """
    try:
        var ticket = _in(ticket_ptr, ticket_len)
        var paths = List[String]()
        var bytes = ticket.as_bytes()
        var start = 0
        for i in range(len(bytes) + 1):
            if i == len(bytes) or bytes[i] == UInt8(10):  # "\n"
                if i > start:
                    paths.append(String(unsafe_from_utf8=bytes[start:i]))
                start = i + 1

        var built = features_batch(paths^)
        var exported = List[ExportedArray]()
        exported.append(export_c(built[0], built[1]))
        _move_into(export_stream_of(exported^), out_ptr, 5, True)  # five words
        return 1
    except e:
        return _fail(err, String(e))
