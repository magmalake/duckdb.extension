"""Turn a directory of recordings into an Arrow batch of features.

This is the half of the audio extension that DuckDB sees as a *table*: give it
a pattern, get back one row per clip with the file's own facts — rate, channel
count, duration — beside the numbers `dsp` computed from its samples. It is
what lets a query join a lakehouse table of sensors against the sound those
sensors actually recorded, which is the thing neither SQL nor the relational
table alone can reach: the samples are in files, and the files are not rows
until something opens them.

## A clip that cannot be read is a row, not a failure

One unreadable file in two thousand should not fail a query, and it should
also not vanish. Every clip therefore produces a row: on success the feature
columns are filled and `error` is null, on failure the feature columns are null
and `error` says what went wrong. `WHERE error IS NOT NULL` is then a
perfectly good way to audit a drop of data, and the count of rows always equals
the count of files.
"""

from std.memory import bitcast
from std.os import listdir
from std.os.path import isdir, isfile

from arrow_mlake.arrow import (
    AT_FLOAT64,
    AT_INT64,
    AT_STRUCT,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    ArrowType,
    bit_set,
    store_u64,
)

from dsp import peak_db, rms_db, spectral_centroid, zero_crossing_rate
from wav import read_wav_file


# ── finding the clips ───────────────────────────────────────────────────────


def _matches(name: Span[UInt8, _], pattern: Span[UInt8, _]) -> Bool:
    """Shell-style `*` and `?` against one name.

    Iterative with a backtrack point rather than recursive: `*` is the only
    construct that needs to try more than one split, and remembering the last
    one it took is enough to match without the stack depth a naive recursion
    would spend on a name full of them.
    """
    var n = 0
    var p = 0
    var star = -1
    var resume = 0
    while n < len(name):
        if p < len(pattern) and (
            pattern[p] == UInt8(63) or pattern[p] == name[n]  # '?'
        ):
            n += 1
            p += 1
        elif p < len(pattern) and pattern[p] == UInt8(42):  # '*'
            star = p
            p += 1
            resume = n
        elif star >= 0:
            p = star + 1
            resume += 1
            n = resume
        else:
            return False
    while p < len(pattern) and pattern[p] == UInt8(42):
        p += 1
    return p == len(pattern)


def _split_at_last_slash(path: String) -> Tuple[String, String]:
    var bytes = path.as_bytes()
    var cut = -1
    for i in range(len(bytes)):
        if bytes[i] == UInt8(47):  # '/'
            cut = i
    if cut < 0:
        return (String("."), path.copy())
    return (
        String(unsafe_from_utf8=bytes[:cut]),
        String(unsafe_from_utf8=bytes[cut + 1 :]),
    )


def _has_wildcard(s: String) -> Bool:
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(42) or bytes[i] == UInt8(63):
            return True
    return False


def _sort(mut items: List[String]):
    """Bottom-up merge sort, so a plan does not depend on the order a
    filesystem happens to hand its entries back in.

    Ticket order is what DuckDB preserves output by, and a query whose row
    order changes between two machines reading the same directory is a bad
    thing to have to explain.
    """
    var n = len(items)
    var scratch = List[String](capacity=n)
    for i in range(n):
        scratch.append(items[i].copy())
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            var hi = lo + 2 * width
            if mid > n:
                mid = n
            if hi > n:
                hi = n
            var a = lo
            var b = mid
            for k in range(lo, hi):
                var take_a: Bool
                if a >= mid:
                    take_a = False
                elif b >= hi:
                    take_a = True
                else:
                    take_a = items[a] <= items[b]
                if take_a:
                    scratch[k] = items[a].copy()
                    a += 1
                else:
                    scratch[k] = items[b].copy()
                    b += 1
            lo = hi
        for k in range(n):
            items[k] = scratch[k].copy()
        width *= 2


def find_clips(pattern: String) raises -> List[String]:
    """Every file matching `pattern`, sorted.

    Three shapes are accepted because all three are what someone types: a
    single file, a directory (which means every file directly in it), and a
    glob over the last path component. Recursive `**` is deliberately absent —
    it would make the plan's cost depend on how deep someone's archive is, and
    a union of two calls says the same thing out loud.
    """
    if not _has_wildcard(pattern):
        if isdir(pattern):
            return find_clips(pattern + String("/*"))
        if isfile(pattern):
            var one = List[String]()
            one.append(pattern.copy())
            return one^
        raise Error("mlake_audio: no such file or directory: '", pattern, "'")

    var parts = _split_at_last_slash(pattern)
    if _has_wildcard(parts[0]):
        raise Error(
            "mlake_audio: wildcards are only supported in the file name, not"
            " in the directory: '",
            pattern,
            "'",
        )
    if not isdir(parts[0]):
        raise Error("mlake_audio: no such directory: '", parts[0], "'")

    var out = List[String]()
    var entries = listdir(parts[0])
    for i in range(len(entries)):
        ref name = entries[i]
        if _matches(name.as_bytes(), parts[1].as_bytes()):
            var full = parts[0] + String("/") + name
            if isfile(full):
                out.append(full^)
    _sort(out)
    return out^


# ── one row per clip ────────────────────────────────────────────────────────


struct _Column(Movable):
    """A column under construction, with the null bookkeeping in one place.

    Arrow wants the validity bitmap, the null count and the values buffer kept
    consistent with each other, and nine columns written by hand is nine
    chances to forget one. Appending through here makes a null cost the same
    call as a value.
    """

    var data: ArrayData

    def __init__(out self, type_id: Int, var name: String):
        self.data = ArrayData(ArrowType(type_id), name^)
        if type_id == AT_UTF8:
            self.data.offsets.append(0)

    def __init__(out self, *, deinit move: Self):
        self.data = move.data^

    def _mark(mut self, valid: Bool):
        bit_set(self.data.validity, self.data.length, valid)
        if not valid:
            self.data.null_count += 1
        self.data.length += 1

    def push_i64(mut self, v: Int64):
        store_u64(self.data.values, UInt64(v))
        self._mark(True)

    def push_f64(mut self, v: Float64):
        store_u64(self.data.values, bitcast[DType.uint64](v))
        self._mark(True)

    def push_str(mut self, s: String):
        self.data.values.extend(s.as_bytes())
        self.data.offsets.append(Int32(len(self.data.values)))
        self._mark(True)

    def push_null(mut self):
        # A null still occupies its slot: fixed-width columns need the bytes so
        # element `i` stays at offset `i * width`, and a varlen column needs the
        # offset repeated so the next value starts where this one would have.
        if self.data.type.id == AT_UTF8:
            self.data.offsets.append(Int32(len(self.data.values)))
        elif self.data.type.id == AT_INT64 or self.data.type.id == AT_FLOAT64:
            store_u64(self.data.values, 0)
        self._mark(False)


comptime COLUMNS = 10
"""Kept beside `_new_columns`: the schema is defined once, there, and the count
is only here so a caller can check it did not drift."""


def _new_columns() -> List[_Column]:
    """The schema of `mlake_audio_features`, in order."""
    var cols = List[_Column]()
    cols.append(_Column(AT_UTF8, String("path")))
    cols.append(_Column(AT_INT64, String("sample_rate")))
    cols.append(_Column(AT_INT64, String("channels")))
    cols.append(_Column(AT_INT64, String("frames")))
    cols.append(_Column(AT_FLOAT64, String("duration_s")))
    cols.append(_Column(AT_FLOAT64, String("rms_db")))
    cols.append(_Column(AT_FLOAT64, String("peak_db")))
    cols.append(_Column(AT_FLOAT64, String("centroid_hz")))
    cols.append(_Column(AT_FLOAT64, String("zcr")))
    cols.append(_Column(AT_UTF8, String("error")))
    return cols^


def _append_clip(mut cols: List[_Column], path: String):
    """One row: the features, or the reason there are none."""
    try:
        var clip = read_wav_file(path)
        var samples = Span(clip.samples)
        cols[0].push_str(path)
        cols[1].push_i64(Int64(clip.sample_rate))
        cols[2].push_i64(Int64(clip.channels))
        cols[3].push_i64(Int64(clip.frames))
        cols[4].push_f64(clip.duration_s())
        cols[5].push_f64(rms_db(samples))
        cols[6].push_f64(peak_db(samples))
        var centroid = spectral_centroid(samples, clip.sample_rate)
        if centroid < 0.0:
            cols[7].push_null()  # silence has no centroid to report
        else:
            cols[7].push_f64(centroid)
        cols[8].push_f64(zero_crossing_rate(samples))
        cols[9].push_null()
    except e:
        cols[0].push_str(path)
        for k in range(1, 9):
            cols[k].push_null()
        cols[9].push_str(String(e))


def features_batch(paths: List[String]) raises -> Tuple[ArrayArena, Int]:
    """Decode every path and build the record batch of their features.

    An empty `paths` is not a special case and must not be: it is how the
    schema is obtained, so a zero-row batch has to carry exactly the columns a
    full one does.
    """
    var cols = _new_columns()
    for i in range(len(paths)):
        _append_clip(cols, paths[i])

    var arena = ArrayArena()
    var children = List[Int]()
    var rows = 0
    for i in range(len(cols)):
        rows = cols[i].data.length
        var node = ArrayData()
        swap(node, cols[i].data)
        children.append(arena.add(node^))

    # A record batch crosses the C Data Interface as a struct array whose
    # children are the columns.
    var row = ArrayData(ArrowType(AT_STRUCT), String("row"))
    row.nullable = False
    row.null_count = 0
    row.length = rows
    row.children = children^
    var root = arena.add(row^)
    return (arena^, root)
