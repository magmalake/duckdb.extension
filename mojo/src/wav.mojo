"""Enough of RIFF/WAVE to get PCM samples out of a sensor recording.

This is not a codec and is not trying to become one. It reads 16-bit
little-endian PCM — what every field recorder and every `scipy.io.wavfile.write`
of an `int16` array produces — and refuses everything else by name, so a file
this cannot read says what it is instead of returning plausible numbers.

Samples come out as `Float64` in [-1, 1), channels averaged to mono. Float64
rather than Float32 because the reference these features are checked against is
numpy, and matching it to the last few digits is worth more here than the
memory: a two-second clip at 16 kHz is 256 KiB of samples and lives only for
the duration of one call.

The input is an address and a length rather than a `List`, because both callers
have one already — a `BLOB` inside a DuckDB vector, or a file this module read
itself — and copying a clip to look at its header would be the single most
expensive thing this file does.
"""

from std.memory import bitcast

comptime FMT_PCM = 1
"""`wFormatTag` for uncompressed integer PCM."""

comptime FMT_FLOAT = 3
"""IEEE float. Recognised only so the error can name it."""

comptime FMT_EXTENSIBLE = 0xFFFE
"""WAVE_FORMAT_EXTENSIBLE: the real tag is in the chunk's extension."""


struct WavClip(Movable):
    """One decoded recording: the samples, and what they were sampled at."""

    var sample_rate: Int
    var channels: Int
    var frames: Int
    """Samples *per channel* — the number of instants, not of numbers."""
    var samples: List[Float64]
    """`frames` mono samples, the channels averaged."""

    def __init__(
        out self,
        sample_rate: Int,
        channels: Int,
        frames: Int,
        var samples: List[Float64],
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = frames
        self.samples = samples^

    def __init__(out self, *, deinit move: Self):
        self.sample_rate = move.sample_rate
        self.channels = move.channels
        self.frames = move.frames
        self.samples = move.samples^

    def duration_s(self) -> Float64:
        if self.sample_rate <= 0:
            return 0.0
        return Float64(self.frames) / Float64(self.sample_rate)


# ── little-endian reads off a raw address ───────────────────────────────────
# Bytes rather than a typed load because a chunk header is not aligned to
# anything: it sits wherever the chunk before it ended.


@always_inline
def _u8(base: Int, i: Int) -> UInt32:
    return UInt32(
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base)[
            unsafe_offset=i
        ]
    )


@always_inline
def _u16(base: Int, i: Int) -> Int:
    return Int(_u8(base, i) | (_u8(base, i + 1) << 8))


@always_inline
def _u32(base: Int, i: Int) -> Int:
    return Int(
        _u8(base, i)
        | (_u8(base, i + 1) << 8)
        | (_u8(base, i + 2) << 16)
        | (_u8(base, i + 3) << 24)
    )


@always_inline
def _tag(base: Int, i: Int) -> Int:
    """A four-character chunk id packed into an Int, so comparing one is a
    single integer compare rather than four."""
    return Int(
        _u8(base, i)
        | (_u8(base, i + 1) << 8)
        | (_u8(base, i + 2) << 16)
        | (_u8(base, i + 3) << 24)
    )


def _fourcc(a: String) -> Int:
    var b = a.as_bytes()
    return Int(
        UInt32(b[0])
        | (UInt32(b[1]) << 8)
        | (UInt32(b[2]) << 16)
        | (UInt32(b[3]) << 24)
    )


def parse_wav(base: Int, length: Int) raises -> WavClip:
    """Decode `length` bytes of RIFF/WAVE at `base`.

    Chunks are walked rather than assumed: a `fmt ` chunk is 16, 18 or 40 bytes
    depending on who wrote it, and real files carry `LIST`, `fact` and `bext`
    chunks between it and the data. Walking costs nothing and is the difference
    between reading files from one tool and files from any.
    """
    if length < 44:
        raise Error("wav: too short to be a WAVE file (", length, " bytes)")
    if _tag(base, 0) != _fourcc(String("RIFF")) or _tag(base, 8) != _fourcc(
        String("WAVE")
    ):
        raise Error("wav: not a RIFF/WAVE file")

    var fmt_tag = -1
    var channels = 0
    var sample_rate = 0
    var bits = 0
    var data_at = -1
    var data_len = 0

    var pos = 12
    while pos + 8 <= length:
        var id = _tag(base, pos)
        var size = _u32(base, pos + 4)
        var body = pos + 8
        if size < 0 or body + size > length:
            # A truncated final chunk: take what is actually there rather than
            # reading off the end. Recordings cut short by a dying battery are
            # a real thing and the samples before the cut are still good.
            size = length - body
        if id == _fourcc(String("fmt ")):
            if size < 16:
                raise Error("wav: fmt chunk is ", size, " bytes, needs 16")
            fmt_tag = _u16(base, body)
            channels = _u16(base, body + 2)
            sample_rate = _u32(base, body + 4)
            bits = _u16(base, body + 14)
            if fmt_tag == FMT_EXTENSIBLE and size >= 26:
                # The real tag is the first two bytes of the GUID in the
                # extension; the rest of the GUID is a fixed suffix.
                fmt_tag = _u16(base, body + 24)
        elif id == _fourcc(String("data")):
            data_at = body
            data_len = size
        pos = body + size + (size & 1)  # chunks are padded to even length

    if fmt_tag < 0:
        raise Error("wav: no fmt chunk")
    if data_at < 0:
        raise Error("wav: no data chunk")
    if fmt_tag == FMT_FLOAT:
        raise Error(
            "wav: IEEE-float samples are not supported, only 16-bit PCM"
        )
    if fmt_tag != FMT_PCM:
        raise Error("wav: unsupported format tag ", fmt_tag, ", expected PCM")
    if bits != 16:
        raise Error("wav: ", bits, "-bit samples, only 16-bit PCM is supported")
    if channels < 1:
        raise Error("wav: zero channels")
    if sample_rate <= 0:
        raise Error("wav: sample rate ", sample_rate)

    var frames = data_len // (2 * channels)
    var samples = List[Float64](capacity=frames)
    comptime SCALE = 1.0 / 32768.0
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base)
    if channels == 1:
        for i in range(frames):
            var lo = UInt16(p[unsafe_offset=data_at + 2 * i])
            var hi = UInt16(p[unsafe_offset=data_at + 2 * i + 1])
            var s = bitcast[DType.int16](lo | (hi << 8))
            samples.append(Float64(Int(s)) * SCALE)
    else:
        for i in range(frames):
            var acc = Float64(0)
            var at = data_at + 2 * channels * i
            for c in range(channels):
                var lo = UInt16(p[unsafe_offset=at + 2 * c])
                var hi = UInt16(p[unsafe_offset=at + 2 * c + 1])
                acc += Float64(Int(bitcast[DType.int16](lo | (hi << 8))))
            samples.append(acc * SCALE / Float64(channels))

    return WavClip(sample_rate, channels, frames, samples^)


def read_wav_file(path: String) raises -> WavClip:
    """Read a WAVE file from disk and decode it."""
    with open(path, "r") as f:
        var bytes = f.read_bytes()
        if len(bytes) == 0:
            raise Error("wav: empty file")
        var size = len(bytes)
        var clip = parse_wav(Int(bytes.unsafe_ptr()), size)
        # Mojo destroys a value at its last *use*, and taking the address is
        # that use — without this the buffer is freed while `parse_wav` reads
        # it, and the samples are whatever the allocator put there next.
        _ = bytes^
        return clip^
