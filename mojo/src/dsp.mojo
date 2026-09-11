"""The numbers SQL cannot compute: loudness, brightness, and how often a
waveform crosses zero.

Every function here is defined tightly enough to be reimplemented in numpy and
diffed against, and `examples/soundlake/reference.py` does exactly that. A
feature nobody can check is a feature nobody should filter on, so the
definitions below are the contract and not an implementation detail:

* **rms_db** — `20 log10 sqrt(mean(x^2))`, x in [-1, 1). Full-scale square wave
  is 0 dBFS; a full-scale sine is -3.01.
* **peak_db** — `20 log10 max|x|`.
* **zcr** — sign changes divided by `n - 1`. A fraction in [0, 1], not a rate:
  it is what librosa calls zero-crossing rate and is comparable across sample
  rates.
* **centroid_hz** — the magnitude-weighted mean frequency of a 1024-point
  Hann-windowed spectrum, hop 512, averaged over the frames that carry any
  energy at all. "Where the sound sits", in hertz.

Both dB figures are floored at `DB_FLOOR` rather than returning `-inf`, because
a silent clip is a real and ordinary thing and `-inf` propagates through every
`avg()` downstream into a result that says nothing.
"""

from std.math import cos, log10, max, sin, sqrt

comptime DB_FLOOR = -120.0
"""Reported instead of `-inf` for a silent clip. Below any real noise floor and
16-bit PCM cannot represent a level within 30 dB of it, so nothing is lost."""

comptime FRAME = 1024
"""Spectrum size. 64 ms at 16 kHz: long enough to resolve ~16 Hz, short enough
that a passing vehicle is several frames rather than one."""

comptime HOP = 512
"""Half a frame, the usual overlap for a Hann window — the windows then sum to
a constant and no instant is weighted more than its neighbours."""

comptime PI = 3.141592653589793


# ── level ───────────────────────────────────────────────────────────────────


@always_inline
def _to_db(amplitude: Float64) -> Float64:
    if amplitude <= 0.0:
        return DB_FLOOR
    var db = 20.0 * log10(amplitude)
    return DB_FLOOR if db < DB_FLOOR else db


def rms_db(samples: Span[Float64, _]) -> Float64:
    """Root-mean-square level in dBFS."""
    var n = len(samples)
    if n == 0:
        return DB_FLOOR
    comptime W = 8
    var acc = SIMD[DType.float64, W](0)
    var p = samples.unsafe_ptr()
    var i = 0
    while i + W <= n:
        var v = p.unsafe_load[width=W](i)
        acc = v.fma(v, acc)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += samples[i] * samples[i]
        i += 1
    return _to_db(sqrt(total / Float64(n)))


def peak_db(samples: Span[Float64, _]) -> Float64:
    """Largest absolute sample, in dBFS."""
    var n = len(samples)
    if n == 0:
        return DB_FLOOR
    comptime W = 8
    var acc = SIMD[DType.float64, W](0)
    var p = samples.unsafe_ptr()
    var i = 0
    while i + W <= n:
        acc = max(acc, abs(p.unsafe_load[width=W](i)))
        i += W
    var top = acc.reduce_max()
    while i < n:
        top = max(top, abs(samples[i]))
        i += 1
    return _to_db(top)


def zero_crossing_rate(samples: Span[Float64, _]) -> Float64:
    """Fraction of adjacent pairs whose sign differs.

    Zero counts as positive, matching numpy's `signbit`: a run of exact zeros
    in the middle of a clip should not read as a crossing on every sample.
    """
    var n = len(samples)
    if n < 2:
        return 0.0
    var crossings = 0
    for i in range(1, n):
        if (samples[i] < 0.0) != (samples[i - 1] < 0.0):
            crossings += 1
    return Float64(crossings) / Float64(n - 1)


# ── spectrum ────────────────────────────────────────────────────────────────


comptime HALF = FRAME // 2
"""The transform is `HALF` points, not `FRAME`. See `Spectra`."""


struct Spectra(Movable):
    """Scratch for one clip's worth of spectra: twiddles, window, and the
    working buffers the transform runs in.

    Held together in a struct because the twiddle table and the Hann window
    depend only on `FRAME` and would otherwise be rebuilt for every one of the
    sixty-odd frames in a two-second clip. It cannot outlive the call that
    makes it — Mojo has no way to hand C a live object with a destructor — but
    within one call it turns the per-frame cost into arithmetic only.

    ## A real signal gets a half-length transform

    Audio samples are real, and a complex transform of real input computes the
    upper half of its output as the mirror image of the lower half: exactly
    twice the necessary work. So the even-numbered samples of a frame go in as
    the real part and the odd-numbered ones as the imaginary part, a *512*-point
    complex transform runs over that, and `centroid_of_frame` untangles the two
    interleaved spectra as it reads them.

    This is the standard real-FFT packing and it is worth the forty lines: it
    halved the time of every query in examples/soundlake that touches a
    centroid, and a textbook complex transform over real input is a hard thing
    to defend against numpy, which does not do it.

    One table serves both halves of that. The untangling step needs
    `e^(-2*pi*i*k/FRAME)`, which is `cos_t[k]`, and the 512-point transform's
    own twiddles are `e^(-2*pi*i*k/512)` = `cos_t[2k]` — the same table at
    stride two.
    """

    var re: List[Float64]
    var im: List[Float64]
    var window: List[Float64]
    var cos_t: List[Float64]
    var sin_t: List[Float64]
    var rev: List[Int32]

    def __init__(out self):
        self.re = List[Float64](length=HALF, fill=0.0)
        self.im = List[Float64](length=HALF, fill=0.0)

        # Periodic Hann — `1 - cos(2*pi*i/N)` over N, not N-1. This is what
        # scipy's `get_window(..., fftbins=True)` and librosa's default give,
        # and the half-sample difference from the symmetric form is visible in
        # the third decimal of a centroid.
        self.window = List[Float64](capacity=FRAME)
        for i in range(FRAME):
            self.window.append(
                0.5 - 0.5 * cos(2.0 * PI * Float64(i) / Float64(FRAME))
            )

        # Twiddles for a decimation-in-time radix-2 transform:
        # e^(-2*pi*i*k/FRAME) for k < FRAME/2.
        self.cos_t = List[Float64](capacity=HALF)
        self.sin_t = List[Float64](capacity=HALF)
        for k in range(HALF):
            var angle = -2.0 * PI * Float64(k) / Float64(FRAME)
            self.cos_t.append(cos(angle))
            self.sin_t.append(sin(angle))

        # Bit-reversal permutation over the *transform* length, tabulated once.
        var bits = 0
        while (1 << bits) < HALF:
            bits += 1
        self.rev = List[Int32](capacity=HALF)
        for i in range(HALF):
            var r = 0
            for b in range(bits):
                if (i >> b) & 1 == 1:
                    r |= 1 << (bits - 1 - b)
            self.rev.append(Int32(r))

    def __init__(out self, *, deinit move: Self):
        self.re = move.re^
        self.im = move.im^
        self.window = move.window^
        self.cos_t = move.cos_t^
        self.sin_t = move.sin_t^
        self.rev = move.rev^

    def _load_frame(mut self, samples: Span[Float64, _], start: Int):
        """Window one frame into the working buffers, zero-padding the tail.

        Windowed, packed into a half-length complex signal, and permuted into
        bit-reversed order in one pass, because all three are a read of
        `samples[start + 2j]` and doing them separately would be three.
        """
        var n = len(samples)
        var re = self.re.unsafe_ptr()
        var im = self.im.unsafe_ptr()
        var window = self.window.unsafe_ptr()
        var source = samples.unsafe_ptr()
        var rev = self.rev.unsafe_ptr()
        for i in range(HALF):
            var j = Int(rev[unsafe_offset=i])
            var at = start + 2 * j
            if at + 1 < n:
                re[unsafe_offset=i] = (
                    source[unsafe_offset=at] * window[unsafe_offset=2 * j]
                )
                im[unsafe_offset=i] = (
                    source[unsafe_offset=at + 1]
                    * window[unsafe_offset=2 * j + 1]
                )
            elif at < n:
                # The very last sample of a frame that ends mid-pair.
                re[unsafe_offset=i] = (
                    source[unsafe_offset=at] * window[unsafe_offset=2 * j]
                )
                im[unsafe_offset=i] = 0.0
            else:
                re[unsafe_offset=i] = 0.0
                im[unsafe_offset=i] = 0.0

    def _transform(mut self):
        """In-place iterative radix-2 Cooley-Tukey over `HALF` points. Input is
        already in bit-reversed order, courtesy of `_load_frame`.

        Through raw pointers rather than `List` subscripts: this is ten stages
        of 256 butterflies each, sixty-odd times per clip, and a bounds check
        on every one of the six loads costs more than the arithmetic does.
        """
        var re = self.re.unsafe_ptr()
        var im = self.im.unsafe_ptr()
        var cos_t = self.cos_t.unsafe_ptr()
        var sin_t = self.sin_t.unsafe_ptr()
        var size = 2
        while size <= HALF:
            var half = size // 2
            # W_size^k = W_FRAME^(2k), so the shared table is walked twice as
            # fast as the transform length alone would suggest.
            var stride = 2 * (HALF // size)
            for start in range(0, HALF, size):
                var t = 0
                for k in range(start, start + half):
                    var wr = cos_t[unsafe_offset=t]
                    var wi = sin_t[unsafe_offset=t]
                    var xr = re[unsafe_offset=k + half]
                    var xi = im[unsafe_offset=k + half]
                    var pr = xr * wr - xi * wi
                    var pi_ = xr * wi + xi * wr
                    re[unsafe_offset=k + half] = re[unsafe_offset=k] - pr
                    im[unsafe_offset=k + half] = im[unsafe_offset=k] - pi_
                    re[unsafe_offset=k] += pr
                    im[unsafe_offset=k] += pi_
                    t += stride
            size *= 2

    def centroid_of_frame(mut self, sample_rate: Int) -> Float64:
        """Magnitude-weighted mean frequency of the transformed buffers, or a
        negative number if the frame holds no energy to weight with.

        This is where the packed transform is undone. `Z` holds two spectra
        superimposed — the even samples' and the odd samples'; they separate
        into the conjugate-even and conjugate-odd parts of `Z[k]` and
        `Z[HALF-k]`, and one twiddle recombines them into the bin of the real
        spectrum that was actually wanted. Only the magnitude is kept, so the
        recombination happens here rather than in a buffer nobody reads.
        """
        var bin_hz = Float64(sample_rate) / Float64(FRAME)
        var re = self.re.unsafe_ptr()
        var im = self.im.unsafe_ptr()
        var cos_t = self.cos_t.unsafe_ptr()
        var sin_t = self.sin_t.unsafe_ptr()

        # k = 0 and k = HALF are the DC and Nyquist bins. Both are real, and
        # both come from Z[0] alone, so neither goes through the general case.
        var z0r = re[unsafe_offset=0]
        var z0i = im[unsafe_offset=0]
        var total = abs(z0r + z0i) + abs(z0r - z0i)
        var weighted = abs(z0r - z0i) * (Float64(HALF) * bin_hz)

        for k in range(1, HALF):
            var zr = re[unsafe_offset=k]
            var zi = im[unsafe_offset=k]
            var mr = re[unsafe_offset=HALF - k]
            var mi = im[unsafe_offset=HALF - k]
            # Even part: (Z[k] + conj(Z[HALF-k])) / 2.
            var er = 0.5 * (zr + mr)
            var ei = 0.5 * (zi - mi)
            # Odd part: (Z[k] - conj(Z[HALF-k])) / 2i.
            var orr = 0.5 * (zi + mi)
            var oi = -0.5 * (zr - mr)
            var wr = cos_t[unsafe_offset=k]
            var wi = sin_t[unsafe_offset=k]
            var xr = er + (wr * orr - wi * oi)
            var xi = ei + (wr * oi + wi * orr)
            var mag = sqrt(xr * xr + xi * xi)
            weighted += mag * (Float64(k) * bin_hz)
            total += mag

        if total <= 0.0:
            return -1.0
        return weighted / total


def spectral_centroid(samples: Span[Float64, _], sample_rate: Int) -> Float64:
    """Mean over frames of the per-frame spectral centroid, in hertz.

    A clip shorter than one frame still gets one zero-padded frame, so a short
    recording gives a real answer rather than a null. Frames with no energy are
    skipped rather than counted as zero hertz: a pause between hammer blows
    should not make the clip read as darker than it is.
    """
    var n = len(samples)
    if n == 0 or sample_rate <= 0:
        return -1.0
    var work = Spectra()
    var total = Float64(0)
    var counted = 0
    var start = 0
    while start == 0 or start + FRAME <= n:
        work._load_frame(samples, start)
        work._transform()
        var c = work.centroid_of_frame(sample_rate)
        if c >= 0.0:
            total += c
            counted += 1
        start += HOP
    if counted == 0:
        return -1.0
    return total / Float64(counted)
