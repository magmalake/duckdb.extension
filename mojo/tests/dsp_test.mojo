"""The feature kernels against signals whose answers are known on paper.

A sine is the whole test corpus on purpose: its RMS, its peak, how often it
crosses zero and where its energy sits are all closed forms, so a wrong window,
a mis-indexed twiddle or a transform that folded the mirror bins in shows up as
a number that is wrong by a lot rather than by a little.
"""

from std.math import sin
from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite

from dsp import (
    DB_FLOOR,
    peak_db,
    rms_db,
    spectral_centroid,
    zero_crossing_rate,
)
from wav import parse_wav

comptime RATE = 16000
comptime PI = 3.141592653589793


def _sine(freq: Float64, amplitude: Float64, seconds: Float64) -> List[Float64]:
    var n = Int(seconds * Float64(RATE))
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(
            amplitude * sin(2.0 * PI * freq * Float64(i) / Float64(RATE))
        )
    return out^


def test_rms_of_a_sine() raises:
    # RMS of a sine is its amplitude over root two, whatever its frequency.
    var s = _sine(1000.0, 0.5, 1.0)
    assert_almost_equal(rms_db(Span(s)), -9.0309, atol=Float64(0.001))
    var quieter = _sine(440.0, 0.05, 1.0)
    assert_almost_equal(rms_db(Span(quieter)), -29.0309, atol=Float64(0.001))


def test_peak_of_a_sine() raises:
    var s = _sine(1000.0, 0.5, 1.0)
    assert_almost_equal(peak_db(Span(s)), -6.0206, atol=Float64(0.01))


def test_silence_hits_the_floor() raises:
    var s = List[Float64](length=RATE, fill=0.0)
    assert_equal(rms_db(Span(s)), DB_FLOOR)
    assert_equal(peak_db(Span(s)), DB_FLOOR)
    # No energy anywhere means no centroid to report, not a centroid of zero.
    assert_true(spectral_centroid(Span(s), RATE) < 0.0)


def test_zero_crossings_count_the_period() raises:
    # A sine crosses zero twice per cycle.
    var s = _sine(1000.0, 0.5, 1.0)
    assert_almost_equal(
        zero_crossing_rate(Span(s)), 0.125, atol=Float64(0.0005)
    )


def test_centroid_finds_the_tone() raises:
    # One pure tone: the magnitude-weighted mean frequency is that tone, give
    # or take the leakage a Hann window spreads into the neighbouring bins.
    var s = _sine(1000.0, 0.5, 1.0)
    assert_almost_equal(
        spectral_centroid(Span(s), RATE), 1000.0, atol=Float64(15.0)
    )
    var bright = _sine(4000.0, 0.5, 1.0)
    assert_almost_equal(
        spectral_centroid(Span(bright), RATE), 4000.0, atol=Float64(15.0)
    )
    # And the ordering that the demo query actually depends on.
    assert_true(spectral_centroid(Span(s), RATE) < spectral_centroid(Span(bright), RATE))


def test_short_clip_still_answers() raises:
    # Shorter than one frame: zero-padded rather than refused.
    var s = _sine(1000.0, 0.5, 0.01)
    assert_true(spectral_centroid(Span(s), RATE) > 0.0)


# ── the container ───────────────────────────────────────────────────────────


def _wav_bytes(var samples: List[Int16], rate: Int, channels: Int) -> List[UInt8]:
    """A minimal 16-bit PCM WAVE file, built the way a recorder would."""
    var data_len = 2 * len(samples)
    var out = List[UInt8]()

    def put(s: String) {mut out}:
        out.extend(s.as_bytes())

    def put32(v: Int) {mut out}:
        for k in range(4):
            out.append(UInt8((v >> (8 * k)) & 0xFF))

    def put16(v: Int) {mut out}:
        for k in range(2):
            out.append(UInt8((v >> (8 * k)) & 0xFF))

    put(String("RIFF"))
    put32(36 + data_len)
    put(String("WAVE"))
    put(String("fmt "))
    put32(16)
    put16(1)  # PCM
    put16(channels)
    put32(rate)
    put32(rate * channels * 2)
    put16(channels * 2)
    put16(16)
    put(String("data"))
    put32(data_len)
    for i in range(len(samples)):
        put16(Int(samples[i]) & 0xFFFF)
    return out^


def test_wav_roundtrip() raises:
    var pcm = List[Int16]()
    for i in range(RATE):
        pcm.append(
            Int16(16384.0 * sin(2.0 * PI * 1000.0 * Float64(i) / Float64(RATE)))
        )
    var bytes = _wav_bytes(pcm^, RATE, 1)
    var size = len(bytes)
    var clip = parse_wav(Int(bytes.unsafe_ptr()), size)
    _ = bytes^
    assert_equal(clip.sample_rate, RATE)
    assert_equal(clip.channels, 1)
    assert_equal(clip.frames, RATE)
    assert_almost_equal(clip.duration_s(), 1.0, atol=Float64(1e-9))
    # 16384/32768 = 0.5 amplitude, so the same -9.03 dBFS as the float sine.
    assert_almost_equal(
        rms_db(Span(clip.samples)), -9.0309, atol=Float64(0.01)
    )


def test_wav_rejects_what_it_cannot_read() raises:
    var junk = List[UInt8](length=100, fill=0)
    var size = len(junk)
    var raised = False
    try:
        var clip = parse_wav(Int(junk.unsafe_ptr()), size)
        _ = clip^
    except e:
        raised = True
        assert_true(String(e).find("RIFF") >= 0)
    _ = junk^
    assert_true(raised, "a buffer of zeros is not a WAVE file")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
