"""The same four features in numpy, so the Mojo ones can be checked and timed.

This exists twice over on purpose. As a **reference** it is what verify.py
diffs the extension against, which is the only reason to believe a number the
extension reports. As a **rival** it is what bench.py times against, and it is
written the way someone who wanted this in numpy would actually write it —
vectorised, one `rfft` over a strided frame matrix, no Python loop over
samples. Beating a straw man would prove nothing.

Every definition here is the one in mojo/src/dsp.mojo, restated:

    rms_db       20 log10 sqrt(mean(x**2)), floored at -120
    peak_db      20 log10 max|x|, floored at -120
    zcr          sign changes / (n - 1), zero counting as positive
    centroid_hz  mean over frames of the magnitude-weighted mean frequency of
                 a 1024-point periodic-Hann spectrum at hop 512, skipping
                 frames with no energy; None if no frame has any
"""

from __future__ import annotations

import io
import wave

import numpy as np

DB_FLOOR = -120.0
FRAME = 1024
HOP = 512

_WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME) / FRAME)
"""Periodic Hann: `N` in the denominator, not `N - 1`. The symmetric form is a
half-sample different and moves a centroid in the third decimal."""


def decode(clip: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAVE to mono float64 in [-1, 1), as wav.mojo does it."""
    with wave.open(io.BytesIO(clip), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{f.getsampwidth() * 8}-bit samples, only 16-bit PCM")
        channels = f.getnchannels()
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1)
        return raw.astype(np.float64) / 32768.0, f.getframerate()


def _to_db(amplitude: float) -> float:
    if amplitude <= 0.0:
        return DB_FLOOR
    return max(DB_FLOOR, 20.0 * np.log10(amplitude))


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return DB_FLOOR
    return _to_db(float(np.sqrt(np.mean(x**2))))


def peak_db(x: np.ndarray) -> float:
    if x.size == 0:
        return DB_FLOOR
    return _to_db(float(np.max(np.abs(x))))


def zcr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    negative = x < 0.0
    return float(np.count_nonzero(negative[1:] != negative[:-1])) / (x.size - 1)


def _frames(x: np.ndarray) -> np.ndarray:
    """Every frame as a row, zero-padded if the clip is shorter than one.

    `as_strided` rather than a loop: the frames overlap by half, so a copy
    would double the data, and the whole point of the comparison is to give
    numpy its best shot.
    """
    if x.size < FRAME:
        padded = np.zeros(FRAME)
        padded[: x.size] = x
        return padded[None, :]
    count = 1 + (x.size - FRAME) // HOP
    return np.lib.stride_tricks.as_strided(
        x, shape=(count, FRAME), strides=(x.strides[0] * HOP, x.strides[0])
    )


def centroid_hz(x: np.ndarray, rate: int) -> float | None:
    if x.size == 0 or rate <= 0:
        return None
    magnitudes = np.abs(np.fft.rfft(_frames(x) * _WINDOW, axis=1))
    total = magnitudes.sum(axis=1)
    live = total > 0.0
    if not live.any():
        return None
    freqs = np.arange(magnitudes.shape[1]) * (rate / FRAME)
    per_frame = (magnitudes[live] @ freqs) / total[live]
    return float(per_frame.mean())


def features(clip: bytes) -> dict:
    """Everything at once, which is how bench.py's rivals are asked for it."""
    x, rate = decode(clip)
    return {
        "sample_rate": rate,
        "duration_s": x.size / rate,
        "rms_db": rms_db(x),
        "peak_db": peak_db(x),
        "centroid_hz": centroid_hz(x, rate),
        "zcr": zcr(x),
    }
