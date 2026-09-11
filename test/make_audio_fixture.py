"""Recordings whose features are known on paper, for test/sql/mlake_audio.test.

Everything here is a sine, silence, or deliberate garbage, because those are
the signals whose RMS, peak, zero-crossing rate and spectral centroid have
closed forms. A fixture of real field recordings would test that the numbers
do not change; these test that they are right.

Written with the standard library's `wave` module and nothing else. The point
of the SQL tests is to check the extension, and a fixture that needs numpy to
exist would put numpy's idea of a WAVE file between the test and the thing
being tested.
"""

from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

RATE = 16000


def sine(freq: float, amplitude: float, seconds: float, rate: int = RATE) -> list[int]:
    n = int(seconds * rate)
    return [
        int(amplitude * 32768.0 * math.sin(2.0 * math.pi * freq * i / rate))
        for i in range(n)
    ]


def write_wav(path: Path, samples: list[int], channels: int = 1, rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def main(out_dir: str) -> None:
    clips = Path(out_dir) / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for stale in clips.glob("*"):
        stale.unlink()

    # A sine at half scale: RMS is amplitude/sqrt(2), so -9.03 dBFS, peak
    # -6.02, two zero crossings per cycle, and all its energy in one bin.
    write_wav(clips / "tone-1000.wav", sine(1000.0, 0.5, 1.0))
    # A quarter of that amplitude and four times the frequency: every feature
    # moves, and each in its own direction.
    write_wav(clips / "tone-4000.wav", sine(4000.0, 0.25, 1.0))
    # No energy at all. RMS and peak hit the floor; the centroid is a null,
    # because silence has no frequency rather than a frequency of zero.
    write_wav(clips / "silence.wav", [0] * (RATE // 2))
    # Two channels, averaged to mono on the way in. Identical channels, so
    # every feature must match tone-1000 exactly — mixing that dropped or
    # double-counted a channel would move the level.
    stereo = sine(1000.0, 0.5, 1.0)
    write_wav(
        clips / "stereo-1000.wav",
        [s for sample in stereo for s in (sample, sample)],
        channels=2,
    )
    # Shorter than one 1024-sample frame: still gets an answer, zero-padded.
    write_wav(clips / "short.wav", sine(1000.0, 0.5, 0.02))
    # A non-WAVE file with a WAVE name. It is a row with an error, not a
    # failed query and not a missing row.
    (clips / "broken.wav").write_bytes(b"this is not a RIFF file, it is a note" * 4)
    # A different extension, so a glob that means "*.wav" can be seen to.
    write_wav(clips / "ignored.dat", sine(1000.0, 0.5, 0.1))

    print(clips.resolve())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures")
