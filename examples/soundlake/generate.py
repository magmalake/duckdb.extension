"""A fortnight of acoustic monitoring at four construction sites, invented.

Twelve sensors, one recording an hour, each two seconds of sound synthesised
from a known cause: ambient hum, passing traffic, a jackhammer, birdsong, a
siren. Every clip's acoustic character follows from its class, so every query
in tour.sql has an answer you can check by reading this file rather than by
trusting the one the database gives back.

Real recordings would be more convincing and much less useful. ESC-50 and
UrbanSound8K are a download away if you want them — the extension does not care
where a WAVE file came from — but a dataset whose ground truth is a CC-BY
annotation file cannot tell you whether a spectral centroid is *right*.

There is one planted story, because a demo query with nothing to find is a
demo of the syntax rather than of the idea: see NIGHT_WORK below.

Writes:
    data/clips/*.wav          the recordings themselves
    data/sensors.csv          who recorded them and where
    data/recordings.csv       when, and the path of each clip
    data/recordings.parquet   the same, with the clip inline as a BLOB
    data/truth.csv            what each clip actually is
"""

from __future__ import annotations

import argparse
import csv
import io
import struct
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

RATE = 16000

SITES = {
    "harbor-north": (53.5461, 9.9661),
    "harbor-south": (53.5311, 9.9512),
    "viaduct": (53.5628, 10.0014),
    "depot": (53.5502, 10.0289),
}

MODELS = ["AM-2200", "AM-2200", "SoundBox 4", "AM-3000"]

PERMIT_HOURS = (7, 18)
"""When work is allowed, inclusive. Everything outside this is quiet by
construction, which is what makes the planted event below the only loud thing
at night — and it has to match the `hour(recorded_at) NOT BETWEEN 7 AND 18` in
tour.sql exactly. A boundary hour that is noisy here and out of hours there
would fill the demo's result with ordinary work and bury what it is looking
for."""

NIGHT_WORK = ("harbor-north", 2, 1, 5)
"""(site, day index, first hour, last hour) of the planted event: a night when
somebody ran a jackhammer between 01:00 and 05:00 at a site where the permit
says 07:00. It is loud, it is broadband, and it is the thing tour.sql's fifth
query is supposed to find without being told where to look.
"""


def _noise(rng: np.random.Generator, n: int, tilt: float) -> np.ndarray:
    """Noise with a `1/f**tilt` spectrum, so "rumble" and "hiss" are one knob.

    Built in the frequency domain because that is where the shape is: draw a
    white spectrum, scale each bin by its frequency, and transform back.
    """
    spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.arange(len(spectrum), dtype=np.float64)
    freqs[0] = 1.0
    spectrum /= freqs**tilt
    out = np.fft.irfft(spectrum, n=n)
    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


def _at(level_db: float, signal: np.ndarray) -> np.ndarray:
    """Scale a signal to a target RMS level in dBFS."""
    rms = np.sqrt(np.mean(signal**2))
    if rms <= 0:
        return signal
    return signal * (10.0 ** (level_db / 20.0)) / rms


def ambient(rng, n, level=-48.0):
    """Wind, distant machinery, the sound of nothing happening."""
    return _at(level, _noise(rng, n, 1.0))


def traffic(rng, n, level=-32.0):
    """Engine rumble: a low harmonic stack over red noise."""
    t = np.arange(n) / RATE
    engine = sum(
        np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / k
        for k, f in enumerate(rng.uniform(70, 130) * np.arange(1, 6), start=1)
    )
    return _at(level, 0.7 * engine / 3 + 0.3 * _noise(rng, n, 1.4))


def jackhammer(rng, n, level=-14.0):
    """Broadband impacts about twelve a second, each decaying fast."""
    t = np.arange(n) / RATE
    period = int(RATE / rng.uniform(10, 14))
    envelope = np.exp(-((np.arange(n) % period) / (RATE * 0.012)))
    return _at(level, envelope * _noise(rng, n, 0.35) + 0.05 * np.sin(2 * np.pi * 95 * t))


def birdsong(rng, n, level=-30.0):
    """Three or four chirps sweeping through the top of the band."""
    out = np.zeros(n)
    for _ in range(rng.integers(3, 6)):
        start = rng.integers(0, max(1, n - RATE // 5))
        length = int(RATE * rng.uniform(0.05, 0.15))
        t = np.arange(length) / RATE
        f0, f1 = rng.uniform(2200, 3200), rng.uniform(4500, 6500)
        sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1])))
        out[start : start + length] += sweep * np.hanning(length)
    return _at(level, out + 0.02 * _noise(rng, n, 1.0))


def siren(rng, n, level=-18.0):
    """A two-tone wail, swept rather than switched."""
    t = np.arange(n) / RATE
    centre = rng.uniform(800, 1100)
    sweep = centre + 300 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(sweep) / RATE
    return _at(level, np.sin(phase) + 0.3 * np.sin(2 * phase))


KINDS = {
    "ambient": ambient,
    "traffic": traffic,
    "jackhammer": jackhammer,
    "birdsong": birdsong,
    "siren": siren,
}


def pick_kind(rng, site: str, hour: int, day: int) -> str:
    """What a sensor is likely to hear, given where and when it is.

    The distribution is the story: sites differ, nights are quiet, and the one
    night at NIGHT_WORK is not.
    """
    night_site, night_day, first, last = NIGHT_WORK
    if site == night_site and day == night_day and first <= hour <= last:
        return "jackhammer"
    if not PERMIT_HOURS[0] <= hour <= PERMIT_HOURS[1]:
        # Out of hours, everywhere: distant traffic and nothing else. Quiet
        # enough that the one night of jackhammering stands out by 20 dB.
        return rng.choice(["ambient", "ambient", "ambient", "traffic"])
    if site == "depot":
        return rng.choice(["traffic", "traffic", "ambient", "siren"])
    if site == "viaduct":
        return rng.choice(["traffic", "traffic", "traffic", "ambient", "birdsong"])
    return rng.choice(["jackhammer", "jackhammer", "traffic", "ambient", "birdsong"])


def wav_bytes(samples: np.ndarray) -> bytes:
    """16-bit mono PCM, the way a field recorder would write it."""
    clipped = np.clip(samples, -1.0, 32767.0 / 32768.0)
    pcm = (clipped * 32768.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(struct.pack(f"<{len(pcm)}h", *pcm.tolist()))
    return buffer.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--per-day", type=int, default=24, help="recordings per sensor per day")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for stale in clips.glob("*.wav"):
        stale.unlink()

    sensors = []
    for s, (site, (lat, lon)) in enumerate(SITES.items()):
        for k in range(3):
            sensors.append(
                {
                    "sensor_id": f"{site.split('-')[0][:3]}-{s}{k}",
                    "site": site,
                    "model": MODELS[(s + k) % len(MODELS)],
                    "lat": round(lat + rng.normal(0, 0.001), 5),
                    "lon": round(lon + rng.normal(0, 0.001), 5),
                    "installed_on": (datetime(2026, 1, 4) + timedelta(days=int(rng.integers(0, 60)))).date(),
                }
            )

    n = int(args.seconds * RATE)
    epoch = datetime(2026, 9, 1)
    step = timedelta(hours=24 / args.per_day)
    recordings, truth, blobs = [], [], []

    for sensor in sensors:
        for day in range(args.days):
            for slot in range(args.per_day):
                when = epoch + timedelta(days=day) + slot * step
                kind = pick_kind(rng, sensor["site"], when.hour, day)
                samples = KINDS[kind](rng, n)
                clip_id = f"{sensor['sensor_id']}-{when:%Y%m%dT%H%M}"
                path = clips / f"{clip_id}.wav"
                payload = wav_bytes(samples)
                path.write_bytes(payload)
                recordings.append(
                    {
                        "clip_id": clip_id,
                        "sensor_id": sensor["sensor_id"],
                        "recorded_at": when.isoformat(sep=" "),
                        "path": str(path.resolve()),
                    }
                )
                truth.append(
                    {
                        "clip_id": clip_id,
                        "kind": kind,
                        "rms_db": round(20 * np.log10(np.sqrt(np.mean(samples**2))), 4),
                    }
                )
                blobs.append(payload)

    def write_csv(name, rows):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv("sensors.csv", sensors)
    write_csv("recordings.csv", recordings)
    # Deliberately not joined to the sensors: keeping them apart is the point.
    # The recordings table knows nothing about sites and the sensors table
    # knows nothing about sound, and the demo query is what puts them together.
    write_csv("truth.csv", truth)

    # The same recordings with the samples inline, for the scalar path. One
    # file so the blob demo needs no directory, and Parquet rather than CSV
    # because a BLOB in a CSV is a base64 round trip nobody learns anything
    # from.
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "clip_id": [r["clip_id"] for r in recordings],
                "sensor_id": [r["sensor_id"] for r in recordings],
                "recorded_at": pa.array(
                    [datetime.fromisoformat(r["recorded_at"]) for r in recordings],
                    type=pa.timestamp("us"),
                ),
                "clip": pa.array(blobs, type=pa.binary()),
            }
        ),
        out / "recordings.parquet",
        compression="zstd",
        # A row group is DuckDB's unit of parallelism. At the default of
        # 122880 rows the whole dataset is one group, one thread does all the
        # work whatever `threads` is set to, and bench.py silently measures a
        # single core. Sixty-four clips is a few megabytes a group, which is
        # small enough to spread and large enough to be worth scheduling.
        row_group_size=64,
    )

    total = sum(len(b) for b in blobs)
    print(f"{len(sensors)} sensors, {len(recordings)} clips, {total / 1e6:.1f} MB of audio")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
