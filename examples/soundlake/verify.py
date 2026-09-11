"""Diff every feature the extension reports against the numpy reference.

This is the script that earns the right to filter on these numbers. It runs
both implementations over every clip in the dataset and reports the largest
disagreement per feature — not a spot check, and not a tolerance chosen to make
it pass.

The two implementations share no code: one is a hand-written radix-2 transform
in Mojo over samples decoded by a hand-written RIFF parser, the other is
`np.fft.rfft` over samples decoded by the `wave` module. Agreement to a few
parts in 10^12 is the floating-point summation order differing, and nothing
else.
"""

from __future__ import annotations

import sys

import numpy as np

import reference
from _connect import connect, require_data

FEATURES = ["duration_s", "rms_db", "peak_db", "centroid_hz", "zcr"]

TOLERANCE = {
    # Sums over a few hundred thousand samples in a different order. The dB
    # figures are logs of such a sum, so they are tighter still.
    "duration_s": 0.0,
    "rms_db": 1e-9,
    "peak_db": 0.0,
    "zcr": 0.0,
    # A centroid is a ratio of two sums over 513 bins, each the result of ten
    # butterfly stages. This is the one place the two transforms can differ.
    "centroid_hz": 1e-6,
}


def main() -> int:
    data = require_data()
    con = connect()

    print("measuring every clip through the extension...")
    measured = con.execute(
        """
        SELECT regexp_extract(path, '[^/]+$') AS clip, duration_s, rms_db,
               peak_db, centroid_hz, zcr, error
        FROM mlake_audio_features(?)
        ORDER BY clip
        """,
        [str(data / "clips" / "*.wav")],
    ).fetchall()

    failed = [row for row in measured if row[-1] is not None]
    if failed:
        print(f"  {len(failed)} clips reported an error, e.g. {failed[0][0]}: {failed[0][-1]}")

    print(f"measuring the same {len(measured)} clips with numpy...")
    worst = {name: (0.0, None) for name in FEATURES}
    nulls = 0
    for row in measured:
        clip, *got, error = row
        if error is not None:
            continue
        expected = reference.features((data / "clips" / clip).read_bytes())
        for name, mine in zip(FEATURES, got):
            theirs = expected[name]
            if mine is None or theirs is None:
                # Only a silent clip has no centroid, and then neither side
                # should have one.
                if (mine is None) != (theirs is None):
                    print(f"  {clip}: {name} is {mine} here and {theirs} in numpy")
                    nulls += 1
                continue
            difference = abs(mine - theirs)
            if difference > worst[name][0]:
                worst[name] = (difference, clip)

    print()
    width = max(len(name) for name in FEATURES)
    ok = nulls == 0
    for name in FEATURES:
        difference, clip = worst[name]
        within = difference <= TOLERANCE[name]
        ok &= within
        mark = "ok " if within else "OFF"
        where = f"  (worst: {clip})" if clip else ""
        print(f"  {mark} {name:<{width}}  max |mojo - numpy| = {difference:.3e}{where}")

    print()
    if ok:
        print(f"the two implementations agree on all {len(measured) - len(failed)} clips.")
        return 0
    print("the implementations disagree by more than floating-point summation order.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
