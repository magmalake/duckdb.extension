# soundlake


Acoustic monitoring: twelve sensors at four construction sites, two seconds of
sound every hour. A relational table says where each sensor is and when each
clip was taken. The clips are 16-bit PCM, as `BLOB`s in a column or as files on
disk. Nothing in the tables knows anything about sound.

SQL does the relational work, Mojo computes the number SQL cannot express,
and neither one leaves the process.

## The query

Noise permits run 07:00 to 19:00. Find recordings from outside those hours that
are far above their own site's baseline for the week.

```sql
WITH measured AS (
    SELECT s.site, r.recorded_at, mlake_rms_db(r.clip) AS loudness_db
    FROM recordings r JOIN sensors s USING (sensor_id)
),
baseline AS (
    SELECT *, median(loudness_db) OVER (PARTITION BY site) AS site_median
    FROM measured
)
SELECT site, recorded_at, round(loudness_db, 1)
FROM baseline
WHERE hour(recorded_at) NOT BETWEEN 7 AND 18
  AND loudness_db - site_median > 12;
```

Told nothing but "loud, and out of hours", it returns fifteen recordings — all
one site, all between 01:00 and 05:00 on one night. That is exactly the night
`generate.py` planted a jackhammer where the permit said 07:00, and the query
never saw the ground truth.

The site and the hour are relational. The loudness is a pass over 32,000
samples in Mojo. Comparing against the site's own baseline is a window function
over the column that pass produced — possible only because the feature is an
*expression*, not a number computed elsewhere and loaded back in.

## Run it

```sh
pixi run build        # in the repository root: the extension and the bridge
./run.sh              # generate the dataset, check it against numpy, take the tour
./run.sh bench        # and time it against the alternatives
```

`tour.sql` is the eight queries in order, with the reasoning in the comments.

## What it adds

Five scalar functions over a WAV `BLOB`, each returning a `DOUBLE`:

| function                  | what it is                                                     |
|---------------------------|-----------------------------------------------------------------|
| `mlake_rms_db(clip)`      | RMS level in dBFS. Full scale is 0; a quiet room is about -50.  |
| `mlake_peak_db(clip)`     | Largest absolute sample, in dBFS. 0 means the clip clipped.     |
| `mlake_centroid_hz(clip)` | Magnitude-weighted mean frequency: where the sound sits.        |
| `mlake_zcr(clip)`         | Fraction of adjacent samples whose sign differs.                 |
| `mlake_duration_s(clip)`  | Length in seconds, from the clip's own header.                   |

and one table function over files, for when the samples are not in a column:

```sql
SELECT * FROM mlake_audio_features('recordings/*.wav', batch_rows => 64);
-- path, sample_rate, channels, frames, duration_s,
-- rms_db, peak_db, centroid_hz, zcr, error
```

A clip that will not decode is a NULL from the scalar functions and a row with
an `error` from the table function — never a failed query.

## Is it right?

`verify.py` runs the kernels and an independent numpy implementation over every
clip and reports the largest disagreement. Exact on duration, RMS, peak and
ZCR; the centroid differs by 2.3e-12 Hz, which is 513 bins summed in a
different order.

## Is it fast?

`bench.py`, 1152 clips, 74 MB of PCM, M4, DuckDB 1.4.1. Same query, same
answer, four ways of getting the column:

```
── rms_db — one pass over the samples ─────────────
  mojo          1 thread       67.6 ms     17047 clips/s    1.00x
  py-arrow      1 thread       90.7 ms     12702 clips/s    0.75x
  py-native     1 thread      129.8 ms      8877 clips/s    0.52x
  fetch+numpy   1 thread       93.8 ms     12278 clips/s    0.72x
  mojo         10 threads      13.8 ms     83340 clips/s    4.89x
  py-arrow     10 threads      45.5 ms     25336 clips/s    1.49x
  py-native    10 threads     148.8 ms      7742 clips/s    0.45x
  fetch+numpy  10 threads      51.7 ms     22287 clips/s    1.31x

── centroid_hz — sixty-odd FFTs per clip ──────────
  mojo          1 thread      271.7 ms      4240 clips/s    1.00x
  py-arrow      1 thread      276.7 ms      4163 clips/s    0.98x
  py-native     1 thread      318.6 ms      3616 clips/s    0.85x
  fetch+numpy   1 thread      279.8 ms      4117 clips/s    0.97x
  mojo         10 threads      57.3 ms     20099 clips/s    4.74x
  py-arrow     10 threads     110.1 ms     10459 clips/s    2.47x
  py-native    10 threads     195.5 ms      5894 clips/s    1.39x
  fetch+numpy  10 threads     235.2 ms      4899 clips/s    1.16x
```

Two different stories. **On one thread the Fourier transform is a tie** —
numpy's is pocketfft and a hand-written transform has no business beating it.
(Reaching parity took a real-input packing: the textbook complex transform this
started as was 2x slower, which is exactly the 2x it was wasting.) **With
threads it is not close**, because DuckDB calls the extension from every worker
and there is no interpreter state to serialise on: 4.9x across ten cores
against 2.5x for the best Python UDF.

So the win is not "Mojo beats numpy at FFTs". It is that a kernel compiled into
the database parallelises with the query and does not cross a language boundary
1152 times — and that you write it in something other than C++.

## What it costs to carry

| | Mojo | Python |
|---|---|---|
| decode + four features | 331 lines (`dsp.mojo`, `wav.mojo`) | 64 lines (`reference.py`) |
| what that leans on | nothing | `numpy` 24 MB, `wave` from the stdlib |
| shipped as | `libmlake_bridge.dylib`, 2.1 MB | an interpreter and its site-packages |

The Mojo is five times the source because it is doing the work rather than
delegating it: chunk-walking the RIFF container that Python's `wave` module
handles, and a radix-2 transform where numpy calls `rfft`. That is the trade —
more code you own, no runtime you have to install beside the database.

## Real recordings

Nothing here is synthetic by necessity. Point `mlake_audio_features` at any
directory of 16-bit PCM WAVE files — [ESC-50][esc50] is 2000 clips with a
metadata CSV that joins straight onto the feature table. Synthetic clips are
the default because ground truth from somebody's annotation file can tell you
the numbers are *stable*; only a dataset you generated can tell you they are
*right*.

[esc50]: https://github.com/karolpiczak/ESC-50
