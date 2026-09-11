-- soundlake: a tour of what a Mojo kernel inside DuckDB is for.
--
-- Run it from this directory, with the extension built:
--
--     ../../build/release/duckdb -unsigned < tour.sql
--
-- or one query at a time from run.sh's shell. The dataset is invented by
-- generate.py and every claim below is checkable against data/truth.csv.

LOAD '../../build/release/extension/mlake/mlake.duckdb_extension';

CREATE VIEW sensors    AS SELECT * FROM read_csv('data/sensors.csv');
CREATE VIEW recordings AS SELECT * FROM read_parquet('data/recordings.parquet');
CREATE VIEW truth      AS SELECT * FROM read_csv('data/truth.csv');

.print
.print ══ 1. the archive ═══════════════════════════════════════════════════
.print
-- Twelve sensors at four sites, and a few thousand two-second recordings
-- sitting next to them as files. Nothing in the relational tables knows
-- anything about sound.
SELECT
    (SELECT count(*) FROM sensors)    AS sensors,
    (SELECT count(DISTINCT site) FROM sensors) AS sites,
    (SELECT count(*) FROM recordings) AS clips,
    (SELECT round(sum(octet_length(clip)) / 1e6, 1) FROM recordings) AS megabytes;

.print
.print ══ 2. a number SQL cannot compute ═══════════════════════════════════
.print
-- mlake_rms_db runs a Mojo kernel over the bytes of the BLOB, in place. It is
-- an ordinary scalar expression, so it composes with everything else.
SELECT clip_id,
       round(mlake_rms_db(clip), 1)      AS loudness_db,
       round(mlake_centroid_hz(clip))    AS brightness_hz,
       round(mlake_zcr(clip), 3)         AS zcr
FROM recordings
ORDER BY clip_id
LIMIT 5;

.print
.print ══ 3. the features really do identify the sound ═════════════════════
.print
-- truth.csv says what each clip was synthesised from; the extension never saw
-- it. If the kernels were wrong these five rows would not separate.
SELECT t.kind,
       count(*)                              AS clips,
       round(avg(mlake_rms_db(r.clip)), 1)    AS loudness_db,
       round(avg(mlake_centroid_hz(r.clip)))  AS brightness_hz,
       round(avg(mlake_zcr(r.clip)), 3)       AS zcr
FROM recordings r JOIN truth t USING (clip_id)
GROUP BY t.kind
ORDER BY loudness_db DESC;

.print
.print ══ 4. joining sound to the relational world ═════════════════════════
.print
-- This is the whole idea: the site and the model come from a table, the
-- loudness comes from the samples, and the GROUP BY does not care which is
-- which.
SELECT s.site,
       count(*)                             AS clips,
       round(avg(mlake_rms_db(r.clip)), 1)   AS avg_db,
       round(max(mlake_rms_db(r.clip)), 1)   AS peak_db,
       round(avg(mlake_centroid_hz(r.clip))) AS brightness_hz
FROM recordings r JOIN sensors s USING (sensor_id)
GROUP BY s.site
ORDER BY avg_db DESC;

.print
.print ══ 5. the query that pays for all of it ═════════════════════════════
.print
-- Noise permits at these sites run 07:00 to 19:00. Find recordings from
-- outside those hours that are more than 12 dB above their own site's median
-- for the week — loud enough, in the wrong place, at the wrong time.
--
-- Every clause here is doing something neither half could do alone. The hour
-- and the site are relational. The loudness is a Mojo kernel. The comparison
-- against the site's own baseline is a window function over the column that
-- kernel produced, which is only possible because the feature is an
-- expression rather than a number somebody computed elsewhere and loaded.
WITH measured AS (
    SELECT s.site, r.recorded_at, r.clip_id,
           mlake_rms_db(r.clip)      AS loudness_db,
           mlake_centroid_hz(r.clip) AS brightness_hz
    FROM recordings r JOIN sensors s USING (sensor_id)
),
baseline AS (
    SELECT *, median(loudness_db) OVER (PARTITION BY site) AS site_median
    FROM measured
)
SELECT site,
       recorded_at,
       round(loudness_db, 1)               AS loudness_db,
       round(loudness_db - site_median, 1) AS above_site_median,
       round(brightness_hz)                AS brightness_hz
FROM baseline
WHERE hour(recorded_at) NOT BETWEEN 7 AND 18
  AND loudness_db - site_median > 12
ORDER BY recorded_at
LIMIT 12;

.print
.print ══ 6. and it was a jackhammer ═══════════════════════════════════════
.print
-- The query above was told nothing but "loud, and out of hours". Here is what
-- it actually found, according to the ground truth it never looked at.
WITH measured AS (
    SELECT s.site, r.recorded_at, r.clip_id, mlake_rms_db(r.clip) AS loudness_db,
           median(mlake_rms_db(r.clip)) OVER (PARTITION BY s.site) AS site_median
    FROM recordings r JOIN sensors s USING (sensor_id)
)
SELECT m.site, t.kind, count(*) AS clips,
       min(m.recorded_at) AS first_seen, max(m.recorded_at) AS last_seen
FROM measured m JOIN truth t USING (clip_id)
WHERE hour(m.recorded_at) NOT BETWEEN 7 AND 18
  AND m.loudness_db - m.site_median > 12
GROUP BY m.site, t.kind
ORDER BY clips DESC;

.print
.print ══ 7. the same thing, from the files ════════════════════════════════
.print
-- The clips do not have to be in a column. mlake_audio_features opens the
-- files itself and returns a row per clip — which is the shape a lakehouse
-- actually has, with metadata in a table and recordings in object storage.
-- A file it cannot read is a row with an error, not a failed query.
SELECT count(*)                                  AS clips,
       count(error)                              AS unreadable,
       round(avg(duration_s), 2)                 AS avg_seconds,
       round(avg(rms_db), 1)                     AS avg_db,
       round(avg(centroid_hz))                   AS avg_brightness_hz
FROM mlake_audio_features('data/clips/*.wav');

.print
.print ══ 8. the two ways in agree exactly ═════════════════════════════════
.print
-- Same kernel, two routes to it. If these ever disagreed, one of them would
-- be lying.
SELECT count(*)                                                 AS clips,
       count(*) FILTER (f.rms_db = mlake_rms_db(r.clip))        AS same_loudness,
       count(*) FILTER (f.centroid_hz = mlake_centroid_hz(r.clip)) AS same_brightness
FROM mlake_audio_features('data/clips/*.wav') f
JOIN recordings r ON r.clip_id = regexp_extract(f.path, '([^/]+)\.wav$', 1);
