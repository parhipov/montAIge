# Construction site: v1 to v3

**Footage:** 5 DJI O4 flights over a construction site (excavators, a sandy courtyard, walls), ~19 min,
4K. **Track:** "Funkorama" (Kevin MacLeod, 101 BPM).
**Task:** cut the flights to a track by the gyro; deterministic, with seeded variants. Then (v3): half the
length, "make my piloting look cool".

| | v1 | v2 | v3 |
|---|---|---|---|
| length | full track 3:21 | full track 3:21 | shortened to 1:37 |
| shots | 91 | 68 in 21 scenes | 35 in 10 scenes |
| flight changes | at every cut | at phrase boundaries | at phrase boundaries |
| motion onsets on a beat (±40 ms) | 47 % | 71 % | 53 % |
| piloting, median percentile | | | 0.79 |
| render | 6 min from 4K | 6 min from 4K | 31 s from 640 px proxies |

At random an onset lands within ±40 ms of a beat ~13 % of the time at 101 BPM.

**v1:** a constant-tempo grid; 16-beat phrases get cut patterns by energy (quiet: long shots at 0.5x, busy:
2-beat shots); every frame of every flight is a candidate start. Score: footage rotation intensity against
music energy (percentiles), onsets on the shot's beats, a penalty for onsets between beats, penalties for
reused or nearby footage; filled greedily, loudest first. Weakness: every cut jumps to another flight.
The owner: "а и неплохо!" ("not bad at all").

**v2:** scenes (one flight per phrase, shots jumping 1-12 s forward, chosen by a DP), cuts that keep the
camera turning the same way (continuity 0.76 against 0.48 random), speed ramps 0.7-1.4x (sync from 44 % to
71 %), hero moments on the music's lifts, fades. Open question: do the ramps read as "breathing"?

**v3:** the track shortened by whole phrases (path [0, 8, 9, 10, 11, 12, 13, 18], joins cost 0.09 of 4; in
this track phrase i equals phrase i+12); the piloting score (parallax 0.45, lean 0.25, rotation 0.30),
checked on a sheet: the best windows are low fast passes around the excavators and dives down the walls into
the courtyard; 12 tricks and 39 dives; four hero ramps, all with a real manoeuvre; loud phrases at 1.1-1.2x;
a look-alike penalty (pairs from 24 to 12; the sandy courtyard is still where the best flying is). Sync drops
to 53 %: hero ramps are rigid and the score now prefers piloting (`W_SKILL` = `W_SYNC` = 3).
