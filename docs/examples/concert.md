# Concert in a factory hall

**Footage:** DJI Avata 2, D-Log M: a rock band in a factory hall with presses, rails and cold-spark
fountains on stage. Some clips without a Gyroflow render were skipped.
**Task:** a concert edit on the v3 scripts. Inside an edit, the owner said: "ты босс" ("you are the boss").

**v1:** "The Complex" (104 BPM; "gets very big around 1:46"): 1:32, 10 scenes, 33 shots, 4 of 4 heroes with
a real manoeuvre (mostly dives here), piloting median 0.73, 43 % of onsets on a beat; ~2 min to render with
the LUT. New: the `.cube` LUT, alignment of two trimmed Gyroflow renders (starts found at frames 224 and 26,
correlation 0.69 and 0.70 against 0.54 and 0.39), the look-alike scale following the footage (σ 15.5 in this
grey hall), music path search rewarding energy and a kept drop (path [0, 14, 15, 16, 17, 24, 25, 26]), the
tempo hint (73 BPM detected for a 110 BPM track).

The owner on v1: "так себе" ("meh"). Too many look-alike manoeuvres (4 identical hero ramps, mostly dives,
plus 2 slow-motion scenes around a manoeuvre), the opening read as mediocre piloting; the rhythm "топ" (top).

**v2,** three tracks with the same rules:

| | The Complex | Industrial Cinematic (`--bpm 110`) | Rising Tide |
|---|---|---|---|
| length | 1:32 | 1:27 | 1:27 |
| music joins cost | 0.28 | 0.07 | 0.00 |
| piloting, median / first 3 shots | 0.89 / 0.96 | 0.85 / 0.82 | 0.87 / 0.88 |
| onsets on a beat | 47 % | 44 % | 59 % |
| heroes | pull-up, dive | pull-up, dive | pull-up, dive |
| look-alike pairs | 16 | 4 | 0 |

New in v2: action and cinematic piloting scores mixed by music energy, an opening bonus, movement
signatures with a variety penalty, manoeuvre kinds (the hall: 48 dives, 28 pull-ups, 2 passes, 3 rotation
tricks), at most two heroes of different kinds. Rising Tide came out the most "concert": the band and the
sparks run through all of it. The near-black frame at 1:00 there is the quad flying through a round opening
in a machine: a gap, the idea of gaps as heroes. All three use the same pull-up and dive as heroes.
