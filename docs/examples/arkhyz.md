# Arkhyz: a ski holiday with a carried camera

**Footage:** 16 GoPro files (4K 29.97p H.264, GPMF with GPS), held in a hand or on a helmet, 16, 17 and 19 Feb
2022, ~66 min, 29 GB; two recordings split into chapters by the camera (GH01/GH02). Read from a disk another
process kept busy.
**Task:** an overview of an old ski holiday, a minute or two ("5 minutes and two tracks if it is worth it"); the
main characters, given by four screenshots the owner made: himself and Marina (a couple) and a friend; beautiful
views. 2-3 previews, done without asking ("I'm not near, don't ask me, just do it").

**What the footage was:** GPS (`gps.py track`) showed it before any picture: nearly all of it is runs down
(13-25 km/h, 100-400 m each), one chairlift ride (0053, +0.8 m/s for 30 s), three days, the 19th overcast.
Five minutes of runs would be dull, so short edits.

**How it went:**
- Not a drone. Everybody wears goggles and holds poles in front of them: the pilot detector would have found
  pilots everywhere. `camera = "carried"`: no pilot, no landings, the camera's pitch is where the head looks.
  The first cuts still had shots of nothing but snow and a shadow (camera 40-45 deg down against -8..+8 for the
  views): `W_GROUND`. A 13 s clip lost 8 s to the take-off and landing trims: carried clips lose 1 s each end.
- GPS speed in the piloting score (`--speed`); a first fix 1.5 km off in altitude (127 m for 1671 m) is
  dropped. The GoPro writes the recording's start into every chapter: chapter 2 now starts after chapter 1.
- The main characters. CLIP of a crisp 4K screenshot against 640 px crops matched every skier in blue (Marina
  "in view" 1608 s); thresholds did not help (her best match 0.84, any skier's 0.80). The owner: "track her from
  the moment you find her, YOLO has trackers". So: `people.py track` (BoT-SORT with camera motion compensation,
  every other frame, 1600 tracks), each screenshot found in the footage (CLIP, then the picture and its
  background on every frame within 0.4 s: the person alone took a stranger in blue), that person's whole track
  as the reference, the people in view at the same moments as somebody else, whole tracks labelled by CLIP plus
  clothes colours. About 3 of 4 rows right: the owner ~400 s in view, the friend ~260 s, Marina ~60 s (she held
  the camera most of the time). One screenshot, a close selfie, is not in the footage.
- "Isn't it time for a database?": `timeline.py`, every feature in one SQLite table, a row per half second.
- First cuts: days shared by "good seconds" gave 16 Feb 8 of 10 scenes and 17 Feb none: by hand
  (`--chapters`). `--lively` at 138 BPM: a median shot of 1.7 s, dropped.

**The edits** (640x360 previews, 16:9):

| edit | track | length | idea |
|---|---|---|---|
| holiday | Super Power Cool Dude, 110 | 1:27 | the queue at the lift on day 1 to a hand switching the camera off on day 3 |
| slopes | Shiny Tech, 138 | 1:03 | runs and speed |
| warm | Porch Swing Days (faster), 130 | 1:14 | views first, us among them |
| together | Montauk Point, 118 | 1:54 | only where it is clearly us: `--focus --focus-min 0.03`, us in 18 of 20 shots |

The owner chose warm, asked for Marina in it, then for a film with everybody, then for the lift. Planning warm
again with Marina as a `--must` reshuffled the other scenes (the friend's close-up opening went), so the warm he
had seen was rebuilt (same options, frame for frame) and one scene swapped (`edl_edit.py --replace 2 --clip
GH010050`). The lift: the detector saw nobody there, "we are ON the lift looking down": the GPS climb found it,
three pairs of skis under the chair; a scene of each liked edit swapped for it. Finals: together_lift and
warm_lift in 4K (3840x2160, 30 fps, HEVC): 15 and 12 min from the busy disk, the Gyroflow pieces 5 and 4
min of it; pieces aligned within 1-4 frames (a smooth chairlift ride correlates 0.67, still 3 frames).

## Prompt and comments

> **Prompt:** An old holiday of ours on skis: make an overview, a minute or two is fine. There are main
> characters, here they are: find them, they are needed. And beautiful views. I don't even know what is in the
> archive; if it makes sense to make 5 minutes with two tracks, do it, you're the boss editor. The screenshots
> are certainly us; where there are two, that's me and Marina, a couple. Do everything yourself, I'm waiting
> for 2-3 previews.

During the job:
- "Write new tools if needed; make them universal: we are at the very beginning of the project."
- On the people search: "track her from the moment you find her: YOLO has trackers, and there are other good
  trackers." "It's not a girl, it's a young woman too."
- "We have a pile of information already: is it convenient for you to read it? Time for a database?"

Comments on the previews:
- "warm is the best of them: add Marina from slopes, there is too little of her. And maybe a common film with
  everybody found, music like warm? A bit longer, but only where the detection is really clear."
- "Both are excellent." "Is there us riding the lift somewhere? Put it in if you can find it." On "the detector
  found nobody there": "but we are ON the lift looking down, you couldn't catch the faces."
- "Great, make 4K."
