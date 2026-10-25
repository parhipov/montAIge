# March: people first

**Footage:** 5 GoPro clips (4K, 25 fps, ~16 min) on a sunny, snowy riverbank with adults and kids. The camera
clock was wrong, so no date captions.
**Task:** people, a pleasant track, about one minute. The owner then asked for a scripts-first way of working:
checkers that print a few lines, pictures only when something is off.

**v1:** `objects.py` (YOLO people, vehicles, animals at 2 samples a second; approach / recede / close events)
and `qc.py` (the OK/WARN report) appear; `--people 4`, 8-beat shots.

| | Rainbows, 110 BPM | Crinoline Dreams, 120 BPM |
|---|---|---|
| length | 1:02, 12 shots | 0:54, 13 shots |
| people | in 100 % of shots | in 85 % of shots |
| piloting, median / worst | 0.81 / 0.50 | 0.78 / 0.01 (the closing shot hovers by the people) |

**The owner:** nearly all the "people" were the pilot, standing in FPV goggles. Then: "пусть там будет
пилот — просто не слишком много" ("let the pilot be there, just not too much"), and "решать не скриптам, а
тебе; главное от скриптов — пилота видеть" ("the decision is yours, not the scripts'; the scripts just have
to see the pilot").

**v2:** `people.py`: poses with radio-pose, waving and facing flags; clothes-colour clustering failed
(backlit jackets, children holding toys); CLIP on person crops found the pilot in 297 of 453 people rows,
matching the owner's account (2 misses on a girl next to the pilot). Close passes as a framework (none
between the legs in this footage). The pilot is worth 0.3 of a person, at most ~15 % of the edit; others
×1.3 facing the camera, ×1.6 waving. The owner on the v2 previews: "оба вполне годные" ("both quite good").
*Crinoline Dreams* v2 (54 s, 13 shots) is one of the two clips at the top of the README; it also has a 1080p
final.

This job also found `trim_range_only` in Gyroflow (pieces drifting at their edges), measured with a trim
test: see notes, Gyroflow.

## Prompt and comments

> **Prompt:** People; pleasant music, your choice; about a minute. Keep it short: nobody watches long videos.

Comments on the previews:
- "Too much of me."
- "Both came out quite good."
