# Tennis: a series from one shoot

**Footage:** kids' tennis practice in an indoor hall: 11 DJI O4 Pro clips from a small quad and 1 from
another (4K 4:3, 50 fps, 10-bit D-Log M, ~30 min; most are 4-minute parts of longer recordings).
**Task:** 3-4 short (30-45 s) lively kids' edits and one about the coach (a woman in a black tracksuit), each
with its own track, previews at 640×480; later one film per child for five girls the owner pointed at in one
frame.

**Edits:** the coach (Motivator, 126 BPM), smiles (Happy Alley), the game (Happy Bee - Surf), fly-bys (Feelin
Good); then one ~45 s film per child, all opening on the flight from the lobby up the stairs onto the court.

**What happened:**
- 10-bit sources: Gyroflow's H.264 on NVENC failed on every frame until `-pix_fmt nv12`.
- Colour: the official DJI LUT looked dim and green indoors, "as if no LUT". After a comparison of 13 LUTs
  the owner picked "официальный + автоуровни — но выровняй картинку в пределах ролика" ("the official one plus
  auto levels, but match the picture within the edit"): `--grade auto`. All official DJI D-Log M LUTs turned
  out to be the same file.
- The coach in words needed a stricter threshold and the children as alternatives (everyone holds a racket).
- The owner's "under a person" moment was the coach in a plank: found as a pass "under (lying)" at 200.4 s
  once passes were scanned at 10 Hz and followed per person.
- The lobby → stairs → court flight: the owner said "не факт что он в начале" ("not necessarily at the
  start"), "ищи объекты — типа дверь, потом лестница" ("look for objects: a door, then the stairs"). A route
  in words found it in two clips in a few seconds (`places.py`).
- Split recordings count as one flight for distance in time: the game edit had opened with two passes over
  the same warm-up 17 s apart.
- Five girls by example (`people.py who`): words cannot tell two girls in pink T-shirts apart. When stricter
  thresholds left two of them with almost nothing, the owner: "немного чужих это ок — не перегни палку" ("a few
  strangers are fine, don't overdo it"). In view: 47 s to 242 s per child.
- Look-alikes standing in: the first per-child cut had another girl at 0:54 (a track jumped from the girl to
  a boy's close-up). Rule: a shot is the child, a group with them or a cutaway, never another child instead.
- Faces: two girls in the same printed T-shirt (one with a headband, one with a ponytail) were told apart only
  by their faces in the 4K originals; one reference face sorted 142 faces.
- 4K 4:3 final: 33 min for a 40 s edit, 25 min with 10-bit pieces, the grade on the GPU and NVENC.
