# Notes: methods and measurements

What the scripts do and why, with the numbers that decided it. The jobs these come from are in
[examples/](examples/).

## The idea

Most AI editing tools cut a video from its text: a transcript or a description of each shot. Drone and
action-camera footage has no text, but the camera often records its own orientation hundreds of times a
second. So the music leads (beats, phrases and energy from the audio), the telemetry and local detectors
describe the footage, and an optimiser places each shot where the footage moves like the music. The same
inputs give the same edit; `--seed N` gives another, repeatable one. Pictures come from whole-clip Gyroflow
renders (same frame count as the original), telemetry from the original.

## Music

- **Beat grid.** Studio tracks keep a constant tempo, so a grid fit (BPM and phase maximising onset strength)
  beats a dynamic-programming tracker: the DP tracker wobbled by 127 ms on one track, the grid stays within
  ±7 ms. The downbeat is the beat position (of 4) with the most kick energy; the phrase phase is where the
  bar-loudness jumps line up. Phrases are 16 beats with an energy class 0-3.
- **Checking a track** (`music_check.py`): a studio track sits within ~10 ms of a constant grid; one sat
  68 ms off and was dropped, a live-played one wandered ±57 ms. The detector can pick a wrong multiple (73 BPM
  against 110 in the catalogue, 107 against 130): pass the catalogue BPM.
- **Shortening** (`music_edit.py`): whole phrases are cut, at most 3 jumps, only where the music sounds the
  same (phrase j like phrase i+1, judged on bar-aligned log spectra), joins on downbeats so the grid stays
  exact. Similarity alone kept 39 s of quiet intro and skipped the big part, so the path cost also rewards
  the mean energy of the kept phrases and keeping a drop. Lengths come in phrase steps (10 s at 96 BPM):
  `--target` gets the nearest path with good joins.
- **Catalogue.** `https://incompetech.com/music/royalty-free/pieces.json` lists every Kevin MacLeod piece
  with BPM, feel, instruments and description; files are at `.../mp3-royaltyfree/<filename>`. Some older
  pieces return 404.

## Telemetry and manoeuvres

- **Attitude.** In DJI's fused frame world up is −Z for `q_cam`. Image-down agrees with world down on
  96-99 % of frames; the camera is upside down on 0.2-2.7 % (flips). On the ground the camera elevation is
  −2..+11°, so "nose down" in the air means speed or a dive, and image parallax tells which. GoPro motion comes
  through Gyroflow's export (`--export-metadata 3`, `org_quat`), also world up −Z.
- **Trimming.** The first 3 s and the last 5 s go; a spike over 2000°/s near the end is a crash and
  everything after it goes; so does any 2 s with neither vibration nor rotation, and blown-out or black frames.
- **Motion onsets** are peaks of the derivative of the visible (1.5 Hz low-passed) rotation rate; they are
  what lands on beats. At random an onset falls within ±40 ms of a beat ~13 % of the time at 101 BPM.
- **Manoeuvre kinds:** roll, spin or loop (more than 250° in 1 s, by axis), dive (camera more than 45° below
  the horizon), pull-up (camera up more than 50° in 1 s), close pass (parallax above the 98.5th percentile,
  little rotation), descent from height (camera more than 18° down for 1.5 s while everything is far), ascent
  (below).
- **Ascents and arcs** come from the picture, since there is no altitude in the telemetry: the CLIP
  probability of "a view from high above" against ground-level views, the sky and a roof up close rises from
  under 0.3 to over 0.6 within 8 s and stays over 0.65 for 2.5 s with everything far (parallax rank under 0.5).
  On one shoot that found all 7 climbs seen on a sheet and 3 more, nothing else; without the sky and the roof
  among the alternatives, flying close over a roof and a chimney against the sky got in. The camera angle does
  not separate them (−8..+6° on most climbs). The way back to the ground closes an arc; "up 24 s, far 0.13"
  ranks how high it went. ([village](examples/village.md))
- **Motion from the picture** (`vis_motion.py`), for clips stabilized elsewhere whose gyro is gone (an owner's
  Gyroflow export without its original): corners tracked between consecutive frames of the proxy (Lucas-Kanade,
  both ways), the camera rotation fitted by RANSAC on pairs of bearings, chained into an orientation per frame
  that edit.py takes like telemetry. Against the gyro of a GoPro clip rendered by our Gyroflow (`stab_quat`, the
  smoothed path the picture shows): correlation 0.99 / 0.95 / 0.99 on tilt / pan / roll, scale 1.03 / 0.98 /
  1.01, error 4-6°/s. Three things made it work: the field of view per axis (GoPro SuperView keeps a 4:3 view in
  16:9: one focal length read tilt at 0.40 of the truth, pan at 0.78); starting the corners where the last
  rotation takes them; and, past 4° a frame, tracking against the previous frame turned by that rotation (a roll
  at 700°/s turns the patches ~25° a frame, and the tracker matches shifted patches only: 14 tracks of ~270
  before, the rate within 3 % after). An essential matrix between consecutive frames is useless (baselines too
  short: thousands of °/s off). The rotation-only fit takes a climb near the ground for a tilt (a take-off read
  +5°/s for a true -10..-20), so the elevation is pulled to a level camera at -4° with a 5 s time constant: a 1 s
  change of elevation correlates 0.65 with the gyro (0.57 at 15 s), flips and dives stay. The field of view is
  found in the picture: with a wrong focal length the edges of a wide frame move unlike its centre and no
  rotation fits, so the grid point whose rotations fit pairs moving 8 px or more best wins (96 x 66° against
  105 x 67° from the gyro; Gyroflow's dynamic zoom moves the focal length ±15 % within a clip anyway). The
  owner's exports came out 103-108 x 71-76°. On the ground the picture stands still: mean flow under 0.12 px
  (160 px wide) for 2 s, which took nothing of a flight for the ground on the gyro clip. Low over a slope the
  ground fills the frame and its parallax reads as a tilt: on a gyro clip flown low over dry hills, pan and roll
  still correlated 0.93 and 0.96, tilt 0.55, and 0.19-0.32 where under 0.6 of the tracks fit one rotation (1 s
  mean), 0.65-0.93 above. So under 0.6 the elevation is pulled level within 1 s and no dive, pull-up or descent
  counts there: elevation then correlated 0.68 with the gyro (0.48 before), its 1 s change 0.70 (0.36).
  Trusted: 57-89 % of the owner's five exports, least where they flew low over hillsides.
- **Movement signature** per frame: dive, pull-up, pitch-down, turn left, turn right, roll, close pass,
  cruise. A shot that moves like one placed within ~25 s of music pays `W_MOVE` (cosine above 0.6).
- **A carried camera** (`camera = "carried"`, `--carried`: in a hand or on a helmet, skiing): nobody pilots it,
  so there is no pilot to find (every skier wears goggles and holds poles in front), no landing, and the pitch is
  where the head looks: no lean in the piloting score, no dives or pull-ups. A helmet looking at its skis shows
  snow and a shadow: such shots were 40-45° down against −8..+8° for the views and −29° for a skier up close,
  so 30-40° down pays `W_GROUND`. Clips lose 1 s at each end (a button), not the 3 s and 5 s of a flight: a 13 s
  clip of us in a lift queue had kept 5 s. ([arkhyz](examples/arkhyz.md))
- **GPS speed and climb** (`gps.py track`, GPS5/GPS9 every ~18th of a second): what a carried camera's action
  is. `--speed W` puts the ground speed into the piloting score (ranked over all footage, the middle rank where
  there is no fix). The climb tells a lift from a run: of 66 min of a ski holiday, one chairlift ride rose
  (+0.8 m/s for 30 s); CLIP's "a chairlift" took 142 s of runs under a lift line for one. The first fixes after
  power-on can be far off (127 m of altitude for 1671 m a second later, DOP 4.9): a jump within the first 3 s
  drops what came before it.

## The picture

- **Image motion** (`analyze.py`): Farneback flow on 160 px grey frames at half the frame rate, a global
  affine fit; the residual (p90) is what camera rotation cannot explain, i.e. parallax from things close by.
  An aggressive flight had a median of 1.7 against 0.9 for calm ones.
- **Look-alikes:** 12×9 Lab thumbnails, RMS distance; σ = 0.56 × the median distance between random frames
  (22 on a construction site, 15.5 in a grey hall; a fixed 22 counted 71 pairs alike in the hall, the scaled
  one 9). Thumbnails do not see one subject from different angles; whole-frame CLIP does not either (random
  windows 0.88, the same roof 0.97, unrelated gardens up to 0.96). Activities in words do (Places).
- **Decoding:** 4K HEVC 10-bit decodes at ~36 fps on the CPU in PyAV; NVDEC through PyAV was slower
  (17 fps, the download to host memory); 640 px proxies decode at ~780 fps. Three Gyroflow renders in parallel
  give ~46 fps in total from 4K HEVC 10-bit. When re-encoding with PyAV, set `frame.time_base` with
  `frame.pts`, or the muxer fails with EINVAL.

## Beautiful views

An option (`--beauty W`), because beauty is the job's own: mountains for one, a beach for another. `beauty.py`
scores every CLIP sample (0.5 s) twice: the LAION aesthetic predictor (a linear head on CLIP ViT-B/32 trained on
people's 1-10 ratings of photos: a good picture whatever it shows) and the probability of the job's beautiful
views in words against what else drone footage shows. A window's view is the mean of the aesthetic rank and
that probability; it pays `W_BEAUTY` everywhere, takes 0.3 of the cinematic piloting score, and in a people edit
excuses a shot with nobody in it. On a Dagestan trip the turquoise river in a canyon topped the check sheet; a
giant seaplane on a beach (a landmark the owner would want) was among the worst by the aesthetic predictor:
a landmark comes in as a theme or a moment, not as beauty.

## Piloting score

A percentile across all footage. Action (loud music): parallax 0.45, forward lean or dive 0.25, rotation
0.30. Cinematic (quiet music): parallax 0.55, lean 0.30, smoothness 0.15 (1 − rank of angular jerk); the two
are mixed by the music energy, because matching quiet music to little rotation picked dull flying for the
openings. With climbs asked for, the cinematic score takes 0.25 of CLIP height. The best windows are low fast
passes and dives; the worst are hovering in front of people. It misses slow descents and climbs (~0.1-0.4),
which owners call beautiful: those come in as editor's pieces.

## Structure of an edit

- **Scenes:** one clip per 16-beat phrase; shots inside jump 1-12 s forward, chosen together by a DP. Inside
  a scene the camera keeps turning the same way across a cut (continuity 0.76 against 0.48 for random pairs).
- **Musical hits:** "on a bang there must be a manoeuvre or a cut". A hit is an onset peak in the top 10 % of the
  track's peaks and 1.8x the envelope around it (80th percentile within ±2 s), on a beat (±80 ms). An orchestral
  track at 130 BPM had 29 in 97 s, often in pairs two beats apart (the last half bar of a phrase and its downbeat)
  or every other beat; with cuts on phrases only, 20 met neither a cut nor a motion onset. Now a cut within 2 beats
  of a hit moves onto it (shots keep 4 beats), a shot is cut at a hit inside it if both parts keep 6 beats (8 with
  `--long`), and a hit left inside a shot pays `W_HIT` 4 x how well a motion onset meets it: 4 of 25 on-beat hits
  on nothing, piloting median unchanged (0.87). `W_HIT` 8 left 6 and took the median to 0.76: the footage has no
  onset there, so a shot with a hit inside bends its speed (0.7-1.4x) to put one on it, and plays at 1x
  elsewhere (most such shots 0.97-1.03x). Another track (120 BPM) had 10 of 15 hits between beats: those wait for
  cuts on half beats.
- **Long flights** (`--long`): a shot per 16-beat phrase, for a calm report where every view is beautiful (a
  kremlin from above, "longer flights: everything there is beautiful").
- **Speed:** ramps bend speed between 0.7x and 1.4x through beat anchors so onsets land on beats (sync from
  44 % to 71 % on the first test; read as "breathing" by some, hence off by default). The hero ramp: beat 0
  rushes in at 1.4x, beats 1-2 at 0.5x with the manoeuvre on beat 2, then base speed; at most 2, where the
  music lifts most, never the same kind twice. `--pace`: loud phrases at 1.1-1.2x, which hides hesitation.
  50 fps shown at 25 fps is true slow motion; 25 fps sources get frame blending.
- **Editor's pieces** (`--must/--start/--end`) are placed before anything else: a hero once took the footage
  next to a piece and the piece silently became another shot. A piece longer than its shots gets the whole
  phrase; longer than the phrase, it plays faster, up to 1.7x (a 25 s climb in a 15 s phrase). `--start`
  folds the pickup before the first phrase into the opening shot. An `--end` moment ends 2 s before the end,
  so the fade-out follows it.
- **Flying scenes** (`--fly N`): one whole phrase around a manoeuvre, people ignored; elsewhere in a people
  edit a shot with nobody in it pays `W_EMPTY`.
- **Chapters** by recording day share the scenes by good flying and, with `--beauty`, beautiful views (a calm
  day over the most beautiful place had 71 good seconds of flying, 285 with the views); `--chapters 3,2,4,2`
  sets them by hand. An editor's moment goes to a scene of its own day: one of 9 Sep placed in a 5 Sep scene
  left that scene with no footage. ([dagestan](examples/dagestan.md))
- **Landings** (the 10 s before a stretch on the ground) are marked, not cut: every clip of one trip ended in a
  slow approach and a flop that read as a roll ("a dull landing"), but "a landing can be an ending", so a
  window pays `W_LANDING` anywhere but in the last scene.
- **Changing a finished edit** (`edl_edit.py`): planning again after a small comment moves other scenes (the
  track is shortened along another path, the scenes per day shared anew), and cutting a scene's phrase out of
  the track moves every later scene off its beats. So the music stays: `--replace K` finds scene K again
  against the rest as it stands; `--end-from K` lets the ending take over from scene K (earlier in its clip,
  at 1x, to the same last frame). "Add Marina to the one I liked": planning again with her as a `--must`
  dropped the opening the owner had liked, so the edit he had seen was rebuilt (same options, deterministic)
  and its weakest scene replaced by her clip (`--replace K --clip`). ([arkhyz](examples/arkhyz.md))
- **Opening pieces and pickups:** an opening piece a little shorter than the opening phrase fills it at down to
  0.8x (a 12 s queue clip against a 14 s opening had gone to the second scene); a pickup under 1.5 s joins the
  first shot instead of flashing by.
- **Distance in time:** footage close to another scene's shot pays `W_NEAR` (2, over 20 s); the parts of a
  split recording count as one flight (an edit once opened with two passes over the same warm-up 17 s apart).
- **Themes** (`--theme "TEXT[:N]"`): a scene shows a theme when one of its shots averages 0.4 or more; once
  a theme has its N scenes, a candidate pays `W_THEME` 8, full from 0.55, nothing under 0.35. At 4 on a
  linear ramp a repeated person still came back (a person is worth `W_PEOPLE`). A shot under 2 s does not use
  a theme up. ([village](examples/village.md))
- Fades: 0.6 s in, 2 s out, the same in the preview and the final.

## People and the pilot

- **Detection** (`people.py detect`): YOLOv8s-pose, 2 samples a second; per person the box, 17 keypoints and
  flags: radio pose (wrists together in front of the torso), waving (a wrist above the shoulders), facing the
  camera, a radio in the hands (COCO `remote`/`cell phone` near the wrists). ~1 min on the GPU for 16 min.
- **People value:** a person covering 2 % of the frame or more counts 1, smaller ones less; ×1.3 facing the
  camera, ×1.6 waving; approaching or receding within ±0.5 s adds 0.5; a pass between the legs, under or over
  adds 1.5. `--people 4` weighs it like piloting.
- **The pilot** is the easiest person to pick: standing still, close to every take-off, in every flight. In
  one people edit nearly all "people" were the pilot. Clothes-colour clustering failed (backlit dark jackets,
  children holding toys strike the radio pose). CLIP zero-shot on each person crop does it: "wearing FPV
  goggles" / "holding a drone radio" against children, women, silhouettes and people with busy hands (a pram,
  a phone, a bag, a child's hand); tracks carry the label to frames where the goggles are not seen. Without
  the busy-hands alternatives a mother pushing a pram was "the pilot" in 241 of 373 rows; with them 70 rows,
  all the pilot. ([march](examples/march.md), [koprino](examples/koprino.md))
- **Pilot budget:** the pilot is worth 0.3 of another person; once 15 % of the edit shows the pilot leading
  the picture (bigger than anybody else in it), every further such shot pays 3× its share. "Alone in the
  picture" was not enough: the pilot in front with someone far behind got ~10 s of a 48 s edit.
- **Close passes** (`people.py passes`): a pass lasts tenths of a second, so it scans at 10 Hz, follows each
  person by box overlap and tells the way out by the edge the body leaves through: between the legs, under,
  over, past on a side, "(lying)" for a horizontal body.
- **Main character in words** (`subject --text`): the crop is widened to take in what is beside the person
  (a pram, a dog); counts 2× another person. With several look-alike groups set the threshold higher (0.8):
  at 0.5 most children holding rackets were "a coach". "A child" found 6 rows; "a child swinging a racket"
  took 88 % of people. ([tennis](examples/tennis.md))
- **By example** (`who --ref NAME=IMAGE|CLIP:SEC:X`): CLIP embeddings of every crop, each row to the
  reference it resembles most, the best finds added as references; names with `_` are other people (needed:
  without the coach as a negative a small girl in black took the coach in black). Defaults 0.80 cosine and a
  0.03 lead; at 0.84 two of five children were left with almost nothing. A label follows a track only to rows
  that still look most like that person. At 640 px it cannot carry the pilot by look: everyone in jeans and
  blue matched, negatives included. `split --k 3` groups one label by look to drop look-alikes.
- **From a screenshot, along tracks** (`people.py track`, then `who`): a crisp 4K screenshot against 640 px crops
  of skiers matched every skier in blue (a woman "in view" 1608 s of 66 min); her best match was 0.84, any
  skier's 0.80, so no threshold worked. Now the screenshot is found in the footage (CLIP over the person rows,
  then the picture: the person and the whole screenshot laid over the frame, on every frame within 0.4 s, from
  0.6; the person alone took a stranger in blue at 0.56), and that person's track becomes the reference: many
  views at the footage's own size and blur. Tracks: BoT-SORT on every other frame, its sparse-flow motion
  compensation keeping ids while the camera skis (1600 tracks over 66 min, 94 % of rows in one; ~1.4x real
  time per process, CPU-bound). People in view at the same moments as a known person are somebody else, for
  free. A whole track goes to a person by the median of its readable rows (pose sure of a body, shoulders and
  hips seen, 8 % of the frame high: skis and snowboards up close had passed for people), at 0.85 at least and
  ahead of the others; CLIP and the colours of the clothes together (CLIP alone took children in pink and
  orange for a woman in turquoise and red trousers, half her rows; with the colours ~3 of 4 rows were right).
  A close selfie screenshot was not in the footage (0.51): kept as an image reference.
- **Somebody filming is not in the picture:** the woman with ~60 s in view held the camera most of the time; on
  the chairlift "we are on it, filming down": the detector finds nobody, the GPS climb finds the ride.
- **Faces** (`faces.py`): two children in the same printed T-shirt are told apart only by face. Eyes are
  ~3 px apart in the renders and ~20 in the 4K originals, so faces are read there (YOLO-pose at 1280 px,
  aligned on eyes and nose, SFace), the frame matched to the render by an ORB homography; one reference face
  sorted 142 faces. Reading 4K by seeking took 35 min; reading on between near samples and caching is the fix.
- **An edit about one person** (`--focus`): a shot is the person, a group with them, or a cutaway (adults,
  the place), never another child of the same kind instead of them (`--peers`, `W_IMPOSTOR`). Several people
  are one subject as `--subject a+b+c`; `--focus-min 0.03` counts them in a shot only at 3 % of the frame or more
  ("only where the detection is really clear": crowds at a lift with one of us far away had counted).
- **Animals** (`objects.py`) count like a person of the same size; in a garden YOLO sees them in cabbage beds
  and clothes on a porch, hence `--animals 0`.

## Places and activities

- **Frame embeddings** (`places.py embed`): CLIP ViT-B/32 every 0.5 s, cached; 1.5 min for 30 min of footage.
- **Search and routes:** softmax over the asked places and alternatives; a route is a stretch of each place
  in order within `--within` s ("a corridor → a staircase → a tennis court" found the flight the owner
  remembered, in a few seconds).
- **One edit per place:** an archive of several trips gets an edit per place, each with its own `clips` in the
  job file. The picture alone put four places 5-25 km apart into one group (autumn birches and water everywhere);
  the GPS told them apart.
- **Light is not a place:** a theme "a sunset over a town" also took a kremlin wall in the low sun for a sunset and
  left that whole flight out; extra sunset scenes were found again in daytime footage with `edl_edit.py
  --replace K --clip CLIP`, choosing the clip whose motion met the same musical hits.
- **Locations:** two clips are alike when the views of each are found in the other (median best cosine, both
  ways); average linkage joins at 0.83. Three shoots mixed (28 clips): pairs of one place 0.85-0.96, of
  different places 0.80 at most.
- **Activities:** a general list against non-activities; on one shoot the roof repair 82 s, mowing 54 s, a
  swing 23 s, the pilot 56 s of 14 min, as seen by eye. Wording matters: "a man working on a roof" caught the
  roof from every angle (0.42-0.97, 0.22 at most elsewhere), "a man on a ladder repairing the roof of a wooden
  shed" missed it from above. `qc.py` reports activities per scene from these.
- **Where it was** (`gps.py`): a place named from the picture is a guess. A turquoise river in a canyon read as
  the Sulak canyon by eye and by CLIP; the GoPro's GPS (GPMF GPS5) put it 30 km south, on the Avar Koysu by the
  Irganay reservoir. Check names against the GPS before captions or a title. ([dagestan](examples/dagestan.md))
- **The owner's picks** (`--picked W`): a clip the owner stabilized themselves and kept without its original was
  chosen out of the shoot ("I didn't stabilize them for nothing"); every window of one pays W (2: half the
  piloting range). At 2 they got 5 to 8 of the 13-19 shots of each edit of one archive.
- **Old exports** (`exports.py`): centre thumbnails at 10 Hz, aspect matched, best correlation over every
  offset; pieces that run on in the same original merge. A 35 s export was found at 72.8-107.8 s (0.84
  against 0.13); a whole-clip export is reported as a copy, not a pick.

## The timeline index

`timeline.py` builds `out/<job>/timeline.sqlite` from the features after every `features` step: a row per clip
per half second (GPS speed, altitude and climb, rotation, flow, parallax, brightness, people and each main
character's size, aesthetics and scenic views, the likeliest activity) and a row per person (track, box, name).
The npz files stay the source; the index answers questions across them in one query ("where is she in view, with
a view behind, at speed") instead of a one-off script loading five files onto one time grid. The owner asked
for it once the information piled up.

## Gyroflow and stabilization

- Whole clips are rendered once at 640 px (`gf_render.py`: ~85 fps from DJI O3 4K 16:9, ~46 fps in total
  from three parallel renders of 4K HEVC 10-bit 4:3); these are the proxies, frame for frame with the original.
- A render with in/out points is aligned by correlating visible gyro rotation with the render's flow
  magnitude: −1..+2 frames on 9 artificial cuts.
- `trim_range_only` (on by default) freezes the camera past the edges of a trim: a 6 s piece differed from the
  whole-clip render by 2-3 (0-255) in the middle and up to 20 in its last 1.5 s; off, 0.2-0.4 and 2-4. Pieces
  keep 1 s of margin.
- The CLI never sets a bitrate below the source's; force it through `encoder_options`. H.264 on NVENC is
  8-bit only (`-pix_fmt nv12` for 10-bit sources) and stops at 4096 px (HEVC above that).
- Propellers or a lens guard in view are framing: Gyroflow's zoom, or `--zoom` (1.16 hid a guard's dark arcs
  and blue corners; a frame-edge darkness metric could not tell them from tree shadows).
- DJI goggle recordings (`DJIG*.mp4`) carry no gyro and are skipped.

## Colour

- A 3D `.cube` is expanded once into a 256³ table (trilinear, 2 s) and applied with one lookup per frame
  (~15 ms at 640×480). Frames are decoded as BT.709 limited range explicitly (untagged proxies would be read
  as BT.601); output is tagged BT.709. A circulating "Avata 2 LUT" is a 1D tone curve, not used.
- Every official DJI D-Log M → Rec.709 LUT (O4, Avata 2, Mini 4 Pro, Air 3/3S, Flip, Pocket 3, Action 4/5,
  Nano, 360) is the same file. Indoors it looked dim and green; `--grade auto` adds per-shot levels (p0.5 → 6,
  p99.5 → 242) and white balance from near-white pixels, constant within a shot, alike across the edit.

## Finals

`final.py` renders only the pieces used (Gyroflow trims with margin, `trim_range_only` off), finds each
piece's start by motion correlation (±1-3 frames from the asked millisecond), applies the LUT and grade on
the GPU and encodes with NVENC. A 45 s 16:9 4K edit: 9.5 min with x264 on the CPU; a 40 s 4:3 4K D-Log edit:
33 min, 25 with 10-bit pieces and NVENC; a 58 s 4:3 4K edit with zoom: ~23 min. Slow motion at 50 fps repeats
frames (no interpolation yet).

## QC

`qc.py` prints about a dozen OK/WARN lines. From the plan: length, shot lengths, chronology, movement
variety, people, the pilot leading the picture, the main character, piloting (shots of people up close and
editor's pieces apart: hovering is their point), activities per scene, flights. From the picture: audio
against video length, dark frames, frozen frames, brightness jumps at cuts.

The cut sheet (`edl_sheet.py --cuts`, made by montage.py's `qc` step) is a row per cut: two frames each side
and, between them, 4 beats of music each side with its hits and the visible rotation of both shots at their
playback speed, faint where the edit does not show it (what the cut left out, what came before the incoming
shot), with motion onsets and manoeuvres. A benchmark of editing agents (Timeline-Bench, 2026) found most of
their failures in the craft: they judged footage and renders by stills and transcripts, and the rhythm of a cut
is not in a still. On one edit (fpv2022, 12 cuts) it showed at a glance a dive starting on the downbeat after a
cut, and two hits before that cut falling on level flight.

## Ideas not done yet

- Stabilization from the picture, for footage without motion data that was never stabilized (phone clips,
  goggle recordings): `vis_motion.py` reads its rotation already (the shake too, which edit.py low-passes);
  smoothing that rotation and turning each frame back would stabilize it. Checkable on the gyro clips.
- An absolute horizon for motion from the picture (sky and ground, or a regression of the camera angle on the
  CLIP embeddings, trained on the gyro clips): the elevation now drifts and is pulled level.
- Tracks without a constant tempo: a neural beat tracker (Beat This!), drum stems for cleaner accents.
- Cuts on half beats, for musical hits between beats (two thirds of the hits of one track).
- Scenes in the order of recording within a day (day, evening, sunset): a sunset placed between daytime scenes
  had to be swapped out by hand.
- Slow motion in finals: frame interpolation, or 100 fps sources.
- Pick the best of N seeds by skill, sync and variety.
- Trick types from the gyro axes (flip, roll, power loop, split-S); flips on snares, rolls on fills.
- Proximity from the flow that pure camera rotation predicts, instead of an affine residual.
- Gaps as heroes: flying through an opening shows as a 0.2-0.4 s brightness dip with high parallax and a new
  scene after it.
- Per-shot Gyroflow settings: horizon lock off for flips.
- An EDL / OTIO export for finishing by hand.
- GoPro HiLight tags: a button pressed while recording is kept in the file, the owner's own pick (what Quik
  builds on): weigh like `--picked`, or offer as `--must`. No footage with tags to try it on yet.
- The footage's own sound, in words (CLAP, the audio twin of CLIP): laughter, a splash, shouting as `find:`
  and themes (Insta360 and GoPro listen to it). Carried cameras only: a drone hears its propellers. No footage
  to try it on yet.
