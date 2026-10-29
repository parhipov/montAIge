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
- **An unsure downbeat.** The downbeat is the position of four with the most kick. A track with the kick on 1 and
  3 alike (+5 % on 1 over the whole track, +3 % on 3 over its cut) got its phrases from the whole track and its
  bars from the cut: 13 of 14 cuts of the edit fell on beat 3. `music_check.py` prints the margin; the tracks that
  worked had +12 % to +60 %: under 10 % take another track.
- **An unsure phrase start.** A phrase starts on the bar of four where the bar-loudness jumps line up. A rap loop
  had them spread alike over the four bars (31.9 / 37.3 / 34.4 / 36.6, +2 % for the chosen one), a bouncy
  sing-along +3 %: their edits' scene changes were "hardly tied to the music" and "all mixed up", with cuts on the
  downbeats and 6 of 9 on hits. An orchestral track at +127 % cut the same footage with its timing liked.
  `music_check.py` prints the margin; under 15 % take another track.
- **Shortening** (`music_edit.py`): whole phrases are cut, at most 3 jumps, only where the music sounds the
  same (phrase j like phrase i+1, judged on bar-aligned log spectra), joins on downbeats so the grid stays
  exact. Similarity alone kept 39 s of quiet intro and skipped the big part, so the path cost also rewards
  the mean energy of the kept phrases and keeping a drop. Lengths come in phrase steps (10 s at 96 BPM):
  `--target` gets the nearest path with good joins. The cut keeps the whole track's phrases (its first piece
  starts at 0): found again by bar-loudness jumps, a song cut from its first verse to its last chorus got its
  phrases a bar late (2.2 s, not 0.2), and the scene cut into the chorus came 2 s after the chorus began. The
  path is a dynamic programme over buckets of phrase count, jumps, energy sum and two flags (the least jump cost
  per bucket, 0.1-0.3 s; trying every set of jumps had taken minutes); ties are common (i -> j costs the same
  as i+1 -> j+1 when j resembles i+1) and go to the fewest jumps, then the earliest cuts.
- **A song with words.** A song the owner names (not the catalogue) has a chorus everybody knows. Whisper
  (large-v3-turbo) on the whole mix gave 60-80 s chunks; the hook's words, placed by their share of the words in
  a chunk, fell within ~2 s of the starts of two sections of every round (a quiet half and a loud half of the
  chorus). The cut verse → last chorus is a join the song makes itself (phrase likeness 0.99); a join from the
  end of a chorus into the next one is not (-0.6).
- **Sections** (`music_edit.sections`, printed by edit.py for every edit): phrases of the whole track joined by
  average linkage on the same bar spectra, the similarity rescaled per track (median pair 0, a copy 1): the
  median pair sat at 0.46 (Shiny Tech) to 0.94 (Morgana Rides), repeats at 0.93-0.99, so a fixed threshold
  joined all of one track and none of another. At 0.5 a block of four phrases came back three times in
  Motivator, and Nonstop's chorus was its phrases 8-11 and 22-23, where the track takes off. The chorus is the
  label that comes back with the highest mean energy; a shortened track takes the labels of the whole one (a
  chorus is the song's, not the cut's). `--sections` puts the heroes and flying scenes on it, the first chorus
  first, and with `--lively` the 4-beat shots only there. A homogeneous track (Morgana Rides) gets many labels
  and a weak chorus.
- **Drums for the hits: no.** Hits from the percussive part (median-filter separation) were the hits of the
  mix: 63 of 69 (Montauk Point), 55 of 57 (Nonstop), 80 of 88 (Motivator), 21 of 25 (Heroic Age). A hit is
  already a top-10 % onset peak on a beat, and those are drums.
- **Bass booms** (`--booms`): in a guitar pop song at 120 BPM the "boom" an owner wanted the cuts on was the bass
  coming in after a short gap on the downbeats of bars 2-4 of every band phrase: the sub-bass (30-90 Hz) +31..+42 dB
  over the 0.2 s before (bar 1: +16, the bass rings on into it; the quiet half of the chorus has no bass). The mix's
  onset peaks found 1 of those and put their hits on the beat before (a strum), so cuts moved onto them came 0.5 s
  early. With the booms as hits 20 of 20 sit on downbeats, and every cut of the edit did too (6 of 16 before).
- **Booms by default, at 10 dB.** A rock track at 128 BPM whose sub-bass never falls silent had its booms at
  +12..+18 dB (two after a silence +52..+55), its plain kicks under +8.1: 20 dB found 3 of 8 and the owner heard
  "booms and no cut" at three of the others. Now the hits are the onsets and the booms together (`--booms` only
  the booms, `--no-booms` only the onsets) and a boom jumps 10 dB. That track also has two extra bars in its
  middle, so the shortened cut put its booms 8 beats into the phrases of the whole track's grid: a scene of one
  shot (an editor's piece of a whole phrase, a flying scene) held a boom uncut. One-shot scenes now go to a
  phrase without a boom inside first, a flying scene over a boom keeps its two shots, and pieces of 3 s (half a
  phrase) leave the cut on the boom.
- **The beat a listener hears** (`beats.py`). The onset grid sits on the strongest onsets; in an oom-pah (country,
  bluegrass, polka) those are the strummed chord between the bass notes: the sub-bass 2.8-5.7x stronger on the
  grid's half-beats than on its beats (13 other tracks 0.2-1.5), the chords changing on its "4 and". Beat This! (a
  trained beat and downbeat tracker, CPJKU 2024) put every beat of two such tracks 0.48 of a beat off the grid, and
  of a steel-drum track with no such bass (its half-beats carry 0.88 of the onsets); on five others it agreed with
  the grid, and it put the downbeat of a track with the kick on 1 and 3 alike on 3. The grid keeps its constant
  tempo and takes the tracker's phase (its beats a quarter beat or more off, circular mean, resultant 0.6 or more)
  and its downbeat (60 % of its downbeats on one place); a shortened track takes the whole track's beats through
  the pieces: on a cut of six phrases the tracker heard bars of 2 and 3 after the joins, on the whole track bars of 4.
- **Catalogue.** `https://incompetech.com/music/royalty-free/pieces.json` lists every Kevin MacLeod piece
  with BPM, feel, instruments and description; files are at `.../mp3-royaltyfree/<filename>`. Some older
  pieces return 404.
- **Tracks like a liked one** (`music_like.py`): picked by the catalogue's words, two of three tracks of one job
  "did not fit at all" (an orchestral "epic, uplifting" one and a "driving" synth one) while a happy steel-drum
  piece (Paradise Found) fitted. So the next ones come by how they sound: CLAP's music model
  (`laion/larger_clap_music`) on 10 s windows, averaged per track, over catalogue pieces within 12 % of the liked
  BPM sharing two of its feel words (86 of 1443). Every track's mean came out 0.97-1.00 like any other's (music
  against all sounds); centred on the mean of all tracks embedded, the liked one against the rejected orchestral
  one went from 0.98 to -0.47, and the candidates spread -0.52..0.69. The top had a steel-drum and marimba piece
  (Moonlight Beach) and an upbeat acoustic one, but also an Irish pub tune first and another steel-drum party
  piece 55th: the sound is a better guess than the words, not an answer; the owner still chooses by ear. A liked
  track from outside the catalogue (the owner's own) is described by `--bpm` and `--feels` in the catalogue's
  words.

- **A performance filmed without sound** (DJI O4 clips carry no audio) could not be aligned to its recording by
  picture (frame brightness, the stage lights and the frame difference against the track's envelopes at every
  lag and tempo stretch: peaks no better than random correlations give). The band stays cutaways where sync
  cannot be read; a sound recording of the room (the goggles' microphone, a phone) would give the offset in one
  cross-correlation.

## Telemetry and manoeuvres

- **Attitude.** In DJI's fused frame world up is −Z for `q_cam`. Image-down agrees with world down on
  96-99 % of frames; the camera is upside down on 0.2-2.7 % (flips). On the ground the camera elevation is
  −2..+11°, so "nose down" in the air means speed or a dive, and image parallax tells which. GoPro motion comes
  through Gyroflow's export (`--export-metadata 3`, `org_quat`), also world up −Z; so does a camera that logs
  its gyro to a `.gcsv` beside the clip (RunCam Thumb Pro W: 1000 Hz, its lens profile named in the header,
  Gyroflow's sync within 20 ms).
- **Trimming.** The first 3 s and the last 5 s go; a spike over 1300°/s near the end is a crash and
  everything after it goes; so does any 2 s on the ground (vibration under 1.5 and rotation under 3°/s, and the
  motors off, vibration under 0.3, or the picture standing still), and blown-out or black frames. Vibration alone
  took a smooth glide past people for the ground (motors off 0.07-0.08, a glide 0.6-1.3 with the picture moving
  0.2-3 px; in the air p5 0.79-1.03), and the hop rule threw away up to 20 s around each: 75 s of one flight.
  Crashes: manoeuvres peaked at 900-950°/s in 136 clips of nine jobs (the pilots' rate limit), five mid-flight
  peaks of 1590-1930 were all crashes, and a RunCam Thumb's gyro stops at 1000°/s per axis, so its crashes peaked
  at 1620-1730 and never met the old 2000; between 1000 and 1300 one was a crash, one a fast turn inside a shed.
  Off the ground (a crash counts as on it) for under 20 s between two stretches on it is the quad flipped back
  onto its props (the turtle tumble read as a loop) or set back by hand; after the last stretch on the ground
  in a clip's last 20 s the pilot carries the quad back (a hand over the lens read as a person filling the
  frame). Manoeuvres in the 4 s before a stretch on the ground are the landing (braking, the flop). Sharpness for
  "the lens in the dirt": no, a dim shed was as blurry by the Laplacian.
- **Motion onsets** are peaks of the derivative of the visible (1.5 Hz low-passed) rotation rate; they are
  what lands on beats. At random an onset falls within ±40 ms of a beat ~13 % of the time at 101 BPM.
- **Manoeuvre kinds:** roll, spin or loop (more than 250° in 1 s, by axis), dive (camera more than 45° below
  the horizon), pull-up (camera up more than 50° in 1 s), close pass (parallax above the 98.5th percentile,
  little rotation), descent from height (camera more than 18° down for 1.5 s while everything is far), ascent
  (below).
- **A view opening up** (`--want reveal`, a cue Insta360's AI Director uses): within 2 s the CLIP probability of a
  wide view far away, against things up close (trees, a wall, snow, the ground, people, the sky), rises by 0.45
  to 0.5 or more from 0.35 or less. On a helmet GoPro (Arkhyz) the best were the head lifting from the snow or its
  shadow to the peaks, and the rotation fell after them (46 -> 15, 38 -> 4, 96 -> 16 deg/s) while weak ones,
  already open, kept turning (11 -> 12, 31 -> 29): a carried camera has to slow to under 0.7 of its rotation
  before. From a drone (fpv2022) nearly all the top ones were real (over a wall to a monastery, past a cliff to a
  lake with a church, above the trees to a village) but the drone turned faster into them as often as not, so no
  slowing there.
- **Ascents and arcs** come from the picture, since there is no altitude in the telemetry: the CLIP
  probability of "a view from high above" against ground-level views, the sky and a roof up close rises from
  under 0.3 to over 0.6 within 8 s and stays over 0.65 for 2.5 s with everything far (parallax rank under 0.5).
  On one shoot that found all 7 climbs seen on a sheet and 3 more, nothing else; without the sky and the roof
  among the alternatives, flying close over a roof and a chimney against the sky got in. The camera angle does
  not separate them (−8..+6° on most climbs). The way back to the ground closes an arc; "up 24 s, far 0.13"
  ranks how high it went. ([village](examples/village.md))
- **Motion from the picture** (`vis_motion.py`), for clips with no gyro to read (an owner's Gyroflow export
  without its original, a DJI file whose motion export does not move, a camera without a gyro taken in under a
  `_stabilized` name): corners tracked between consecutive frames of the proxy (Lucas-Kanade,
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
  owner's exports came out 103-108 x 71-76°. Calm orbits round a boat on open water (glare, a moving subject)
  fooled it: two DJI O3 exports read pixel aspects 3.5 and 1.6 (neither square nor SuperView's 1.33, now a
  WARN), two others too little motion; with square pixels the hfov alone had a clear best at 108-112° on all
  four, so `--hfov` alone (job `hfov`) gives the vertical one from square pixels. That search cost 1-3 min a
  clip whatever its length (20-40 % of the step) and still read 97 x 92° on a GoPro looking down at its own skis
  (they move with the camera), while the exact view matters little: it scales the rotation rates, not when the
  camera turns or which way, and Gyroflow's zoom moves it ±15 % anyway. Now the view comes from the camera that
  named the file, after stabilization (GoPro 100°: its exports read 87-90° in a hand, 96-114° on a drone; DJI
  110°), square pixels; a file no camera named gets a quick estimate in the picture (square pixels, 40 pairs,
  5° steps: 11-21 s, within 2° of the full search on clips that turn). Tracking every pair was the rest of the
  cost: pairs of consecutive frames sampled 6 a second (every pair while turning over 60°/s) give the same 1.5 Hz
  rates (correlation 0.997-1.000, 0.5-2.2°/s off) and trust, 3.4-5.2x faster; pairs of frames further apart lose
  inliers to parallax (97 % one frame apart, 67 % four, 52 % eight on a flight). On the ground the picture
  stands still: mean flow under 0.12 px
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
- **Flown or carried, per clip** (`camera.py`, `camera = "auto"`: a trip filmed with a drone and a GoPro in a
  hand). A GoPro is either (fpv2022 on a drone, Arkhyz on a helmet), so not by the file name: by its own
  accelerometer (GPMF ACCL, ~200 a second). A drone's thrust moves the acceleration under 5 Hz by 1.6-5.4 m/s²
  RMS over every flight of over 15 s, a skier's helmet or hand by 0.2-1.1; the threshold is 1.35. Blind on four
  more shoots (Dagestan FPV, two SUP shoots, a birthday: 33 clips) it called every one flown, and the frames were
  all from the air (a prop guard in view on one). The camera's rotation does not tell them apart (a calm flight
  peaked at 93-138 deg/s, a run down a slope at 55-134); vibration over 20 Hz overlapped too. Clips of a few
  seconds before a take-off read carried (1.1-1.2: on the ground or in a hand), which they were. A clip without
  an accelerometer (DJI, stabilized elsewhere) is flown. edit.py `--cameras` makes every carried rule per clip
  (trims, landings, lean, dives, the ground, views opening); montage.py looks for the pilot and close passes in
  the flown clips only. Missed once: a gentle cinewhoop with a soft-mounted GoPro (moose) read carried, 1.06 over
  the clip with 28 s on the grass before the take-off, 0.6-1.8 per 10 s in the air, and no walking cadence
  (1.5-2.5 Hz) either; the job file said `camera = "drone"`.
- **A hand camera beside a drone** (an Avata 2 flown in a gym, an Osmo Pocket 3 and GoPros on the floor and the
  ring posts). Without the accelerometer (GoPro exports stabilized elsewhere, DJI files) there is no verdict, so
  the job file names the carried clips (`carried`, into `cameras.json`). What the drone rules did to the rest: a
  camera standing still is "on the ground", and the GoPros set down kept 10-23 % of their footage (a still hand camera is a shot now); flying
  scenes took the GoPros, whose motion from the picture scored gloves past the lens as manoeuvres (7 of 10
  scenes; `--fly` takes only flown clips when a job has both); a whip of the hand camera to the ceiling scored
  "turn" 0.94. Turning over 60 deg/s told it apart: 46 % of its frames against 0-3 % in close-ups of pads, a bag,
  wraps or sparring (the hand camera's 90th percentile 22 deg/s), and it pays `W_SWING` 10 per share. Neither
  the people boxes (a person at the frame's edge), CLIP ("a blurry floor", "a glove over the lens") nor sharpness
  against the clip's median told the shots "looking past" from the good ones: a GoPro swinging inside the ring
  or picked up by someone is left out of the job instead.
- **No gyro in the file** (an Osmo Action 3 with RockSteady): its DJI track has no fused-attitude quaternions,
  and Gyroflow's export of it is the identity all along. montage.py exports Gyroflow's motion for such a clip;
  an export that does not move sends the clip to `vis_motion.py` like one stabilized before (80 deg wide where
  it calibrated, 77-86).
- **A clock never set** (an Osmo dating its clips 2000-03-30 beside a drone's 2026-02-27): a clip before 2010 has
  no recording time and takes the day of the others when they share one (a day of its own had penned that
  camera's close-ups into three scenes).
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
- **Two aspects in one edit** (a 4:3 drone, 16:9 hand cameras): every clip is scaled to cover the frame and cut
  in its centre, never stretched (`--width` sets the preview's aspect), the look thumbnails compare on their 4:3
  centre, and `final.py` renders each piece at its clip's own aspect covering the final (a 16:9 hand camera at
  5120x2880 for a 3840x2880 final) and cuts the centre, as the preview does.
- **Look-alikes:** 12×9 Lab thumbnails, RMS distance; σ = 0.56 × the median distance between random frames
  (22 on a construction site, 15.5 in a grey hall; a fixed 22 counted 71 pairs alike in the hall, the scaled
  one 9). Thumbnails do not see one subject from different angles; whole-frame CLIP does not either (random
  windows 0.88, the same roof 0.97, unrelated gardens up to 0.96). Activities in words do (Places).
- **Decoding:** 4K HEVC 10-bit decodes at ~36 fps on the CPU in PyAV; NVDEC through PyAV's hwaccel was slower
  (17 fps, the download to host memory), the `hevc_cuvid` decoder (frames in host memory from the start) gave
  38 fps beside two Gyroflow renders; 640 px proxies decode at ~2500 fps. When re-encoding with PyAV, set
  `frame.time_base` with `frame.pts`, or the muxer fails with EINVAL.
- **Frames into arrays** (`pix.py`): PyAV's `frame.to_ndarray(format=..., width=...)` and `frame.reformat(...)`
  build a new swscale context for every frame, 3-7 ms each at 640 px: a 100 s proxy decoded in 1.2 s and converted
  in 7.5 s (gray 10 s), and every script paid it per frame. One kept `VideoReformatter` per kind of conversion
  (and per thread: a shared one crashed under four grading threads) gives the same pixels bit for bit at ~0.3 s.
  With it and the clips analysed `--workers` at a time, the `features` step of 25 min of GoPro footage takes
  7.7 min (`analyze.py` 8x faster, `vis_motion.py` ~125 s per clip, its tracking), every feature file identical.
  Half precision in YOLO saved nothing worth its differences.
- **Model loads:** importing torch and transformers costs 8-9 s per process (the CLIP weights themselves 0.5 s
  from the cache). The CLIP model is loaded only when a script needs it (`places.Lazy`), text embeddings are kept
  in `models/clip/texts.npz` and the aesthetic head beside its `.pth`, so an edit's find:TEXT, `--want` and
  `--theme` do not load the model at all (beauty.py on cached frame embeddings: 0.4 s instead of 14).

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
  elsewhere (most such shots 0.97-1.03x). Hits between beats wait for cuts on half beats (Ideas).
- **Long flights** (`--long`): a shot per 16-beat phrase, for a calm report where every view is beautiful (a
  kremlin from above, "longer flights: everything there is beautiful").
- **Speed:** ramps bend speed between 0.7x and 1.4x through beat anchors so onsets land on beats (sync from
  44 % to 71 % on the first test; read as "breathing" by some, hence off by default). The hero ramp: beat 0
  rushes in at 1.4x, beats 1-2 at 0.5x with the manoeuvre on beat 2, then base speed; at most 2, where the
  music lifts most, never the same kind twice. `--pace`: loud phrases at 1.1-1.2x, which hides hesitation.
  50 fps shown at 25 fps is true slow motion. Slower than 0.95x of real time a picture is the two source frames
  around its time mixed (`SLOW_BLEND`, previews and finals alike): 30 fps at 0.5x into 25 fps had shown every
  other frame twice (qc.py: 19 frozen frames, 1.4 %, all in the two hero ramps; 0 after). At 1x nothing is
  mixed (30 fps into 50 fps repeats frames rather than ghost every one).
- **Editor's pieces** (`--must/--start/--end`) are placed before anything else: a hero once took the footage
  next to a piece and the piece silently became another shot. A piece longer than its shots gets the whole
  phrase; longer than the phrase, it plays faster, up to 1.7x (a 25 s climb in a 15 s phrase). A piece with no
  manoeuvre in it whose one shot runs over a musical hit with no motion onset within 0.1 s of it is cut on the
  strongest such hit: its first part, then the same clip 1-12 s on. A car followed far ahead for 16 beats over a
  hit "did not react to the beats" (the owner, twice: cut by hand on the downbeat it was "круто"); the rule cut
  both test edits there, hits met by a cut 8 -> 9 each. `--skip` holds around a piece (its 3 s opening once let
  a swaying stretch back in); a piece inside a skip still wins. `--start`
  folds the pickup before the first phrase into the opening shot, and takes the first phrase even when a
  flying scene or a hero was planned there (with `--sections` a chorus opened one track: the pickup stayed a 1 s
  scene of another clip, "an off-topic piece for a couple of seconds at the very start"). An `--end` moment
  (`CLIP:SEC`) lands on the track's last hit, with 2 s of run-up in its shot and the rest of the music after it
  (2-6 s; else it ends 2 s before the end): as a piece a crash had ended the edit 2 s after it, in silence; as a
  moment it met the last hit within 12 ms. The plan puts the moment within 0.15 s of the hit and the last shot is
  then slid onto it at constant speed (bending onto hits had moved it 0.3-0.5 s). A track's last phrase can be a
  tail of a few seconds (3.3-5.4 s on three tracks): an `--end` piece that would run over 1.25x there takes the
  scene before it too (unless that is a hero) and plays near 1x, as `edl_edit.py --end-from` does after the fact.
- **Flying scenes** (`--fly N`): one whole phrase around a manoeuvre, people ignored; elsewhere in a people
  edit a shot with nobody in it pays `W_EMPTY`.
- **Chapters** by recording day share the scenes by good flying and, with `--beauty`, beautiful views (a calm
  day over the most beautiful place had 71 good seconds of flying, 285 with the views); `--chapters 3,2,4,2`
  sets them by hand. An editor's moment goes to a scene of its own day: one of 9 Sep placed in a 5 Sep scene
  left that scene with no footage. ([dagestan](examples/dagestan.md)) One day can hold two shoots (flights by day
  and one at sunset 1.6 h later: the evening's 77 good seconds against 475 got 2 scenes of 9): `--sessions H`
  makes a day's clips more than H hours apart chapters of their own (labelled by date and time, `--chapters 5,4`
  sets them).
- **Landings** (the 10 s before a stretch on the ground) are marked, not cut: every clip of one trip ended in a
  slow approach and a flop that read as a roll ("a dull landing"), but "a landing can be an ending", so a
  window pays `W_LANDING` anywhere but in the last scene.
- **Changing a finished edit** (`edl_edit.py`): planning again after a small comment moves other scenes (the
  track is shortened along another path, the scenes per day shared anew), and cutting a scene's phrase out of
  the track moves every later scene off its beats. So the music stays: `--replace K` finds scene K again
  against the rest as it stands; `--end-from K` lets the ending take over from scene K (earlier in its clip,
  at 1x, to the same last frame). "Add Marina to the one I liked" is `--replace K --clip` on the weakest scene of
  the edit the owner saw (rebuilt: same options, deterministic), never a new plan (one with her as a `--must` had
  dropped the opening he liked). ([arkhyz](examples/arkhyz.md))
- **Opening pieces and pickups:** an opening piece a little shorter than the opening phrase fills it at down to
  0.8x (a 12 s queue clip against a 14 s opening had gone to the second scene); a pickup under 1.5 s joins the
  first shot instead of flashing by.
- **Scene cuts that match** (`--joins W`, from match cutting: DIRECT 2026, Netflix's Match Cutting 2023): inside
  a scene the camera keeps turning across a cut; between scenes nothing held them together. A join is the mean of
  four likenesses across the cut: turning on the same way (`continuity`), looking as high (15 deg), as bright (25
  of 255) and of the same colour (Lab a/b of the look thumbnails, 8). A scene is scored against the neighbours
  placed before it (scenes are placed by priority, not in order), so every cut is scored once. edit.py prints the
  matching cuts always. On fpv2022 (late, 7 scene cuts): 3 matched (over 0.6) without it, mean 0.57; `--joins 2`
  4, mean 0.66, piloting and hits as they were; `--joins 4` 5, mean 0.75, but the piloting median fell from 0.86
  to 0.76. So 2.
- **Distance in time:** footage close to another scene's shot pays `W_NEAR` (2, over 20 s); the parts of a
  split recording count as one flight (an edit once opened with two passes over the same warm-up 17 s apart).
- **The order of recording within a day** (`--order W`): a sunset placed between daytime scenes had to be swapped
  out by hand. The K scenes of a day are K slots of the day's usable footage in recording order, and a scene's
  first shot pays W per slot its frame is off its own (a penalty against the scenes already placed had anchored
  everything on the first one placed). On a forest edit of three days (10 scenes): 4 of 9 cuts stepped back
  without it, 2 at W 2, 0 at 4; the price was the piloting median 0.87 -> 0.79 -> 0.78 and look-alike pairs
  3 -> 2 -> 6 (the same hour side by side). edit.py and qc.py count the cuts that step back within a day always.
- **The best of N seeds** (`--seeds N`): one score per plan from the summary numbers, 0.35 the median piloting,
  0.3 the share of musical hits met by a cut or a motion onset, 0.15 the onsets on beats, 0.2 the variety of
  movements (the most common kind's share, runs over 3), 0.03 off per look-alike pair, plus 0.2 the share of
  shots with people when people are weighted and 0.1 the mean join when joins are. Four seeds of one edit scored
  0.79-0.86 (piloting 0.80-0.85, hits met 16-18 of 22, look-alikes 2-4): the differences are real but small, ~10 s
  a seed, the chosen seed goes into the EDL.
- **Themes** (`--theme "TEXT[:N]"`): a scene shows a theme when one of its shots averages 0.4 or more; once
  a theme has its N scenes, a candidate pays `W_THEME` 8, full from 0.55, nothing under 0.35. At 4 on a
  linear ramp a repeated person still came back (a person is worth `W_PEOPLE`). A shot under 2 s does not use
  a theme up. ([village](examples/village.md))
- Fades: 0.6 s in, 2 s out, the same in the preview and the final.
- **Build and release** (`--build`): editors build tension with shorter shots and let it go with a long one. The
  phrase before a drop (the next phrase two energy classes up) is cut 8, 4, 2, 2 beats while a shot stays 1.2 s
  (8, 4, 4 at 120 BPM and above), and the drop opens with a shot of 8 beats or more. Not under `--long`.
- **Shot sizes** (`--sizes W`): editing by shot size: two shots of one size in a row read as a jump ("a wide on a
  wide on a wide", an editor on an FPV forest edit). A frame's size runs 0 extreme wide, 1 wide, 2 medium, 3 close,
  4 detail, the closest of: the view (CLIP: from high above 0, from the ground 1, up close 2.5), a person's full
  height in frame heights (the box, or 3.4 torsos, or 4 shoulder widths when the figure is cut: 0.08 -> 0, 0.3 -> 1,
  0.9 -> 1.5, 2 -> 2, 4 -> 3, 8 -> 4) and a vehicle's or an animal's box area (0.3 % -> 0, 2 % -> 1, 10 % -> 2,
  30 % -> 3). CLIP's words for sizes ("a medium shot", "a close-up") ordered nothing (a person among trees read as
  the widest), parallax neither (Spearman 0.05-0.28 against them on five jobs). A cut pays W x (1 - the step
  between the second before it and the second after it), inside a scene and across its cuts; `--seeds` adds 0.1 x
  the share of cuts that step. The first version counted people only and called the rest "nobody": an FPV edit of
  13 shots, 11 of them nobody, printed 0 cuts of one size. On that job (most footage 1-2 m over snow, size 2): one
  size across 6 of 12 and 7 of 8 cuts without it; W 2 with 8 seeds 3 of 12 and 2 of 8, the piloting median
  0.88 -> 0.84 and 0.86 -> 0.78; W 3-4 lost the closing shot (`--roles 3`). edit.py prints the sizes always.
  Views from above are few in low flying: two side by side pay, so the rule keeps one, and a liked one goes back
  as a `--must`; two pieces of one length fall into the track's long phrases, side by side when those are.
- **Opening and closing shots** (`--roles W`): "an off-topic piece for a couple of seconds at the very start", "the
  ending flew past them". The first shot earns W x how wide its view is (the REVEAL_WIDE texts against the
  close ones on the frame embeddings), the last shot W x the best of people waving, the biggest person receding,
  the wide view and the landing over its last 1.5 s; the edit prints both (0..1).
- **A piece at its own speed** (`CLIP:SEC@0.5`): a jump off a yacht or a pass under someone in true slow motion
  (50 fps shown at 25). The shot that holds the piece plays at that speed and is never bent; a piece too long for
  its phrase at that speed plays as slow as fits (a 7 s piece asked at 0.5x in a 14 s phrase came out at 0.76x).
  One or two per edit: editors warn against ramping everything.
- **Dissolves** (`--dissolve S`): hard cuts are right for FPV, a calm report asks for a 0.5-1 s cross-dissolve.
  The cut into a scene dissolves unless its beat is one of the track's stronger hits (the top half: cuts are
  moved onto hits, so every hit would have left a calm track with no dissolves at all), the scene is a hero or
  flying scene, or either shot is shorter than 2 S; the preview renders a pre-roll and a tail along the shot's
  time map run straight at its end speeds, the final has them inside its 1 s margins. The EDL marks the scenes
  (`dissolve_in`), so the final dissolves where the preview did.
- **Sound bites** (`--bites N`, `sounds = true`; a `hear:TEXT` piece always): the clip's own sound under the music
  for 2.5 s at a laugh, a shout, a splash, a crash, faded 0.25 s at both ends and brought to the music's level,
  the music ducked to 0.3 with 0.3 s ramps. The moments come from CLAP's windows (sounds.py): its softmax over the
  texts saturates at 1.0 on nearly every window, so a moment is ranked by the margin in cosine between its best
  bite text and the best other sound: on 66 min of a ski holiday 13 % of the windows had a margin over 0, 3.5 %
  over 0.05 (the threshold), and the snow under the skis reads as water splashing.

## The footage in words

`captions.py` (job `captions = true`): a local vision-language model (Qwen3-VL 2B, 4 GB of GPU memory) describes
every 3 s of a render as a short clip (12 frames at 4 fps, 448 px wide, 295 tokens) in one sentence. Single frames
every 2 s took 4.7 s each in eager attention and missed what happened between them (a jump off a roof read as
standing before it and swimming after). Clips of 3 s with SDPA attention take 0.5 s each in batches of 6 (25 min
of footage in ~10 min) and describe the action ("a man is jumping into the water from a boat"). A boy's dance it
did not see (it saw a dolphin that was not there), and a question listing the actions to choose from primed it
(the jump in 21 of 55 chunks of a clip with one jump); asked as a careful observer to name only an action it sees
happen, it said so in 9, the jump among them. So a caption is a first answer, found by the words' stems
(`--search`, `--must cap:TEXT`, the timeline's caption column and its FTS5 table), and the frame sheet of the
stretch found is the proof.
`review.py` asks the same model about the start, middle and end of every shot of a finished edit: a sentence on
what it shows, then three questions with one-word answers (the subject visible or hidden, action or nothing,
pleasant or dull); asked for the shot's problems from a menu of seven, the 2B model had copied the whole menu for
17 of 18 shots. Whether it catches a man just standing there or a dull landing (the owner's own complaints on
earlier jobs) is to be seen on such an edit. montage.py's `qc` prints the review's WARN lines.

**Smiles** (`smiles.py`): the same model, asked "Is this person smiling?" about candidate heads only (on whole
640 px frames it had said yes mostly to faces of 10-20 px: a guess). The candidates are people facing the camera
from 6 % of the frame; the model sees a square around the head (nose, eyes, ears; at least the shoulders) scaled
to 336 px. The false yeses were of three kinds, read off a sheet across the sharpness range: smeared or covered
heads (Laplacian variance at 96 px 16-230 against 240-3100 for faces), heads of 32-36 px, and single yeses with
no other within 1.6 s; so heads under 48 px or under 250 are not asked and a yes counts only with another: 28-29
of 32 on the sheet plainly smiling. Few and right on purpose: the owner would rather miss smiles than be sent
wrong ones.

## Speech and voices

`speech.py` (job `speech = "ru"`; a venv of its own, since pyannote.audio 4 wants NumPy 2): pyannote's
speaker-diarization-community-1 says who speaks when in a clip, Whisper large-v3-turbo writes the words of that
speech only, CLAP tells sung from spoken, and the clips' speakers are joined into the job's voices.
- **Whisper only where pyannote hears speech** (turns 0.5 s apart joined, pieces up to 28 s): over music and
  silence it writes subtitle credits and "thanks for watching". On singing it still loops: one word sixty times in
  6 s, one vowel 400 letters long in half a second. Its own test catches that: the text packs over 2.4 by zlib
  (loops 4.8-47; every real phrase 2.0 or less, a refrain with its repeats included) or comes faster than 30
  letters a second (speech 5-15).
- **Word times cost memory**: they keep the encoder's attention of all 32 layers, ~2.9 GB per 30 s piece in fp16.
  Eight pieces at once asked for 30 GB on an 8 GB card (Windows spilled into shared memory, then gave out); one at
  a time peaks at 6.1 GB with pyannote and CLAP loaded. 39 min of sound in 13 clips: 11 min.
- **Sung or spoken** (CLAP: the best of three singing texts against the best of three speaking ones): a
  children's choir at a median margin of +0.16, a host talking with music under her at +0.01. Over 0, 37 of her 62
  phrases were "sung"; over 0.08, 3 of them and 50 of the choir's 60.
- **Clearness**: a phrase's level over the clip's quietest moments put all the phrases of one clip first (its floor
  20 dB under the others'), whatever played under them. Over what is heard within 5 s of the phrase without speech
  (music, the hall), the clearest came from five clips; spoken phrases had a median of 2 dB, `--talk` takes 6 and up.
- **Voices across clips**: a 39 min single take cut into 3 min pieces, against the whole take diarized at once (VBx:
  4 speakers). The pieces had 38 speakers. Joined by average linkage on their WeSpeaker embeddings at a cosine of
  0.45-0.5: 8 voices, 84 % of the talk time in its voice's main reference speaker, 89 % of each reference speaker
  in one voice. Under 0.45 they fell into 3-4 voices (purity 0.6), over 0.6 they split into 14-35 (completeness
  0.8 down to 0.4). Children reciting in turn are one or two voices to pyannote.
- **What `--talk` picks and what an editor picks**: the clearest spoken lines of 1.2-7 s made no sense outside the
  play ("is there or not?"); the lines that told its story were found by reading the transcript (260 phrases for 39
  min, cheap), two of them at -3 and -9 dB with music starting right after. Interesting is the editor's call.
- **A voice as a person in view**: not measurable in a hall where somebody is always in view (people in 98.6 % of
  the half seconds with a voice, 96.9 % without); it needs footage where people come and go.
- **Lines in order**: placed where each scored best, a play's lines came out shuffled; with `--order` the editor's
  pieces of a day now keep their recording order, each between the scenes of those recorded before and after it.
- **The review model and speech**: a static wide shot of a child speaking is "nothing happening" to the stills;
  a shot under a phrase of speech is not flagged for it.

## People and the pilot

- **Detection** (`people.py detect`): YOLOv8s-pose, 2 samples a second; per person the box, 17 keypoints and
  flags: radio pose (wrists together in front of the torso), waving (a wrist above the shoulders), facing the
  camera, a radio in the hands (COCO `remote`/`cell phone` near the wrists). ~1 min on the GPU for 16 min.
- **People value:** a person covering 2 % of the frame or more counts 1, smaller ones less; ×1.3 facing the
  camera, ×1.6 waving; approaching or receding within ±0.5 s adds 0.5; a pass between the legs, under or over
  adds 1.5. `--people 4` weighs it like piloting.
- **A head cut off:** the people value grows with the person's area, so a body standing right at the camera
  with the head out of the frame scored as the best close-up. YOLO-pose puts a head out of the picture at the
  frame's edge with a low confidence; the rule: a person of 6 % of the frame or more, the box at the top edge,
  and the nose and eyes under 0.5 or the eyes within 4 % of the edge. Shots with the head cut off: 93 % of the
  samples; close-ups with the head in view: 0 %; wide shots of a crowd: ~10 %. Low face confidence alone caught
  a third (YOLO also guesses a face at the edge), the head room from the shoulder width half, and it fired on
  bowed heads. The biggest person cut off makes the sample worth -0.5 of a person, and not the main character
  in view.
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
  sorted 142 faces. A seek in 4K HEVC costs a GOP (35 min for 142 faces by seeking), so frames are read on when
  the next sample is near, and the face embeddings are cached.
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
- **The job's own activities** (`places.py activities --text ...`, in the timeline's `activity` column) joined
  with people in view answer "a worker welding, sparks flying" as a list: welding p 0.62 with a worker, the crane
  0.89, in seconds. The top "welding with sparks" were the stage's cold-spark fountains, told apart by the day.
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
- **The general activity list is a compromise:** Kinetics-400 whole (400 actions, "a photo of a person ...") was
  exact on sport (ice hockey 0.89) but named a random class where nothing happens ("disc golfing" on forests from
  above, dance styles on a dance) and passed 0.4 on 22 % of half seconds; 33 hand-made phrases passed on 48 %, a
  tenth of it "a person wearing FPV goggles", on drone shots of most jobs. Both together, a selection of Kinetics in the
  same phrasing, 84 phrases: 33 %, hockey 0.98. Neither the softmax probability nor the plain cosine tells a true
  activity from a false one (a false welding in an empty hall 0.77 / 0.24, a true paddleboard 0.48 / 0.28): false
  ones are answered by background phrases ("the inside of a factory hall": welding 566 s -> 104 s), a shoot's own
  trades by `--only`.
- **The footage's own sound** (`sounds.py`, CLAP `laion/larger_clap_general`; job `sounds = true`; edit.py
  `hear:TEXT` pieces; Insta360 and GoPro listen to it too): 2 s windows every second, a sound in words against
  what else footage hears. A carried camera hears people, a drone its propellers: 66 min on the slopes were wind
  and skis mostly, a shout of joy 110 s, laughter 11 s (embedding 4 min). The laughs were the right moments with a
  weak picture, hence the sound bites (`--bites`).
- **Where it was** (`gps.py`): a place named from the picture is a guess. A turquoise river in a canyon read as
  the Sulak canyon by eye and by CLIP; the GoPro's GPS (GPMF GPS5) put it 30 km south, on the Avar Koysu by the
  Irganay reservoir. Check names against the GPS before captions or a title. ([dagestan](examples/dagestan.md))
- **The owner's picks** (`--picked W`): a clip the owner stabilized themselves and kept without its original was
  chosen out of the shoot ("I didn't stabilize them for nothing"); every window of one pays W (2: half the
  piloting range). At 2 they got 5 to 8 of the 13-19 shots of each edit of one archive. An export of the whole
  flight beside its original (Gyroflow's trim takes the ends off: 296 of 313 s, 197 of 220, 291 of 303) is a pick
  too, and montage.py takes the original, gyro and all; an export under 0.8 of its original is a piece of the
  flight, found by `exports.py`, and does not make the whole flight a pick.
- **Old exports** (`exports.py`): centre thumbnails at 10 Hz, aspect matched, best correlation over every
  offset; pieces that run on in the same original merge. A 35 s export was found at 72.8-107.8 s (0.84
  against 0.13); a whole-clip export is reported as a copy, not a pick.

## The timeline index

`embedia/timeline.py` builds its tables in the index's `embedia.sqlite` after every `features` step: a row per clip
per half second (GPS speed, altitude and climb, rotation, flow, parallax, brightness, people and each main
character's size, aesthetics and scenic views, the likeliest activity, from the job's own `_activities.npz` when
`places.py activities` ran with the job's texts) and a row per person (track, box, name).
The npz files stay the source; the index answers questions across them in one query ("where is she in view, with
a view behind, at speed") instead of a one-off script loading five files onto one time grid.

The index is task-agnostic: a feature is measured once per clip with nothing from the job; what needs the job's
words or screenshots (who, a main character in words, beautiful views, the shoot's activities) is a question to the
stored embeddings and tracks, kept with its input in `questions` and asked again for clips added later, its answer
in the `labels` table and the `t` columns. Searches in words are not kept: they read the embeddings in seconds.
Vehicles (objects.py): YOLO saw one car as a car and a truck at once, so a car chase had "two vehicles" in 10 %
of its samples, every one a single car; boxes of one group overlapping by half or more count once: 3.6 %, the
two oncoming-car stretches. Each analyzer is a `Step` in `embedia/run.py` with a version; `features` records which version made a clip's
features, and a raised version makes them again.

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
- A clip with no motion data under its own name (no DJI or GoPro track, no `.gcsv`) is left out by the scan: a
  bare mp4 beside originals is usually the owner's export of one (`exports.py`). Footage that is such a clip (a
  phone, a camera with its own stabilization) is taken in under a `_stabilized` name: `proxy.py` scales it,
  `vis_motion.py` reads its motion, finals cut it directly.
- The scan names every video it leaves out, a line per reason (`not taken:`). The container's encoder tag tells
  an export from a camera: the owner's six named exports of one folder and Gyroflow's `_stabilized` files all
  say `Lavf` (ffmpeg); DJI originals say `DJI O4`, `DJI O4P`, `DJI DEFAULT ENCODING`; GoPro and RunCam write none.
  No motion data and not `Lavf` is a WARN (a camera without a gyro?), so is a video that is not mp4 (a phone's
  .MOV); an export is listed without one.
- A clip of a fraction of a second (a GoPro's 0.2 s false start) gets no motion export and a render with no
  video stream (752 bytes); every later step failed on it. The render step keeps such a render as
  `<clip>_stabilized.failed` and the scan leaves the clip out.

## Colour

- A 3D `.cube` is expanded once into a 256³ table (trilinear, 2 s) and applied with one lookup per frame
  (~15 ms at 640×480). Frames are decoded as BT.709 limited range explicitly (untagged proxies would be read
  as BT.601); output is tagged BT.709. A circulating "Avata 2 LUT" is a 1D tone curve, not used.
- **10-bit frames into 8-bit RGB:** swscale's `rgb24` straight from a 10-bit frame (p010le or yuv420p10le,
  limited to full range) comes out 1.2-1.6 of 255 darker in every channel than the same frame through `rgb48le`
  and a rounding of one's own (a graded 4K frame against final.py's path: −1.4 mean through `rgb24`, +0.08
  through `rgb48le`). Deep frames are read as `rgb48le` (final.py's Reader does) and quantized by us, and a
  10-bit output is compared in 16 bits, or the check itself reports a darkening that is not there.
- Every official DJI D-Log M → Rec.709 LUT (O4, Avata 2, Mini 4 Pro, Air 3/3S, Flip, Pocket 3, Action 4/5,
  Nano, 360) is the same file. Indoors it looked dim and green; `--grade auto` adds per-shot levels (p0.5 → 6,
  p99.5 → 242) and white balance from near-white pixels, constant within a shot, alike across the edit.
- **Two cameras** (`--match CLIP`, `edl_edit.py --match`): the grade levels each shot, but a RunCam Thumb stayed
  yellow-orange next to a GoPro of the same afternoon (Lab b 34-39 against 20, a 0..+6 against −6; the owner:
  "align the colour"). Each other clip's Lab a, b (mean and spread over 24 frames of its usable footage) are
  moved to the reference clip's, fitted as one affine RGB map per clip, applied after the LUT and before the
  grade (so the grade's levels come after it). Per clip, not per shot: a shot of grass is not a shot of sky.
  After it and the grade: b 23-27, a −2..−7. The owner then: "the colour is right, contrast and tone are not":
  the grade's levels move only the ends, and the Thumb's shadows sat lifted (lightness p5 14-15 against 6, the
  median 40-43 against 35). A luma curve through 17 quantiles matched the histograms, but its slopes of 0.5-2.5,
  times the colour map and the grade's gain, made the Thumb's compression blocks stand out ("everything in steps,
  discrete") and a brown moose near black. So the match starts with a gamma between the clip's black and white
  (where the grade's levels go), fitted to the reference's quantiles 10-90 and taken at half (gamma 1.14-1.26),
  each pixel scaled whole (the same curve on R, G, B reddened the moose): p5 10-12, median 36-38, no more blocks
  than the colour alone. A cheap camera's 8-bit footage takes a gentle curve only.

## Finals

`final.py` renders only the pieces used (Gyroflow trims with margin, `trim_range_only` off), finds each
piece's start by motion correlation (±1-3 frames from the asked millisecond), applies the LUT and grade on
the GPU and encodes with NVENC: about half an hour per minute of a 4:3 4K D-Log edit (10-bit pieces), ~13 min
per minute of a 16:9 one with x264 on the CPU.

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
is not in a still.

## Ideas not done yet

- Stabilization from the picture, for footage without motion data that was never stabilized (phone clips taken
  in under a `_stabilized` name, goggle recordings): `vis_motion.py` reads its rotation already (the shake too,
  which edit.py low-passes);
  smoothing that rotation and turning each frame back would stabilize it. Checkable on the gyro clips.
- An absolute horizon for motion from the picture (sky and ground, or a regression of the camera angle on the
  CLIP embeddings, trained on the gyro clips): the elevation now drifts and is pulled level.
- Tracks without a constant tempo: a neural beat tracker (Beat This!).
- Cuts on half beats, for musical hits between beats (two thirds of the hits of one track).
- Slow motion in finals: frame interpolation, or 100 fps sources.
- Trick types from the gyro axes (flip, roll, power loop, split-S); flips on snares, rolls on fills.
- People's own motion at the 2 Hz samples of `detect`, measured and set aside: limb motion (wrists and ankles
  against the hips between a person's consecutive samples, body heights per second) found a jump off a yacht's
  side (0.94-0.95, the clip's top 3) but not the dancing boy or a jump off the roof: the greedy 2 Hz tracks break
  at exactly those moments (half the rows have no previous sample), and the box's rise mixes with the drone's own
  pitch. It needs the tracker's keypoints (people.py track, every other frame), which only jobs with screenshots
  run today; "where he dances" is asked of the captions first (`captions.py --search`, `cap:`).
- Proximity from the flow that pure camera rotation predicts, instead of an affine residual.
- Gaps as heroes: flying through an opening shows as a 0.2-0.4 s brightness dip with high parallax and a new
  scene after it.
- Per-shot Gyroflow settings: horizon lock off for flips.
- An EDL / OTIO export for finishing by hand.
- Visual beats from people (Davis & Agrawala 2018: motion has beats that can be put on the music's): a limb
  stopping against its own body (wrists and ankles from the hips, in body heights), so the camera's motion drops
  out and no flow is needed. The cost is the tracker with poses (people.py track, ~1.4x real time), unless a job
  tracks people anyway (screenshots of its main characters). On a minute of an FPV drone chasing children on a
  tennis court the strongest "beats" were running steps and bodies cut by the frame's edge: nothing to cut on.
  Worth trying on footage of accented moves towards a steadier camera (a serve, a jump), on tracks the job has.
- Sound bites the job names in words (`bites = ["laughter"]`): CLAP hears the snow under the skis as water
  splashing; a job could say what its sounds are, as `beauty` says what its views are.
- Voices beyond one job: a library of known voices (the owner's, the family's: a folder of clips per person
  compared by WeSpeaker embedding, as a live speaker-ID does), names given by example in the job file
  (`name = "CLIP:SEC"`), a voice in the people score (`--voices W`) once measured where people come and go, a
  phrase's sound running on over the next shot (an L-cut: a long line without a long shot), Whisper's own token
  log-probabilities as clearness.
- Captions by a bigger model when the GPU allows (Qwen3-VL 4B or 8B): the 2B one misses a dance and invents animals.
