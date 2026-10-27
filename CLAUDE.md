# montAIge: notes for Claude

You edit footage to music with the scripts in `scripts/`. The owner of the footage says what it is and what
they want; you run the steps, judge from the numbers the scripts print, render previews, and change the edit
after their comments. `docs/notes.md` explains the methods and the numbers behind them; `docs/examples/` has
the early jobs with the owner's comments (later ones stay local, `out/<job>/example.md`).

## Layout

```
scripts/                    the toolset
docs/notes.md               methods, measurements, ideas not done yet
docs/examples/<job>.md      early jobs: footage, task, edits, comments, what it taught (new ones: out/<job>/)
models/                     inputs that are not footage (not in git): music/, luts/, yolo/, clip/, faces/
out/<job>/                  a job (not in git): job.toml, previews, EDLs, QC, a snapshot of scripts/
out/diagnostics/            proxy_<job>/ (stabilized proxies and their features), .telemetry_cache/
LOCAL.md                    this machine: tool paths, footage folders of the jobs (not in git)
```

## How we work

- Short edits: about a minute ("nobody watches long"). Shots of 8 beats for people, 8 to 16 for calm
  reports (`--long`: whole flights, a shot per phrase); never chop into tiny pieces (at 145 BPM even 4 beats is
  1.7 s).
- A strong hit in the music gets a cut or a manoeuvre: `edit.py` moves cuts onto the hits, picks footage whose
  motion meets them and bends speed onto them; read its `musical hits:` line (on a cut / on a motion onset / on
  nothing).
- Previews first: 640 px, 25 fps is enough to judge. 4K only when asked.
- Scripts perceive, you decide. Checkers write per-frame numbers to `.npz` and print a few summary lines.
  Look at pictures only on a WARN, once when a detector meets new footage (`pilot_check.jpg`,
  `passes_check.jpg`, `subject_check.jpg`), and the shot sheet and cut sheet of each edit (`<edit>_sheet.jpg`,
  `<edit>_cuts.jpg`, made by the `qc` step) before handing previews over: one activity filling an edit from
  different angles passes every number, and a cut is judged by the music and motion around it, not by stills.
- Editorial calls are yours: which track, how much pilot, pacing, which moments. When the owner names a
  moment, or you see one the scores miss, it goes in with `--must` / `--start` / `--end`, not by bending
  weights. A described moment is searched over all footage (`find:TEXT`, `places.py`, passes), never guessed
  from where it might be in a flight.
- The pilot (FPV goggles, or a radio in the hands) may appear, about 15 % of the edit at most. Anyone else
  with busy hands (a pram, a phone, a bag) is not the pilot.
- The main character comes from the task, in words (`people.py subject --text`), or by example
  (`people.py who`; screenshots the owner sends go into the job file's `[who]`: found in the footage, the
  person followed along their track). A few wrong people in the picks are fine; do not tighten thresholds
  until a person is lost. Whoever holds the camera is not in the picture.
- No flights to nowhere: in a people edit, footage with nobody in it belongs only to flying scenes built
  around a manoeuvre. With little footage, a shorter edit is better than filler.
- Read the footage folder first: the owner's old exports are often the best moments (`exports.py`), and clips
  the owner stabilized themselves are their picks (`--picked`).
- An archive of several places: one edit per place (`places.py locations`, the GPS), each with its `clips`.
- Variants differ by idea (track, pacing, opening); spread the best moments across them.
- Split DJI recordings: ask the owner before joining when they are there (`montage.py --join` renames the
  parts); overnight, join.
- Propellers or a lens guard in view after stabilization are framing: Gyroflow's zoom, or `--zoom`. Horizon
  tilt is not a fault.
- Music: the incompetech catalogue (`docs/notes.md`, Music), constant tempo only (`music_check.py`), the
  catalogue BPM as `bpm`. Once the owner has liked a track, the next ones by how they sound
  (`music_like.py LIKED --not DISLIKED`), not by the catalogue's words. Kevin MacLeod, CC BY 4.0: credit when
  publishing.
- `models/` is filled on demand: download what a job needs into `models/<kind>/` (`yolo/yolov8s.pt`,
  `yolo/yolov8s-pose.pt` from Ultralytics; `clip/` and `clap/` fill themselves from Hugging Face; `faces/` SFace from the
  OpenCV Model Zoo; `aesthetic/sa_0_4_vit_b_32_linear.pth` from LAION-AI/aesthetic-predictor; `luts/` DJI's D-Log M to Rec.709 LUT from DJI's site, any camera's; `music/<job>/`).
- When the owner is there: keep steps short, say what is running, take comments in between. Left for the night
  ("do it yourself, I'll watch in the morning"), nothing waits for an answer: the task text is the whole brief,
  every choice is yours (the track, from the catalogue by the footage and by what the owner liked before; the
  length, about a minute; the subject, from the task; `camera = "auto"` when unsure), two or three previews with
  different ideas go to `out/<job>/previews/` as numbered files, and whatever you would have asked goes into
  `out/<job>/example.md` under "Questions", to be answered in the morning. Split DJI recordings are joined then
  (the parts stay, renamed) and said so in the morning.
- A new job in a new session: every request re-reads the whole conversation, and a job done late in a long
  session cost 3.5 times as many tokens as it would on its own. What enters the context is paid for on every
  later step: numbers first (poses, CLIP, `timeline.py`), then one picture to confirm; no progress output or long
  dumps. No `/compact`: a summary loses the exact seconds and the reasons.
- Searching the footage for a moment ("where does he dance", the jumps, a calm ending with them in view) goes to a
  subagent with the clips, the tools and the answer wanted (CLIP:SEC ranges, a line each): its frame sheets stay in
  its context. One edit: 33M tokens, of which the search for one dancing boy ~8.5M in the main context.
- A toolkit, not a fixed pipeline: `montage.py` does what every job needs; locations, activities, exports,
  climbs and themes are called when a job asks for them. A one-off can stay a one-off script in the job.
- The scripts are long (`edit.py` 2900 lines, 40k tokens): to change one, grep for the function and read that
  range, never the whole file; the options are all in this file. Frames become arrays through `pix.py` (one
  conversion context: a new one per frame cost more than decoding), CPU steps take `--workers`
  (`analyze.py`, `vis_motion.py`), CLIP text embeddings are cached (`models/clip/texts.npz`).
- What a job teaches goes into `scripts/` as a general rule (a sharper definition, an option), never as a
  special case for that footage; footage-specific things come in as inputs (texts, moments, themes).

## A job

```bash
export GYROFLOW="/path/to/Gyroflow.exe"       # Gyroflow 1.6.3 CLI, keeps the settings saved in its GUI
PY="/path/to/python"                           # CUDA torch, ultralytics, transformers, av, cv2
$PY scripts/music_check.py models/music/<job>/<track>.mp3 <bpm> [...]   # constant grid? phrase length?
$PY scripts/montage.py out/<job>/job.toml       # scan -> render -> features -> edits -> qc
$PY scripts/montage.py out/<job>/job.toml --only edits --only qc         # after changing the edits
$PY scripts/montage.py out/<job>/job.toml --final <edit>[,<edit>] --size 3840x2880   # on request
```

Every step skips what is done; edits are redone when the job file is newer than their EDL. Split DJI
recordings make the scan stop and list them. The ML steps need CUDA and stop with a message without it.
Long renders go through a small script in `out/<job>/` (it can wait for the hour the owner names); finals
side by side in `out/<job>/4k/`.

```toml
footage = ["/path/to/footage"]                    # folders (with subfolders) or clips
camera = "carried"                                # in a hand or on a helmet, not flown (default "drone");
                                                  # "auto": per clip by its accelerometer (camera.py, GoPro)
lut = "models/luts/DJI O4 Air Unit Series D-Log M to Rec.709 V1.cube"   # log footage only
objects = true                                    # vehicles and animals (objects.py)
sounds = true                                     # the clips' own sound (hear: pieces, --bites 2 on every edit)
captions = true                                   # the footage in words every 3 s (cap: pieces, review in qc)
args = ["--people", "4"]                          # edit.py options of every edit, before each edit's own

[[edit]]
name = "coach"
music = "models/music/tennis/Motivator.mp3"
bpm = 126
target = 40                                       # seconds, or "auto" for an edit about one person
series = true                                     # keeps the footage of earlier series edits out
clips = ["GH01043*", "GH010440"]                  # only these clips (names, wildcards): one edit per place
args = ["--subject", "coach", "--focus",
        "--start", "find:a corridor -> a staircase -> an indoor tennis court"]
```

`montage.py` checks every edit's options with `edit.py`'s parser and prints them, one line per edit, before
anything runs: read that line (a TOML comment once swallowed half an args line).

## edit.py options

- Length and music: `--target` (seconds; the track is shortened by whole phrases), `--bpm`, `--seed N`,
  `--seeds N` (N plans, the best kept by one score: piloting, hits met, onsets on beats, variety, look-alikes; a
  line per seed, ~10 s each), `--long` (a shot per phrase), `--no-hits` (cuts and speed not moved onto the
  musical hits); the hits are the mix's onsets and the bass booms (the sub-bass coming in after a gap: "cuts on
  the boom") together, `--booms` only the booms, `--no-booms` only the onsets.
- People: `--people W` (0 ignores people, 4 counts them as much as piloting), `--animals W` (0 when YOLO sees
  animals in clutter), `--subject TAG` (`a+b+c`: any of several) with `--focus` (an edit about them;
  `--focus-min 0.03`: only where they are clearly in view), `--minor TAG` (another
  edit's main character counts like the pilot), `--peers` (look-alikes that must not stand in), `--avoid
  EDL` (a series from one shoot: keep earlier edits' footage out).
- A camera carried by a person (skiing, cycling, walking; `camera = "carried"` in the job file): `--carried`
  (no pilot, no landings, no lean, dives or pull-ups: the pitch is where the head looks), `--speed W` (GPS
  ground speed from `gps.py track` in the piloting score, e.g. 0.5; a drone with a GoPro too). A trip filmed both
  ways: `camera = "auto"`, `--cameras FILE` (camera.py's verdict per clip).
- Flying: `--fly N` (flying scenes around a manoeuvre on the biggest lifts), `--ramps`, `--heroes N` (the
  hero ramp, never the same manoeuvre kind twice), `--pace` (loud phrases 1.1-1.2x), `--lively` (4-beat shots
  in loud phrases), `--want KIND` (e.g. `ascent`; `reveal`: a view opening up, any camera), `--arcs N` (whole
  climbs and descents).
- Music structure: edit.py prints the track's sections per phrase and its chorus (the section that comes back
  loudest); `--sections` puts the heroes and flying scenes on the chorus, the first one first, and with
  `--lively` the 4-beat shots only there.
- Variety: `--theme "TEXT[:N]"` (an activity in words, at most N scenes of it, default 1).
- Scene cuts that match: `--joins W` (e.g. 2: the camera turning on the same way, looking as high, as bright and
  of the same colour across a scene cut; edit.py prints how many match either way).
- The owner's picks: `--picked W` (clips the owner stabilized themselves: a `_stabilized` export without its
  original, or an export of the whole flight beside it; a shorter export is a piece, found by `exports.py`).
- Beautiful views: `--beauty W` (off by default; `beauty = [...]` in the job file says what is beautiful in
  words, `beauty.py`; `plain = [...]` what fills the footage without being a view, e.g. snow up close): a
  beautiful view also excuses a shot with nobody in it. A landmark is a theme or a moment.
- Days: `--chapters N,N,...` scenes per recording day (by default shared by good flying and beauty);
  `--sessions H`: a day's clips more than H hours apart are chapters of their own (the flights by day and the
  evening's), counted by `--chapters`; `--order W` (e.g. 4): scenes in the order of recording within a day (day,
  evening, sunset); edit.py and qc.py count the scene cuts that step back in time. Landings are marked and kept
  for the last scene.
- Editor's pieces: `--must`, `--start`, `--end` as `CLIP:SEC`, `CLIP:START-END` or `find:TEXT` / `find:A ->
  B -> C`, or `hear:TEXT` (the clips' own sound, job `sounds = true`), or `cap:TEXT` (what the captions say, job
  `captions = true`: "cap:jumping into the water"); a piece longer than its phrase plays faster, up to 1.7x.
  `@0.5` after a piece plays it at that speed (`CLIP:SEC@0.5`: true slow motion from 50 fps; one or two per
  edit, no more). The `--start` piece takes the first phrase; an `--end` piece over 1.25x in a short last
  phrase takes the scene before it too (not a hero). An `--end` moment (`CLIP:SEC`, e.g. a crash) lands on the
  track's last hit, the music after it as the ending.
- Craft (use them on new jobs; old jobs keep their numbers): `--build` (the phrase before a drop cut shorter and
  shorter, a long shot on the drop), `--sizes 2` (shot sizes alternate in a people edit), `--roles 3` (an
  establishing first shot, a closing last shot: waving, receding, a wide view, the landing), `--dissolve 0.8`
  (cross-dissolves into scenes on calm music; never onto a hit or into a hero), `--bites 2` (the footage's own
  sound under the music at a laugh, a shout, a splash; montage.py adds it with `sounds = true`; a `hear:` piece
  always gets its sound). edit.py prints what each did: shot sizes in a row, the opening and closing scores,
  the dissolves, the bites.
- Seeing the footage in words: job `captions = true` runs `captions.py` (a local vision-language model, 3 s
  chunks, ~0.5 s each) in `features` and `review.py` in `qc` (the model's line per shot of an edit, WARN on
  nothing happening / blurred / dull view / the subject cut off: read the line, then `edl_edit.py --replace K`).
  `captions.py --search "words"` and the timeline's `caption` column answer "where does he dance" without
  frame sheets; the model is small (2B), so a stretch found is a first answer, the sheet of that stretch the
  proof.
- Picture: `--lut FILE.cube`, `--grade auto`, `--zoom Z`, `--match CLIP` (two cameras of one shoot: every other
  clip's colour matched to that clip's) (kept in the EDL for `final.py`).
- `--plan-only`, `--debug`, `--height`, `--width` (a preview of another aspect: every clip cropped to it, never
  stretched; a 4:3 drone beside 16:9 hand cameras), `--fps`, `--brief` (no footage table: montage.py prints it for the first
  edit of a run only).

## Other tools

- `people.py track|subject|who|split|passes|summary`: tracks through every other frame, main characters in
  words or by example (a screenshot found in the footage), look-alike groups, close passes; `faces.py` tells two people in the same clothes apart by their faces in the 4K originals.
- `sounds.py embed|search`: the clips' own sound in words (laughter, a shout of joy): a carried camera hears
  people, a drone its propellers.
- `places.py search|route|locations|activities`: places in words, routes through places, clips of one
  location, what happens in the footage (seconds per activity; the ones that fill it become `--theme`s).
- `exports.py FOLDER --originals ... --proxy-dir ...`: the owner's old exports, found in the originals.
- `timeline.py out/<job> --sql "SELECT ..."`: every feature in one SQLite table, a row per clip per 0.5 s (speed,
  altitude, rotation, parallax, people and each main character, beauty, activity, the caption of that moment) and
  a row per person: questions across features in one query instead of a one-off script. montage.py rebuilds it
  after `features`.
- `captions.py RENDER [...]` / `--search "words"`: the footage in words every 3 s (Qwen3-VL 2B in `models/vlm`),
  stretches whose captions say the words; `review.py EDL`: the model's line per shot of an edit.
- `camera.py CLIP|--list FILE`: flown or carried, from a GoPro's accelerometer.
- `gps.py CLIP` (where), `gps.py track` (speed, altitude, climb per GPS point; montage.py runs it on GoPro
  clips): a lift going up, a run going down, how fast.
- `music_like.py LIKED.mp3 [--not DISLIKED.mp3]`: catalogue tracks that sound like the liked one (CLAP's music
  model), downloaded into `models/music/catalogue/`; then `music_check.py`.
- `objects.py --summary`, `qc.py EDL VIDEO`, `edl_sheet.py EDL OUT.jpg [--cuts]`, `final.py`, `dji_join.py`,
  `djimeta.py`.
- `edl_edit.py`: a comment on one scene of a finished edit: `--replace K` (the same music, the rest frame for
  frame), `--end-from K` (the ending takes over), `--put K.J CLIP:SEC` (a moment the owner names, in shot J of
  scene K at 1x), `--match CLIP` (the colour of the cameras matched). The music is never cut: that moves every
  later scene.
- `vis_motion.py`: camera motion from the picture of a clip stabilized elsewhere (a `_stabilized` export
  without its original: montage.py takes it in, scales it down with `proxy.py`, finals cut it directly).

## After a job

Everything general that will be useful again goes into the repository: rules into `scripts/`, measurements and
methods into `docs/notes.md`, new options into this file and the READMEs. Session data does not, unless the owner
says so: the job's write-up (footage, task, edits with their numbers, the owner's comments, what it taught) stays
in `out/<job>/example.md` (not in git), and notes give the numbers without linking a job. Keep the job's
`scripts/` snapshot in `out/<job>/`. Docs and code comments in English; `README.ru.md` is the Russian twin of
`README.md`.
