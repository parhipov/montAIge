# montAIge: notes for Claude

You edit footage to music with the scripts in `scripts/`. The owner of the footage says what it is and what
they want; you run the steps, judge from the numbers the scripts print, render previews, and change the edit
after their comments. `docs/notes.md` explains the methods and the numbers behind them; `docs/examples/` has
every job done so far, with the owner's comments.

## Layout

```
scripts/                    the toolset
docs/notes.md               methods, measurements, ideas not done yet
docs/examples/<job>.md      one file per job: footage, task, edits, comments, what it taught
models/                     inputs that are not footage (not in git): music/, luts/, yolo/, clip/, faces/
out/<job>/                  a job (not in git): job.toml, previews, EDLs, QC, a snapshot of scripts/
out/diagnostics/            proxy_<job>/ (stabilized proxies and their features), .telemetry_cache/
LOCAL.md                    this machine: tool paths, footage folders of the jobs (not in git)
```

## How we work

- Short edits: about a minute ("nobody watches long"). Shots of 8 beats for people, 8 to 16 for calm
  reports; never chop into tiny pieces (at 145 BPM even 4 beats is 1.7 s).
- Previews first: 640 px, 25 fps is enough to judge. 4K only when asked.
- Scripts perceive, you decide. Checkers write per-frame numbers to `.npz` and print a few summary lines.
  Look at pictures only on a WARN, once when a detector meets new footage (`pilot_check.jpg`,
  `passes_check.jpg`, `subject_check.jpg`), and the shot sheet of each edit (`edl_sheet.py`) before handing
  previews over: one activity filling an edit from different angles passes every number.
- Editorial calls are yours: which track, how much pilot, pacing, which moments. When the owner names a
  moment, or you see one the scores miss, it goes in with `--must` / `--start` / `--end`, not by bending
  weights. A described moment is searched over all footage (`find:TEXT`, `places.py`, passes), never guessed
  from where it might be in a flight.
- The pilot (FPV goggles, or a radio in the hands) may appear, about 15 % of the edit at most. Anyone else
  with busy hands (a pram, a phone, a bag) is not the pilot.
- The main character comes from the task, in words (`people.py subject --text`), or by example
  (`people.py who`). A few wrong people in the picks are fine; do not tighten thresholds until a person is lost.
- No flights to nowhere: in a people edit, footage with nobody in it belongs only to flying scenes built
  around a manoeuvre. With little footage, a shorter edit is better than filler.
- Read the footage folder first: the owner's old exports are often the best moments (`exports.py`).
- Variants differ by idea (track, pacing, opening); spread the best moments across them.
- Split DJI recordings: ask the owner before joining (`montage.py --join` renames the parts).
- Propellers or a lens guard in view after stabilization are framing: Gyroflow's zoom, or `--zoom`. Horizon
  tilt is not a fault.
- Music: the incompetech catalogue (`docs/notes.md`, Music), constant tempo only (`music_check.py`), the
  catalogue BPM as `bpm`. Kevin MacLeod, CC BY 4.0: credit when publishing.
- `models/` is filled on demand: download what a job needs into `models/<kind>/` (`yolo/yolov8s.pt`,
  `yolo/yolov8s-pose.pt` from Ultralytics; `clip/` fills itself from Hugging Face; `faces/` SFace from the
  OpenCV Model Zoo; `luts/` DJI's D-Log M to Rec.709 LUT from DJI's site, any camera's; `music/<job>/`).
- The owner watches as it goes: keep steps short, say what is running, take comments in between.
- A new job in a new session: every request re-reads the whole conversation, and a job done late in a long
  session cost 3.5 times as many tokens as it would on its own.
- A toolkit, not a fixed pipeline: `montage.py` does what every job needs; locations, activities, exports,
  climbs and themes are called when a job asks for them. A one-off can stay a one-off script in the job.
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
lut = "models/luts/DJI O4 Air Unit Series D-Log M to Rec.709 V1.cube"   # log footage only
objects = true                                    # vehicles and animals (objects.py)
args = ["--people", "4"]                          # edit.py options of every edit, before each edit's own

[[edit]]
name = "coach"
music = "models/music/tennis/Motivator.mp3"
bpm = 126
target = 40                                       # seconds, or "auto" for an edit about one person
series = true                                     # keeps the footage of earlier series edits out
args = ["--subject", "coach", "--focus",
        "--start", "find:a corridor -> a staircase -> an indoor tennis court"]
```

`montage.py` checks every edit's options with `edit.py`'s parser and prints them, one line per edit, before
anything runs: read that line (a TOML comment once swallowed half an args line).

## edit.py options

- Length and music: `--target` (seconds; the track is shortened by whole phrases), `--bpm`, `--seed N`.
- People: `--people W` (0 ignores people, 4 counts them as much as piloting), `--animals W` (0 when YOLO sees
  animals in clutter), `--subject TAG` with `--focus` (an edit about one person), `--minor TAG` (another
  edit's main character counts like the pilot), `--peers` (look-alikes that must not stand in), `--avoid
  EDL` (a series from one shoot: keep earlier edits' footage out).
- Flying: `--fly N` (flying scenes around a manoeuvre on the biggest lifts), `--ramps`, `--heroes N` (the
  hero ramp, never the same manoeuvre kind twice), `--pace` (loud phrases 1.1-1.2x), `--lively` (4-beat shots
  in loud phrases), `--want KIND` (e.g. `ascent`), `--arcs N` (whole climbs and descents).
- Variety: `--theme "TEXT[:N]"` (an activity in words, at most N scenes of it, default 1).
- Editor's pieces: `--must`, `--start`, `--end` as `CLIP:SEC`, `CLIP:START-END` or `find:TEXT` / `find:A ->
  B -> C`; a piece longer than its phrase plays faster, up to 1.7x.
- Picture: `--lut FILE.cube`, `--grade auto`, `--zoom Z` (kept in the EDL for `final.py`).
- `--plan-only`, `--debug`, `--height`, `--fps`.

## Other tools

- `people.py subject|who|split|passes|summary`: main characters in words or by example, look-alike groups,
  close passes; `faces.py` tells two people in the same clothes apart by their faces in the 4K originals.
- `places.py search|route|locations|activities`: places in words, routes through places, clips of one
  location, what happens in the footage (seconds per activity; the ones that fill it become `--theme`s).
- `exports.py FOLDER --originals ... --proxy-dir ...`: the owner's old exports, found in the originals.
- `objects.py --summary`, `qc.py EDL VIDEO`, `edl_sheet.py EDL OUT.jpg`, `final.py`, `dji_join.py`,
  `djimeta.py`.

## After a job

Add `docs/examples/<job>.md` (footage, task, edits with their numbers, the owner's comments, what it
taught), put general lessons into `docs/notes.md` and `scripts/`, keep the job's `scripts/` snapshot in
`out/<job>/`. Docs and code comments in English; `README.ru.md` is the Russian twin of `README.md`.
