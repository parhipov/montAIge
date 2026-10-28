# embedia

Footage in, an index out. Every clip of a shoot is measured once on your GPU: camera motion, people with their
poses and tracks, objects, CLIP embeddings of the frames, CLAP embeddings of the sound, captions by a local
vision-language model, speech with its words and voices. All of it stays in one folder with a SQLite database,
and is asked afterwards in words or by example, in seconds: "where is she in view while we go fast", "a jump
into the water", "the person on this screenshot", "where somebody laughs".

It knows nothing of editing. [montAIge](../README.md) cuts films to music from it; any other project can open
the same folder.

## Quick start

```bash
pip install -e /path/to/montAIge                    # or run from the repository: python -m embedia ...
export GYROFLOW=/path/to/Gyroflow.exe               # GoPro motion; stabilized proxies (stabilize = true)
python -m embedia init trip/index --footage /path/to/footage [--camera carried] [--set captions=true]
python -m embedia run trip/index                    # scan, proxies, features, timeline; skips what is done
python -m embedia status trip/index
python -m embedia search trip/index "a jump into the water"           # --by image | sound | caption | speech
python -m embedia sql trip/index "SELECT clip, t, speed FROM t WHERE people > 2 AND speed > 20"
python -m embedia who trip/index marina=refs/marina.jpg kid=GH010051:12.5:0.3
```

```python
import embedia
ix = embedia.open('trip/index')
for src in ix.originals():
    people = ix.get(src, 'people')           # np.load of the feature, None when the clip has none
ix.sql('SELECT name, clip, t FROM labels WHERE area > 0.05')
ix.search('welding with sparks', by='image')  # [{clip, t0, t1, score}]
```

Requirements: an NVIDIA GPU, Python 3.11+, PyTorch with CUDA, ultralytics, transformers, accelerate, PyAV,
OpenCV, NumPy, SciPy, Pillow; Gyroflow 1.6.3; speech in a venv of its own (pyannote.audio 4 wants NumPy 2; see
the main README). Model weights go to `models/` beside `embedia/` (found by walking up from the package).

## What is indexed

`embedia.toml` says which footage and which analyzers (`run.py` has every key). What the footage is (FPV drone
footage or not: `fpv`) is said there by whoever indexes it, never guessed. Each step runs on the clips it
is for that lack its features, or have them from an older version of the step.

| step | what | per clip | on by |
|---|---|---|---|
| analyze | image motion (flow, parallax, approach), brightness, sharpness; Lab look thumbnails | `flow`, `look` | always |
| vis_motion | camera rotation from the picture, for clips with no gyro to read | `vis` | clips stabilized before |
| gps | GoPro GPS: position, speed, altitude, climb | `gps` | GoPro clips |
| people | poses (YOLOv8-pose) 2 a second, clothes colours, waving, facing | `people` | `people = true` |
| pclip | CLIP of every person row: a tight crop and one with what is beside them (who, subject need no picture) | `pclip`, `pclipw` | `people = true` |
| pilot, passes | the FPV pilot (goggles, a radio); flying close past, between legs, under, over | `pilot`, `passes` | `fpv = true`, flown clips |
| track | every person through every other frame (BoT-SORT) | `ptrack`, `tid` | `track = true`, `who` |
| objects | vehicles and animals | `objects` | `objects = true` |
| places | CLIP ViT-B/32 of the whole frame every 0.5 s | `clip` | `places = true` |
| sounds | CLAP of the clip's sound | `clap` | `sounds = true` |
| speech | phrases with their words (Whisper), speakers (pyannote), voices across clips | `speech` | `speech = "ru"` |
| captions | the picture in words every 3 s (Qwen3-VL 2B) | `captions` | `captions = true` |

A feature is a file `proxy/<stem>_stabilized_<feature>.npz` (`.npy`, `.json`), read with `ix.get(clip, feature)`.
Times are seconds of the original. The steps read a 640 px proxy, frame for frame with the original (4K HEVC
10-bit decodes at ~36 fps on the CPU, a 640 px proxy at ~2500: one decode of the original serves every step). What it is, is the
clip's frame, kept in its recipe (`ix.recipe(clip)`: `<stem>_stabilized_recipe.json`, with the size, fps, frame
count and how Gyroflow was called) and in `clips.frame`:

| frame | the proxy | boxes are shares of | when |
|---|---|---|---|
| `stabilized` | Gyroflow's render of a clip with motion data, kept | the stabilized frame | `stabilize = true` (default with `fpv`; the montage sets it) |
| `as-is` | the original scaled down, made while a step reads the picture and removed after the run | the original's frame | otherwise, and clips stabilized before |

Everything that answers works without the proxies: SQL, `search`, the questions. A stabilized render is never
removed for a scaled one; when it is gone and `stabilize` is off, the clip is measured again as it is. A clip
whose frame changes is measured again by the steps that read the picture, and its questions asked again.

## Questions

What needs words or examples from a task is asked of the stored features, not measured again, and kept with its
input in the database: asked again by itself for clips added later; asked with another input, its answers are
replaced.

| question | input | answer |
|---|---|---|
| `who` | names with screenshots or `CLIP:SECONDS:X` (`_name`: somebody else, kept apart) | `subject_<name>` per person row; `labels` |
| `subject NAME` | the person in words: "a woman pushing a pram" | `subject_<name>`; `labels` |
| `beauty` | beautiful views in words, and what fills the footage without being one | `beauty`: aesthetic and scenic every 0.5 s |
| `activities` | the shoot's own activities, added to a general list | `activities`: the likeliest every 0.5 s |

Searches in words are not kept: `ix.search` reads the embeddings and captions in seconds.

## The database

`embedia.sqlite`. `python -m embedia schema INDEX` prints the tables of an index with their columns and row counts.
Times are seconds of the original; areas, boxes are shares of the proxy's frame; NULL where a feature is missing.
Rebuilt from the features after every run, except `features`, `questions`, `meta`.

| table | columns |
|---|---|
| `t` | a row per clip per 0.5 s: `clip`, `t`, `frame`; GPS `speed` (km/h), `alt` (m), `climb` (m/s); `rot` (camera rotation, deg/s), `flow` (px per frame pair at 160 px), `parallax` (what a rotation cannot explain: close and fast), `bright` (0-255), `sharp` (Laplacian variance); `people` (count), `people_area` (the biggest), `facing`, `waving`; `vehicles`, `vehicle_area`, `animals`; a column per main character (their area, 0 out of view); `aes` (LAION aesthetic), `scenic` (the `beauty` question's views); `activity`, `activity_p`; `caption`; `voice`, `speech` (the phrase's words), `sung` |
| `clips` | `clip`, `path` (the original), `render` (the proxy, NULL when not kept), `frame` (`stabilized`, `as-is`), `start` (ISO), `day`, `seconds`, `fps`, `gps` (share with a fix), `lat`, `lon`, `alt_min`, `alt_max`, `speed_p50` |
| `people` | a person seen (2 a second): `clip`, `t`, `frame`, `track` (-1 without `track`), `x`, `y`, `w`, `h` (centre, size), `area`, `facing`, `waving`, `name` (a main character or NULL) |
| `labels` | a main character's rows (`who`, `subject`): `name`, `clip`, `t`, `frame`, `track`, `area` |
| `passes` | close passes (`fpv`): `clip`, `t`, `kind` (between legs, under, over, past on its left / right, close; lying), `size`, `name` (who was passed) |
| `captions` | `clip`, `t0`, `t1`, `text`; `captions_fts` searches it: `WHERE captions_fts MATCH 'jump*'` |
| `phrases` | speech: `clip`, `t0`, `t1`, `voice`, `sung`, `snr`, `level`, `text`; `phrases_fts` as above |
| `features` | `clip`, `step`, `version`, `made`, `embedia`, `frame`: which step version and embedia version made a clip's features, on which frame |
| `questions` | `kind`, `name`, `input` (JSON), `made`, `embedia` |
| `meta` | `key`, `value`: `embedia_version`, `embedia_code` (git commit, +dirty), `created`, `built` |

## Adding an analyzer

A module that takes proxies (or originals) on its command line and writes `<proxy>_<feature>.npz` beside each,
skipping what exists; then a `Step` line in `run.py`: its name, version, outputs, which clips, the `embedia.toml`
key that turns it on. Raise the version when its output changes: the old features are made again. Before tuning
a new detector, run it on a clip with the thing and one without, and compare the two numbers.
