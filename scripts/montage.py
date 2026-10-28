"""One command per job: footage -> stabilized proxies -> features -> edits -> QC (-> finals on request).

    GPU_PY scripts/montage.py out/JOB/job.toml                 every step; each skips what is done already
    GPU_PY scripts/montage.py out/JOB/job.toml --join          also join split DJI recordings (renames the parts!)
    GPU_PY scripts/montage.py out/JOB/job.toml --only edits    one step: scan, render, features, edits, qc
    GPU_PY scripts/montage.py out/JOB/job.toml --final coach,flyby [--size 3840x2880]

Run from ai_montage/ with a Python that has CUDA torch (GPU_PY): the ML steps run on the GPU only.
Gyroflow comes from the job's `gyroflow` or the GYROFLOW environment variable.

Job file (TOML) out/<job>/job.toml: everything of the job stays in out/<job>/ (local, like the footage
paths in it); the job's name is its folder (or `name = ...`):

    footage = ["/path/to/footage"]           # folders (subfolders too) or single clips
    lut = "models/luts/DJI O4 Air Unit Series D-Log M to Rec.709 V1.cube"   # only for log footage
    renders = 3                              # Gyroflow renders at once (default 3)
    camera = "carried"                       # carried by a person (hand, helmet, chest), not flown: no pilot,
                                             # no close passes, edit.py --carried (default "drone"); "auto": per
                                             # clip by its accelerometer (camera.py), a trip filmed both ways
    people = true                            # poses, pilot, close passes (default true)
    objects = false                          # animals, vehicles (objects.py)
    places = true                            # CLIP frame embeddings for find:TEXT pieces (default true)
    sounds = false                           # CLAP of the clips' sound for hear:TEXT pieces (sounds.py; a carried
                                             # camera hears people, a drone its propellers); edits get --bites 2
    captions = false                         # every 3 s of each render in words by a local vision-language model
                                             # (captions.py: cap:TEXT pieces, the timeline's caption column) and
                                             # the model's review of every shot in `qc` (review.py)
    speech = "ru"                            # what people say and who says it (speech.py: Whisper, pyannote; the
                                             # footage's language, or true to let Whisper find it): say:TEXT
                                             # pieces, the timeline's speech and voice columns, out/<job>/voices/;
                                             # --talk N only in the args of an edit that asks for it. Runs in a
                                             # venv of its own (pyannote wants numpy 2): `speech_py`, SPEECH_PY
                                             # or models/venvs/speech
    beauty = ["mountains", "a canyon"]       # beautiful views in words (beauty.py), for edit.py --beauty W
    plain = ["snow up close"]                # what fills this footage without being a view (beauty.py --plain)
    hfov = 92                                # field of view of clips stabilized before (vis_motion.py; square
                                             # pixels; default: by the camera that named the files, else estimated)
    args = ["--people", "4"]                 # edit.py options of every edit (before the edit's own)

    [who]                                    # main characters by example (people.py who, in `features`):
    marina = ["refs/marina.jpg", "GH010051:12.5:0.3"]   # screenshots in out/<job>/, or CLIP:SECONDS:X
    _others = ["refs/lookalike.jpg"]         # _NAME: somebody else, kept apart

    [[edit]]
    name = "coach"
    music = "models/music/tennis/Motivator.mp3"
    bpm = 126
    target = 40                              # seconds, or "auto" for an edit about one person (--subject)
    series = true                            # keep the footage of the earlier series edits out (--avoid)
    clips = ["GH01043*", "GH010440"]         # only these clips (names without extension, wildcards): one
                                             # edit per place of an archive (places.py locations)
    args = ["--subject", "coach", "--focus",
            "--start", "find:a corridor -> a staircase -> an indoor tennis court"]

Steps:
  scan      the clips: originals only (not renders, not goggle recordings, not the _parts of a joined
            recording), and `_stabilized` clips whose original is not there; not a clip Gyroflow gave no picture of
            (a fraction of a second: its render is kept as <clip>_stabilized.failed). Every video file left out
            is listed with why and what would take it in ("not taken:", WARN where it may be footage: no motion
            data and not an ffmpeg export, or not mp4). Split DJI recordings are
            listed and the job stops: they are joined (dji_join.py)
            only with --join, after the owner of the footage said yes. -> out/<name>/originals.txt
  render    whole-clip Gyroflow renders 640 px wide, several at once (gf_render.py); GoPro and .gcsv motion through
            Gyroflow's export (gf_motion.py). A `_stabilized` clip without its original (stabilized before,
            no motion data left) is only scaled down (proxy.py); its motion comes from the picture
            (vis_motion.py in `features`, `hfov = 92` in the job file: the field of view of its picture)
  features  per render: image motion and looks (analyze.py), GPS speed and altitude of GoPro clips (gps.py
            track), people (people.py detect, pilotclip, passes),
            objects (objects.py), places (places.py embed),
            beautiful views (beauty.py, with `beauty`); then timeline.py: all of it in out/<job>/timeline.sqlite,
            one row per half second, for questions across the features (timeline.py out/<job> --sql ...)
  edits     edit.py per [[edit]], previews 640 px wide, 25 fps; redone when the job file is newer than the EDL.
            "auto" length: 25 s + 0.25 x seconds the person is in view at 2 % of the frame or more + 0.5 x
            seconds at 6 % or more, 25..75 s (the tennis kids: 29..65 s)
  qc        qc.py per edit -> <edit>_qc.txt; edl_sheet.py: the shot sheet <edit>_sheet.jpg and the cut sheet
            <edit>_cuts.jpg (+ _cuts.txt, its summary printed)
Main characters (people.py subject / who / split) are a job's own business: run them before `edits`.
"""
import argparse
import fnmatch
import glob
import json
import os
import subprocess
import sys
import time
import tomllib

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
ROOT = paths.ROOT
sys.path.insert(0, HERE)
STEPS = ('scan', 'render', 'features', 'edits', 'qc')
PROXY_W = 640
MODELS = paths.MODELS


def short_cmd(cmd):
    """The command as a line to read: the script's name, and a run of three or more files of one folder as
    `folder/*.ext (n)` (a step over 16 renders printed 16 paths)."""
    out, grp = [], []

    def flush():
        if len(grp) > 2:
            d, ext = os.path.dirname(grp[0]), os.path.splitext(grp[0])[1]
            out.append(f'{d}/*{ext} ({len(grp)})')
        else:
            out.extend(grp)
        grp.clear()
    for i, c in enumerate(cmd):
        if i < 2:
            out.append(os.path.basename(c))
            continue
        if os.path.splitext(c)[1] and not c.startswith('-') and (not grp or os.path.dirname(c) == os.path.dirname(grp[0])
                                                                 and os.path.splitext(c)[1] == os.path.splitext(grp[0])[1]):
            grp.append(c)
            continue
        flush()
        out.append(c)
    flush()
    return ' '.join(out)


def run(args, log=None, py=None):
    """A step's script with this Python (or py); its output to the console (or a log file); the echo ends with
    the time the step took."""
    cmd = [py or sys.executable] + [str(a) for a in args]
    print('  $ ' + short_cmd(cmd), end='' if log else '\n', flush=True)
    t0 = time.time()
    if log:
        with open(log, 'a', encoding='utf-8') as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run(cmd).returncode
    print(f'  ({time.time() - t0:.0f}s)' + (f'  FAILED, exit code {rc}' + (f': see {log}' if log else '') if rc else ''),
          flush=True)
    return rc


def has_track(path, fourcc):
    from mp4parse import open_mp4
    try:
        f, _, _, tracks = open_mp4(path)
        f.close()
    except Exception:
        return False
    return any(t.formats and t.formats[0] == fourcc for t in tracks)


def gyroflow_motion(path, cache):
    """The clip's motion comes from Gyroflow's export (gf_motion.py): no DJI `djmd` track (GoPro), or one without
    fused-attitude quaternions (Osmo Action 3 keeps only its raw gyro there)."""
    if stabilized_before(path):
        return False
    if os.path.exists(os.path.join(cache, os.path.splitext(os.path.basename(path))[0] + '_gyroflow_camera.json')):
        return True
    if not has_track(path, b'djmd'):
        return True
    import djimeta
    try:
        djimeta.frame_quats(path)
    except SystemExit:
        return True
    return False


def motionless_export(path, cache):
    """Gyroflow's motion export of the clip stands still all along: no gyro in the file (an Osmo Action 3
    stabilized in the camera). Its motion comes from the picture, like a clip stabilized before."""
    gj = os.path.join(cache, os.path.splitext(os.path.basename(path))[0] + '_gyroflow_camera.json')
    if not os.path.exists(gj):
        return False
    with open(gj, encoding='utf-8') as fh:
        q = np.array([r['org_quat'] for r in json.load(fh)], float)
    return len(q) == 0 or np.ptp(q, axis=0).max() < 1e-9


def has_gcsv(path):
    """A Gyroflow IMU log next to the clip (<stem>.gcsv: RunCam Thumb, other cameras without a motion track in
    the file): Gyroflow reads it, lens profile included, for the render and the motion export (gf_motion.py)."""
    return os.path.isfile(os.path.splitext(path)[0] + '.gcsv')


# ------------------------------------------------------------------------------------------- scan
OTHER_VIDEO = ('.mov', '.m4v', '.mts', '.m2ts', '.avi', '.mkv', '.insv', '.360')   # not LRV/LRF: a clip's preview


def encoder(path):
    """The container's encoder tag: `Lavf...` is ffmpeg (Gyroflow, an editor), so an export; a camera writes its
    own (`DJI O4P`) or none (GoPro, RunCam)."""
    try:
        with av.open(path) as c:
            return c.metadata.get('encoder', '')
    except Exception:
        return ''


def not_taken(bare, other, exports, goggles):
    """The video files the scan left out, a line per reason with what would take them in: nothing is dropped
    unseen. WARN where it may be footage (a camera without a gyro, a phone's .MOV)."""
    made = [f for f in bare if encoder(f).startswith('Lavf')]
    cams = [f for f in bare if f not in made]
    for warn, fs, why in (
            (True, cams, 'no motion data and not made by ffmpeg: footage of a camera without a gyro? renamed to '
                         '<name>_stabilized.mp4 it comes in, motion from the picture'),
            (True, other, 'not mp4 (a phone\'s .MOV, a camcorder\'s .MTS): converted to mp4 it can come in'),
            (False, made, 'no motion data, made by ffmpeg (encoder Lavf): an export or a finished edit; '
                          'exports.py finds its pieces in the originals'),
            (False, exports, 'the owner\'s exports beside their originals: picks (--picked, exports.py)'),
            (False, goggles, 'goggle recordings (DJIG*): the goggles\' view, no gyro')):
        if fs:
            names = sorted(os.path.basename(f) for f in fs)
            print(f'  {"WARN " if warn else ""}not taken: {len(fs)} {why}: '
                  + ', '.join(names[:3]) + (f' +{len(names) - 3}' if len(names) > 3 else ''))


def scan(job, out, join, proxy):
    import dji_join
    roots = [p for p in job['footage'] if os.path.isdir(p)]
    todo = [(o, c) for r in roots for o, c in dji_join.chains_under(r)]
    if todo:
        print(f'split recordings: {len(todo)}')
        for o, c in todo:
            print(f'  {os.path.basename(str(o))} <- ' + ' + '.join(p.name for p in c))
        if not join:
            raise SystemExit('ask the owner of the footage, then run again with --join '
                             '(the parts are renamed to _name next to the joined file)')
        for r in roots:
            if run([os.path.join(HERE, 'dji_join.py'), r]):
                raise SystemExit('joining failed')
    files = []
    for p in job['footage']:
        files += [p] if os.path.isfile(p) else [os.path.join(d, n) for d, _, ns in os.walk(p) for n in ns]
    files = [f for f in files if not os.path.basename(f).startswith(('_', '.'))]
    other = [f for f in files if f.lower().endswith(OTHER_VIDEO)]
    files = [f for f in files if f.lower().endswith('.mp4')]
    stems = {os.path.splitext(os.path.basename(f))[0].lower() for f in files}
    clips, elsewhere, exports, goggles, bare = [], [], [], [], []
    for f in files:
        n = os.path.basename(f)
        if n.upper().startswith('DJIG'):                         # goggle recordings carry no gyro
            goggles.append(f)
            continue
        if '_stabilized' in n.lower():
            # an export of an original that is here is the owner's old export (exports.py); without its original
            # it is the footage itself, already stabilized: picture as it is, motion from it (vis_motion.py)
            (exports if n.lower().split('_stabilized')[0] in stems else elsewhere).append(f)
            continue
        clips.append(f)
    keep = []
    for c in sorted(set(clips)):
        if os.path.exists(failed_render(proxy, c)):
            print(f'  skip {os.path.basename(c)}: Gyroflow gave no picture of it before ({failed_render(proxy, c)})')
        elif has_track(c, b'djmd') or has_track(c, b'gpmd') or has_gcsv(c):
            keep.append(c)
        else:
            bare.append(c)
    for c in sorted(set(elsewhere)):
        print(f'  {os.path.basename(c)}: stabilized before, without its original: motion from the picture')
    keep += sorted(set(elsewhere))
    not_taken(bare, other, exports, goggles)
    with open(os.path.join(out, 'originals.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(keep) + '\n')
    print(f'scan: {len(keep)} clips -> {os.path.join(out, "originals.txt")}')
    return keep


def stabilized_before(path):
    """A clip the scan took in as already stabilized (a `_stabilized` export without its original)."""
    return '_stabilized' in os.path.basename(path).lower()


def originals(out):
    with open(os.path.join(out, 'originals.txt'), encoding='utf-8') as fh:
        return [x.strip() for x in fh if x.strip()]


def failed_render(proxy, clip):
    """Where a render Gyroflow made no picture in is kept (a clip of a fraction of a second: a header, no video
    stream), so that the scan leaves its clip out instead of every later step failing on it."""
    return os.path.join(proxy, os.path.splitext(os.path.basename(clip))[0] + '_stabilized.failed')


def has_picture(path):
    try:
        with av.open(path) as c:
            return bool(c.streams.video) and next(c.decode(video=0), None) is not None
    except (av.error.FFmpegError, OSError):
        return False


# ------------------------------------------------------------------------------------------- render
def render(job, out, proxy, cache):
    clips = originals(out)
    gyroflow = job.get('gyroflow') or os.environ.get('GYROFLOW')
    if not gyroflow or not os.path.isfile(gyroflow):
        raise SystemExit('Gyroflow not found: set GYROFLOW or `gyroflow` in the job file')
    os.environ['GYROFLOW'] = gyroflow
    gopro = [c for c in clips if gyroflow_motion(c, cache)]
    if gopro:
        lst = os.path.join(out, 'gopro.txt')
        open(lst, 'w', encoding='utf-8').write('\n'.join(gopro) + '\n')
        run([os.path.join(HERE, 'gf_motion.py'), cache, '--list', lst])
    by_size = {}
    for c in clips:
        stem = os.path.splitext(os.path.basename(c))[0]
        if os.path.exists(os.path.join(proxy, stem + '_stabilized.mp4')):
            continue
        with av.open(c) as ct:
            cc = ct.streams.video[0].codec_context
            h = int(round(PROXY_W * cc.height / cc.width / 2)) * 2
        # Gyroflow renders the originals; a clip stabilized before is only scaled down (proxy.py)
        by_size.setdefault((h, 'proxy' if stabilized_before(c) else 'render'), []).append(c)
    n = int(job.get('renders', 3))
    for (h, tool), cl in by_size.items():
        procs = []
        for k in range(n):
            part = cl[k::n]
            if not part:
                continue
            lst = os.path.join(out, f'{tool}_{h}_{k}.txt')
            open(lst, 'w', encoding='utf-8').write('\n'.join(part) + '\n')
            log = open(os.path.join(out, f'{tool}_{h}_{k}.log'), 'a', encoding='utf-8')
            cmd = [sys.executable, os.path.join(HERE, 'gf_render.py' if tool == 'render' else 'proxy.py'), proxy,
                   '--list', lst, '--width', str(PROXY_W), '--height', str(h)]
            procs.append(subprocess.Popen(cmd + (['--bitrate', '3M'] if tool == 'render' else []),
                                          stdout=log, stderr=subprocess.STDOUT))
        t0 = time.time()
        print(f'{tool}: {len(cl)} clips at {PROXY_W}x{h}, {len(procs)} at once', flush=True)
        for p in procs:
            p.wait()
        print(f'{tool}: done in {time.time() - t0:.0f}s')
    made = {c: os.path.join(proxy, os.path.splitext(os.path.basename(c))[0] + '_stabilized.mp4') for c in clips}
    bad = [c for c, r in made.items() if os.path.exists(r) and not has_picture(r)]
    for c in bad:
        os.replace(made[c], failed_render(proxy, c))
        print(f'  {os.path.basename(c)}: no picture in its render (see the render logs in {out}): left out')
    if bad:
        with open(os.path.join(out, 'originals.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(c for c in clips if c not in bad) + '\n')


# ------------------------------------------------------------------------------------------- features
def features(job, out, proxy, cache):
    renders = [os.path.join(proxy, os.path.splitext(os.path.basename(c))[0] + '_stabilized.mp4')
               for c in originals(out)]
    missing = [r for r in renders if not os.path.exists(r)]
    if missing:
        raise SystemExit(f'{len(missing)} renders missing, e.g. {missing[0]}: run the render step')
    yolo = os.path.join(MODELS, 'yolo')
    log = os.path.join(out, 'features.log')
    if job.get('camera') == 'auto' and not os.path.exists(os.path.join(out, 'cameras.json')):
        run([os.path.join(HERE, 'camera.py'), '--list', os.path.join(out, 'originals.txt'),
             '--out', os.path.join(out, 'cameras.json')])
    run([os.path.join(HERE, 'analyze.py')] + renders, log)
    before = [r for c, r in zip(originals(out), renders) if stabilized_before(c) or motionless_export(c, cache)]
    if before:                                                   # their motion, from the picture
        run([os.path.join(HERE, 'vis_motion.py')] + before + (['--hfov', job['hfov']] if job.get('hfov') else []), log)
    gopro = [c for c in originals(out) if not stabilized_before(c) and has_track(c, b'gpmd')]
    if gopro:                                                    # speed and altitude (gps.py track)
        lst = os.path.join(out, 'gopro.txt')
        open(lst, 'w', encoding='utf-8').write('\n'.join(gopro) + '\n')
        run([os.path.join(HERE, 'gps.py'), 'track', '--list', lst, '--proxy-dir', proxy], log)
    if job.get('people', True):
        run([os.path.join(HERE, 'people.py'), 'detect'] + renders +
            ['--weights', os.path.join(yolo, 'yolov8s-pose.pt'), '--objects', os.path.join(yolo, 'yolov8s.pt')], log)
        kinds = camera_kinds(job, out)
        flown = [r for c, r in zip(originals(out), renders) if kinds(c) == 'drone']
        for c, r in zip(originals(out), renders):                # nobody pilots a carried camera
            b = os.path.splitext(r)[0]
            if kinds(c) == 'carried' and os.path.exists(b + '_people.npz'):
                np.save(b + '_pilot.npy', np.zeros(len(np.load(b + '_people.npz')['frame']), bool))
        if flown:
            run([os.path.join(HERE, 'people.py'), 'pilotclip'] + flown +
                ['--clip-cache', os.path.join(MODELS, 'clip'), '--sheet', os.path.join(out, 'pilot_check.jpg')], log)
            run([os.path.join(HERE, 'people.py'), 'passes'] + flown +
                ['--weights', os.path.join(yolo, 'yolov8s-pose.pt'), '--sheet', os.path.join(out, 'passes_check.jpg')],
                log)
    if job.get('who') and job.get('people', True):               # main characters by example
        who(job, out, renders, log)
    if job.get('objects', False):
        run([os.path.join(HERE, 'objects.py'), '--weights', os.path.join(yolo, 'yolov8s.pt')] + renders, log)
    if job.get('places', True):
        run([os.path.join(HERE, 'places.py'), 'embed'] + renders + ['--clip-cache', os.path.join(MODELS, 'clip')], log)
    if job.get('sounds', False):                                 # the clips' own sound in words (hear:TEXT pieces)
        run([os.path.join(HERE, 'sounds.py'), 'embed', '--list', os.path.join(out, 'originals.txt'),
             '--proxy-dir', proxy], log)
    if job.get('speech'):                                        # what people say, who says it (speech.py)
        own = [os.path.join(MODELS, 'venvs', 'speech', *x) for x in (('Scripts', 'python.exe'), ('bin', 'python'))]
        py = job.get('speech_py') or os.environ.get('SPEECH_PY') or next((x for x in own if os.path.exists(x)), None)
        if not py:
            raise SystemExit('speech = ... needs its venv (pyannote.audio 4, README): models/venvs/speech, '
                             'SPEECH_PY or `speech_py`')
        lang = job['speech'] if isinstance(job['speech'], str) else None
        run([os.path.join(HERE, 'speech.py'), 'run', '--list', os.path.join(out, 'originals.txt'), '--proxy-dir', proxy,
             '--out', out] + (['--language', lang] if lang else []), log, py=py)
        vj = os.path.join(out, 'voices.json')
        if os.path.exists(vj):
            with open(vj, encoding='utf-8') as fh:
                vs = json.load(fh)['voices']
            print(f'  voices: {len(vs)}' + ''.join(f', {k} {v["talk"]:.0f}s in {len(v["clips"])} clips'
                                                   for k, v in list(vs.items())[:6]) +
                  f' (out/{os.path.basename(out)}/voices/, transcript.txt)')
    if job.get('beauty'):                                        # beautiful views in the job's words (edit.py --beauty)
        run([os.path.join(HERE, 'beauty.py')] + renders + [x for t in job['beauty'] for x in ('--text', t)] +
            [x for t in job.get('plain', []) for x in ('--plain', t)] +
            ['--clip-cache', os.path.join(MODELS, 'clip'), '--sheet', os.path.join(out, 'beauty_check.jpg')], log)
    if job.get('captions', False):                               # every 3 s in words (captions.py; cap:TEXT pieces)
        run([os.path.join(HERE, 'captions.py')] + renders, log)
    run([os.path.join(HERE, 'timeline.py'), out, '--proxy-dir', proxy, '--clip-cache', os.path.join(MODELS, 'clip')],
        log)                                                     # one table of it all: out/<job>/timeline.sqlite
    print(f'features: done (log {log}); check {os.path.join(out, "pilot_check.jpg")} and passes_check.jpg once')


def who(job, out, renders, log):
    """people.py who with the job's `[who]` references: name = [images (paths in the job folder: screenshots the
    owner sent) or CLIP:SECONDS:X]; a name starting with _ is somebody else (a look-alike to keep apart). Run
    again only when the references change (out/<job>/who.txt) -> <render>_subject_NAME.npy, who_check.jpg.
    Before it, people.py track: every person through every other frame, so that a screenshot found in the footage
    gives that person's whole track as references (people.py, find_who_tracked)."""
    if run([os.path.join(HERE, 'people.py'), 'track'] + renders +
           ['--weights', os.path.join(MODELS, 'yolo', 'yolov8s-pose.pt')], log):
        raise SystemExit(f'people.py track failed: see {log}')
    refs = ['tracked']
    for name, items in job['who'].items():
        for x in ([items] if isinstance(items, str) else items):
            p = os.path.join(out, x)
            refs.append(f'{name}={p if os.path.exists(p) else x}')
    stamp = os.path.join(out, 'who.txt')
    done = os.path.exists(stamp) and open(stamp, encoding='utf-8').read().split('\n') == refs and all(
        os.path.exists(os.path.splitext(r)[0] + f'_subject_{n}.npy') for r in renders for n in job['who']
        if not n.startswith('_'))
    if done:
        return
    refs = refs[1:]
    rc = run([os.path.join(HERE, 'people.py'), 'who'] + renders + [x for r in refs for x in ('--ref', r)] +
             ['--originals', os.path.join(out, 'originals.txt'), '--ref-dir', os.path.join(out, 'refs'),
              '--clip-cache', os.path.join(MODELS, 'clip'), '--sheet', os.path.join(out, 'who_check.jpg')], log)
    if rc:
        raise SystemExit(f'people.py who failed: see {log}')
    open(stamp, 'w', encoding='utf-8').write('\n'.join(['tracked'] + refs))


# ------------------------------------------------------------------------------------------- edits
def auto_target(proxy, tag):
    """Seconds of an edit about one person, from how much of them there is (2 samples a second)."""
    import people
    mid = close = 0.0
    for r in glob.glob(os.path.join(proxy, '*_stabilized.mp4')):
        b = os.path.splitext(r)[0]
        sf = people.subject_file(b, tag)
        if not os.path.exists(sf):
            continue
        z = np.load(b + '_people.npz')
        s = np.load(sf)
        best = {}
        for f, a in zip(z['frame'][s], z['box'][s, 2] * z['box'][s, 3]):
            best[f] = max(best.get(f, 0.0), a)
        v = np.array(list(best.values()))
        mid += (v > 0.02).sum() / 2
        close += (v > 0.06).sum() / 2
    t = float(np.clip(25 + 0.25 * mid + 0.5 * close, 25, 75))
    print(f'  {tag}: {mid:.0f} s in view at 2 %+, {close:.0f} s at 6 %+ -> {t:.0f} s')
    return t


def carried(job):
    """The camera of the job is carried by a person (hand, helmet, chest), not flown: `camera = "carried"`."""
    cam = job.get('camera', 'drone')
    if cam not in ('drone', 'carried', 'auto'):
        raise SystemExit(f'camera = {cam!r}: "drone" (default), "carried" or "auto" (per clip, camera.py)')
    return cam == 'carried'


def camera_kinds(job, out):
    """clip path -> 'drone' | 'carried': the job's camera, or with "auto" camera.py's verdict per clip (out/
    cameras.json; a clip without one, e.g. DJI or stabilized elsewhere, is flown)."""
    if job.get('camera') != 'auto':
        k = 'carried' if carried(job) else 'drone'
        return lambda c: k
    with open(os.path.join(out, 'cameras.json'), encoding='utf-8') as fh:
        v = json.load(fh)
    return lambda c: v.get(os.path.splitext(os.path.basename(c))[0].lower(), 'drone')


def edit_args(job, e):
    """edit.py options of an edit: the job's `args` (all edits: the people weight, themes of the shoot),
    then the edit's own (a later value wins); --carried for a carried camera, --cameras for "auto"."""
    cam = ['--carried'] if carried(job) else \
        ['--cameras', os.path.join(job['_out'], 'cameras.json')] if job.get('camera') == 'auto' else []
    return cam + [str(a) for a in job.get('args', []) + e.get('args', [])]


def check_edits(job):
    """Every edit's options through edit.py's own parser before anything runs, and one line each to read: a
    TOML comment once swallowed half an args line (--heroes, --pace, --fly) without any error."""
    import edit
    for e in job.get('edit', []):
        args = edit_args(job, e)
        try:
            edit.make_parser().parse_args([e['music'], '-o', 'check.mp4'] + args)
        except SystemExit:
            raise SystemExit(f'{e["name"]}: edit.py does not take these options: {" ".join(args)}')
        clips = ''
        if e.get('clips') and os.path.exists(os.path.join(job['_out'], 'originals.txt')):
            clips = f', {len(edit_clips(job, e))} clips'
        print(f'  {e["name"]}: {e["music"].rsplit("/", 1)[-1]}, {e.get("target", 60)} s{clips}, {" ".join(args)}')


def edit_clips(job, e):
    """The job's originals an edit's `clips` patterns take (names without extension, case ignored); a pattern
    that takes nothing stops the job: a typo would quietly leave a place out."""
    with open(os.path.join(job['_out'], 'originals.txt'), encoding='utf-8') as fh:
        srcs = [x.strip() for x in fh if x.strip()]
    stems = [os.path.splitext(os.path.basename(s))[0].lower() for s in srcs]
    for p in e['clips']:
        if not any(fnmatch.fnmatchcase(st, p.lower()) for st in stems):
            raise SystemExit(f'{e["name"]}: clips pattern {p!r} takes no clip of the job')
    return [s for s, st in zip(srcs, stems) if any(fnmatch.fnmatchcase(st, p.lower()) for p in e['clips'])]


def preview_height(proxy):
    """Previews are PROXY_W wide whatever the footage's aspect: 480 for 4:3, 360 for 16:9."""
    r = next(iter(sorted(glob.glob(os.path.join(proxy, '*_stabilized.mp4')))), None)
    if r is None:
        return 480
    with av.open(r) as c:
        cc = c.streams.video[0].codec_context
        return int(round(PROXY_W * cc.height / cc.width / 2)) * 2


def edits(job, job_path, out, proxy, cache, plan_only=False):
    series = []
    brief = False                                                # the footage table once per run (edit.py --brief)
    for e in job.get('edit', []):
        name = e['name']
        dst = os.path.join(out, name, name + '.mp4')
        edl = os.path.join(out, name, name + '_edl.json')
        args = edit_args(job, e)
        target = e.get('target', 60)
        if target == 'auto':
            if '--subject' not in args:
                raise SystemExit(f'{name}: target "auto" is for an edit about one person (--subject)')
            target = auto_target(proxy, args[args.index('--subject') + 1])
        fresh = os.path.exists(edl) and os.path.getmtime(edl) > os.path.getmtime(job_path) and \
            (plan_only or os.path.exists(dst))
        if not fresh:
            originals = os.path.join(out, 'originals.txt')
            if e.get('clips'):
                os.makedirs(os.path.join(out, name), exist_ok=True)
                originals = os.path.join(out, name, 'originals.txt')
                with open(originals, 'w', encoding='utf-8') as fh:
                    fh.write(''.join(s + '\n' for s in edit_clips(job, e)))
            cmd = [os.path.join(HERE, 'edit.py'), e['music'], '-o', dst, '--originals', originals,
                   '--proxy-dir', proxy, '--cache', cache, '--target', target, '--height', preview_height(proxy),
                   '--fps', 25]
            if e.get('bpm'):
                cmd += ['--bpm', e['bpm']]
            if job.get('lut'):
                cmd += ['--lut', job['lut'], '--grade', 'auto']
            if job.get('sounds'):                                # the clips' own sound under the music at its moments
                cmd += ['--bites', 2]
            if e.get('series'):
                for prev in series:
                    cmd += ['--avoid', prev]
            if plan_only:
                cmd += ['--plan-only']
            if brief:
                cmd += ['--brief']
            if run(cmd + args):
                raise SystemExit(f'{name}: edit.py failed')
            brief = True
        else:
            print(f'  {name}: up to date')
        if e.get('series'):
            series.append(edl)


def qc(job, out):
    for e in job.get('edit', []):
        name = e['name']
        edl, mp4 = (os.path.join(out, name, name + x) for x in ('_edl.json', '.mp4'))
        if os.path.exists(mp4):
            r = subprocess.run([sys.executable, os.path.join(HERE, 'qc.py'), edl, mp4], capture_output=True, text=True)
            open(os.path.join(out, name, name + '_qc.txt'), 'w', encoding='utf-8').write(r.stdout)
            warn = [x for x in r.stdout.splitlines() if x.startswith('WARN')]
            print(f'  {name}: {len(warn)} WARN' + ''.join(f'\n    {w}' for w in warn))
            sheet = [sys.executable, os.path.join(HERE, 'edl_sheet.py'), edl]
            subprocess.run(sheet + [os.path.join(out, name, name + '_sheet.jpg')], capture_output=True)
            r = subprocess.run(sheet + [os.path.join(out, name, name + '_cuts.jpg'), '--cuts'], capture_output=True,
                               text=True)
            open(os.path.join(out, name, name + '_cuts.txt'), 'w', encoding='utf-8').write(r.stdout + r.stderr)
            print(f'    {(r.stdout.strip().splitlines() or ["cut sheet failed, see _cuts.txt"])[-1]}')
            if job.get('captions', False):                       # the model's look at every shot (review.py)
                r = subprocess.run([sys.executable, os.path.join(HERE, 'review.py'), edl], capture_output=True, text=True)
                lines = [x for x in r.stdout.splitlines() if x.startswith(('WARN', 'review:'))]
                print(''.join(f'    {x}\n' for x in lines) or f'    review failed: {r.stderr.strip().splitlines()[-1:]}')


def final(job, out, names, size):
    w, h = (int(v) for v in size.lower().split('x'))
    if job.get('gyroflow'):                                      # final.py renders its pieces through Gyroflow too
        os.environ['GYROFLOW'] = job['gyroflow']
    for name in names:
        edl = os.path.join(out, name, name + '_edl.json')
        run([os.path.join(HERE, 'final.py'), edl, os.path.join(out, name, f'{name}_{w}x{h}.mp4'),
             '--width', w, '--height', h, '--fps', 50, '--bitrate', '120M', '--crf', 18])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('job')
    ap.add_argument('--only', choices=STEPS, action='append', default=None)
    ap.add_argument('--join', action='store_true', help='join split DJI recordings (asked the owner first)')
    ap.add_argument('--plan-only', action='store_true', help='edits: EDL and plan picture, no preview')
    ap.add_argument('--final', default=None, help='comma-separated edit names to render in full quality')
    ap.add_argument('--size', default='3840x2880', help='final size, e.g. 3840x2880 (4:3) or 1920x1080')
    a = ap.parse_args()
    import gpu
    gpu.device()
    job_path = os.path.abspath(a.job)
    with open(job_path, 'rb') as fh:
        job = tomllib.load(fh)
    os.chdir(ROOT)
    name = job.get('name') or os.path.basename(os.path.dirname(job_path))
    out = os.path.join('out', name)
    job['_out'] = out
    proxy = os.path.join('out', 'diagnostics', 'proxy_' + name)
    cache = os.path.join('out', 'diagnostics', '.telemetry_cache')
    os.makedirs(out, exist_ok=True)
    os.makedirs(proxy, exist_ok=True)
    if a.final:
        final(job, out, a.final.split(','), a.size)
        return
    steps = a.only or STEPS
    t0 = time.time()
    check_edits(job)
    if 'scan' in steps:
        scan(job, out, a.join, proxy)
    if 'render' in steps:
        render(job, out, proxy, cache)
    if 'features' in steps:
        features(job, out, proxy, cache)
    if 'edits' in steps:
        edits(job, job_path, out, proxy, cache, a.plan_only)
    if 'qc' in steps and not a.plan_only:
        qc(job, out)
    print(f'montage: {", ".join(steps)} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
