"""Footage in, an index out: scan -> proxies -> features -> questions asked again -> timeline. Every step skips
what is done.

    python -m embedia run INDEX                  every step (INDEX: a folder with embedia.toml)
    python -m embedia analyze INDEX [--only STEP] [--redo STEP]

embedia.toml (written by hand, by `python -m embedia init`, or by the montage from its job file):

    footage = ["/path/to/footage"]           # folders (subfolders too) or single clips
    camera = "drone"                         # "carried": in a hand, on a helmet, not flown (no pilot, no passes);
                                             # "auto": per clip by its accelerometer (camera.py, GoPro)
    carried = ["GX*", "DJI_2000*"]           # with "auto": clip names (wildcards) carried, said (a hand camera with
                                             # no accelerometer, a GoPro stabilized elsewhere); the rest by camera.py
    gyroflow = "/path/to/Gyroflow.exe"       # or the GYROFLOW environment variable: proxies of clips with motion data
    stabilize = false                        # true: clips with motion data are measured on Gyroflow's stabilized
                                             # render, kept as their proxy; false: on the original as it is, scaled
                                             # down while a step reads it, removed after the run (default: fpv's
                                             # value; the montage sets true: it cuts previews from the proxies)
    renders = 3                              # Gyroflow renders at once
    workers = 4                              # CPU steps' processes at once (analyze, vis_motion; default half the CPUs)
    hfov = 92                                # field of view of clips stabilized before (vis_motion.py)
    people = true                            # poses, waving, facing the camera (default true)
    fpv = true                               # FPV drone footage: who the pilot is (goggles, a radio), close passes
                                             # past people (default false: the session says what the footage is)
    track = false                            # every person through every other frame (people.py track; who asks it)
    objects = false                          # animals, vehicles (objects.py)
    places = true                            # CLIP of the frames every 0.5 s (search in words, activities, beauty)
    sounds = false                           # CLAP of the clips' sound (search sounds in words)
    captions = false                         # the footage in words every 3 s (captions.py, a local VLM)
    speech = "ru"                            # what people say, who says it (speech.py; true: Whisper finds the
                                             # language); runs in its own venv: speech_py, SPEECH_PY or
                                             # models/venvs/speech

Steps:
  scan      the clips: originals only (not renders, not goggle recordings, not the _parts of a joined recording),
            and `_stabilized` clips whose original is not there; not a clip Gyroflow gave no picture of (a fraction of
            a second: its render is kept as <clip>_stabilized.failed). Every video file left out is listed with why
            and what would take it in ("not taken:", WARN where it may be footage: no motion data and not an ffmpeg
            export, or not mp4). Split DJI recordings are listed and the scan stops: they are joined (dji_join.py)
            only with join=True, after the owner of the footage said yes. -> originals.txt
  proxies   640 px wide, several at once: with `stabilize`, whole-clip Gyroflow renders (gf_render.py) of the clips
            with motion data, kept; otherwise the original scaled down (proxy.py), made when a step reads the
            picture and removed after the run. A `_stabilized` clip without its original (stabilized before, no
            motion data left) is always scaled; its motion comes from the picture (vis_motion.py). GoPro and .gcsv
            motion through Gyroflow's export (gf_motion.py) either way. Beside each clip's features its recipe
            (<stem>_stabilized_recipe.json): the frame they are measured on ("stabilized" | "as-is"), the
            proxy's size, fps, frame count, how Gyroflow was called
  features  the analyzers of STEPS, each on the clips it is for and that lack its features, have them from an
            older version of the step (`features` in embedia.sqlite), or measured on another frame (a picture
            step: the clip is now stabilized, or its stabilized render is gone); the questions' answers of a clip
            measured again go, to be asked again
  questions the questions asked before (who, subject, beauty, activities) asked again when a clip lacks its answer
  timeline  timeline.py: every feature on one grid in embedia.sqlite, for questions across them
"""
import json
import os
import subprocess
import sys
import time

import av
import numpy as np

from embedia import paths
from embedia.read import export_of, stabilized_before

ROOT = paths.ROOT
MODELS = paths.MODELS
PROXY_W = 640
OTHER_VIDEO = ('.mov', '.m4v', '.mts', '.m2ts', '.avi', '.mkv', '.insv', '.360')   # not LRV/LRF: a clip's preview


# ------------------------------------------------------------------------------------------- running a module
def short_cmd(cmd):
    """The command as a line to read: the module's name, and a run of three or more files of one folder as
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
        if i < 3:
            if c != '-m':
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


def command(module, args, py=None):
    """`python -m embedia.<module> args`, from the repository (another Python, e.g. the speech venv, too)."""
    return [py or sys.executable, '-m', f'embedia.{module}'] + [str(a) for a in args]


def env():
    e = dict(os.environ)
    e['PYTHONPATH'] = ROOT + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    return e


def run(module, args, log=None, py=None):
    """A step's module with this Python (or py); its output to the console (or a log file); the echo ends with the
    time the step took. The exit code."""
    cmd = command(module, args, py)
    print('  $ ' + short_cmd(cmd), end='' if log else '\n', flush=True)
    t0 = time.time()
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, 'a', encoding='utf-8') as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env()).returncode
    else:
        rc = subprocess.run(cmd, cwd=ROOT, env=env()).returncode
    print(f'  ({time.time() - t0:.0f}s)' + (f'  FAILED, exit code {rc}' + (f': see {log}' if log else '') if rc else ''),
          flush=True)
    return rc


# ------------------------------------------------------------------------------------------- what a clip is
def has_track(path, fourcc):
    from embedia.mp4parse import open_mp4
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
    from embedia import djimeta
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


def camera_kinds(ix):
    """clip path -> 'drone' | 'carried': the index's camera, or with "auto" camera.py's verdict per clip
    (cameras.json; a clip without one, e.g. DJI or stabilized elsewhere, is flown)."""
    cam = ix.cfg.get('camera', 'drone')
    if cam not in ('drone', 'carried', 'auto'):
        raise SystemExit(f'camera = {cam!r}: "drone" (default), "carried" or "auto" (per clip, camera.py)')
    if cam != 'auto':
        return lambda c: cam
    with open(os.path.join(ix.root, 'cameras.json'), encoding='utf-8') as fh:
        v = json.load(fh)
    return lambda c: v.get(os.path.splitext(os.path.basename(c))[0].lower(), 'drone')


# ------------------------------------------------------------------------------------------- scan
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


def failed_render(ix, clip):
    """Where a render Gyroflow made no picture in is kept (a clip of a fraction of a second: a header, no video
    stream), so that the scan leaves its clip out instead of every later step failing on it."""
    return os.path.splitext(ix.render(clip))[0] + '.failed'


def has_picture(path):
    try:
        with av.open(path) as c:
            return bool(c.streams.video) and next(c.decode(video=0), None) is not None
    except (av.error.FFmpegError, OSError):
        return False


def scan(ix, join=False):
    from embedia import dji_join
    footage = ix.cfg.get('footage') or []
    if not footage:
        raise SystemExit(f'{ix.root}: no footage in embedia.toml')
    roots = [p for p in footage if os.path.isdir(p)]
    todo = [(o, c) for r in roots for o, c in dji_join.chains_under(r)]
    if todo:
        print(f'split recordings: {len(todo)}')
        for o, c in todo:
            print(f'  {os.path.basename(str(o))} <- ' + ' + '.join(p.name for p in c))
        if not join:
            raise SystemExit('ask the owner of the footage, then run again with --join '
                             '(the parts are renamed to _name next to the joined file)')
        for r in roots:
            if run('dji_join', [r]):
                raise SystemExit('joining failed')
    files = []
    for p in footage:
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
            (exports if export_of(n, stems) else elsewhere).append(f)
            continue
        clips.append(f)
    keep = []
    for c in sorted(set(clips)):
        if os.path.exists(failed_render(ix, c)):
            print(f'  skip {os.path.basename(c)}: Gyroflow gave no picture of it before ({failed_render(ix, c)})')
        elif has_track(c, b'djmd') or has_track(c, b'gpmd') or has_gcsv(c):
            keep.append(c)
        else:
            bare.append(c)
    for c in sorted(set(elsewhere)):
        print(f'  {os.path.basename(c)}: stabilized before, without its original: motion from the picture')
    keep += sorted(set(elsewhere))
    not_taken(bare, other, exports, goggles)
    ix.write_originals(keep)
    print(f'scan: {len(keep)} clips -> {ix.originals_txt}')
    return keep


# ------------------------------------------------------------------------------------------- proxies
def gyroflow_exe(ix):
    g = ix.cfg.get('gyroflow') or os.environ.get('GYROFLOW')
    if not g or not os.path.isfile(g):
        raise SystemExit('Gyroflow not found: set GYROFLOW or `gyroflow` in embedia.toml')
    os.environ['GYROFLOW'] = g
    return g


def stabilize(cfg):
    """Clips with motion data are measured on Gyroflow's stabilized render, kept as their proxy (`stabilize = true`;
    the default with `fpv = true`; the montage sets it), or on the original as it is, scaled down only while a step
    reads the picture (the default)."""
    return bool(cfg.get('stabilize', cfg.get('fpv', False)))


def frame_for(ix, c):
    """What a clip is measured on now: 'stabilized' (Gyroflow's render) or 'as-is' (the original scaled down). A
    stabilized render that is there stays what its clip is measured on: it is never given up for a scaled one."""
    if stabilized_before(c):
        return 'as-is'
    if stabilize(ix.cfg) or ix.frame_kind(c) == 'stabilized' and os.path.exists(ix.render(c)):
        return 'stabilized'
    return 'as-is'


def write_recipe(ix, c, tool, call=None):
    """The clip's recipe beside its features: the frame they are measured on, the proxy's size, fps and frame count
    (the timeline reads them when the proxy is gone), when the recording began, how Gyroflow was called."""
    from embedia.read import creation_time
    with av.open(ix.render(c)) as ct:
        st = ct.streams.video[0]
        n, fps, w, h = st.frames, float(st.average_rate), st.codec_context.width, st.codec_context.height
    here = os.path.exists(c)
    rec = dict(frame='stabilized' if tool == 'gf_render' else 'as-is', tool=tool, width=w, height=h, frames=n,
               fps=fps, original=c, size=os.path.getsize(c) if here else None,
               start=creation_time(c).isoformat() if here else None,
               gyroflow=dict(version=call.get('gyroflow'), args=call.get('args')) if call else None)
    with open(ix.path(c, 'recipe'), 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)


def adopt_recipes(ix):
    """Recipes of the proxies made before the index kept them: a clip with motion data was rendered by Gyroflow."""
    for c in ix.originals():
        if os.path.exists(ix.render(c)) and not ix.has(c, 'recipe'):
            write_recipe(ix, c, 'proxy' if stabilized_before(c) else 'gf_render')


def motion(ix, clips):
    """GoPro and .gcsv motion through Gyroflow's export (gf_motion.py), once per clip, whatever it is measured on."""
    gopro = [c for c in clips if gyroflow_motion(c, ix.cache_dir) and not os.path.exists(
        os.path.join(ix.cache_dir, ix.stem(c) + '_gyroflow_camera.json'))]
    if gopro:
        gyroflow_exe(ix)
        lst = os.path.join(ix.log_dir, 'gopro.txt')
        open(lst, 'w', encoding='utf-8').write('\n'.join(gopro) + '\n')
        run('gf_motion', [ix.cache_dir, '--list', lst])


def proxies(ix, clips=None):
    """Proxies of the clips (all by default) that lack one, 640 px wide, several at once: Gyroflow's render of a
    clip measured stabilized (gf_render.py), the original scaled down otherwise (proxy.py); a recipe beside each.
    A scaled proxy of a clip now to be measured stabilized goes first. An original not there is skipped."""
    clips = ix.originals() if clips is None else clips
    os.makedirs(ix.proxy_dir, exist_ok=True)
    os.makedirs(ix.log_dir, exist_ok=True)
    adopt_recipes(ix)
    for c in clips:
        if ix.frame_kind(c) == 'as-is' and frame_for(ix, c) == 'stabilized' and os.path.exists(ix.render(c)):
            os.remove(ix.render(c))
    gone = [c for c in clips if not os.path.exists(ix.render(c)) and not os.path.exists(c)]
    if gone:
        print(f'  WARN proxies: {len(gone)} originals not found (a disk not connected?): '
              + ', '.join(ix.stem(c) for c in gone[:5]))
    here = [c for c in clips if os.path.exists(c)]
    motion(ix, here)
    todo = [c for c in here if not os.path.exists(ix.render(c))]
    by_size, tool_of = {}, {}
    for c in todo:
        with av.open(c) as ct:
            cc = ct.streams.video[0].codec_context
            h = int(round(PROXY_W * cc.height / cc.width / 2)) * 2
        tool_of[c] = 'gf_render' if frame_for(ix, c) == 'stabilized' else 'proxy'
        by_size.setdefault((h, tool_of[c]), []).append(c)
    if 'gf_render' in tool_of.values():
        gyroflow_exe(ix)
    n = int(ix.cfg.get('renders', 3))
    calls = {}
    for (h, tool), cl in by_size.items():
        procs = []
        for k in range(n):
            part = cl[k::n]
            if not part:
                continue
            lst = os.path.join(ix.log_dir, f'{tool}_{h}_{k}.txt')
            open(lst, 'w', encoding='utf-8').write('\n'.join(part) + '\n')
            log = open(os.path.join(ix.log_dir, f'{tool}_{h}_{k}.log'), 'a', encoding='utf-8')
            cf = os.path.join(ix.log_dir, f'{tool}_{h}_{k}_calls.jsonl')
            if os.path.exists(cf):
                os.remove(cf)
            cmd = command(tool, [ix.proxy_dir, '--list', lst, '--width', PROXY_W, '--height', h]
                          + (['--bitrate', '3M', '--calls', cf] if tool == 'gf_render' else []))
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env()), cf))
        t0 = time.time()
        print(f'{tool}: {len(cl)} clips at {PROXY_W}x{h}, {len(procs)} at once', flush=True)
        for p, cf in procs:
            p.wait()
            if os.path.exists(cf):
                with open(cf, encoding='utf-8') as fh:
                    calls.update({os.path.normcase(x['clip']): x for x in map(json.loads, fh)})
        print(f'{tool}: done in {time.time() - t0:.0f}s')
    bad = [c for c in todo if os.path.exists(ix.render(c)) and not has_picture(ix.render(c))]
    for c in bad:
        os.replace(ix.render(c), failed_render(ix, c))
        print(f'  {os.path.basename(c)}: no picture in its render (see the logs in {ix.log_dir}): left out')
    if bad:
        ix.write_originals([c for c in ix.originals() if c not in bad])
    for c in todo:
        if c not in bad and os.path.exists(ix.render(c)):
            write_recipe(ix, c, tool_of[c], calls.get(os.path.normcase(os.path.abspath(c))))


def tidy(ix):
    """After a run: the proxies of the clips measured as they are go, their recipes stay (made again when a step
    needs the picture), unless `stabilize = true` keeps every proxy. A stabilized render always stays."""
    if stabilize(ix.cfg):
        return
    gone = [c for c in ix.originals() if ix.frame_kind(c) == 'as-is' and os.path.exists(ix.render(c))]
    for c in gone:
        os.remove(ix.render(c))
    if gone:
        print(f'proxies: {len(gone)} of clips measured as they are removed (made again when a step reads the picture)')


# ------------------------------------------------------------------------------------------- features
class Step:
    """An analyzer as the runner sees it: what it writes per clip (`outputs`, feature names; `optional`: a clip may
    get none, e.g. no GPS fix), which clips it is for, when it runs at all (`on`: from embedia.toml), and its
    command over the clips that need it (`cmd(ix, srcs, renders)`). A higher `version` makes the features of the
    older one stale: they are made again. `whole`: the step takes every clip it is for each time it runs (it
    joins them: voices across clips). `picture`: it reads the proxies (made for it where gone; measured again when
    the clip's frame changes); the others read the originals' sound and metadata."""
    def __init__(self, name, version, outputs, cmd, on=lambda cfg: True, clips='all', whole=False, optional=False,
                 py=None, files=(), picture=True):
        self.name, self.version, self.outputs, self.cmd, self.on = name, version, outputs, cmd, on
        self.clips, self.whole, self.optional, self.py, self.files = clips, whole, optional, py, files
        self.picture = picture


def yolo(name):
    return os.path.join(MODELS, 'yolo', name)


def clip_cache():
    return os.path.join(MODELS, 'clip')


def speech_py(ix):
    own = [os.path.join(MODELS, 'venvs', 'speech', *x) for x in (('Scripts', 'python.exe'), ('bin', 'python'))]
    py = ix.cfg.get('speech_py') or os.environ.get('SPEECH_PY') or next((x for x in own if os.path.exists(x)), None)
    if not py:
        raise SystemExit('speech = ... needs its venv (pyannote.audio 4, README): models/venvs/speech, SPEECH_PY or '
                         '`speech_py` in embedia.toml')
    return py


def workers(ix):
    """The CPU steps' processes at once (`workers`; default half the CPUs): fewer on a machine others work on."""
    return ['--workers', str(ix.cfg['workers'])] if ix.cfg.get('workers') else []


def lst(ix, name, srcs):
    """A list file of clip paths (paths with spaces or Cyrillic go through a file, not the command line)."""
    os.makedirs(ix.log_dir, exist_ok=True)
    p = os.path.join(ix.log_dir, name + '.txt')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(''.join(s + '\n' for s in srcs))
    return p


def sheet(ix, name):
    os.makedirs(ix.checks_dir, exist_ok=True)
    return os.path.join(ix.checks_dir, name)


def language(cfg):
    return ['--language', cfg['speech']] if isinstance(cfg.get('speech'), str) else []


STEPS = [
    Step('cameras', 1, (), lambda ix, s, r: ['--list', ix.originals_txt, '--out', os.path.join(ix.root, 'cameras.json'),
                                             '--carried', *ix.cfg.get('carried', [])],
         on=lambda c: c.get('camera') == 'auto', whole=True, files=('cameras.json',), picture=False),
    Step('analyze', 1, ('flow', 'look'), lambda ix, s, r: r + workers(ix)),
    Step('vis_motion', 1, ('vis',), lambda ix, s, r: r + (['--hfov', ix.cfg['hfov']] if ix.cfg.get('hfov') else [])
         + workers(ix), clips='picture'),
    Step('gps', 1, ('gps',), lambda ix, s, r: ['track', '--list', lst(ix, 'gps', s), '--proxy-dir', ix.proxy_dir],
         clips='gopro', optional=True, picture=False),
    Step('people', 1, ('people',), lambda ix, s, r: ['detect'] + r + ['--weights', yolo('yolov8s-pose.pt'),
                                                                      '--objects', yolo('yolov8s.pt')],
         on=lambda c: c.get('people', True)),
    # CLIP of each person row (tight, and widened for subject): who and subject then need no picture
    Step('pclip', 1, ('pclip', 'pclipw'), lambda ix, s, r: ['embed'] + r + ['--clip-cache', clip_cache()],
         on=lambda c: c.get('people', True)),
    Step('pilot', 1, ('pilot',), lambda ix, s, r: ['pilotclip'] + r + [
        '--clip-cache', clip_cache(), '--sheet', sheet(ix, 'pilot_check.jpg')],
         on=lambda c: c.get('people', True) and c.get('fpv', False), clips='flown'),
    Step('passes', 1, ('passes',), lambda ix, s, r: ['passes'] + r + [
        '--weights', yolo('yolov8s-pose.pt'), '--sheet', sheet(ix, 'passes_check.jpg')],
         on=lambda c: c.get('people', True) and c.get('fpv', False), clips='flown'),
    Step('track', 1, ('ptrack', 'tid'), lambda ix, s, r: ['track'] + r + ['--weights', yolo('yolov8s-pose.pt')],
         on=lambda c: c.get('people', True) and c.get('track', False)),
    Step('objects', 2, ('objects',), lambda ix, s, r: ['--weights', yolo('yolov8s.pt')] + r,
         on=lambda c: c.get('objects', False)),
    Step('places', 1, ('clip',), lambda ix, s, r: ['embed'] + r + ['--clip-cache', clip_cache()],
         on=lambda c: c.get('places', True)),
    Step('sounds', 1, ('clap',), lambda ix, s, r: ['embed', '--list', lst(ix, 'sounds', s), '--proxy-dir', ix.proxy_dir],
         on=lambda c: c.get('sounds', False), picture=False),
    Step('speech', 1, ('speech',), lambda ix, s, r: ['run', '--list', ix.originals_txt, '--proxy-dir', ix.proxy_dir,
                                                     '--out', ix.root] + language(ix.cfg),
         on=lambda c: bool(c.get('speech')), whole=True, optional=True, py='speech', picture=False),
    Step('captions', 1, ('captions',), lambda ix, s, r: r, on=lambda c: c.get('captions', False)),
]
MODULE = {'cameras': 'camera', 'pilot': 'people', 'passes': 'people', 'track': 'people', 'pclip': 'people',
          'sounds': 'sounds'}


def step_clips(ix, step, kinds):
    """The originals a step is for."""
    srcs = ix.originals()
    if step.clips == 'flown':
        return [c for c in srcs if kinds(c) == 'drone']
    if step.clips == 'picture':
        return [c for c in srcs if stabilized_before(c) or motionless_export(c, ix.cache_dir)]
    if step.clips == 'gopro':
        return [c for c in srcs if not stabilized_before(c) and has_track(c, b'gpmd')]
    return srcs


def stale(ix, step, srcs):
    """The clips whose features of the step are missing or from an older version of it. A clip with the features
    and no record (made before the index kept records) is taken as made by this version."""
    made = ix.made(step.name)
    on = ix.made_on(step.name)
    out, adopt = [], []
    for c in srcs:
        v = made.get(ix.stem(c))
        have = all(ix.has(c, o) for o in step.outputs) and all(os.path.exists(os.path.join(ix.root, f))
                                                                for f in step.files)
        if v is None:
            (adopt if have and step.outputs else out).append(c)
        elif v < step.version or not (have or step.optional) or step.picture and moved(ix, c, on):
            out.append(c)
    if adopt:
        ix.record(step.name, step.version, adopt, step.picture)
    return out


def moved(ix, c, on):
    """The clip's features of a step were measured on another frame than the clip is measured on now (on: the
    step's ix.made_on; a record without a frame is from before frames were kept: as now)."""
    was = on.get(ix.stem(c))
    return was is not None and was != frame_for(ix, c)


# which steps a question's answers are made from: a clip measured again by one of them is asked again
ASKED_OF = {'who': ('people', 'pclip', 'track'), 'subject': ('people', 'pclip'), 'beauty': ('places',),
            'activities': ('places',)}


def forget(ix, again):
    """{step: clips measured again} -> the answers of the questions made from those steps go, for those clips."""
    for q in ix.questions():
        clips = {c for s in ASKED_OF.get(q['kind'], ()) for c in again.get(s, ())}
        if q['kind'] not in QUESTIONS or not clips:
            continue
        for c in clips:
            for f in QUESTIONS[q['kind']][0](ix, q['input'], q['name']):
                if ix.has(c, f):
                    os.remove(ix.path(c, f))


def drop(ix, step, srcs):
    """Remove a step's features of these clips (made again)."""
    for c in srcs:
        for o in step.outputs:
            if ix.has(c, o):
                os.remove(ix.path(c, o))


def carried_pilot(ix, kinds):
    """Nobody pilots a carried camera: an empty pilot for every person row of such a clip."""
    for c in ix.originals():
        if kinds(c) == 'carried' and ix.has(c, 'people') and not ix.has(c, 'pilot'):
            np.save(ix.path(c, 'pilot'), np.zeros(len(ix.get(c, 'people')['frame']), bool))


def features(ix, only=None, redo=()):
    """Every step of STEPS that embedia.toml turns on, over the clips it is for that need it; a step that reads the
    picture gets the proxies it lacks first."""
    os.makedirs(ix.proxy_dir, exist_ok=True)
    adopt_recipes(ix)
    log = os.path.join(ix.log_dir, 'features.log')
    ran = False
    again = {}
    for step in STEPS:
        if only and step.name not in only or not step.on(ix.cfg):
            continue
        kinds = camera_kinds(ix) if step.clips == 'flown' or step.name == 'pilot' else None
        srcs = step_clips(ix, step, kinds)
        if not srcs:
            continue
        if step.name in redo:
            drop(ix, step, srcs)
            with ix.db() as con:
                con.execute('DELETE FROM features WHERE step = ?', (step.name,))
        todo = stale(ix, step, srcs)
        if not todo:
            continue
        if not ran:
            from embedia import gpu
            gpu.device()
            ran = True
        on = ix.made_on(step.name)
        old = [c for c in todo if ix.made(step.name).get(ix.stem(c), step.version) < step.version
               or step.picture and moved(ix, c, on)]
        drop(ix, step, old)
        again[step.name] = srcs if step.name in redo else old
        take = srcs if step.whole else todo
        if step.picture:
            proxies(ix, take)
            take = [c for c in take if os.path.exists(ix.render(c))]
            if not take:
                continue
        py = speech_py(ix) if step.py == 'speech' else None
        rc = run(MODULE.get(step.name, step.name), step.cmd(ix, take, [ix.render(c) for c in take]), log, py)
        if rc:
            raise SystemExit(f'{step.name} failed: see {log}')
        done = [c for c in take if step.optional or all(ix.has(c, o) for o in step.outputs)]
        ix.record(step.name, step.version, done, step.picture)
        if step.name == 'speech':
            voices(ix)
    forget(ix, again)
    if ix.cfg.get('people', True) and ix.cfg.get('fpv', False):
        carried_pilot(ix, camera_kinds(ix))


def voices(ix):
    vj = os.path.join(ix.root, 'voices.json')
    if os.path.exists(vj):
        with open(vj, encoding='utf-8') as fh:
            vs = json.load(fh)['voices']
        print(f'  voices: {len(vs)}' + ''.join(f', {k} {v["talk"]:.0f}s in {len(v["clips"])} clips'
                                               for k, v in list(vs.items())[:6]) + f' ({ix.root}/voices/, transcript.txt)')


# ------------------------------------------------------------------------------------------- questions
# kind -> (the answer files per clip for its input, the command asking it). A question's input is kept in
# embedia.sqlite (`questions`): asked again when a clip lacks its answer (a clip added later) or the input changes.
def who_answers(ix, inp, name=''):
    return [f'subject_{n}' for n in inp if not n.startswith('_')]


def who_cmd(ix, inp, name=''):
    refs = [x for n, items in inp.items() for p in ([items] if isinstance(items, str) else items)
            for x in ('--ref', f'{n}={p}')]
    # the person on a screenshot is found in the picture and brings their whole track: the proxies, people.py track
    # first (without the originals, who goes by the kept embeddings alone)
    proxies(ix)
    track = STEPS[[s.name for s in STEPS].index('track')]
    todo = stale(ix, track, ix.originals())
    if todo and run('people', track.cmd(ix, todo, [ix.render(c) for c in todo]), log_of(ix)):
        raise SystemExit(f'people.py track failed: see {log_of(ix)}')
    ix.record('track', track.version, todo)
    return ['who'] + ix.renders() + refs + ['--originals', ix.originals_txt, '--ref-dir', os.path.join(ix.root, 'refs'),
                                            '--clip-cache', clip_cache(), '--sheet', sheet(ix, 'who_check.jpg')]


def subject_cmd(ix, inp, name=''):
    # the widened crops' embeddings are kept by the pclip step; a clip measured before it needs its picture
    proxies(ix, [c for c in ix.originals() if not ix.has(c, 'pclipw')])
    return (['subject'] + ix.renders() + [x for t in inp['texts'] for x in ('--text', t)]
            + [x for t in inp.get('nots', []) for x in ('--not', t)] + (['--tag', name] if name else [])
            + ['--clip-cache', clip_cache(), '--sheet', sheet(ix, f'subject_{name or "check"}.jpg')])


QUESTIONS = {
    # main characters by example: {name: [screenshots or CLIP:SECONDS:X]}, _name somebody else (people.py who)
    'who': (who_answers, who_cmd, 'people'),
    # a main character in words: {texts, nots} (people.py subject --tag NAME)
    'subject': (lambda ix, inp, name='': [f'subject_{name}' if name else 'subject'], subject_cmd, 'people'),
    # beautiful views in the job's words: {texts, plain} (beauty.py)
    'beauty': (lambda ix, inp, name='': ['beauty'],
               lambda ix, inp, name='': ix.renders() + [x for t in inp['texts'] for x in ('--text', t)]
               + [x for t in inp.get('plain', []) for x in ('--plain', t)]
               + ['--clip-cache', clip_cache(), '--sheet', sheet(ix, 'beauty_check.jpg')], 'beauty'),
    # what happens in the footage, the shoot's own activities added: {texts}, or alone with only (places.py activities)
    'activities': (lambda ix, inp, name='': ['activities'],
                   lambda ix, inp, name='': ['activities'] + ix.renders() + [x for t in inp.get('texts', [])
                                                                              for x in ('--text', t)]
                   + (['--only'] if inp.get('only') else [])
                   + ['--clip-cache', clip_cache(), '--sheet', sheet(ix, 'activities.jpg')], 'places'),
}


def log_of(ix):
    return os.path.join(ix.log_dir, 'features.log')


def answered(ix, kind, inp, name=''):
    files, _, _ = QUESTIONS[kind]
    want = files(ix, inp, name)
    return all(ix.has(c, f) for c in ix.originals() for f in want)


def ask(ix, kind, inp, name='', force=False):
    """Ask the footage a question and keep it: asked again only when its input changes or a clip lacks its answer
    (force: always). True when it was asked now."""
    if kind not in QUESTIONS:
        raise SystemExit(f'{kind}: not a question ({", ".join(QUESTIONS)})')
    old = ix.question(kind, name)
    same = old is not None and json.dumps(old['input'], sort_keys=True) == json.dumps(inp, sort_keys=True)
    if same and not force and answered(ix, kind, inp, name):
        return False
    files, cmd, module = QUESTIONS[kind]
    if not same:                                                 # a new input: the old answers go
        for c in ix.originals():
            for f in (files(ix, old['input'], name) if old else []):
                if ix.has(c, f):
                    os.remove(ix.path(c, f))
    from embedia import gpu
    gpu.device()
    args = cmd(ix, inp, name)
    if run(module, args, log_of(ix)):
        raise SystemExit(f'{kind} {name}: failed, see {log_of(ix)}')
    ix.remember(kind, name, inp)
    return True


def reask(ix):
    """The questions asked before, for the clips that lack their answers."""
    n = 0
    for q in ix.questions():
        if q['kind'] in QUESTIONS and not answered(ix, q['kind'], q['input'], q['name']):
            n += ask(ix, q['kind'], q['input'], q['name'], force=True)
    return n


# ------------------------------------------------------------------------------------------- all of it
def timeline(ix):
    from embedia import timeline as tl
    tl.build(ix)


def everything(ix, join=False, only=None, redo=()):
    """scan, proxies, features, the questions again, the timeline: each skips what is done."""
    t0 = time.time()
    scan(ix, join)
    if stabilize(ix.cfg):                                        # kept: every clip has its proxy
        proxies(ix)
    analyze(ix, only, redo)
    print(f'embedia: {len(ix.originals())} clips in {time.time() - t0:.0f}s -> {ix.root}')


def analyze(ix, only=None, redo=()):
    features(ix, only, redo)
    reask(ix)
    tidy(ix)
    timeline(ix)
