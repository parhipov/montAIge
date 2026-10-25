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
    people = true                            # poses, pilot, close passes (default true)
    objects = false                          # animals, vehicles (objects.py)
    places = true                            # CLIP frame embeddings for find:TEXT pieces (default true)
    beauty = ["mountains", "a canyon"]       # beautiful views in words (beauty.py), for edit.py --beauty W
    hfov = 92                                # field of view of clips stabilized before (vis_motion.py)
    args = ["--people", "4"]                 # edit.py options of every edit (before the edit's own)

    [[edit]]
    name = "coach"
    music = "models/music/tennis/Motivator.mp3"
    bpm = 126
    target = 40                              # seconds, or "auto" for an edit about one person (--subject)
    series = true                            # keep the footage of the earlier series edits out (--avoid)
    args = ["--subject", "coach", "--focus",
            "--start", "find:a corridor -> a staircase -> an indoor tennis court"]

Steps:
  scan      the clips: originals only (not renders, not goggle recordings, not the _parts of a joined
            recording), and `_stabilized` clips whose original is not there. Split DJI recordings are listed and the job stops: they are joined (dji_join.py)
            only with --join, after the owner of the footage said yes. -> out/<name>/originals.txt
  render    whole-clip Gyroflow renders 640 px wide, several at once (gf_render.py); GoPro motion through
            Gyroflow's export (gf_motion.py). A `_stabilized` clip without its original (stabilized before,
            no motion data left) is only scaled down (proxy.py); its motion comes from the picture
            (vis_motion.py in `features`, `hfov = 92` in the job file: the field of view of its picture)
  features  per render: image motion and looks (analyze.py), people (people.py detect, pilotclip, passes),
            objects (objects.py), places (places.py embed),
            beautiful views (beauty.py, with `beauty`)
  edits     edit.py per [[edit]], previews 640 px 25 fps; redone when the job file is newer than the EDL.
            "auto" length: 25 s + 0.25 x seconds the person is in view at 2 % of the frame or more + 0.5 x
            seconds at 6 % or more, 25..75 s (the tennis kids: 29..65 s)
  qc        qc.py per edit -> <edit>_qc.txt
Main characters (people.py subject / who / split) are a job's own business: run them before `edits`.
"""
import argparse
import glob
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


def run(args, log=None):
    """A step's script with this Python; its output to the console (or a log file)."""
    cmd = [sys.executable] + [str(a) for a in args]
    print('  $ ' + ' '.join(os.path.basename(c) if i < 2 else c for i, c in enumerate(cmd)), flush=True)
    if log:
        with open(log, 'a', encoding='utf-8') as fh:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd).returncode


def has_track(path, fourcc):
    from mp4parse import open_mp4
    try:
        f, _, _, tracks = open_mp4(path)
        f.close()
    except Exception:
        return False
    return any(t.formats and t.formats[0] == fourcc for t in tracks)


# ------------------------------------------------------------------------------------------- scan
def scan(job, out, join):
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
    files = [f for f in files if f.lower().endswith('.mp4') and not os.path.basename(f).startswith(('_', '.'))]
    stems = {os.path.splitext(os.path.basename(f))[0].lower() for f in files}
    clips, elsewhere = [], []
    for f in files:
        n = os.path.basename(f)
        if n.upper().startswith('DJIG'):                         # goggle recordings carry no gyro
            continue
        if '_stabilized' in n.lower():
            # an export of an original that is here is the owner's old export (exports.py); without its original
            # it is the footage itself, already stabilized: picture as it is, motion from it (vis_motion.py)
            if n.lower().split('_stabilized')[0] not in stems:
                elsewhere.append(f)
            continue
        clips.append(f)
    keep = []
    for c in sorted(set(clips)):
        if has_track(c, b'djmd') or has_track(c, b'gpmd'):
            keep.append(c)
        else:
            print(f'  skip {os.path.basename(c)}: no motion data (DJI djmd / GoPro gpmd)')
    for c in sorted(set(elsewhere)):
        print(f'  {os.path.basename(c)}: stabilized before, without its original: motion from the picture')
    keep += sorted(set(elsewhere))
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


# ------------------------------------------------------------------------------------------- render
def render(job, out, proxy, cache):
    clips = originals(out)
    gyroflow = job.get('gyroflow') or os.environ.get('GYROFLOW')
    if not gyroflow or not os.path.isfile(gyroflow):
        raise SystemExit('Gyroflow not found: set GYROFLOW or `gyroflow` in the job file')
    os.environ['GYROFLOW'] = gyroflow
    gopro = [c for c in clips if not stabilized_before(c) and not has_track(c, b'djmd')]
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


# ------------------------------------------------------------------------------------------- features
def features(job, out, proxy):
    renders = [os.path.join(proxy, os.path.splitext(os.path.basename(c))[0] + '_stabilized.mp4')
               for c in originals(out)]
    missing = [r for r in renders if not os.path.exists(r)]
    if missing:
        raise SystemExit(f'{len(missing)} renders missing, e.g. {missing[0]}: run the render step')
    yolo = os.path.join(MODELS, 'yolo')
    log = os.path.join(out, 'features.log')
    run([os.path.join(HERE, 'analyze.py')] + renders, log)
    before = [r for c, r in zip(originals(out), renders) if stabilized_before(c)]
    if before:                                                   # their motion, from the picture
        run([os.path.join(HERE, 'vis_motion.py')] + before + (['--hfov', job['hfov']] if job.get('hfov') else []), log)
    if job.get('people', True):
        run([os.path.join(HERE, 'people.py'), 'detect'] + renders +
            ['--weights', os.path.join(yolo, 'yolov8s-pose.pt'), '--objects', os.path.join(yolo, 'yolov8s.pt')], log)
        run([os.path.join(HERE, 'people.py'), 'pilotclip'] + renders +
            ['--clip-cache', os.path.join(MODELS, 'clip'), '--sheet', os.path.join(out, 'pilot_check.jpg')], log)
        run([os.path.join(HERE, 'people.py'), 'passes'] + renders +
            ['--weights', os.path.join(yolo, 'yolov8s-pose.pt'), '--sheet', os.path.join(out, 'passes_check.jpg')], log)
    if job.get('objects', False):
        run([os.path.join(HERE, 'objects.py'), '--weights', os.path.join(yolo, 'yolov8s.pt')] + renders, log)
    if job.get('places', True):
        run([os.path.join(HERE, 'places.py'), 'embed'] + renders + ['--clip-cache', os.path.join(MODELS, 'clip')], log)
    if job.get('beauty'):                                        # beautiful views in the job's words (edit.py --beauty)
        run([os.path.join(HERE, 'beauty.py')] + renders + [x for t in job['beauty'] for x in ('--text', t)] +
            ['--clip-cache', os.path.join(MODELS, 'clip'), '--sheet', os.path.join(out, 'beauty_check.jpg')], log)
    print(f'features: done (log {log}); check {os.path.join(out, "pilot_check.jpg")} and passes_check.jpg once')


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


def edit_args(job, e):
    """edit.py options of an edit: the job's `args` (all edits: the people weight, themes of the shoot),
    then the edit's own (a later value wins)."""
    return [str(a) for a in job.get('args', []) + e.get('args', [])]


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
        print(f'  {e["name"]}: {e["music"].rsplit("/", 1)[-1]}, {e.get("target", 60)} s, {" ".join(args)}')


def edits(job, job_path, out, proxy, cache, plan_only=False):
    series = []
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
            cmd = [os.path.join(HERE, 'edit.py'), e['music'], '-o', dst, '--originals', os.path.join(out, 'originals.txt'),
                   '--proxy-dir', proxy, '--cache', cache, '--target', target, '--height', 480, '--fps', 25]
            if e.get('bpm'):
                cmd += ['--bpm', e['bpm']]
            if job.get('lut'):
                cmd += ['--lut', job['lut'], '--grade', 'auto']
            if e.get('series'):
                for prev in series:
                    cmd += ['--avoid', prev]
            if plan_only:
                cmd += ['--plan-only']
            if run(cmd + args):
                raise SystemExit(f'{name}: edit.py failed')
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


def final(job, out, names, size):
    w, h = (int(v) for v in size.lower().split('x'))
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
        scan(job, out, a.join)
    if 'render' in steps:
        render(job, out, proxy, cache)
    if 'features' in steps:
        features(job, out, proxy)
    if 'edits' in steps:
        edits(job, job_path, out, proxy, cache, a.plan_only)
    if 'qc' in steps and not a.plan_only:
        qc(job, out)
    print(f'montage: {", ".join(steps)} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
