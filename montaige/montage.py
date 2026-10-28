"""One command per job: footage -> its index (embedia: proxies, features, the job's questions) -> edits -> QC
(-> finals on request).

    GPU_PY -m montaige out/JOB/job.toml                 every step; each skips what is done already
    GPU_PY -m montaige out/JOB/job.toml --join          also join split DJI recordings (renames the parts!)
    GPU_PY -m montaige out/JOB/job.toml --only edits    one step: scan, render, features, edits, qc
    GPU_PY -m montaige out/JOB/job.toml --only edits --only qc --edit 3_zima    a new version only
    GPU_PY -m montaige out/JOB/job.toml --final coach,flyby [--size 3840x2880]
(or GPU_PY montaige/montage.py ...)

Run with a Python that has CUDA torch (GPU_PY): the ML steps run on the GPU only. Gyroflow comes from the job's
`gyroflow` or the GYROFLOW environment variable.

Job file (TOML) out/<job>/job.toml: everything of the job stays in out/<job>/ (local, like the footage paths in it);
the job's name is its folder (or `name = ...`). What the footage is and which analyzers run goes to the index
(out/<job>/index/embedia.toml, written from these keys every run; embedia/run.py says what each does):

    footage = ["/path/to/footage"]           # folders (subfolders too) or single clips
    camera = "carried"                       # carried by a person (hand, helmet, chest), not flown: no pilot, no close
                                             # passes, edit.py --carried (default "drone"); "auto": per clip by its
                                             # accelerometer (camera.py), a trip filmed both ways
    carried = ["GX*", "DJI_2000*"]           # with "auto": clips carried, said (no accelerometer to tell)
    renders = 3                              # Gyroflow renders at once (default 3)
    workers = 4                              # CPU steps' processes at once (default half the CPUs)
    people = true                            # poses (default true)
    fpv = true                               # FPV drone footage: the pilot, close passes (default false: say it)
    objects = false                          # animals, vehicles (objects.py)
    places = true                            # CLIP frame embeddings for find:TEXT pieces (default true)
    sounds = false                           # CLAP of the clips' sound for hear:TEXT pieces (a carried camera hears
                                             # people, a drone its propellers); edits get --bites 2
    captions = false                         # every 3 s in words by a local vision-language model (cap:TEXT pieces,
                                             # the timeline's caption column) and its review of every shot in `qc`
    speech = "ru"                            # what people say and who says it (say:TEXT pieces, the timeline's speech
                                             # and voice columns, index/voices/); --talk N only in the args of an edit
                                             # that asks for it. Its own venv: `speech_py`, SPEECH_PY or
                                             # models/venvs/speech
    hfov = 92                                # field of view of clips stabilized before (vis_motion.py)
    index = "out/trip/index"                 # another index than out/<job>/index (one shoot, several jobs)

The job's own questions to the index (kept there with their input, asked again for clips added later):

    beauty = ["mountains", "a canyon"]       # beautiful views in words (beauty.py), for edit.py --beauty W
    plain = ["snow up close"]                # what fills this footage without being a view (beauty.py --plain)
    [who]                                    # main characters by example (people.py who):
    marina = ["refs/marina.jpg", "GH010051:12.5:0.3"]   # screenshots in out/<job>/, or CLIP:SECONDS:X
    _others = ["refs/lookalike.jpg"]         # _NAME: somebody else, kept apart

The montage's own:

    lut = "models/luts/DJI O4 Air Unit Series D-Log M to Rec.709 V1.cube"   # only for log footage
    lut_clips = ["DJI_2026*"]                # the clips in log, when the others are not (default every clip)
    args = ["--people", "4"]                 # edit.py options of every edit (before the edit's own)

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
  scan      embedia: the clips taken in, every video left out listed with why; split DJI recordings stop the job
            until --join (ask the owner of the footage first) -> index/originals.txt
  render    embedia: the proxies, 640 px wide (Gyroflow renders; clips stabilized before only scaled down)
  features  embedia: the analyzers, the job's questions (who, beauty), the timeline (index/embedia.sqlite)
  edits     edit.py per [[edit]], previews 640 px wide, 25 fps; redone when the job file is newer than the EDL.
            "auto" length: 25 s + 0.25 x seconds the person is in view at 2 % of the frame or more + 0.5 x
            seconds at 6 % or more, 25..75 s (the tennis kids: 29..65 s)
  qc        qc.py per edit -> <edit>_qc.txt; edl_sheet.py: the shot sheet <edit>_sheet.jpg and the cut sheet
            <edit>_cuts.jpg (+ _cuts.txt, its summary printed)
Main characters in words (python -m embedia subject) and look-alike groups (people.py split) are a job's own
business: ask them before `edits`.
"""
import argparse
import fnmatch
import glob
import os
import subprocess
import sys
import time
import tomllib

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # the repository: embedia, montaige (run as a file)
import embedia  # noqa: E402
from embedia import paths  # noqa: E402
from embedia import run as er  # noqa: E402
from montaige import beats  # noqa: E402
ROOT = paths.ROOT
STEPS = ('scan', 'render', 'features', 'edits', 'qc')
# a step's printed lines read back: UTF-8 both ways (captions and clip names carry Cyrillic; Windows' default is cp1252)
TEXT = dict(capture_output=True, encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
PROXY_W = er.PROXY_W
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


# ------------------------------------------------------------------------------------------- the index
# the job file's keys that describe the footage and the analyzers: the index's (embedia.toml)
INDEX_KEYS = ('footage', 'camera', 'carried', 'gyroflow', 'stabilize', 'renders', 'workers', 'hfov', 'people', 'fpv', 'track', 'objects',
              'places', 'sounds', 'captions', 'speech', 'speech_py')


def index_of(job, out):
    """The job's index: out/<job>/index (or `index = ...`), its embedia.toml written from the job file. The montage
    cuts its previews from the proxies: they are stabilized and kept."""
    cfg = {k: job[k] for k in INDEX_KEYS if k in job}
    cfg.setdefault('stabilize', True)
    if job.get('who'):                                           # a screenshot's person is found along their track
        cfg['track'] = True
    return embedia.create(job.get('index') or os.path.join(out, 'index'), cfg)


def questions(job, job_dir, ix):
    """The job's questions to its index: main characters by example ([who]: screenshots in the job folder or
    CLIP:SECONDS:X), beautiful views in words (beauty, plain). True when one was asked now."""
    asked = False
    if job.get('who') and job.get('people', True):
        inp = {}
        for name, items in job['who'].items():
            for x in ([items] if isinstance(items, str) else items):
                p = os.path.join(job_dir, x)
                inp.setdefault(name, []).append(os.path.abspath(p) if os.path.exists(p) else x)
        asked |= er.ask(ix, 'who', inp)
    if job.get('beauty'):
        asked |= er.ask(ix, 'beauty', dict(texts=list(job['beauty']), plain=list(job.get('plain', []))))
    return asked


def features(job, job_dir, ix):
    er.features(ix)
    questions(job, job_dir, ix)
    er.reask(ix)
    er.timeline(ix)
    print(f'features: done (log {os.path.join(ix.log_dir, "features.log")}); check {ix.checks_dir}'
          f'/pilot_check.jpg and passes_check.jpg once')


# ------------------------------------------------------------------------------------------- edits
def auto_target(ix, tag):
    """Seconds of an edit about one person, from how much of them there is (2 samples a second)."""
    from embedia import people
    mid = close = 0.0
    for r in ix.renders():
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


def edit_args(job, e):
    """edit.py options of an edit: the job's `args` (all edits: the people weight, themes of the shoot),
    then the edit's own (a later value wins); --carried for a carried camera, --cameras for "auto"."""
    cam = ['--carried'] if carried(job) else \
        ['--cameras', os.path.join(job['_ix'].root, 'cameras.json')] if job.get('camera') == 'auto' else []
    return cam + [str(a) for a in job.get('args', []) + e.get('args', [])]


def check_edits(job):
    """Every edit's options through edit.py's own parser before anything runs, and one line each to read: a
    TOML comment once swallowed half an args line (--heroes, --pace, --fly) without any error."""
    from montaige import edit
    for e in job.get('edit', []):
        args = edit_args(job, e)
        try:
            edit.make_parser().parse_args([e['music'], '-o', 'check.mp4'] + args)
        except SystemExit:
            raise SystemExit(f'{e["name"]}: edit.py does not take these options: {" ".join(args)}')
        clips = ''
        if e.get('clips') and job['_ix'].originals():
            clips = f', {len(edit_clips(job, e))} clips'
        print(f'  {e["name"]}: {e["music"].rsplit("/", 1)[-1]}, {e.get("target", 60)} s{clips}, {" ".join(args)}')


def edit_clips(job, e):
    """The job's originals an edit's `clips` patterns take (names without extension, case ignored); a pattern
    that takes nothing stops the job: a typo would quietly leave a place out."""
    srcs = job['_ix'].originals()
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


def edits(job, job_path, out, ix, plan_only=False):
    series = []
    brief = False                                                # the footage table once per run (edit.py --brief)
    beats.ensure([e['music'] for e in job.get('edit', [])])      # the beats a listener hears, for edit.py's grid
    for e in job.get('edit', []):
        name = e['name']
        dst = os.path.join(out, name, name + '.mp4')
        edl = os.path.join(out, name, name + '_edl.json')
        if job.get('_edits') and name not in job['_edits']:     # --edit: the others stay as they are
            if e.get('series'):
                series.append(edl)
            continue
        args = edit_args(job, e)
        target = e.get('target', 60)
        if target == 'auto':
            if '--subject' not in args:
                raise SystemExit(f'{name}: target "auto" is for an edit about one person (--subject)')
            target = auto_target(ix, args[args.index('--subject') + 1])
        fresh = os.path.exists(edl) and os.path.getmtime(edl) > os.path.getmtime(job_path) and \
            (plan_only or os.path.exists(dst))
        if not fresh:
            originals = ix.originals_txt
            if e.get('clips'):
                os.makedirs(os.path.join(out, name), exist_ok=True)
                originals = os.path.join(out, name, 'originals.txt')
                with open(originals, 'w', encoding='utf-8') as fh:
                    fh.write(''.join(s + '\n' for s in edit_clips(job, e)))
            cmd = [os.path.join(HERE, 'edit.py'), e['music'], '-o', dst, '--originals', originals,
                   '--proxy-dir', ix.proxy_dir, '--cache', ix.cache_dir, '--target', target,
                   '--height', preview_height(ix.proxy_dir),
                   '--fps', 25]
            if e.get('bpm'):
                cmd += ['--bpm', e['bpm']]
            if job.get('lut'):
                cmd += ['--lut', job['lut'], '--grade', 'auto']
                cmd += ['--lut-clips', *job['lut_clips']] if job.get('lut_clips') else []
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
        if job.get('_edits') and name not in job['_edits']:
            continue
        edl, mp4 = (os.path.join(out, name, name + x) for x in ('_edl.json', '.mp4'))
        if os.path.exists(mp4):
            r = subprocess.run([sys.executable, os.path.join(HERE, 'qc.py'), edl, mp4], **TEXT)
            open(os.path.join(out, name, name + '_qc.txt'), 'w', encoding='utf-8').write(r.stdout)
            warn = [x for x in r.stdout.splitlines() if x.startswith('WARN')]
            print(f'  {name}: {len(warn)} WARN' + ''.join(f'\n    {w}' for w in warn))
            sheet = [sys.executable, os.path.join(HERE, 'edl_sheet.py'), edl]
            subprocess.run(sheet + [os.path.join(out, name, name + '_sheet.jpg')], capture_output=True)
            r = subprocess.run(sheet + [os.path.join(out, name, name + '_cuts.jpg'), '--cuts'], **TEXT)
            open(os.path.join(out, name, name + '_cuts.txt'), 'w', encoding='utf-8').write(r.stdout + r.stderr)
            print(f'    {(r.stdout.strip().splitlines() or ["cut sheet failed, see _cuts.txt"])[-1]}')
            if job.get('captions', False):                       # the model's look at every shot (review.py)
                r = subprocess.run([sys.executable, os.path.join(HERE, 'review.py'), edl], **TEXT)
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
    ap.add_argument('--edit', default=None, help='edits and qc of these edits only (comma-separated names): a new '
                    'version is a new [[edit]], the ones watched before are not made again')
    ap.add_argument('--size', default='3840x2880', help='final size, e.g. 3840x2880 (4:3) or 1920x1080')
    a = ap.parse_args()
    from embedia import gpu
    gpu.device()
    job_path = os.path.abspath(a.job)
    with open(job_path, 'rb') as fh:
        job = tomllib.load(fh)
    os.chdir(ROOT)
    name = job.get('name') or os.path.basename(os.path.dirname(job_path))
    out = os.path.join('out', name)
    job['_out'] = out
    job['_edits'] = a.edit.split(',') if a.edit else None
    os.makedirs(out, exist_ok=True)
    ix = index_of(job, out)
    job['_ix'] = ix
    if a.final:
        final(job, out, a.final.split(','), a.size)
        return
    steps = a.only or STEPS
    t0 = time.time()
    check_edits(job)
    if 'scan' in steps:
        er.scan(ix, a.join)
    if 'render' in steps:
        er.proxies(ix)
    if 'features' in steps:
        features(job, os.path.dirname(job_path), ix)
    if 'edits' in steps:
        edits(job, job_path, out, ix, a.plan_only)
    if 'qc' in steps and not a.plan_only:
        qc(job, out)
    print(f'montage: {", ".join(steps)} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
