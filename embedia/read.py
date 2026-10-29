"""The footage as the features describe it, read for whoever uses them (the montage, timeline.py, other
projects): camera motion per frame (DJI `djmd`, Gyroflow's export of GoPro and .gcsv motion, or vis_motion.py's
reading of the picture), recording times, image motion, look thumbnails, beauty and GPS speed on a clip's frame
grid. Paths: the original clip (`src`) and its proxy (`render`, `stab`: <stem>_stabilized.mp4 in the index's proxy
folder, the features beside it as <stem>_stabilized_<feature>.npz)."""
import json
import os

import av
import numpy as np
from scipy.ndimage import uniform_filter1d

SPEED_SMOOTH_S = 1.0    # GPS ground speed is smoothed over this before use
PICK_WHOLE = 0.8        # an export beside its original this long or longer is the whole flight (ends trimmed)
# look thumbnails of two aspects compare on the 4:3 window (a 16:9 hand camera beside a 4:3 drone)
LOOK_ROWS, LOOK_COLS = 9, 12


def subject_file(base, tag=''):
    """Where a main character's label lives: per people.npz row, bool (people.py subject / who)."""
    return base + '_subject' + (f'_{tag}' if tag else '') + '.npy'


def subject_rows(base, tags):
    """Per people.npz row: the main character (people.py subject / who), or None without any file. Tags A+B+C take
    any of several people (a family, a couple: the film is about all of them)."""
    masks = [np.load(subject_file(base, t)) for t in tags.split('+') if os.path.exists(subject_file(base, t))]
    if not masks:
        return None
    return np.logical_or.reduce(masks)


def frame_count(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return s.frames, float(s.average_rate)


def rates_from_quats(q, fps):
    """Camera angular velocity (deg/s, x=tilt y=pan z=roll) between consecutive frames."""
    a, b = q[:-1], q[1:]
    b = b * np.where(np.sum(a * b, axis=1) < 0, -1.0, 1.0)[:, None]
    aw, ax, ay, az = a[:, 0], -a[:, 1], -a[:, 2], -a[:, 3]        # conj(a)
    bw, bx, by_, bz = b.T
    w = aw * bw - ax * bx - ay * by_ - az * bz
    v = np.stack([aw * bx + ax * bw + ay * bz - az * by_,
                  aw * by_ - ax * bz + ay * bw + az * bx,
                  aw * bz + ax * by_ - ay * bx + az * bw], 1)
    sn = np.linalg.norm(v, axis=1)
    ang = 2 * np.arctan2(sn, np.clip(w, -1, 1))
    rv = v / np.maximum(sn, 1e-12)[:, None] * ang[:, None]
    rates = np.degrees(rv) * fps
    return np.vstack([rates, rates[-1:]])


def rotate(q, v):
    """Camera-frame vector v in world coordinates for every orientation q (N, 4 wxyz)."""
    w, x, y, z = q.T
    u = np.stack([x, y, z], 1)
    t = 2 * np.cross(u, v)
    return v + w[:, None] * t + np.cross(u, t)


def load_telemetry(src, cache_dir):
    st = os.stat(src)
    key = f'{os.path.splitext(os.path.basename(src))[0]}_{st.st_size}_{int(st.st_mtime)}_q.npz'
    cp = os.path.join(cache_dir, key)
    if os.path.exists(cp):
        z = np.load(cp)
        return z['q'].astype(np.float64), float(z['fps'])
    from .media import djimeta
    n, fps = frame_count(src)
    q = djimeta.per_frame(src, n)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cp, q=q.astype(np.float32), fps=fps)
    return q, fps


def stabilized_before(path):
    """A clip montage.py's scan took in as already stabilized: a `_stabilized` export without its original."""
    return '_stabilized' in os.path.basename(path).lower()


def export_of(name, stems):
    """The original (one of `stems`, lower case) a `_stabilized` export is of, or None: its name before
    `_stabilized` is the original's, or ends with it after words the owner put in front ("tubing, the good take
    GH018650_stabilized.mp4")."""
    head = os.path.basename(name).lower().split('_stabilized')[0]
    for s in stems:
        if head == s or head.endswith(s) and not head[-len(s) - 1].isalnum():
            return s
    return None


def owner_picked(path):
    """A clip the owner stabilized themselves: a `_stabilized` export without its original, or an original with
    the owner's `_stabilized` export of the whole flight beside it (PICK_WHOLE of its length or more: Gyroflow's
    trim takes the ends off; montage.py takes the original then, gyro and all). A shorter export is a piece of
    the flight, not all of it: exports.py finds where it is, and it comes in as a --must."""
    if stabilized_before(path):
        return True
    d, n = os.path.split(path)
    stem = os.path.splitext(n)[0].lower()
    exp = [m for m in (os.listdir(d or '.') if os.path.isdir(d or '.') else [])
           if os.path.splitext(m)[0].lower().endswith('_stabilized') and export_of(m, [stem])]
    for m in exp:
        with av.open(path) as a, av.open(os.path.join(d, m)) as b:
            if b.duration >= PICK_WHOLE * a.duration:
                return True
    return False


def creation_time(path):
    """When the recording of a file began. A GoPro chapter (GH02xxxx, GX03xxxx: the camera splits a recording at
    4 GB) carries the start of the whole recording, so it begins after the chapters before it."""
    import datetime
    import re
    with av.open(path) as c:
        t = c.metadata.get('creation_time') or c.streams.video[0].metadata.get('creation_time')
    try:
        when = datetime.datetime.fromisoformat(t.replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc)
    folder, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    m = re.fullmatch(r'(G[HX])(\d\d)(\d{4})', stem, re.I)
    if m and int(m.group(2)) > 1:
        before = [os.path.join(folder, f'{m.group(1)}{k:02d}{m.group(3)}{ext}') for k in range(1, int(m.group(2)))]
        if all(os.path.exists(p) for p in before):
            for p in before:
                with av.open(p) as c:
                    when += datetime.timedelta(seconds=float(c.duration) / 1e6)
    return when


def load_motion(src, cache_dir, render=None):
    """Per-frame camera orientation and where it comes from: Gyroflow's export if the cache has one ('gyro');
    for a clip stabilized before (a `_stabilized` export without its original, see montage.py scan) or one whose
    export does not move (no gyro in the file: stabilized in the camera) the orientation vis_motion.py found in
    its picture ('picture'); else DJI `djmd` ('gyro')."""
    stem = os.path.splitext(os.path.basename(src))[0]
    gj = os.path.join(cache_dir, stem + '_gyroflow_camera.json')
    flat = False
    if os.path.exists(gj):
        with open(gj, encoding='utf-8') as fh:
            d = json.load(fh)
        q = np.array([r['org_quat'] for r in d], float)
        flat = np.ptp(q, axis=0).max() < 1e-9                    # no gyro in the file (Osmo Action 3, RockSteady)
        if not flat:
            return q / np.linalg.norm(q, axis=1, keepdims=True), frame_count(src)[1], 'gyro'
    if flat or stabilized_before(src):
        vp = os.path.splitext(render or '')[0] + '_vis.npz'
        if not os.path.exists(vp):
            raise SystemExit(f'stabilized before, no motion from its picture yet: run vis_motion.py on {render}')
        z = np.load(vp)
        return z['q'].astype(np.float64), float(z['fps']), 'picture'
    return (*load_telemetry(src, cache_dir), 'gyro')


def load_flow(stab, proxy_dir, n):
    """Image motion features of the proxy pass on the full frame grid, or None."""
    if not proxy_dir:
        return None
    p = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_flow.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    fr = z['frame'] - 1.0                  # flow between frames f-2 and f is centred on f-1
    i = np.arange(n)
    out = {k: np.interp(i, fr, z[k]) for k in ('flow', 'resid50', 'resid90', 'div', 'bottom', 'bright')}
    pp = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_props.npz')
    if os.path.exists(pp):                 # props.py: propellers inside the band the final crop keeps
        zp = np.load(pp)
        if 'kept' in zp.files:
            out['props'] = np.interp(i, zp['frame'], zp['kept'])
    return out


def load_look(stab, proxy_dir):
    """(step, thumbnails every `step` frames) from looks.py, or None."""
    if not proxy_dir:
        return None
    p = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_look.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    d = z['desc'].astype(np.float64)
    w = d.shape[1] // (LOOK_ROWS * 3)                            # analyze.py: 9 rows, the width from the aspect
    if w > LOOK_COLS:                                            # wider than 4:3: its centre, as render crops it
        x0 = (w - LOOK_COLS) // 2
        d = d.reshape(len(d), LOOK_ROWS, w, 3)[:, :, x0:x0 + LOOK_COLS].reshape(len(d), -1)
    return int(z['step']), d


def load_beauty(stab, n):
    """Aesthetic score and probability of the job's beautiful views (beauty.py) on the full frame grid, or None."""
    p = os.path.splitext(stab)[0] + '_beauty.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p)
    i = np.arange(n)
    return np.interp(i, z['frame'], z['aes']), np.interp(i, z['frame'], z['scenic'])


def load_gps_speed(render, n, fps, off=0):
    """GPS ground speed (m/s, gps.py track) on the render's frame grid, NaN where there is no fix; None without a
    track."""
    p = os.path.splitext(render)[0] + '_gps.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p)
    ok = z['fix'] > 0
    if ok.sum() < 2:
        return np.full(n, np.nan)
    t = (np.arange(n) + off) / fps
    sp = np.interp(t, z['t'][ok], z['speed'][ok])
    sp = uniform_filter1d(sp, max(int(SPEED_SMOOTH_S * fps), 1))
    # no fix for more than 2 s around a frame: unknown
    gap = np.interp(t, z['t'], (~ok).astype(float))
    return np.where((t < z['t'][ok][0] - 1) | (t > z['t'][ok][-1] + 1) | (gap > 0.99), np.nan, sp)
