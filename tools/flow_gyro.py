"""A gyro track from the picture, for Gyroflow, for a camera with no gyro: on a tripod, with a zoom lens.

    PY tools/flow_gyro.py CLIP.MP4 [...] [--f0 PX] [--zoom-still]
    PY tools/flow_gyro.py CLIP.MP4 --check RENDER.mp4 --from S      # how much shake a render of it has left

Beside each clip: CLIP.gyroflow (a Gyroflow project: open it, or render it `Gyroflow CLIP.gyroflow`: its own output
section wins over -p, the file goes beside the clip; a test render goes through the gcsv instead), and
what it is made of: CLIP_flow.npz (the measurement; a second run only rewrites the rest), CLIP.gcsv and CLIP_lens.json
(`Gyroflow CLIP.MP4 CLIP_lens.json -g CLIP.gcsv`). The GUI does not take a gcsv by its name when the clip's own
format is known (a Sony MP4: "no motion data").

Per pair of consecutive frames (the picture MEAS_W wide): corners tracked both ways (vis_motion.track) and a
similarity fitted by RANSAC: shift, roll and scale about the image centre. The scale is the zoom; Gyroflow cannot
follow one, so it is left out.

The gyro written is the picture's motion over a fixed focal length f0, not the camera's true rotation, and the
lens profile is a pinhole of the same f0: Gyroflow then takes away the shift that is seen. After a zoom in the same
shake of the tripod moves the picture more and its gyro is larger with it; a true gyro under a fixed profile would
be corrected f/f0 times too little. Smoothing and cropping work in the picture's pixels as well.

During a zoom (the scale changing over ZOOM_RATE a frame, ZOOM_WIN mean) the shift and the roll are measured as
anywhere else, the scale taken about the image centre: the stabilization goes on through the zoom (a x2.1 zoom over
4 s on a6400 footage kept 0.12 of its shake, the 20 s after it 0.13). A zoom centre fitted per zoom spread from -400 to +1200 px (the operator
panning while zooming; a clean zoom: +4, +13): one fitted for the clip and taken out would add a pan to every zoom.
--zoom-still: a zero gyro during zooms instead (Gyroflow does nothing there).

A weak stretch (no fit, or few tracks agreeing: frames smeared by a jerk) is measured across, from the frame before
to the frame after, and the difference spread over its weak pairs (bridge); --no-bridge: as measured pair by pair.

No rolling shutter correction (the project's frame_readout_time 0). On the a6400 at 4K 25p it is there and
measurable (the shear and stretch of a frame pair against the change of its shift, corr +0.7-0.8, 24-39 ms, rows top
to bottom), but on a tripod the shake changes by 1-2 px a frame: about 1 px between the top and the bottom of the
picture, not to be seen. Gyroflow would take it (frame k/fps is the middle row, row y at k/fps - R/2 + y R/H, the
orientation slerped between the frames') from stabilization.frame_readout_time.

--check: the shake left in a render (Gyroflow's export of a range starting at --from seconds) against the clip's own
over the same frames, per axis: the high-pass (1 s) spread of the shift across, the shift up and the roll. Under 0.5
the axis is stabilized, ~1 with a high correlation not driven, ~2 the wrong sign. Gyroflow's zooming (dynamic by
default) changes the scale a frame and blurs it: a check render goes without it.
"""
import argparse
import datetime
import json
import os
import sys
import time
import urllib.parse

import av
import cv2
import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository (run as a file)
from embedia import pix  # noqa: E402
from embedia.vis_motion import track  # noqa: E402

MEAS_W = 960            # the picture is measured this wide
RANSAC_PX = 1.0         # inlier distance of the similarity, px at MEAS_W
MIN_TRACKS = 12         # fewer tracks or inliers: no fit, a zero gyro for that pair
ZOOM_RATE, ZOOM_WIN, ZOOM_PAD = 0.002, 5, 3   # |log scale| a frame (ZOOM_WIN mean) that is a zoom; frames added
F0_MM, SENSOR_MM = 16.0, 23.5                 # f0: the kit lens wide, on APS-C (a6400 4K reads the full width)
HP_S = 1.0              # --check: the shake is what a HP_S moving mean takes away
UP = 8                  # gyro samples a frame (200 Hz at 25 fps)
WEAK_INL = 0.4          # a pair with fewer tracks agreeing (or no fit) is weak: measured across instead (bridge)
BRIDGE_INL, BRIDGE_PAD = 0.2, 2               # a measurement across holds from this share; ends moved out up to this


def similarity(a, b, cen):
    """(scale, roll rad, shift x, shift y about cen, inlier share, tracks) of tracks a -> b; NaN without a fit."""
    if len(a) < MIN_TRACKS:
        return np.nan, np.nan, np.nan, np.nan, 0.0, len(a)
    M, m = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
                                       maxIters=500, refineIters=10)
    if M is None or m.sum() < MIN_TRACKS:
        return np.nan, np.nan, np.nan, np.nan, 0.0, len(a)
    t = M[:, 2] + M[:, :2] @ cen - cen                       # p' - c = A (p - c) + t
    return np.hypot(M[0, 0], M[1, 0]), np.arctan2(M[1, 0], M[0, 0]), t[0], t[1], float(m.mean()), len(a)


def measure(path, width=MEAS_W, start=0.0):
    """Every pair of consecutive frames of a video from `start` s -> dict of per-frame arrays (frame k: the pair
    k-1 -> k; the first frame none), shifts in the video's own pixels."""
    c = av.open(path)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    W, H = st.codec_context.width, st.codec_context.height
    h = int(round(H * width / W / 2)) * 2
    cen = np.array([(width - 1) / 2, (h - 1) / 2])
    if start > 0:
        c.seek(int(start / st.time_base), stream=st, backward=True)
    rows, prev = [], None
    for fr in c.decode(st):
        if fr.pts is not None and float(fr.pts * st.time_base) < start - 0.5 / fps:
            continue
        g = pix.ndarray(fr, format='gray', width=width, height=h)
        rows.append((1.0, 0.0, 0.0, 0.0, 1.0, 0) if prev is None else similarity(*track(prev, g), cen))
        prev = g
    c.close()
    r = np.array(rows, float)
    sc = W / width
    return dict(s=r[:, 0], th=r[:, 1], tx=r[:, 2] * sc, ty=r[:, 3] * sc, inl=r[:, 4], npts=r[:, 5],
                fps=fps, W=W, H=H)


def frames_at(path, ks, width=MEAS_W):
    """Grey frames of a video by number, MEAS_W wide -> {k: array}."""
    out = {}
    with av.open(path) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate)
        W, H = st.codec_context.width, st.codec_context.height
        h = int(round(H * width / W / 2)) * 2
        c.seek(max(int((min(ks) / fps - 1.0) / st.time_base), 0), stream=st, backward=True)
        for fr in c.decode(st):
            k = int(round(float(fr.pts * st.time_base) * fps))
            if k in ks:
                out[k] = pix.ndarray(fr, format='gray', width=width, height=h)
            if k >= max(ks):
                break
    return out


def bridge(clip, m):
    """Weak stretches (no fit, or under WEAK_INL of the tracks agreeing: a frame smeared by a jerk at a long
    exposure) measured across: from the frame before to the frame after (BRIDGE_PAD more on each side when the ends
    do not hold), the tracks started where the measured shifts put them; the difference between that and the sum of
    the stretch's good pairs spread evenly over its weak ones. A jerk at C0028 1336.4 s (a6400, 1/25 s): one frame
    with no fit, three at 19-23 %; 244 px measured pair by pair, 355 across, and Gyroflow left the 110 as a 61 px jump
    (1080p) in two frames. -> new tx, ty, th; lines to print."""
    tx, ty, th = (np.nan_to_num(m[k]).copy() for k in ('tx', 'ty', 'th'))
    weak = (m['inl'] < WEAK_INL) | np.isnan(m['s'])
    weak[0] = False
    edges = np.flatnonzero(np.diff(np.r_[0, weak.astype(int), 0]))
    fps, W, n = float(m['fps']), int(m['W']), len(tx)
    sc = W / MEAS_W
    lines = []
    for i, j in zip(edges[::2], edges[1::2] - 1):            # weak pairs i..j: frames i-1 .. j
        best = None
        for pad in range(BRIDGE_PAD + 1):
            a, b = max(i - 1 - pad, 0), min(j + pad, n - 1)
            f = frames_at(clip, {a, b})
            if a not in f or b not in f:
                continue
            h = f[a].shape[0]
            cen = np.array([(MEAS_W - 1) / 2, (h - 1) / 2])
            d = np.array([tx[a + 1:b + 1].sum(), ty[a + 1:b + 1].sum()]) / sc
            for guess in (lambda p, d=d: p + d, None):
                r = similarity(*track(f[a], f[b], guess, levels=4), cen)
                if np.isfinite(r[0]) and r[4] >= BRIDGE_INL and (best is None or r[4] > best[1][4]):
                    best = ((a, b), r)
            if best is not None and best[1][4] >= 2 * BRIDGE_INL:
                break
        if best is None:
            lines.append(f'    {i / fps:8.2f}-{j / fps:8.2f} s: not measured across, left as it was')
            continue
        (a, b), r = best
        w = np.arange(a + 1, b + 1)
        wk = w[weak[w]]
        good = w[~weak[w]]
        before = (tx[w].sum(), ty[w].sum())
        for v, total in ((tx, r[2] * sc), (ty, r[3] * sc), (th, r[1])):
            v[wk] = (total - v[good].sum()) / len(wk)
        lines.append(f'    {i / fps:8.2f}-{j / fps:8.2f} s ({len(wk)} weak of {len(w)} pairs): {before[0]:+.0f}, '
                     f'{before[1]:+.0f} px pair by pair -> {tx[w].sum():+.0f}, {ty[w].sum():+.0f} across '
                     f'({r[4] * 100:.0f} % of {r[5]:.0f} tracks)')
    return tx, ty, th, lines


def zooms(z, fps):
    """Frames of zooming from the log scale per frame -> bool mask, list of (first, last) frames."""
    zoom = np.abs(uniform_filter1d(np.nan_to_num(z), ZOOM_WIN)) > ZOOM_RATE
    zoom = binary_dilation(zoom, iterations=ZOOM_PAD)
    edges = np.flatnonzero(np.diff(np.r_[0, zoom.astype(int), 0]))
    return zoom, list(zip(edges[::2], edges[1::2] - 1))


def shake(tx, ty, th, fps):
    """High-pass spread of the shifts and the roll (what a HP_S moving mean takes away)."""
    n = max(int(round(HP_S * fps)), 1)
    return [float(np.nanstd(v - uniform_filter1d(v, n))) for v in (tx, ty, th)]


def held(g, fps):
    """Rates per pair (n x 3) -> (sample times s, rates) UP samples a frame, the pair k-1 -> k on the samples in
    ((k-1)/fps, k/fps]. Gyroflow (1.6.3 sources): frame k is looked up at k/fps (+ half the readout, none here), and
    an integrator turns by sample i's rate over (t[i-1], t[i]]; rates held around k/fps instead gave frame k 0.63 of
    its pair and 0.37 of the one before (single-frame shake a third corrected)."""
    n = len(g)
    j = np.arange(n * UP)
    k = -(-j // UP)                                          # ceil(j / UP): the pair whose rate holds at sample j
    keep = (k >= 1) & (k < n)
    out = np.zeros((len(j), 3))
    out[keep] = g[k[keep]]
    return j / (fps * UP), out


def orientations(g, fps):
    """Gyroflow's orientation at every frame k / fps, wxyz -> xyzw rows, from rates per pair (rad/s, gcsv axes), the way
    its SimpleGyroIntegrator turns them (start Rx(90 deg), omega = (-gy, gx, gz), q <- q * exp(omega dt))."""
    from scipy.spatial.transform import Rotation
    steps = Rotation.from_rotvec(np.column_stack([-g[:, 1], g[:, 0], g[:, 2]]) / fps)
    q = np.zeros((len(g), 4))
    r = Rotation.from_euler('x', np.pi / 2)
    q[0] = r.as_quat()
    for k in range(1, len(g)):
        r = r * steps[k]
        q[k] = r.as_quat()                                   # x, y, z, w: nalgebra's storage, as the project takes it
    return q


def write_project(path, clip, gcsv, lens, g, fps, W, H):
    """A Gyroflow project: the clip, the lens, and in gyro_source the orientation of every frame (quaternions: Gyroflow
    takes them as they are, integration_method 0, no VQF with its rest-bias estimate) and the rates (raw_imu, UP a
    frame: the motion charts, the GUI's sample-rate check, another integrator if chosen there). filepath is the gcsv:
    a project's motion data is read only from a file other than the video."""
    n = len(g)
    q = orientations(g, fps)
    t, r = held(g, fps)
    deg = np.degrees(r)
    imu = ','.join(f'{{"timestamp_ms":{ti * 1000:.4f},"gyro":[{a:.6f},{b:.6f},{c:.6f}],"accl":null,"magn":null}}'
                   for ti, (a, b, c) in zip(t, deg))
    quats = ','.join(f'"{int(round(k * 1e6 / fps))}":[{x:.12f},{y:.12f},{z:.12f},{w:.12f}]'
                     for k, (x, y, z, w) in enumerate(q))
    with av.open(clip) as c:
        rate = (c.streams.video[0].bit_rate or c.bit_rate or 0) / 1e6
    folder, name = os.path.split(os.path.abspath(clip))
    url = 'file:///' + urllib.parse.quote(folder.replace(os.sep, '/') + '/')
    stem = os.path.splitext(name)[0]
    head = {'title': 'Gyroflow data file', 'version': 3, 'app_version': '1.6.3',
            'videofile': os.path.abspath(clip).replace(os.sep, '/'), 'calibration_data': lens,
            'video_info': {'width': W, 'height': H, 'rotation': 0.0, 'num_frames': n, 'fps': fps,
                           'duration_ms': n * 1000.0 / fps},
            # without it the CLI's render queue never adds the job (render_queue.rs: `loaded` only from "output")
            'output': {'codec': 'H.265/HEVC', 'codec_options': '', 'output_folder': url,
                       'output_filename': stem + '_stabilized.mp4', 'output_width': W, 'output_height': H,
                       'input_filename': name, 'input_url': url + urllib.parse.quote(name), 'bitrate': rate,
                       'use_gpu': True, 'audio': True, 'pixel_format': '', 'encoder_options': '',
                       'metadata': {'comment': ''}, 'keyframe_distance': 1.0, 'preserve_other_tracks': False,
                       'pad_with_black': False, 'export_trims_separately': False, 'audio_codec': 'AAC',
                       'interpolation': 'Lanczos4'}}
    src = json.dumps({'filepath': os.path.abspath(gcsv).replace(os.sep, '/'), 'detected_source': 'flow_gyro',
                      'imu_orientation': 'XYZ', 'integration_method': 0}, ensure_ascii=False)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(head, ensure_ascii=False)[:-1] + ',"gyro_source":' + src[:-1] +
                 f',"raw_imu":[{imu}],"quaternions":{{{quats}}}}}}}')


def write_gcsv(path, g, fps, video):
    """The rates (rad/s, Gyroflow's axes) as a gcsv, UP samples a frame (the GUI warns under 50 Hz, asks for 200)."""
    t, g = held(g, fps)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('GYROFLOW IMU LOG\nversion,1.3\nid,flow_gyro\n'
                 'orientation,XYZ\nnote,gyro from the picture (tools/flow_gyro.py)\n'
                 f'videofilename,{os.path.basename(video)}\ntscale,0.000001\ngscale,0.000001\nascale,1\n'
                 't,gx,gy,gz\n')
        for ti, (a, b, c) in zip(t, g):
            fh.write(f'{int(round(ti * 1e6))},{int(round(a * 1e6))},{int(round(b * 1e6))},{int(round(c * 1e6))}\n')


def lens_profile(W, H, f0, fps):
    """A pinhole lens profile of focal f0 px (no distortion: the camera corrects its kit lens in video)."""
    return {'name': f'flow_gyro pinhole f0 {f0:.0f}px {W}x{H}', 'note': 'tools/flow_gyro.py', 'calibrated_by': 'flow_gyro',
            'camera_brand': 'flow_gyro', 'camera_model': 'pinhole', 'lens_model': f'f0 {f0:.0f}px', 'camera_setting': '',
            'calib_dimension': {'w': W, 'h': H}, 'orig_dimension': {'w': W, 'h': H},
            'output_dimension': {'w': W, 'h': H}, 'frame_readout_time': None, 'gyro_lpf': None,
            'input_horizontal_stretch': 1.0, 'input_vertical_stretch': 1.0, 'num_images': 0, 'fps': fps,
            'official': False, 'asymmetrical': False, 'compatible_settings': [], 'sync_settings': None,
            'distortion_model': 'opencv_standard', 'digital_lens': None, 'digital_lens_params': None, 'crop': None,
            'interpolations': None, 'focal_length': None, 'crop_factor': None, 'global_shutter': False,
            # an empty calibrator_version: Gyroflow 1.6.3 refuses the profile ("Loading lens profile: InvalidData")
            'date': datetime.date.today().isoformat(), 'calibrator_version': '1.6.3', 'identifier': '',
            'fisheye_params': {'RMS_error': 0.0, 'camera_matrix': [[f0, 0.0, W / 2], [0.0, f0, H / 2], [0.0, 0.0, 1.0]],
                               'distortion_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0], 'radial_distortion_limit': None}}


def run(clip, a):
    stem = os.path.splitext(clip)[0]
    npz = stem + '_flow.npz'
    t0 = time.time()
    if os.path.isfile(npz):
        m = dict(np.load(npz))
        m['fps'], m['W'], m['H'] = float(m['fps']), int(m['W']), int(m['H'])
    else:
        m = measure(clip)
        np.savez(npz, **m)
    fps, W, H = m['fps'], m['W'], m['H']
    n = len(m['s'])
    ok = np.isfinite(m['s'])
    z = np.log(np.where(ok, m['s'], 1.0))
    zoom, spans = zooms(z, fps)
    t = np.column_stack([np.nan_to_num(m['tx']), np.nan_to_num(m['ty'])])
    th = np.nan_to_num(m['th'])
    bridged = []
    if not a.no_bridge:
        tx, ty, th, bridged = bridge(clip, m)
        t = np.column_stack([tx, ty])
    if a.zoom_still:
        t[zoom], th[zoom] = 0.0, 0.0
    f0 = a.f0 or F0_MM / SENSOR_MM * W
    # in Gyroflow's gcsv axes (orientation XYZ), measured on renders with one axis at a time: gx takes the picture's
    # shift across away, gy (negative) the shift up, gz the roll
    g = np.column_stack([t[:, 0] / f0 * fps, -t[:, 1] / f0 * fps, th * fps])
    lens = lens_profile(W, H, f0, fps)
    write_gcsv(stem + '.gcsv', g, fps, clip)
    with open(stem + '_lens.json', 'w', encoding='utf-8') as fh:
        json.dump(lens, fh, indent=1)
    write_project(stem + '.gyroflow', clip, stem + '.gcsv', lens, g, fps, W, H)
    sx, sy, sr = shake(t[:, 0], t[:, 1], th, fps)
    quiet = ~binary_dilation(zoom, iterations=int(fps))
    print(f'{os.path.basename(clip)}: {n} frames at {fps:g} fps, {W}x{H}; fitted {ok.mean() * 100:.1f} %, median '
          f'{np.median(m["npts"]):.0f} tracks, inliers {np.median(m["inl"][ok]) * 100:.0f} %; in {time.time() - t0:.0f}s')
    print(f'  zoom: {len(spans)} stretches, {zoom.mean() * 100:.1f} % of the clip; |log scale| a frame away from zooms '
          f'p50 {np.median(np.abs(z[quiet & ok])):.5f} p99 {np.percentile(np.abs(z[quiet & ok]), 99):.5f} '
          f'(a zoom over {ZOOM_RATE})')
    for i, j in spans[:a.list]:
        print(f'    {i / fps:7.1f}-{j / fps:7.1f} s  x{np.exp(z[i:j + 1].sum()):.2f}')
    print(f'  shake (1 s high-pass): {sx:.1f} px across, {sy:.1f} px up, {np.degrees(sr) * 60:.1f} arcmin roll a frame; '
          f'f0 {f0:.0f} px -> {os.path.basename(stem)}.gyroflow')
    hp = max(int(round(HP_S * fps)), 1)
    e = np.hypot(*(v - uniform_filter1d(v, hp) for v in (t[:, 0], t[:, 1])))
    e = uniform_filter1d(e ** 2, int(10 * fps)) ** 0.5            # 10 s windows
    worst = []
    for k in np.argsort(-e):
        if all(abs(k - w) > 10 * fps for w in worst):
            worst.append(k)
        if len(worst) == 3:
            break
    print('  shakiest 10 s: ' + ', '.join(f'{k / fps - 5:.0f}-{k / fps + 5:.0f} s ({e[k]:.1f} px)' for k in worst))
    if bridged:
        print(f'  weak stretches measured across ({len(bridged)}):')
        print('\n'.join(bridged))


def check(clip, a):
    """Shake left in a render against the clip over the same frames, per axis."""
    m = dict(np.load(os.path.splitext(clip)[0] + '_flow.npz'))
    fps, W = float(m['fps']), int(m['W'])
    r = measure(a.check)                         # as wide as the clip was measured: a render at 640 px measured
    k0 = int(round(a.start * fps))               # at 640 has a noise floor (~1 px = 6 px at 4K) over a tripod's shake
    n = min(len(r['s']), len(m['s']) - k0)
    o = [np.nan_to_num(m['tx'][k0:k0 + n]) / W, np.nan_to_num(m['ty'][k0:k0 + n]) / W, np.nan_to_num(m['th'][k0:k0 + n])]
    b = [np.nan_to_num(r['tx'][:n]) / r['W'], np.nan_to_num(r['ty'][:n]) / r['W'], np.nan_to_num(r['th'][:n])]
    hp = max(int(round(HP_S * fps)), 1)
    o = [v - uniform_filter1d(v, hp) for v in o]
    b = [v - uniform_filter1d(v, hp) for v in b]
    names = ('across', 'up', 'roll')
    # left / the clip's: 1 and correlation ~1 is an axis not driven, ~2 the wrong sign, ~0 stabilized
    print(f'{os.path.basename(a.check)}: {n} frames from {a.start:g} s; shake left of the clip\'s (correlation with '
          f'it): ' + ', '.join(f'{nm} {np.std(y) / max(np.std(x), 1e-12):.2f} ({np.corrcoef(x, y)[0, 1]:+.2f})'
                                for nm, x, y in zip(names, o, b)))
    zoom = zooms(np.log(np.where(np.isfinite(m['s']), m['s'], 1.0)), fps)[0][k0:k0 + n]
    for name, sel in (('during zooms', zoom), ('away from zooms', ~zoom)):
        if sel.sum() >= 10:
            print(f'  {name} ({sel.sum()} frames): ' + ', '.join(
                f'{nm} {np.std(y[sel]) / max(np.std(x[sel]), 1e-12):.2f}' for nm, x, y in zip(names, o, b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clips', nargs='+')
    ap.add_argument('--f0', type=float, default=None, help='the profile focal length, px (default: 16 mm on APS-C)')
    ap.add_argument('--zoom-still', action='store_true', help='a zero gyro during zooms')
    ap.add_argument('--no-bridge', action='store_true', help='weak stretches as measured pair by pair')
    ap.add_argument('--list', type=int, default=12, help='zoom stretches printed')
    ap.add_argument('--check', default=None, help='a render of the clip: the shake it has left, per axis')
    ap.add_argument('--from', dest='start', type=float, default=0.0, help='--check: the render starts here, s')
    a = ap.parse_args()
    for clip in a.clips:
        (check if a.check else run)(clip, a)


if __name__ == '__main__':
    main()
