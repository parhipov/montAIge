"""Camera motion from the picture, for footage that was stabilized elsewhere and has no gyro left in the file.

    python vis_motion.py RENDER.mp4 [...] [--hfov DEG | --calibrate] [--rate HZ]   ->  <render>_vis.npz next to each

The proxy is the stabilized picture, so what it shows is the smoothed camera path: the motion edit.py looks at
anyway (the visible, 1.5 Hz low-passed rotation). Per pair of frames: corners tracked with pyramidal Lucas-Kanade
both ways (kept where the way back lands within FB_PX), turned into bearings through a pinhole, and the camera
rotation fitted by RANSAC on pairs of bearings (Kabsch), refined on the inliers. The rotations are chained into
an orientation per frame in the gyro's convention (camera x right, y down, z forward; world up -Z), so edit.py
treats it like telemetry. There is no absolute reference in a picture, so the up direction is pulled towards a
level camera at ELEV0 with time constant TAU_S: drift goes, a dive or a flip (seconds) stays.

The pairs of consecutive frames are sampled --rate a second while the camera turns under FAST_DPS, every pair
above; the rotation is linear between.

The field of view comes from the camera that named the file (CAMERAS, after stabilization), or --hfov (the job's
`hfov`); for a file no camera named, or with --calibrate, it is estimated in the picture (calibrate). Pixels are
square after stabilization, so the vertical view follows. A wrong view scales the rotation rates, not when the
camera turns or which way.

Saved: frame, rot (n x 3, rad, rotation vector of frame k-1 -> k, camera axes), inl (inlier share; under TRUST_INL
the tilt is not trusted: flying low over a slope), npts,
q (n x 4 wxyz), fps, hfov, vfov.
"""
import argparse
import multiprocessing
import os
import sys
import time

import av
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if not __package__:                               # run as a file: the package of its folder (PEP 366)
    sys.path.insert(0, os.path.dirname(HERE))
    __package__ = os.path.basename(HERE)
from . import pix  # noqa: E402

FB_PX = 1.0             # forward-backward tracking error that keeps a track
RANSAC_PX = 1.5         # inlier distance of the rotation fit, px
MAX_CORNERS, MIN_DIST = 400, 12
FAST_DEG = 4.0          # last rotation over this many degrees a frame: track against the turned previous frame
# up pulled to a level camera at ELEV0 (median of the GoPro flights' -20..+6) with TAU_S: the rotation-only fit
# reads a climb near the ground as a tilt (GH010401 take-off: +5 deg/s for a true -10..-20); against the gyro,
# a 1 s change of elevation correlated 0.65 at 5 s, 0.57 at 15 s, 0.45 at 30 s
ELEV0, TAU_S = -4.0, 5.0
# low over a slope the ground fills the frame and its parallax reads as a tilt: where under TRUST_INL of the tracks
# fit one rotation (1 s mean), the tilt is noise (GH010403, low over dry hills: tilt against the gyro correlates
# 0.19 under 0.4, 0.32 at 0.4-0.6, 0.65-0.93 above; pan and roll stay good, 0.93 and 0.96). There the up direction
# is pulled with TAU_LOW_S, and edit.py counts no tilt manoeuvre (dive, pull-up, descent)
TRUST_INL, TAU_LOW_S = 0.6, 1.0
# pairs of consecutive frames sampled a second while the camera turns under FAST_DPS, every pair above (a flip):
# edit.py reads the 1.5 Hz low-passed rotation. Against every pair, at 6 a second: rates (1.5 Hz) correlated 0.997-
# 1.000, off by 0.5-2.2 deg/s of 9-44, inliers and trust the same, 5.2x faster on a ski run at 50 fps and 3.4x on
# a drone flight at 30 fps (turning over 60 deg/s 10 % of it); decoding is 4 s of it. Pairs of frames further apart
# instead cost as much (a weak fit is tracked again 3-4 times) and lose inliers to parallax (97 % one frame apart,
# 67 % four, 52 % eight on that flight)
RATE_HZ, FAST_DPS = 6.0, 60.0
# the horizontal field of view after stabilization by the name the camera gave the file (Gyroflow's export and the
# proxy keep it): the owners' exports estimated in the picture, GoPro 87-90 deg in a hand or on a helmet (Gyroflow's
# crop) and 96-114 on a drone, DJI O3 110; within ~15 % either way, and the rotation rates scale with it
CAMERAS = {'GoPro': (('GX', 'GH', 'GP', 'GOPR'), 100.0), 'DJI': (('DJI',), 110.0)}
# a file no camera named: the view estimated in the picture (square pixels), on CAL_PAIRS frame pairs whose tracks
# move CAL_MOVE_PX (median), so that turning outweighs travel; with too little turning, HFOV. The full search (both
# axes, 160 pairs, 1 deg steps) cost 1-3 min a clip whatever its length and read 97 x 92 on a camera looking down
# at its own skis, 100 x 134 over calm water; GH010401 with its gyro: 105 x 67 by the rates, found 96 x 66
CAL_PAIRS, CAL_MOVE_PX, HFOV_GRID, HFOV = 40, 8.0, np.arange(50, 131, 5.0), 100.0


def kabsch(a, b):
    """Rotations R (..., 3, 3) with b ~ R a for point sets a, b (..., n, 3)."""
    h = np.swapaxes(a, -1, -2) @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.swapaxes(vt, -1, -2) @ np.swapaxes(u, -1, -2)))
    dd = np.zeros(h.shape)
    dd[..., 0, 0] = dd[..., 1, 1] = 1.0
    dd[..., 2, 2] = d
    return np.swapaxes(vt, -1, -2) @ dd @ np.swapaxes(u, -1, -2)


class Pinhole:
    def __init__(self, W, H, hfov, vfov):
        self.fx = W / 2 / np.tan(np.radians(hfov) / 2)
        self.fy = H / 2 / np.tan(np.radians(vfov) / 2)
        self.cx, self.cy = (W - 1) / 2, (H - 1) / 2
        self.hfov, self.vfov = hfov, vfov

    def bearings(self, p):
        v = np.column_stack([(p[:, 0] - self.cx) / self.fx, (p[:, 1] - self.cy) / self.fy, np.ones(len(p))])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def homography(self, R):
        """Pixel map of a pure rotation R (bearing v -> R v)."""
        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])
        return K @ R @ np.linalg.inv(K)

    def project(self, v):
        z = np.maximum(v[:, 2], 1e-6)
        return np.column_stack([v[:, 0] / z * self.fx + self.cx, v[:, 1] / z * self.fy + self.cy])


def fit_rotation(a, b, cam, rng, iters=64):
    """Pure rotation between pixel tracks a -> b: RANSAC on pairs of bearings, refined on the inliers twice.
    -> R (b ~ R a), inlier share, truncated squared pixel error (the calibration's cost)."""
    A, B = cam.bearings(a), cam.bearings(b)
    thr = RANSAC_PX / max(cam.fx, cam.fy)
    idx = rng.integers(0, len(A), (iters, 2))
    Rs = kabsch(A[idx], B[idx])
    err = np.linalg.norm(B[None] - A[None] @ np.swapaxes(Rs, -1, -2), axis=2)
    best = int(np.argmax((err < thr).sum(1)))
    R, m = Rs[best], err[best] < thr
    for _ in range(2):
        if m.sum() < 3:
            break
        R = kabsch(A[m], B[m])
        m = np.linalg.norm(B - A @ R.T, axis=1) < thr
    px = np.linalg.norm(cam.project(A @ R.T) - b, axis=1)
    return R, m.mean(), float(np.mean(np.minimum(px, 3 * RANSAC_PX) ** 2))


def track(prev, g, guess=None, levels=3):
    """Corners of prev and where they are in g, both ways checked; guess(points) -> predicted positions."""
    p0 = cv2.goodFeaturesToTrack(prev, MAX_CORNERS, 0.01, MIN_DIST, blockSize=7)
    if p0 is None or len(p0) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    flags, p1 = 0, None
    if guess is not None:
        p1 = guess(p0.reshape(-1, 2)).astype(np.float32).reshape(-1, 1, 2)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, p1, winSize=(21, 21), maxLevel=levels, flags=flags)
    pb, s2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, p0.copy(), winSize=(21, 21), maxLevel=levels,
                                         flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1) < FB_PX)
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]


def track_warped(prev, g, cam, R):
    """track() after turning prev by the predicted rotation R, so the patches are not rotated against g (a roll
    at 700 deg/s turns them ~25 deg a frame, and the tracker matches shifted patches only); points come back
    in prev's own pixels."""
    H = cam.homography(R)
    a, b = track(cv2.warpPerspective(prev, H, (prev.shape[1], prev.shape[0])), g)
    if len(a):
        a = cv2.perspectiveTransform(a.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(H)).reshape(-1, 2)
    return a, b


def tracks(path, pick=None):
    """(frame k, points in frame k-1, the same points in frame k) for consecutive frames (only k in pick);
    also yields the video's size and rate first."""
    c = av.open(path)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    yield st.codec_context.width, st.codec_context.height, fps
    prev = None
    for fr in c.decode(st):
        k = int(round(float(fr.pts * st.time_base) * fps))
        want = pick is None or k in pick
        if not want and (pick is None or k + 1 not in pick):
            prev = None
            continue
        g = pix.ndarray(fr, format='gray')
        if want and prev is not None:
            yield (k, *track(prev, g))
        elif want:
            yield k, None, None
        prev = g
    c.close()


def frame_count(path):
    with av.open(path) as c:
        return c.streams.video[0].frames


def calibrate(path, rng):
    """The horizontal field of view (square pixels) whose rotations fit the tracks of moving frame pairs best
    -> (hfov, pairs, cost, median cost over the grid), None: too little turning."""
    n = frame_count(path)
    pick = set(np.linspace(1, max(n - 1, 1), 12 * CAL_PAIRS).astype(int).tolist())
    it = tracks(path, pick)
    W, H, _ = next(it)
    pairs = [(a, b) for _, a, b in it if a is not None and len(a) >= 30
             and np.median(np.linalg.norm(b - a, axis=1)) > CAL_MOVE_PX]
    pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, min(CAL_PAIRS, len(pairs))).astype(int)] if pairs else []
    if len(pairs) < CAL_PAIRS // 2:
        return None
    cost = np.array([np.mean([fit_rotation(a, b, Pinhole(W, H, hf, square(hf, W, H)), rng)[2] for a, b in pairs])
                     for hf in HFOV_GRID])
    i = int(np.argmin(cost))
    return float(HFOV_GRID[i]), len(pairs), float(cost[i]), float(np.median(cost))


def square(hfov, W, H):
    """The vertical field of view of a picture with square pixels and this horizontal one."""
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(hfov) / 2) * H / W)))


def view(path, hfov=None, estimate=False):
    """The horizontal field of view of a render -> (deg, where it comes from)."""
    if hfov is not None:
        return hfov, 'given'
    name = os.path.basename(path).upper()
    cam = next((c for c, (names, _) in CAMERAS.items() if name.startswith(names)), None)
    if cam and not estimate:
        return CAMERAS[cam][1], f"a {cam}'s after stabilization, by the file's name"
    cal = calibrate(path, np.random.default_rng(0))
    if cal is None:
        return HFOV, 'too little turning to estimate it in the picture'
    return cal[0], (f'estimated in the picture from {cal[1]} moving frame pairs (cost {cal[2]:.2f} px^2, median over '
                    f'the grid {cal[3]:.2f})')


def pair_rotations(path, hfov, vfov, rate=RATE_HZ):
    """Rotation of the camera between frames k-1 and k for every frame k of a video (frame 0: none), measured on
    `rate` pairs a second while the camera turns under FAST_DPS and on every pair above, linear between.
    The corners start where the turning so far would take them (a flip turns the picture ~15 deg a frame, beyond
    what the tracker finds on its own: at over 200 deg/s it kept 12-60 tracks of ~270); a fast turn tracks
    against prev turned by that rotation; a weak fit is tried again with a deeper pyramid, with and without the
    guess, and the fit with the most inliers is kept. A pair with no fit keeps the last rate of turning (constant
    turning through a blurred frame or two)."""
    rng = np.random.default_rng(0)
    c = av.open(path)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    step = max(1, int(round(fps / rate)))
    cam = Pinhole(st.codec_context.width, st.codec_context.height, hfov, vfov)
    frames, rots, inls, npts = [], [], [], []
    prev, ip, w, inl, npt = None, 0, np.zeros(3), 0.0, 0     # w: rotation vector a frame, of the last pair
    for i, fr in enumerate(c.decode(st)):
        frames.append(int(round(float(fr.pts * st.time_base) * fps)))
        rots.append(w)
        inls.append(inl)
        npts.append(npt)
        every = np.degrees(np.linalg.norm(w)) * fps > FAST_DPS
        if not every and i - ip < step - 1:                      # neither this pair nor the next is measured
            prev = None
            continue
        g = pix.ndarray(fr, format='gray')
        if prev is not None and (every or i - ip >= step):
            R_last = cv2.Rodrigues(w)[0]
            def guess(p, R=R_last):
                return cam.project(cam.bearings(p) @ R.T)
            best = None
            fast = np.linalg.norm(w) > np.radians(FAST_DEG)
            for gs, lv in ((('warp', 3),) if fast else ()) + ((guess, 3), (guess, 5), (None, 5)):
                a, b = track_warped(prev, g, cam, R_last) if gs == 'warp' else track(prev, g, gs, lv)
                if len(a) >= 8:
                    R, share, _ = fit_rotation(a, b, cam, rng)
                    if best is None or share * len(a) > best[1] * best[2]:
                        best = (R, share, len(a))
                if best is not None and best[1] >= 0.8 and best[2] >= 60:
                    break
            w0, inl, npt = w, 0.0, 0
            if best is not None and best[1] * best[2] >= 8:
                w = cv2.Rodrigues(best[0])[0].ravel()
                inl, npt = best[1], best[2]
            s = i - ip
            rots[ip + 1:] = list(w0 + np.outer(np.arange(1, s + 1) / s, w - w0))
            inls[ip + 1:], npts[ip + 1:] = [inl] * s, [npt] * s
            ip = i
        prev = g
    c.close()
    return np.array(frames), np.array(rots), np.array(inls), np.array(npts), fps


def level_up(elev):
    """World up in camera coordinates for a camera without roll, elev degrees above the horizon."""
    e = np.radians(elev)
    return np.array([0.0, -np.cos(e), np.sin(e)])


def trusted(inl, fps):
    """Frames whose rotation is trusted: the inlier share, 1 s mean, at least TRUST_INL."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(np.asarray(inl, float), max(int(fps), 1)) >= TRUST_INL


def integrate(rots, fps, elev0=ELEV0, tau=TAU_S, trust=None):
    """Orientation per frame (wxyz, camera -> world, world up -Z) from the frame-to-frame rotations; the up
    direction pulled towards a level camera at elev0 with time constant tau seconds (TAU_LOW_S where not trust)."""
    from scipy.spatial.transform import Rotation
    up_w = np.array([0.0, 0.0, -1.0])
    prior = level_up(elev0)
    # start level at elev0: camera axes in world, x right = world X, y down = world +Z
    C = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    C = C @ Rotation.from_rotvec([-np.radians(elev0), 0, 0]).as_matrix()
    alpha = 1.0 / max(tau * fps, 1.0)
    alpha_low = 1.0 / max(TAU_LOW_S * fps, 1.0)
    out = np.zeros((len(rots), 3, 3))
    for k, rv in enumerate(rots):
        if k:
            C = C @ cv2.Rodrigues(np.asarray(rv, float))[0].T   # X_k = R X_{k-1}: camera k -> k-1 is R^T
            u = C.T @ up_w
            ax = np.cross(u, prior)
            s = np.linalg.norm(ax)
            if s > 1e-9:
                al = alpha if trust is None or trust[k] else alpha_low
                G = cv2.Rodrigues(ax / s * al * np.arctan2(s, u @ prior))[0]      # turns u towards the prior
                C = C @ G.T
        out[k] = C
    q = Rotation.from_matrix(out).as_quat()                      # xyzw
    return np.column_stack([q[:, 3], q[:, :3]])


def process(job):
    """One render's motion from its picture -> lines to print."""
    r, hf, estimate, rate = job
    t0 = time.time()
    hf, where = view(r, hf, estimate)
    with av.open(r) as c:
        W, H = c.streams.video[0].codec_context.width, c.streams.video[0].codec_context.height
    vf = square(hf, W, H)
    fr, rot, inl, npt, fps = pair_rotations(r, hf, vf, rate)
    q = integrate(rot, fps, trust=trusted(inl, fps))
    np.savez(os.path.splitext(r)[0] + '_vis.npz', frame=fr, rot=rot.astype(np.float32), inl=inl.astype(np.float32),
             npts=npt, q=q, fps=fps, hfov=hf, vfov=vf)
    return [f'{os.path.basename(r)}: field of view {hf:.0f} x {vf:.0f} deg, {where}; {len(fr)} frames, median '
            f'{np.median(npt):.0f} tracks, inliers {np.median(inl[1:]) * 100:.0f} %, rotation trusted on '
            f'{trusted(inl, fps).mean() * 100:.0f} %, in {time.time() - t0:.0f}s']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--hfov', type=float, default=None,
                    help='horizontal field of view of the picture, deg (default: by the camera that named the file)')
    ap.add_argument('--calibrate', action='store_true', help='estimate the field of view in the picture, any file')
    ap.add_argument('--rate', type=float, default=RATE_HZ, help='frame pairs a second while the camera turns slowly')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--integrate', action='store_true', help='only chain the saved rotations again (new pull settings)')
    ap.add_argument('--workers', type=int, default=None,
                    help='clips at once (processes; the tracking is single-threaded CPU work; default half the CPUs)')
    a = ap.parse_args()
    todo = []
    for r in a.renders:
        dst = os.path.splitext(r)[0] + '_vis.npz'
        if a.integrate:
            z = dict(np.load(dst))
            z['q'] = integrate(z['rot'], float(z['fps']), trust=trusted(z['inl'], float(z['fps'])))
            np.savez(dst, **z)
            print(f'{os.path.basename(r)}: chained again, rotation trusted on {trusted(z["inl"], float(z["fps"])).mean() * 100:.0f} %')
            continue
        if os.path.exists(dst) and not a.force:
            print(os.path.basename(r), 'done before')
            continue
        todo.append((r, a.hfov, a.calibrate, a.rate))
    w = a.workers or max(1, min(len(todo), (os.cpu_count() or 2) // 2))
    for lines in (map if w <= 1 else multiprocessing.Pool(w).imap_unordered)(process, todo):
        print('\n'.join(lines), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
