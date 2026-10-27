"""Camera motion from the picture, for footage that was stabilized elsewhere and has no gyro left in the file.

    python vis_motion.py RENDER.mp4 [...] [--hfov DEG --vfov DEG]     ->  <render>_vis.npz next to each

The proxy is the stabilized picture, so what it shows is the smoothed camera path: the motion edit.py looks at
anyway (the visible, 1.5 Hz low-passed rotation). Per pair of consecutive frames: corners tracked with
pyramidal Lucas-Kanade both ways (kept where the way back lands within FB_PX), turned into bearings through a
pinhole, and the camera rotation fitted by RANSAC on pairs of bearings (Kabsch), refined on the inliers.
The rotations are chained into an orientation per frame in the gyro's convention (camera x right, y down,
z forward; world up -Z), so edit.py treats it like telemetry. There is no absolute reference in a picture, so
the up direction is pulled towards a level camera at ELEV0 with time constant TAU_S: drift goes, a dive or a
flip (seconds) stays.

The pinhole has its own horizontal and vertical field of view: GoPro SuperView squeezes a 4:3 view into 16:9,
and Gyroflow keeps that. Without --hfov/--vfov both are found in the picture (calibrate): a wide lens shows
the focal length, because with a wrong one the edges of the frame move unlike its centre and no rotation
fits; the field of view whose rotations fit the tracks best wins. Calm flying over open water fools it (a WARN
on an aspect neither square nor SuperView's): there --hfov alone gives a picture of square pixels (DJI).

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
sys.path.insert(0, HERE)
import pix  # noqa: E402

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
# calibration: frame pairs whose tracks move this much (median), so that turning outweighs travel. GH010401 with
# its gyro: 105 x 67 deg by the rates; calibrated 100 x 70 from pairs moving 3 px, 96 x 66 from 8 px, 94 x 63 from
# 15 px (25 pairs). Gyroflow's dynamic zoom moves the focal length of that render +-15 % anyway (fov_scale 0.77-1.04)
CAL_PAIRS, CAL_MOVE_PX = 160, 8.0
HFOV_GRID, VFOV_GRID, CAL_FINE = np.arange(70, 136, 5.0), np.arange(40, 136, 5.0), np.arange(-4, 5, 1.0)
# a calibrated view is believed when its pixel aspect (tan(vfov/2) / tan(hfov/2) * W / H) is square or SuperView's
# 4:3 in 16:9 (1.33). Calm orbits over open water (glare, a moving boat; 25-28 pairs moving 8 px of 160) gave
# 100 x 134 and 104 x 97 deg on DJI O3 exports (3.5 and 1.6); with square pixels the hfov alone had a clear best at
# 108-112 deg on all four flights of that trip (pairs moving 4 px)
PIXEL_ASPECTS = ((0.9, 1.1), (1.25, 1.45))


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
    """Horizontal and vertical field of view whose rotations fit the tracks of moving frame pairs best."""
    n = frame_count(path)
    pick = set(np.linspace(1, max(n - 1, 1), 6 * CAL_PAIRS).astype(int).tolist())
    it = tracks(path, pick)
    W, H, _ = next(it)
    pairs = [(a, b) for _, a, b in it if a is not None and len(a) >= 30
             and np.median(np.linalg.norm(b - a, axis=1)) > CAL_MOVE_PX]
    pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, min(CAL_PAIRS, len(pairs))).astype(int)] if pairs else []
    if len(pairs) < 20:
        return None
    def search(hs, vs):
        cost = np.array([[np.mean([fit_rotation(a, b, Pinhole(W, H, hf, vf), rng)[2] for a, b in pairs])
                          for vf in vs] for hf in hs])
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        return float(hs[i]), float(vs[j]), float(cost[i, j]), float(np.median(cost))
    hf, vf, _, med = search(HFOV_GRID, VFOV_GRID)                # 5 deg steps, then 1 deg around the best
    hf, vf, best, _ = search(hf + CAL_FINE, vf + CAL_FINE)
    return hf, vf, len(pairs), best, med


def square_vfov(path, hfov):
    """The vertical field of view of a picture with square pixels and this horizontal one."""
    with av.open(path) as c:
        W, H = c.streams.video[0].codec_context.width, c.streams.video[0].codec_context.height
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(hfov) / 2) * H / W))), W, H


def pair_rotations(path, hfov, vfov):
    """Rotation of the camera between frames k-1 and k for every frame k of a video (frame 0: none).
    The corners start where the last rotation would take them (a flip turns the picture ~15 deg a frame, beyond
    what the tracker finds on its own: at over 200 deg/s it kept 12-60 tracks of ~270); a fast turn tracks
    against prev turned by that rotation; a weak fit is tried again with a deeper pyramid, with and without the
    guess, and the fit with the most inliers is kept. A pair with no fit takes the last rotation (constant
    turning through a blurred frame or two)."""
    rng = np.random.default_rng(0)
    c = av.open(path)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    cam = Pinhole(st.codec_context.width, st.codec_context.height, hfov, vfov)
    frames, rots, inls, npts = [], [], [], []
    prev, R_last = None, np.eye(3)
    for fr in c.decode(st):
        k = int(round(float(fr.pts * st.time_base) * fps))
        g = pix.ndarray(fr, format='gray')
        rv, inl, npt = np.zeros(3), 0.0, 0
        if prev is not None:
            def guess(p, R=R_last):
                return cam.project(cam.bearings(p) @ R.T)
            best = None
            fast = np.linalg.norm(cv2.Rodrigues(R_last)[0]) > np.radians(FAST_DEG)
            for gs, lv in ((('warp', 3),) if fast else ()) + ((guess, 3), (guess, 5), (None, 5)):
                a, b = track_warped(prev, g, cam, R_last) if gs == 'warp' else track(prev, g, gs, lv)
                if len(a) >= 8:
                    R, share, _ = fit_rotation(a, b, cam, rng)
                    if best is None or share * len(a) > best[1] * best[2]:
                        best = (R, share, len(a))
                if best is not None and best[1] >= 0.8 and best[2] >= 60:
                    break
            if best is not None and best[1] * best[2] >= 8:
                R_last, inl, npt = best
            rv = cv2.Rodrigues(R_last)[0].ravel()
        frames.append(k)
        rots.append(rv)
        inls.append(inl)
        npts.append(npt)
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
    """One render's motion from its picture -> (lines to print, failed?)."""
    r, hf, vf = job
    dst = os.path.splitext(r)[0] + '_vis.npz'
    t0 = time.time()
    lines = []
    if hf is not None and vf is None:
        vf = square_vfov(r, hf)[0]
    if hf is None or vf is None:
        cal = calibrate(r, np.random.default_rng(0))
        if cal is None:
            lines.append(f'{os.path.basename(r)}: too little motion to find the field of view; pass --hfov (square '
                         f'pixels) or --hfov --vfov')
            return lines, True
        hf, vf = hf or cal[0], vf or cal[1]
        _, W, H = square_vfov(r, hf)
        aspect = np.tan(np.radians(vf) / 2) / np.tan(np.radians(hf) / 2) * W / H
        lines.append(f'{os.path.basename(r)}: field of view {cal[0]:.0f} x {cal[1]:.0f} deg from {cal[2]} moving frame '
                     f'pairs (cost {cal[3]:.2f} px^2, median over the grid {cal[4]:.2f})'
                     + ('' if any(lo <= aspect <= hi for lo, hi in PIXEL_ASPECTS) else
                        f'; WARN pixel aspect {aspect:.2f}, neither square nor SuperView: pass --hfov'))
    fr, rot, inl, npt, fps = pair_rotations(r, hf, vf)
    q = integrate(rot, fps, trust=trusted(inl, fps))
    np.savez(dst, frame=fr, rot=rot.astype(np.float32), inl=inl.astype(np.float32), npts=npt, q=q, fps=fps,
             hfov=hf, vfov=vf)
    lines.append(f'{os.path.basename(r)}: {len(fr)} frames, median {np.median(npt):.0f} tracks, inliers '
                 f'{np.median(inl[1:]) * 100:.0f} %, rotation trusted on {trusted(inl, fps).mean() * 100:.0f} %, '
                 f'in {time.time() - t0:.0f}s')
    return lines, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--hfov', type=float, default=None, help='horizontal field of view of the picture, deg')
    ap.add_argument('--vfov', type=float, default=None,
                    help='vertical field of view (default: from --hfov with square pixels, else calibrated)')
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
        todo.append((r, a.hfov, a.vfov))
    failed = 0
    w = a.workers or max(1, min(len(todo), (os.cpu_count() or 2) // 2))
    if w <= 1:
        results = map(process, todo)
    else:
        pool = multiprocessing.Pool(w)
        results = pool.imap_unordered(process, todo)
    for lines, bad in results:
        print('\n'.join(lines), flush=True)
        failed += bad
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
