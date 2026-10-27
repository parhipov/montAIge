"""Faces: tell apart people dressed alike (two girls in the same T-shirt), from the original clips.

    GPU_PY faces.py RENDER.mp4 [...] --originals originals.txt --check kid_pink [--also kid_pink_not]
        --ref kid_pink=CLIP:SECONDS:X [--ref _ponytail=CLIP:SECONDS:X ...] [--sheet faces.jpg]

The small renders are too small for faces (eyes ~3 px apart at 640 px), so at every sample where a
person labelled --check (or --also) is in view, the ORIGINAL frame is read (4K, the full field of view),
people are found there with YOLO-pose at 1280 px, and each face is aligned on the eyes and the nose to the
112x112 ArcFace layout and embedded by SFace (OpenCV Model Zoo, Apache 2.0, models/faces/; onnxruntime on
the GPU). The original frame is matched to the stabilized render by a homography (ORB), which tells which
render row each face belongs to.

References (--ref NAME=CLIP:SECONDS:X, the person nearest to X across the frame at that moment, or an image
of a face) name the faces: NAME = the --check person, _OTHER = someone dressed like them. A face goes to the
reference it resembles most (SFace cosine >= --sim and ahead by --margin). Then the --check label:
  - a track whose faces are mostly someone else loses it as a whole;
  - otherwise a row whose face is the person gets it (also from --also, e.g. the look-alikes split
    dropped), and so do the rows of its track within VOUCH_S that have no face of their own (faces are
    seen only while the person looks at the camera, a shot needs them for longer); a row whose face is
    someone else loses it; other rows keep what they had (a track is not trusted further: in FPV footage
    tracks jump between people).
"""
import argparse
import glob
import os

import av
import cv2
import numpy as np

import gpu
import paths
import pix
import people as pp

HERE = os.path.dirname(os.path.abspath(__file__))
SFACE = os.path.join(paths.MODELS, 'faces', 'face_recognition_sface_2021dec.onnx')
POSE = os.path.join(paths.MODELS, 'yolo', 'yolov8s-pose.pt')
# where the left eye, the right eye and the nose sit in a 112x112 ArcFace face
TEMPLATE = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366]])
EYES_MIN_PX, SIM, MARGIN, NEAR = 10.0, 0.36, 0.05, 0.08
VOUCH_S = 1.0           # a face vouches for its own track this far around it (rows without a face of their own)
NOT_BELOW = 0.20        # a face this unlike the person (SFace cosine) is someone else, even without their reference


class Embedder:
    def __init__(self, model):
        gpu.device()
        import torch  # noqa: F401  (onnxruntime takes CUDA and cuDNN from torch)
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        self.s = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
        if 'CUDAExecutionProvider' not in self.s.get_providers():
            raise SystemExit('onnxruntime has no CUDA here: pip install onnxruntime-gpu built for the CUDA of torch')

    def __call__(self, face_bgr):
        x = face_bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)   # SFace takes RGB 0..255
        e = self.s.run(None, {'data': np.ascontiguousarray(x)})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


def face_of(img, kp):
    """Aligned 112x112 face from keypoints (pixels, 17 x 3), or None when it is too small or not seen."""
    if min(kp[pp.LEYE, 2], kp[pp.REYE, 2], kp[pp.NOSE, 2]) < 0.5:
        return None
    src = np.float32([kp[pp.REYE, :2], kp[pp.LEYE, :2], kp[pp.NOSE, :2]])  # the person's right eye is on the left
    if np.hypot(*(src[0] - src[1])) < EYES_MIN_PX:
        return None
    M, _ = cv2.estimateAffinePartial2D(src, TEMPLATE)
    return None if M is None else cv2.warpAffine(img, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)


def frame_at(c, st, t):
    c.seek(int(max(t - 0.6, 0) / st.time_base), stream=st, backward=True)
    for fr in c.decode(st):
        if float(fr.pts * st.time_base) >= t - 0.005:
            return pix.ndarray(fr, format='bgr24')
    return None


class SeqReader:
    """Frames at increasing times: decodes on when the next time is near (4K HEVC seeks cost a GOP each:
    the first run took 35 min for 142 faces), seeks when it is far."""
    def __init__(self, path):
        self.c = av.open(path)
        self.st = self.c.streams.video[0]
        self.st.thread_type = 'AUTO'
        self.it, self.t = None, -1e9

    def at(self, t):
        if self.it is None or t < self.t or t - self.t > 3.0:
            self.c.seek(int(max(t - 0.6, 0) / self.st.time_base), stream=self.st, backward=True)
            self.it = self.c.decode(self.st)
        for fr in self.it:
            self.t = float(fr.pts * self.st.time_base)
            if self.t >= t - 0.005:
                return pix.ndarray(fr, format='bgr24')
        self.it = None
        return None

    def close(self):
        self.c.close()


def original_people(model, img):
    """People in a big frame: boxes (n, 4) cx cy w h in pixels, keypoints (n, 17, 3) in pixels."""
    r = model.predict(img, imgsz=1280, conf=0.3, verbose=False)[0]
    if r.boxes is None or not len(r.boxes):
        return np.zeros((0, 4)), np.zeros((0, 17, 3))
    return r.boxes.xywh.cpu().numpy(), r.keypoints.data.cpu().numpy()


class Matcher:
    """Homography from an original frame to its stabilized render frame (ORB on grey, 960 px wide)."""
    def __init__(self):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, orig, rend):
        so = 960.0 / orig.shape[1]
        a = cv2.cvtColor(cv2.resize(orig, None, fx=so, fy=so), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
        ka, da = self.orb.detectAndCompute(a, None)
        kb, db = self.orb.detectAndCompute(b, None)
        if da is None or db is None:
            return None
        good = [m for m, n in (p for p in self.bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 15:
            return None
        H, inl = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]) / so,
                                    np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 4.0)
        return H if H is not None and inl.sum() >= 12 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--originals', required=True)
    ap.add_argument('--check', required=True, help='the label (subject tag) to check by face')
    ap.add_argument('--also', action='append', default=[], help='labels whose rows may turn out to be the person')
    ap.add_argument('--ref', action='append', default=[], help='NAME=CLIP:SECONDS:X or NAME=face.jpg (repeatable)')
    ap.add_argument('--sim', type=float, default=SIM)
    ap.add_argument('--margin', type=float, default=MARGIN)
    ap.add_argument('--not-below', type=float, default=NOT_BELOW,
                    help='a face less like the person than this is someone else (blurred faces sit in between)')
    ap.add_argument('--model', default=SFACE)
    ap.add_argument('--pose', default=POSE)
    ap.add_argument('--ref-dir', default='refs')
    ap.add_argument('--sheet', default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    emb, pose, match = Embedder(a.model), YOLO(a.pose), Matcher()
    renders = sorted({os.path.normcase(p): p for r in a.renders for p in glob.glob(r)}.values())
    originals = [x.strip() for x in open(a.originals, encoding='utf-8') if x.strip()]
    orig_of = {os.path.splitext(os.path.basename(o))[0].lower(): o for o in originals}

    # references
    refs, ref_faces = {}, []
    for spec in a.ref:
        name, what = spec.split('=', 1)
        if os.path.exists(what):
            face = cv2.resize(cv2.imread(what), (112, 112))
        else:
            clip, sec, x = what.rsplit(':', 2)
            c = av.open(orig_of[os.path.splitext(os.path.basename(clip))[0].lower()])
            img = frame_at(c, c.streams.video[0], float(sec))
            c.close()
            bx, kp = original_people(pose, img)
            if not len(bx):
                raise SystemExit(f'{spec}: nobody there')
            i = int(np.argmin(np.abs(bx[:, 0] / img.shape[1] - float(x))))
            face = face_of(img, kp[i])
            if face is None:
                raise SystemExit(f'{spec}: the face is not seen well enough (eyes and nose, {EYES_MIN_PX:.0f} px apart)')
        os.makedirs(a.ref_dir, exist_ok=True)
        cv2.imwrite(os.path.join(a.ref_dir, f'face_{name}_{len(refs.get(name, []))}.jpg'), face)
        refs.setdefault(name, []).append(emb(face))
        ref_faces.append((name, face))
    if a.check not in refs:
        raise SystemExit(f'give a reference of the person: --ref {a.check}=CLIP:SECONDS:X')
    names = list(refs)

    found = {}                                          # (render, row) -> (name or None, sim, face)
    for r in renders:
        base = os.path.splitext(r)[0]
        z = np.load(base + '_people.npz')
        cand = np.load(pp.subject_file(base, a.check)) if os.path.exists(pp.subject_file(base, a.check)) \
            else np.zeros(len(z['frame']), bool)
        for t in a.also:
            if os.path.exists(pp.subject_file(base, t)):
                cand |= np.load(pp.subject_file(base, t))
        if not cand.any():
            continue
        stem = os.path.basename(base).replace('_stabilized', '').lower()
        if stem not in orig_of:
            print(f'{stem}: no original in the list, skipped')
            continue
        cache = {}
        if os.path.exists(base + '_faces.npz'):
            cz = np.load(base + '_faces.npz')
            cache = {int(i): (e, fc) for i, e, fc in zip(cz['row'], cz['emb'], cz['face'])}
            tried = set(cz['tried'].tolist())
        else:
            tried = set()
        fps = float(z['fps'])
        todo = sorted(set(z['frame'][cand & ~np.isin(np.arange(len(cand)), list(tried))].tolist()))
        co, cr = SeqReader(orig_of[stem]), SeqReader(r)
        for f in todo:
            tried |= {int(i) for i in np.flatnonzero(cand & (z['frame'] == f))}
            img, rend = co.at(f / fps), cr.at(f / fps)
            if img is None or rend is None:
                continue
            H = match(img, rend)
            if H is None:
                continue
            bx, kp = original_people(pose, img)
            if not len(bx):
                continue
            # each original person's box centre, top and bottom in the render, normalized
            pts = np.concatenate([np.stack([bx[:, 0], bx[:, 1]], 1), np.stack([bx[:, 0], bx[:, 1] - bx[:, 3] / 2], 1),
                                  np.stack([bx[:, 0], bx[:, 1] + bx[:, 3] / 2], 1)]).astype(np.float32)
            m = cv2.perspectiveTransform(pts[None], H)[0] / np.float32([rend.shape[1], rend.shape[0]])
            k = len(bx)
            centre, height = m[:k], m[2 * k:, 1] - m[k:2 * k, 1]
            for i in np.flatnonzero(cand & (z['frame'] == f)):
                cx, cy, _, h = z['box'][i]
                d = np.hypot(centre[:, 0] - cx, centre[:, 1] - cy)
                j = int(np.argmin(d))
                if d[j] > NEAR or not (0.6 < height[j] / max(h, 1e-6) < 1.6):
                    continue
                face = face_of(img, kp[j])
                if face is not None:
                    cache[int(i)] = (emb(face), face)
        co.close()
        cr.close()
        if cache or tried:
            rows_ = sorted(cache)
            np.savez(base + '_faces.npz', row=np.array(rows_, int), tried=np.array(sorted(tried), int),
                     emb=np.array([cache[i][0] for i in rows_], np.float32).reshape(-1, 128),
                     face=np.array([cache[i][1] for i in rows_], np.uint8).reshape(-1, 112, 112, 3))
        for i, (e, face) in cache.items():
            if not cand[i]:
                continue
            sims = np.array([max(float(e @ v) for v in refs[n]) for n in names])
            o = np.argsort(-sims)
            lead = sims[o[0]] - (sims[o[1]] if len(o) > 1 else 0.0)
            name = names[o[0]] if sims[o[0]] >= a.sim and lead >= a.margin else None
            hero = float(sims[names.index(a.check)])
            if name is None and hero < a.not_below:
                name = '_unlike'                                 # clearly not the person, whoever it is
            found[(r, int(i))] = (name, hero, face)
        print(f'{stem}: {sum(1 for k in found if k[0] == r)} faces ({len(todo)} new samples read)', flush=True)

    # the label, by face, carried along tracks
    lost = won = kept = 0
    for r in renders:
        base = os.path.splitext(r)[0]
        if not os.path.exists(pp.subject_file(base, a.check)):
            continue
        z = np.load(base + '_people.npz')
        lab = np.load(pp.subject_file(base, a.check))
        tid = pp.tracks(z)
        for t in np.unique(tid):
            rows = np.flatnonzero(tid == t)
            said = [found[(r, int(i))][0] for i in rows if (r, int(i)) in found]
            if not said:
                continue
            yes = sum(1 for s in said if s == a.check)
            no = sum(1 for s in said if s is not None and s != a.check)
            if no > yes and lab[rows].any():                     # the track is someone else: all of it
                lost += int(lab[rows].sum())
                lab[rows] = False
            else:                                                # only the rows a face vouches for: tracks
                fps = float(z['fps'])                            # jump between people in FPV footage, so a
                yes_f = [z['frame'][i] for i in rows             # face vouches only for VOUCH_S around it
                         if found.get((r, int(i)), (None,))[0] == a.check]
                for i in rows:
                    v = found.get((r, int(i)))
                    near = any(abs(int(z['frame'][i]) - int(f)) <= VOUCH_S * fps for f in yes_f)
                    if (v and v[0] == a.check) or (v is None and near):
                        won += int(not lab[i])
                        kept += int(lab[i])
                        lab[i] = True
                    elif v and v[0] is not None:
                        lost += int(lab[i])
                        lab[i] = False
        np.save(pp.subject_file(base, a.check), lab)
    names_seen = [v[0] for v in found.values()]
    print(f'faces {a.check}: {len(found)} faces read; {names_seen.count(a.check)} are {a.check}, '
          + ', '.join(f'{names_seen.count(n)} {n}' for n in names if n != a.check)
          + f', {names_seen.count("_unlike")} clearly not, {names_seen.count(None)} unsure; label: {lost} rows dropped, {won} added, {kept} confirmed')
    if a.sheet:
        rows = []
        for name in names + ['_unlike', None]:
            got = sorted(((v[1], v[2]) for v in found.values() if v[0] == name), key=lambda x: -x[0])[:15]
            tiles = [f for n, f in ref_faces if n == name][:1] + [f for _, f in got]
            tiles += [np.zeros((112, 112, 3), np.uint8)] * (16 - len(tiles))
            row = np.hstack(tiles[:16])
            cv2.putText(row, str(name or 'unsure'), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(row)
        cv2.imwrite(a.sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()
