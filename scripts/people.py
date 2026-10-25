"""People in the picture, person by person: pose (YOLOv8-pose), who the pilot is, close passes.

    python people.py detect --weights models/yolo/yolov8s-pose.pt RENDER.mp4 [...]   -> <render>_people.npz
    python people.py pilot  RENDER.mp4 [...] [--sheet check.jpg]                    -> <render>_pilot.npy
    python people.py passes --weights ... RENDER.mp4 [...]                           -> <render>_passes.npz
    python people.py subject RENDER.mp4 [...] --text "a woman pushing a baby stroller" [--text ...] [--not ...] [--tag NAME]
                    [--sheet check.jpg]
    python people.py who RENDER.mp4 [...] --ref NAME=IMAGE.jpg [--ref NAME=IMAGE2.jpg ...] [--sheet check.jpg]
    python people.py split RENDER.mp4 [...] --tag NAME --k 3 --sheet looks.jpg     then  --keep 0,2
    python people.py summary RENDER.mp4 [...]

detect   2 samples a second; per person: box, 17 keypoints, a colour print of the clothes (top and
         bottom half: hue histogram weighted by saturation, value histogram, mean saturation).
pilot    The pilot stands still holding the radio, wears goggles and is in nearly every flight. Clothes
         prints of all clips are clustered; the cluster that most often holds a radio (wrists together in
         front of the torso) and shows up in most clips is the pilot. Every person close to it in clothes,
         or holding a radio, is flagged. --sheet saves crops of the pilot cluster and of the others, to
         confirm by eye once.
passes   Its own scan at 10 Hz (a pass takes tenths of a second) for "a big person, then gone"; around
         each, pose on every frame: the last visible frames tell whether the quad went between the legs
         (legs left and right of the centre, hips above it), under (out through the top edge, feet above
         the centre), over (out through the bottom, head below it), or past on one side; "(lying)" when
         the body is horizontal (under a plank). --sheet: one row of frames per pass, to check by eye.
pilotclip  CLIP zero-shot on every person crop: the pilot wears FPV goggles or holds a radio. People whose
         hands are busy in front of them with something else (a pram, a phone, a bag, a child's hand)
         look like a pilot to the pose check, so they are listed among the other people CLIP chooses from.
subject  The film's main character, described in words (--text, from the task): CLIP zero-shot on every
         person crop, widened to take in what is beside or in front of them (a pram, a dog, a bike)
         -> <render>_subject.npy (<render>_subject_NAME.npy with --tag NAME, so one shoot can have
         several main characters, one per edit); tracks carry the label like the pilot's.
who      Main characters by example: a crop of each person (--ref NAME=IMAGE, several per name allowed).
         Every person row goes to the one it resembles most in CLIP (whole look: clothes, hair, build),
         or to nobody; the best finds widen the references (other sides of the person) and rows are
         assigned again -> <render>_subject_NAME.npy for each NAME (edit.py --subject NAME). For telling
         apart people who look alike in words: "the girl in the green tracksuit" vs her friends. Give the
         others too, as --ref _NAME=IMAGE (the coach in black took the small girl in black otherwise).
         --ref NAME=CLIP:SECONDS:X cuts the reference from the original clip: the person nearest to X (0..1
         across the frame) at that moment, as on a screenshot the user sends (--originals clip list).
         --ref NAME=@TAG takes the rows labelled TAG as references: after `split --keep`, a second `who`
         with --ref kid=@kid --ref _alike=@kid_not finds the child again, minus the look-alikes.
split    The rows of one main character (--tag) in --k groups by look, a sheet row per group; --keep lists
         the groups that are that person, the rest (children dressed the same, a parent) are dropped.
summary  One line per clip: people other than the pilot, pilot share, waving, facing the camera, passes,
         the subject.
Raw boxes and keypoints are never printed.
"""
import argparse
import glob
import os
import time

import av
import cv2
import numpy as np

import gpu

NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(17)
KP_OK = 0.35


# ------------------------------------------------------------------------------- per-person data
def clothes_print(img, box):
    """Colour print of a person crop: per half, 12 hue bins (weighted by saturation), 4 value bins, mean sat."""
    H, W = img.shape[:2]
    cx, cy, w, h = box
    x0, x1 = int(max((cx - w / 2) * W, 0)), int(min((cx + w / 2) * W, W))
    y0, y1 = int(max((cy - h / 2) * H, 0)), int(min((cy + h / 2) * H, H))
    out = []
    for a, b in ((0.15, 0.5), (0.5, 0.95)):
        ya, yb = y0 + int(a * (y1 - y0)), y0 + int(b * (y1 - y0))
        crop = img[ya:max(yb, ya + 1), x0:max(x1, x0 + 1)]
        if crop.size == 0:
            out.append(np.zeros(17))
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        hh = np.histogram(hsv[:, 0], bins=12, range=(0, 180), weights=hsv[:, 1])[0]
        hh = hh / (hh.sum() + 1e-6)
        vv = np.histogram(hsv[:, 2], bins=4, range=(0, 256))[0].astype(np.float32)
        vv = vv / (vv.sum() + 1e-6)
        out.append(np.concatenate([hh, vv, [hsv[:, 1].mean() / 255.0]]))
    return np.concatenate(out)


def flags(kp):
    """kp (17, 3) normalized x, y, conf -> holding a radio, waving, facing the camera, eyes hidden."""
    ok = kp[:, 2] > KP_OK
    hold = wave = face = goggles = False
    if ok[LSH] and ok[RSH]:
        sw = abs(kp[LSH, 0] - kp[RSH, 0]) + 1e-6
        sy = (kp[LSH, 1] + kp[RSH, 1]) / 2
        if ok[LWR] and ok[RWR]:
            hy = (kp[LHIP, 1] + kp[RHIP, 1]) / 2 if ok[LHIP] and ok[RHIP] else sy + 2.5 * sw
            together = np.hypot(*(kp[LWR, :2] - kp[RWR, :2])) < 0.8 * sw
            front = sy < (kp[LWR, 1] + kp[RWR, 1]) / 2 < hy + 0.3 * sw
            hold = bool(together and front)
        wave = bool((ok[LWR] and kp[LWR, 1] < sy - 0.2 * sw) or (ok[RWR] and kp[RWR, 1] < sy - 0.2 * sw))
    face = bool(ok[NOSE] and ok[LEYE] and ok[REYE])
    goggles = bool(ok[NOSE] and (ok[LEAR] or ok[REAR]) and not (ok[LEYE] or ok[REYE]))
    return hold, wave, face, goggles


def run_pose(model, imgs):
    """-> list per image of (boxes n x 4 normalized cx cy w h, conf n, kp n x 17 x 3 normalized)."""
    out = []
    for b0 in range(0, len(imgs), 16):
        for r, img in zip(model.predict(imgs[b0:b0 + 16], imgsz=640, conf=0.35, verbose=False), imgs[b0:b0 + 16]):
            H, W = img.shape[:2]
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))))
                continue
            bx = r.boxes.xywh.cpu().numpy() / np.array([W, H, W, H])
            kp = r.keypoints.data.cpu().numpy().copy()
            kp[..., 0] /= W
            kp[..., 1] /= H
            out.append((bx, r.boxes.conf.cpu().numpy(), kp))
    return out


RADIO_CLASSES = [65, 67]     # COCO "remote", "cell phone": what a radio controller usually passes for


def radio_flags(obj_model, imgs, poses):
    """Per image, per person: a remote / phone inside the person's box, near the wrists if they are seen."""
    out = []
    res = []
    for b0 in range(0, len(imgs), 16):
        res += obj_model.predict(imgs[b0:b0 + 16], imgsz=640, conf=0.25, classes=RADIO_CLASSES, verbose=False)
    for r, img, (bx, cf, kp) in zip(res, imgs, poses):
        H, W = img.shape[:2]
        objs = r.boxes.xywh.cpu().numpy() / np.array([W, H, W, H]) if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        fl = []
        for b, k in zip(bx, kp):
            hit = False
            for o in objs:
                inside = abs(o[0] - b[0]) < b[2] / 2 and abs(o[1] - b[1]) < b[3] / 2
                wr = [k[j, :2] for j in (LWR, RWR) if k[j, 2] > KP_OK]
                near = (not wr) or min(np.hypot(*(o[:2] - w)) for w in wr) < 0.35 * b[3]
                hit |= bool(inside and near)
            fl.append(hit)
        out.append(fl)
    return out


def detect(model, src, every, obj_model=None):
    c = av.open(src)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    samples, imgs = [], []
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i % every == 0:
            samples.append(i)
            imgs.append(fr.to_ndarray(format='bgr24'))
    c.close()
    rows = dict(frame=[], box=[], conf=[], kp=[], app=[], hold=[], wave=[], face=[], goggles=[], radio=[])
    poses = run_pose(model, imgs)
    radios = radio_flags(obj_model, imgs, poses) if obj_model is not None else [[False] * len(p[0]) for p in poses]
    for f, img, (bx, cf, kp), rad in zip(samples, imgs, poses, radios):
        for b, c_, k, rd in zip(bx, cf, kp, rad):
            rows['frame'].append(f)
            rows['box'].append(b)
            rows['conf'].append(c_)
            rows['kp'].append(k.reshape(-1))
            rows['app'].append(clothes_print(img, b))
            rows['radio'].append(rd)
            for name, v in zip(('hold', 'wave', 'face', 'goggles'), flags(k)):
                rows[name].append(v)
    arr = {k: np.array(v, dtype=np.float32 if k not in ('frame',) else np.int64) for k, v in rows.items()}
    if not rows['frame']:
        arr.update(box=np.zeros((0, 4), np.float32), kp=np.zeros((0, 51), np.float32), app=np.zeros((0, 34), np.float32))
    return dict(samples=np.array(samples), fps=fps, **arr)


# ------------------------------------------------------------------------------- who is the pilot
def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):                                        # k-means++ seeding
        d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
    return C, lab


def find_pilot(renders, k=6, sheet=None):
    data = [(r, np.load(os.path.splitext(r)[0] + '_people.npz')) for r in renders]
    X, src, hold, size = [], [], [], []
    for ci, (r, z) in enumerate(data):
        big = z['box'][:, 3] > 0.12 if len(z['frame']) else np.zeros(0, bool)
        X.append(z['app'][big])
        src += [ci] * int(big.sum())
        hold.append(np.maximum(z['hold'][big], z['radio'][big]))     # holding a radio: pose, or a radio seen
        size.append(z['box'][big, 3])
    X, src, hold = np.concatenate(X), np.array(src), np.concatenate(hold)
    if len(X) < k * 3:
        print('too few people to find the pilot')
        return None
    C, lab = kmeans(X, k)
    stats = []
    for j in range(k):
        m = lab == j
        clips = len(set(src[m].tolist())) / len(data)
        stats.append((j, int(m.sum()), clips, float(hold[m].mean()) if m.any() else 0.0))
    score = [s[2] * s[3] for s in stats]
    pj = int(np.argmax(score))
    others = sorted(score)[-2] if k > 1 else 0
    radius = np.percentile(np.sqrt(((X[lab == pj] - C[pj]) ** 2).sum(-1)), 80)
    print('clothes clusters (size, share of clips, holding a radio):')
    for j, n, cl, hr in stats:
        print(f'  #{j}: {n:4d} people, in {cl * 100:3.0f} % of clips, holding {hr * 100:3.0f} %{"  <- pilot" if j == pj else ""}')
    if score[pj] < 1.5 * others + 0.02:
        print('  (weak: no cluster clearly holds a radio more than the others)')
    for r, z in data:                                            # flag every row
        if not len(z['frame']):
            np.save(os.path.splitext(r)[0] + '_pilot.npy', np.zeros(0, bool))
            continue
        d = np.sqrt(((z['app'] - C[pj]) ** 2).sum(-1))
        nearest = np.argmin(((z['app'][:, None, :] - C[None]) ** 2).sum(-1), 1)
        pilot = ((nearest == pj) & (d < 1.5 * radius)) | (z['hold'] > 0) | (z['radio'] > 0)
        np.save(os.path.splitext(r)[0] + '_pilot.npy', pilot)
    if sheet:
        crops = {True: [], False: []}
        for r, z in data:
            if not len(z['frame']):
                continue
            pil = np.load(os.path.splitext(r)[0] + '_pilot.npy')
            order = np.argsort(-z['box'][:, 3])
            for i in order[:40]:
                crops[bool(pil[i])].append((r, int(z['frame'][i]), z['box'][i]))
        rows = []
        for is_p in (True, False):
            tiles = []
            for r, f, b in crops[is_p][::max(1, len(crops[is_p]) // 10)][:10]:
                c = av.open(r)
                st = c.streams.video[0]
                fps = float(st.average_rate)
                c.seek(int(max(f - 2, 0) / fps / st.time_base), stream=st, backward=True)
                for fr in c.decode(st):
                    if int(round(float(fr.pts * st.time_base) * fps)) >= f:
                        img = fr.to_ndarray(format='bgr24')
                        break
                c.close()
                H, W = img.shape[:2]
                x0, x1 = int(max((b[0] - b[2] / 2) * W, 0)), int(min((b[0] + b[2] / 2) * W, W))
                y0, y1 = int(max((b[1] - b[3] / 2) * H, 0)), int(min((b[1] + b[3] / 2) * H, H))
                tiles.append(cv2.resize(img[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)], (96, 160)))
            while len(tiles) < 10:
                tiles.append(np.zeros((160, 96, 3), np.uint8))
            row = np.hstack(tiles)
            cv2.putText(row, 'PILOT' if is_p else 'others', (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(row)
        cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return pj


# ------------------------------------------------------------------------------- the pilot, by meaning (CLIP)
PILOT_TEXTS = ['a person wearing FPV drone goggles over the eyes', 'a man wearing a video headset that covers his eyes',
               'a person holding a drone radio controller with both hands']
OTHER_TEXTS = ['a child playing', 'a girl in a jacket', 'a woman', 'people walking on a beach', 'a person with a bare face',
               'a backlit dark silhouette of a person',
               # not the pilot, though their hands are busy in front of them like a pilot's
               'a person pushing a baby stroller', 'a person looking at a phone in their hands',
               'a person carrying a bag', 'a person holding a child by the hand',
               'a person pushing a cart or carrying a basket', 'a person holding a sports racket',
               'a person holding a ball']
NOT_PERSON_TEXTS = ['a tree trunk', 'sand and snow on the ground', 'a blurry patch of sky', 'rocks on a beach']


def crops_at(render, z, pad=0.1, pad_y=None):
    """PIL crops of every person row, from the sample frames of the render (pad: share of the box
    added on each side; pad_y for the height, default the same)."""
    from PIL import Image
    want = {}
    for i, f in enumerate(z['frame']):
        want.setdefault(int(f), []).append(i)
    out = [None] * len(z['frame'])
    c = av.open(render)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    for fr in c.decode(st):
        f = int(round(float(fr.pts * st.time_base) * fps))
        if f not in want:
            continue
        img = fr.to_ndarray(format='rgb24')
        H, W = img.shape[:2]
        for i in want[f]:
            cx, cy, w, h = z['box'][i]
            x0, x1 = int(max((cx - w / 2 - pad * w) * W, 0)), int(min((cx + w / 2 + pad * w) * W, W))
            py = pad if pad_y is None else pad_y
            y0, y1 = int(max((cy - h / 2 - py * h) * H, 0)), int(min((cy + h / 2 + py * h) * H, H))
            out[i] = Image.fromarray(img[y0:max(y1, y0 + 4), x0:max(x1, x0 + 4)])
    c.close()
    return out


def clip_probs(cache):
    """CLIP ViT-B/32: a function (PIL crops, texts) -> softmax over the texts per crop, (N, len(texts))."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = gpu.device()
    name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(name, cache_dir=cache).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(name, cache_dir=cache)

    def feats(out):
        out = out if torch.is_tensor(out) else getattr(out, 'pooler_output', out[0])
        return out / out.norm(dim=-1, keepdim=True)

    def probs(crops, texts):
        with torch.no_grad():
            tf = feats(model.get_text_features(**proc(text=texts, return_tensors='pt', padding=True).to(dev)))
        out = np.zeros((len(crops), len(texts)), np.float32)
        for b0 in range(0, len(crops), 64):
            batch = crops[b0:b0 + 64]
            with torch.no_grad():
                inp = proc(images=batch, return_tensors='pt').to(dev)
                out[b0:b0 + len(batch)] = (100.0 * feats(model.get_image_features(**inp)) @ tf.T).softmax(-1).cpu().numpy()
        return out
    return probs


def clip_scores(renders, cache):
    """Per person row: probability (CLIP, zero-shot) that it is the pilot -> <render>_pilotclip.npy."""
    probs = clip_probs(cache)
    n_p, n_o = len(PILOT_TEXTS), len(OTHER_TEXTS)
    for r in renders:
        base = os.path.splitext(r)[0]
        z = np.load(base + '_people.npz')
        p = probs(crops_at(r, z), PILOT_TEXTS + OTHER_TEXTS + NOT_PERSON_TEXTS)
        np.save(base + '_pilotclip.npy', p[:, :n_p].sum(-1))
        np.save(base + '_notperson.npy', p[:, n_p + n_o:].sum(-1))     # YOLO said person, CLIP says sand / sky / tree


# ------------------------------------------------------------------------------- the main character, in words (CLIP)
# what the subject is told apart from (texts equal to the subject's own are dropped)
SUBJECT_OTHER_TEXTS = ['a person walking alone', 'a person standing still', 'people talking to each other',
                       'a child playing', 'a person with a dog']


def subject_file(base, tag=''):
    return base + '_subject' + (f'_{tag}' if tag else '') + '.npy'


def find_subject(renders, cache, texts, row_thr=0.5, track_frac=0.3, sheet=None, tag='', nots=()):
    """Per person row: is it the film's main character described by `texts` -> <render>_subject[_tag].npy (bool).
    The crop is widened (0.8 of the box width each side, 0.15 of the height) to take in what the person
    has beside or in front of them. `nots`: who the main character is not, in words from the task (the
    others around them in the same place: "a child with a tennis racket" next to the coach); generic
    alternatives alone lose to a subject text that matches the place."""
    probs = clip_probs(cache)
    neg = [t for t in list(nots) + SUBJECT_OTHER_TEXTS + PILOT_TEXTS + NOT_PERSON_TEXTS if t not in texts]
    tot = dict(rows=0, by_clip=0, by_track=0)
    picks = []
    for r in renders:
        base = os.path.splitext(r)[0]
        z = np.load(base + '_people.npz')
        if not len(z['frame']):
            np.save(subject_file(base, tag), np.zeros(0, bool))
            continue
        p = probs(crops_at(r, z, pad=0.8, pad_y=0.15), list(texts) + neg)[:, :len(texts)].sum(-1)
        strong = p > row_thr
        tid = tracks(z)
        sub = strong.copy()
        for t in np.unique(tid):
            m = tid == t
            if m.sum() >= 2 and strong[m].mean() >= track_frac:
                sub |= m
        np.save(subject_file(base, tag), sub)
        tot['rows'] += len(sub)
        tot['by_clip'] += int(strong.sum())
        tot['by_track'] += int((sub & ~strong).sum())
        picks += [(float(p[i]), r, int(i)) for i in np.flatnonzero(strong)]
    print(f'subject {texts} (CLIP + tracks): {tot["by_clip"]} rows by CLIP, {tot["by_track"]} more by following '
          f'their tracks, of {tot["rows"]} people rows')
    if sheet and picks:                                  # the most and the least confident picks, to check by eye
        picks.sort(reverse=True)
        sel = picks[:12] + picks[-12:] if len(picks) > 24 else picks
        tiles = []
        for _, r, i in sel:
            z = np.load(os.path.splitext(r)[0] + '_people.npz')
            one = {k: z[k][i:i + 1] for k in ('frame', 'box')}
            tiles.append(np.asarray(crops_at(r, one, pad=0.8, pad_y=0.15)[0].resize((120, 120))))
        blank = np.zeros_like(tiles[0])
        rows = [np.hstack(tiles[k:k + 12] + [blank] * (12 - len(tiles[k:k + 12]))) for k in range(0, len(tiles), 12)]
        cv2.imwrite(sheet, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def tracks(z, max_gap=2, dist=0.12):
    """Greedy tracks over the samples: a person keeps its id if one in the next samples sits close by."""
    samples = z['samples']
    k = np.searchsorted(samples, z['frame'])
    tid = -np.ones(len(k), int)
    live = []                                                    # (track id, last sample index, cx, cy, h)
    nxt = 0
    for s in range(len(samples)):
        rows = np.flatnonzero(k == s)
        live = [t for t in live if s - t[1] <= max_gap]
        used = set()
        for i in rows[np.argsort(-z['box'][rows, 3])]:
            cx, cy, _, h = z['box'][i]
            best = None
            for j, (t, ls, tx, ty, th) in enumerate(live):
                if j in used or not (0.5 < h / max(th, 1e-6) < 2.0):
                    continue
                d = np.hypot(cx - tx, cy - ty)
                if d < dist * (s - ls) ** 0.5 + 0.02 and (best is None or d < best[0]):
                    best = (d, j)
            if best is None:
                tid[i] = nxt
                live.append((nxt, s, cx, cy, h))
                used.add(len(live) - 1)
                nxt += 1
            else:
                j = best[1]
                tid[i] = live[j][0]
                live[j] = (live[j][0], s, cx, cy, h)
                used.add(j)
    return tid


def find_pilot_clip(renders, sheet=None, row_thr=0.5, track_frac=0.3):
    tot = dict(rows=0, by_clip=0, by_track=0, by_radio=0)
    for r in renders:
        base = os.path.splitext(r)[0]
        z = np.load(base + '_people.npz')
        if not len(z['frame']):
            np.save(base + '_pilot.npy', np.zeros(0, bool))
            continue
        p = np.load(base + '_pilotclip.npy')
        notp = np.load(base + '_notperson.npy') if os.path.exists(base + '_notperson.npy') else np.zeros(len(p))
        p = np.where(notp > 0.5, 0.0, p)
        strong = (p > row_thr) | ((z['radio'] > 0) & (z['hold'] > 0))
        tid = tracks(z)
        pilot = strong.copy()
        for t in np.unique(tid):
            m = tid == t
            if m.sum() >= 2 and strong[m].mean() >= track_frac:
                pilot |= m
        np.save(base + '_pilot.npy', pilot)
        tot['rows'] += len(pilot)
        tot['by_clip'] += int((p > row_thr).sum())
        tot['by_radio'] += int(((z['radio'] > 0) & (z['hold'] > 0)).sum())
        tot['by_track'] += int((pilot & ~strong).sum())
    print(f'pilot (CLIP + tracks): {tot["by_clip"]} rows by CLIP, {tot["by_radio"]} by radio seen in hands, '
          f'{tot["by_track"]} more by following their tracks, of {tot["rows"]} people rows')
    if sheet:
        pilot_sheet(renders, sheet)


def pilot_sheet(renders, sheet):
    crops = {True: [], False: []}
    for r in renders:
        base = os.path.splitext(r)[0]
        z = np.load(base + '_people.npz')
        if not len(z['frame']):
            continue
        pil = np.load(base + '_pilot.npy')
        for i in np.argsort(-z['box'][:, 3])[:40]:
            crops[bool(pil[i])].append((r, z, i))
    rows = []
    for is_p in (True, False):
        pick = crops[is_p][::max(1, len(crops[is_p]) // 10)][:10]
        tiles = []
        for r, z, i in pick:
            im = crops_at(r, {'frame': z['frame'][i:i + 1], 'box': z['box'][i:i + 1]})[0]
            tiles.append(cv2.resize(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), (96, 160)))
        while len(tiles) < 10:
            tiles.append(np.zeros((160, 96, 3), np.uint8))
        row = np.hstack(tiles)
        cv2.putText(row, 'PILOT' if is_p else 'others', (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows.append(row)
    cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ------------------------------------------------------------------------------- the main characters, by example
WHO_SIM, WHO_MARGIN, WHO_EXPAND = 0.80, 0.03, 12   # cosine to a person's references, lead over the next person,
                                                 # most confident finds added as references (other views)


def cut_ref(original, sec, x, out_path, weights):
    """A reference crop from the original clip (the full field of view, as on a screenshot of it): the person
    whose box centre is nearest to x (share of the frame width) at `sec` seconds -> out_path."""
    from ultralytics import YOLO
    c = av.open(original)
    st = c.streams.video[0]
    c.seek(int(max(sec - 1.0, 0) / st.time_base), stream=st, backward=True)
    img = None
    for fr in c.decode(st):
        if float(fr.pts * st.time_base) >= sec:
            img = fr.to_ndarray(format='bgr24')
            break
    c.close()
    if img is None:
        raise SystemExit(f'{original}: no frame at {sec} s')
    r = YOLO(weights).predict(img, imgsz=1280, conf=0.3, classes=[0], verbose=False)[0]
    H, W = img.shape[:2]
    bx = r.boxes.xywh.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
    if not len(bx):
        raise SystemExit(f'{original} {sec} s: nobody found')
    cx, cy, w, h = bx[int(np.argmin(np.abs(bx[:, 0] / W - x)))]
    cv2.imwrite(out_path, img[int(max(cy - h / 2, 0)):int(cy + h / 2), int(max(cx - w / 2, 0)):int(cx + w / 2)])
    return out_path


def clip_image_feats(cache):
    """CLIP ViT-B/32: a function (PIL images) -> unit image embeddings (N, 512)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = gpu.device()
    name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(name, cache_dir=cache).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(name, cache_dir=cache)

    def run(images):
        out = np.zeros((len(images), 512), np.float32)
        for b0 in range(0, len(images), 64):
            with torch.no_grad():
                f = model.get_image_features(**proc(images=images[b0:b0 + 64], return_tensors='pt').to(dev))
                f = f if torch.is_tensor(f) else getattr(f, 'pooler_output', f[0])
                out[b0:b0 + 64] = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
        return out
    return run


def person_embeddings(renders, feats):
    """Per render, the CLIP embedding of every person row (tight crop) -> cached <render>_pclip.npy."""
    out = {}
    for r in renders:
        base = os.path.splitext(r)[0]
        z = np.load(base + '_people.npz')
        if os.path.exists(base + '_pclip.npy') and len(np.load(base + '_pclip.npy')) == len(z['frame']):
            out[r] = np.load(base + '_pclip.npy').astype(np.float32)
            continue
        e = feats(crops_at(r, z, pad=0.05)) if len(z['frame']) else np.zeros((0, 512), np.float32)
        np.save(base + '_pclip.npy', e.astype(np.float16))
        out[r] = e
    return out


def find_who(renders, cache, refs, sheet=None, sim=WHO_SIM, margin=WHO_MARGIN):
    """refs: {name: [image paths]}. Every person row goes to the reference person it resembles most (CLIP
    cosine at least `sim`, ahead of the next person by `margin`), or to nobody; the pilot is nobody. The
    WHO_EXPAND most confident rows of each person then join their references (a back view, a side view)
    and rows are assigned again. Tracks carry the label. -> <render>_subject_<name>.npy for each name.
    Names starting with '_' are other people (the coach, a parent): rows that look more like them go to
    nobody, and no file is written for them."""
    from PIL import Image
    feats = clip_image_feats(cache)
    names = list(refs)
    emb = person_embeddings(renders, feats)

    def labelled(tag, most=60):
        """Embeddings of the rows labelled `tag` so far (the biggest crops first)."""
        got = []
        for r in renders:
            b = os.path.splitext(r)[0]
            if os.path.exists(subject_file(b, tag)):
                z = np.load(b + '_people.npz')
                lab = np.load(subject_file(b, tag))
                got += [(float(z['box'][i, 2] * z['box'][i, 3]), emb[r][i]) for i in np.flatnonzero(lab)]
        got.sort(key=lambda g: -g[0])
        return [e for _, e in got[:most]]
    ref = {}
    for n in names:
        es = [feats([Image.open(p).convert('RGB')])[0] for p in refs[n] if not p.startswith('@')]
        for p in refs[n]:
            if p.startswith('@'):
                es += labelled(p[1:])
        ref[n] = np.array(es, np.float32).reshape(-1, 512)
    rows = [(r, i) for r in renders for i in range(len(emb[r]))]
    E = np.concatenate([emb[r] for r in renders]) if rows else np.zeros((0, 512), np.float32)
    pil = np.concatenate([np.load(os.path.splitext(r)[0] + '_pilot.npy') if os.path.exists(os.path.splitext(r)[0] + '_pilot.npy')
                          else np.zeros(len(emb[r]), bool) for r in renders]) if rows else np.zeros(0, bool)
    for it in range(2):
        S = np.stack([np.max(E @ ref[n].T, axis=1) for n in names], 1)        # best match among each person's refs
        top = np.argmax(S, 1)
        srt = np.sort(S, 1)
        lead = srt[:, -1] - (srt[:, -2] if len(names) > 1 else 0)
        ok = (srt[:, -1] >= sim) & (lead >= margin) & ~pil
        if it == 0:                                                         # widen the references once
            for k, n in enumerate(names):
                cand = np.flatnonzero(ok & (top == k))
                best = cand[np.argsort(-S[cand, k])][:WHO_EXPAND]
                ref[n] = np.concatenate([ref[n], E[best]])
    tot = {n: 0 for n in names}
    picks = {n: [] for n in names}
    off = 0
    for r in renders:
        base = os.path.splitext(r)[0]
        z = np.load(base + '_people.npz')
        m = len(emb[r])
        tid = tracks(z) if m else np.zeros(0, int)
        for k, n in enumerate(names):
            if n.startswith('_'):
                continue
            strong = ok[off:off + m] & (top[off:off + m] == k)
            lab = strong.copy()
            for t in np.unique(tid):
                mm = tid == t
                if mm.sum() >= 2 and strong[mm].mean() >= 0.3:
                    # only rows that still look most like this person: in fast FPV footage a track jumps
                    # from one person to the next (a girl's track ended on a boy's close-up)
                    lab |= mm & ~pil[off:off + m] & (top[off:off + m] == k)
            np.save(subject_file(base, n), lab)
            tot[n] += int(lab.sum())
            picks[n] += [(float(S[off + i, k]), r, int(i)) for i in np.flatnonzero(strong)]
        off += m
    names = [n for n in names if not n.startswith('_')]
    for n in names:
        secs = sum(len(set(np.load(os.path.splitext(r)[0] + '_people.npz')['frame'][np.load(subject_file(os.path.splitext(r)[0], n))].tolist()))
                   for r in renders) * 0.5
        print(f'who {n}: {tot[n]} person rows, in view ~{secs:.0f} s (2 samples a second)')
    if sheet:
        out = []
        for n in names:
            pk = sorted(picks[n], reverse=True)
            sel = (pk[:8] + pk[len(pk) // 2:len(pk) // 2 + 4] + pk[-8:]) if len(pk) > 20 else pk
            tiles = []
            for _, r, i in sel:
                z = np.load(os.path.splitext(r)[0] + '_people.npz')
                im = crops_at(r, {'frame': z['frame'][i:i + 1], 'box': z['box'][i:i + 1]}, pad=0.1)[0]
                tiles.append(cv2.cvtColor(np.asarray(im.resize((72, 128))), cv2.COLOR_RGB2BGR))
            tiles = [cv2.resize(cv2.imread(refs[n][0]), (72, 128))] + tiles
            tiles += [np.zeros((128, 72, 3), np.uint8)] * (21 - len(tiles))
            row = np.hstack(tiles[:21])
            cv2.putText(row, n, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            out.append(row)
        cv2.imwrite(sheet, np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 85])


def split_looks(renders, tag, k, sheet=None, keep=None):
    """The rows labelled `tag` (who / subject) grouped into k looks: CLIP embedding plus the clothes colours
    (top and bottom half), k-means with a fixed seed, so the same call gives the same groups. --sheet: one
    row of crops per group, to see which groups are that person and which are someone dressed alike.
    --keep 0,2: only those groups stay labelled (the others were look-alikes, saved as tag_not: a negative
    for the next `who`, --ref _alike=@tag_not)."""
    rows, X = [], []
    for r in renders:
        b = os.path.splitext(r)[0]
        if not (os.path.exists(subject_file(b, tag)) and os.path.exists(b + '_pclip.npy')):
            continue
        z = np.load(b + '_people.npz')
        lab = np.load(subject_file(b, tag))
        e = np.load(b + '_pclip.npy').astype(np.float32)
        for i in np.flatnonzero(lab):
            rows.append((r, int(i), float(z['box'][i, 2] * z['box'][i, 3])))
            X.append(np.concatenate([e[i], 0.5 * z['app'][i]]))
    if len(rows) < k:
        print(f'split {tag}: only {len(rows)} rows')
        return
    X = np.array(X)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    _, grp = kmeans(X, k)
    for g in range(k):
        print(f'split {tag}: group {g}: {int((grp == g).sum())} rows')
    if sheet:
        want = {}                                                # render -> [(group, tile, row)]
        for g in range(k):
            idx = np.flatnonzero(grp == g)
            near = idx[np.argsort(((X[idx] - X[idx].mean(0)) ** 2).sum(1))][:40]
            for t, j in enumerate(sorted(near, key=lambda j: -rows[j][2])[:14]):
                want.setdefault(rows[j][0], []).append((g, t, rows[j][1]))
        tiles = {}
        for r, items in want.items():
            z = np.load(os.path.splitext(r)[0] + '_people.npz')
            ii = np.array([i for _, _, i in items])
            for (g, t, _), im in zip(items, crops_at(r, {'frame': z['frame'][ii], 'box': z['box'][ii]}, pad=0.05)):
                tiles[(g, t)] = cv2.cvtColor(np.asarray(im.resize((72, 128))), cv2.COLOR_RGB2BGR)
        out = []
        for g in range(k):
            row = np.hstack([tiles.get((g, t), np.zeros((128, 72, 3), np.uint8)) for t in range(14)])
            cv2.putText(row, f'{tag} group {g}: {int((grp == g).sum())}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
            out.append(row)
        cv2.imwrite(sheet, np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if keep is not None:
        drop = {(r, i) for (r, i, _), g in zip(rows, grp) if g not in keep}
        for r in renders:
            b = os.path.splitext(r)[0]
            if not os.path.exists(subject_file(b, tag)):
                continue
            lab = np.load(subject_file(b, tag))
            nope = np.zeros(len(lab), bool)
            for i in np.flatnonzero(lab):
                if (r, int(i)) in drop:
                    lab[i], nope[i] = False, True
            np.save(subject_file(b, tag), lab)
            np.save(subject_file(b, tag + '_not'), nope)
        print(f'split {tag}: kept groups {sorted(keep)}, {len(rows) - len(drop)} of {len(rows)} rows')


# ------------------------------------------------------------------------------- close passes
# A pass takes a few tenths of a second, so the 2-a-second samples of `detect` mostly miss it: `passes` scans
# PASS_SCAN_HZ itself. A candidate is a big person (box area PASS_AREA, or PASS_H of the frame height, or
# PASS_W of its width; YOLO confidence PASS_CONF, as nets and ball baskets pass for people at lower) who is
# not there in the next sample. "Who" is followed by box overlap, not as "the biggest person": right after
# the quad goes under someone, the next person is often just as big (the coach in a plank, then a child).
# Every candidate is then followed frame by frame; it is a pass when that person, at full size somewhere,
# is lost for PASS_LOST frames, or shrinks below PASS_GONE of their size, within PASS_WITHIN_S. The scan
# takes candidates from PASS_SCAN_MIN of the full size: up close YOLO flickers, and the coach in a plank was
# big for 0.16 s, none of it on a 10 Hz sample.
PASS_SCAN_HZ, PASS_AREA, PASS_H, PASS_W, PASS_CONF, PASS_GONE, PASS_WITHIN_S = 10.0, 0.12, 0.6, 0.5, 0.5, 0.3, 0.6
PASS_IOU, PASS_LOST, PASS_SCAN_MIN = 0.2, 3, 0.6
PASS_EXIT = 0.12        # the box edge moves this much (of the frame) towards the edge it leaves through


def box_size(b):
    """1 = just big enough to be passed close."""
    return float(max(b[2] * b[3] / PASS_AREA, b[3] / PASS_H, b[2] / PASS_W))


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    w, h = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    return w * h / (a[2] * a[3] + b[2] * b[3] - w * h + 1e-9)


def match(box, boxes):
    """Index of the box in `boxes` that is the same person as `box` (best overlap), or -1."""
    if not len(boxes):
        return -1
    ov = [iou(box, b) for b in boxes]
    j = int(np.argmax(ov))
    return j if ov[j] >= PASS_IOU else -1


def lying(b, kp, aspect):
    """The body is horizontal (a plank, a push-up, lying on the ground): shoulders to ankles (or hips) run
    more across the picture than down it; without those keypoints, a box much wider than tall."""
    ok = kp[:, 2] > KP_OK
    top = [kp[j, :2] for j in (LSH, RSH) if ok[j]]
    low = [kp[j, :2] for j in (LAN, RAN) if ok[j]] or [kp[j, :2] for j in (LHIP, RHIP) if ok[j]]
    if top and low:
        dx, dy = np.abs(np.mean(top, 0) - np.mean(low, 0))
        return bool(dx * aspect > 1.5 * dy)
    return bool(b[2] * aspect > 1.4 * b[3])                     # box width against height, in pixels


def classify_pass(seq, aspect=4 / 3, fps=50.0):
    """seq: per frame (box, kp) of one person up to the frame it was last seen -> (kind, size).
    Where the person leaves the picture tells the pass: between the legs (limbs left and right of the centre,
    the body above it; for a body lying horizontally, a plank, that is under it), under (the body goes out
    through the top edge, or the feet are above the centre), over (out through the bottom), or past on one
    side. ' (lying)' is added when the body is horizontal: under a plank, over someone lying on the ground."""
    if not seq:
        return 'close', 0.0
    # how the box moved over the last ~0.4 s: a body that leaves through the top edge has its bottom edge
    # rising (the quad goes under it); through the bottom edge, its top edge sinking (over it). A close
    # person always fills the frame down to its bottom edge, so touching an edge alone says nothing.
    run = [b for b, _ in seq[-max(3, int(0.4 * fps)):]]
    rise = (run[0][1] + run[0][3] / 2) - (run[-1][1] + run[-1][3] / 2)          # bottom edge going up
    sink = (run[-1][1] - run[-1][3] / 2) - (run[0][1] - run[0][3] / 2)          # top edge going down
    # ... and the box as a whole goes more up / down than sideways: a child running past the camera fills it
    # top to bottom and leaves through a side, which only the centre's way out tells apart
    dcx, dcy = run[-1][0] - run[0][0], run[-1][1] - run[0][1]
    up, down = -dcy > 1.5 * abs(dcx), dcy > 1.5 * abs(dcx)              # a diagonal way out is a pass on that side
    # a box cut by the left or right edge at the end is someone leaving sideways, however its centre moved
    # (the last frames show only an arm or a leg in a bottom corner)
    last = run[-1]
    side = last[0] - last[2] / 2 < 0.02 or last[0] + last[2] / 2 > 0.98
    up, down = up and not side, down and not side
    flat = sum(lying(b, kp, aspect) for b, kp in seq[-10:]) >= max(2, len(seq[-10:]) // 2)
    kinds = []
    for b, kp in seq[-6:]:
        ok = kp[:, 2] > KP_OK
        limbs = [kp[j, 0] for j in ((LKN, RKN, LAN, RAN, LWR, RWR, LEL, REL) if flat else (LKN, RKN, LAN, RAN))
                 if ok[j]]
        body = [kp[j, 1] for j in ((LHIP, RHIP, LSH, RSH) if flat else (LHIP, RHIP)) if ok[j]]
        feet = [kp[j, 1] for j in (LAN, RAN) if ok[j]]
        top, bottom = b[1] - b[3] / 2, b[1] + b[3] / 2
        if len(limbs) >= 2 and min(limbs) < 0.5 < max(limbs) and (not body or min(body) < 0.5):
            kinds.append('under' if flat else 'between legs')
        elif (feet and max(feet) < 0.45) or (top < 0.03 and rise > PASS_EXIT and up):
            kinds.append('under')
        elif bottom > 0.97 and sink > PASS_EXIT and down:
            kinds.append('over')
        elif b[0] < 0.3:
            kinds.append('past, on its left')
        elif b[0] > 0.7:
            kinds.append('past, on its right')
        else:
            kinds.append('close')
    kind = max(set(kinds), key=kinds.count)
    return kind + (' (lying)' if flat else ''), float(seq[-1][0][3])


def poses_of(model, c, st, fps, a, b):
    """Confident people (boxes, keypoints) on every frame a..b of an open render."""
    c.seek(int(max(a, 0) / fps / st.time_base), stream=st, backward=True)
    frames, imgs = [], []
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i > b:
            break
        if i >= a:
            frames.append(i)
            imgs.append(fr.to_ndarray(format='bgr24'))
    out = []
    for bx, cf, kp in run_pose(model, imgs):
        m = cf >= PASS_CONF
        out.append((bx[m], kp[m]))
    aspect = imgs[0].shape[1] / imgs[0].shape[0] if imgs else 4 / 3
    return frames, out, aspect


def scan_big(model, src):
    """Confident people at PASS_SCAN_HZ: (frames, [boxes per sample], fps)."""
    c = av.open(src)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    every = max(1, int(round(fps / PASS_SCAN_HZ)))
    frames, boxes, imgs, idx = [], [], [], []

    def flush():
        for (bx, cf, kp) in run_pose(model, imgs):
            boxes.append(bx[cf >= PASS_CONF])
        frames.extend(idx)
        imgs.clear()
        idx.clear()
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i % every == 0:
            idx.append(i)
            imgs.append(fr.to_ndarray(format='bgr24'))
            if len(imgs) == 64:
                flush()
    flush()
    c.close()
    return np.array(frames), boxes, fps


def follow(frames, dets, k0, box0):
    """Follow one person from frame index k0 (box0) back to the start and forward until lost.
    -> (list of (box, kp) up to the last frame seen, index of that frame, lost?)"""
    def walk(ks):
        out, cur, miss = [], box0, 0
        for k in ks:
            j = match(cur, dets[k][0])
            if j < 0:
                miss += 1
                if miss >= PASS_LOST:
                    return out, True
                continue
            miss = 0
            cur = dets[k][0][j]
            out.append((k, dets[k][0][j], dets[k][1][j]))
        return out, False
    back, _ = walk(range(k0, -1, -1))
    fwd, lost = walk(range(k0 + 1, len(frames)))
    seq = back[::-1] + fwd
    if not seq:
        return [], k0, False
    peak = max(box_size(b) for _, b, _ in seq)
    for n, (k, b, _) in enumerate(seq):                          # shrinking far away counts as gone
        if k > k0 and box_size(b) < PASS_GONE * peak:
            return [(b, kp) for _, b, kp in seq[:n]], seq[n - 1][0], True
    return [(b, kp) for _, b, kp in seq], seq[-1][0], lost


def passes(model, src):
    samples, boxes, fps = scan_big(model, src)
    cand = [(samples[i], b) for i in range(len(samples) - 1) for b in boxes[i]
            if box_size(b) >= PASS_SCAN_MIN and match(b, boxes[i + 1]) < 0]
    out = []
    c = av.open(src)
    st = c.streams.video[0]
    for f0, box0 in cand:
        frames, dets, aspect = poses_of(model, c, st, fps, int(f0 - 1.5 * fps), int(f0 + PASS_WITHIN_S * fps))
        if not frames:
            continue
        k0 = int(np.argmin(np.abs(np.array(frames) - f0)))
        seq, k_last, lost = follow(frames, dets, k0, box0)
        if not lost or len(seq) < 3 or max(box_size(b) for b, _ in seq) < 1.0:
            continue                                             # still there (a missed detection), or never close
        kind, size = classify_pass(seq, aspect, fps)
        f, last = int(frames[k_last]), seq[-1][0]
        # the same person's pass found from two samples (the quad often passes one person, then the next)
        same = [n for n, o in enumerate(out) if abs(f - o[0]) < 0.5 * fps and iou(last, o[3]) >= PASS_IOU]
        if same:
            if size > out[same[0]][2]:
                out[same[0]] = (f, kind, size, last)
            continue
        out.append((f, kind, size, last))
    c.close()
    out.sort(key=lambda o: o[0])
    np.savez(os.path.splitext(src)[0] + '_passes.npz', frame=np.array([o[0] for o in out], int),
             kind=np.array([o[1] for o in out]), size=np.array([o[2] for o in out]))
    return out


def passes_sheet(renders, sheet, per_row=8):
    """One row per pass other than "past" (those are many and plain): frames from 1.2 s before the moment
    to 0.4 s after it, labelled with the kind."""
    rows = []
    for r in renders:
        base = os.path.splitext(r)[0]
        if not os.path.exists(base + '_passes.npz'):
            continue
        p = np.load(base + '_passes.npz')
        c = av.open(r)
        st = c.streams.video[0]
        fps = float(st.average_rate)
        for f0, kind in zip(p['frame'], p['kind']):
            if str(kind).startswith('past'):
                continue
            want = list(np.linspace(f0 - 1.2 * fps, f0 + 0.4 * fps, per_row).round().astype(int))
            c.seek(int(max(want[0], 0) / fps / st.time_base), stream=st, backward=True)
            tiles = []
            for fr in c.decode(st):
                i = int(round(float(fr.pts * st.time_base) * fps))
                if want and i >= want[0]:
                    tiles.append(cv2.resize(fr.to_ndarray(format='bgr24'), (160, 120)))
                    want.pop(0)
                if not want:
                    break
            tiles += [np.zeros((120, 160, 3), np.uint8)] * (per_row - len(tiles))
            row = np.hstack(tiles)
            cv2.putText(row, f'{os.path.basename(r)[:30]} {f0 / fps:.1f}s {kind}', (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(row)
        c.close()
    if rows:
        cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


# ------------------------------------------------------------------------------- report
def summary(renders, tag=''):
    for r in renders:
        base = os.path.splitext(r)[0]
        if not os.path.exists(base + '_people.npz'):
            continue
        z = np.load(base + '_people.npz')
        fps, ns = float(z['fps']), len(z['samples'])
        pil = np.load(base + '_pilot.npy') if os.path.exists(base + '_pilot.npy') else np.zeros(len(z['frame']), bool)
        other = ~pil
        s_other = len(set(z['frame'][other].tolist())) / max(ns, 1)
        s_pilot = len(set(z['frame'][pil].tolist())) / max(ns, 1)
        line = (f'{os.path.basename(r).replace("_stabilized.mp4", "")}: others in {s_other * 100:3.0f} % of samples, '
                f'pilot {s_pilot * 100:3.0f} % (radio seen {int(z["radio"].sum())}, radio pose {int(z["hold"].sum())}), '
                f'waving {int(z["wave"][other].sum())}, facing camera {int(z["face"][other].sum())}')
        if os.path.exists(subject_file(base, tag)):
            sb = np.load(subject_file(base, tag)) & other
            line += f', subject in {len(set(z["frame"][sb].tolist())) / max(ns, 1) * 100:3.0f} % of samples'
        if os.path.exists(base + '_passes.npz'):
            p = np.load(base + '_passes.npz')
            line += '; passes: ' + (', '.join(f'{f / fps:.1f}s {k}' for f, k in zip(p['frame'], p['kind'])) or 'none')
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('detect', 'pilot', 'pilotclip', 'subject', 'who', 'split', 'passes', 'summary'))
    ap.add_argument('--k', type=int, default=3, help='split: number of looks')
    ap.add_argument('--keep', default=None, help='split: the groups that are the person, e.g. 0,2')
    ap.add_argument('--ref', action='append', default=[], help='who: NAME=IMAGE, a crop of that person (repeatable)')
    ap.add_argument('--sim', type=float, default=WHO_SIM, help='who: CLIP cosine to the references')
    ap.add_argument('--originals', default=None, help='who: clip list, for references given as CLIP:SECONDS:X')
    ap.add_argument('--ref-dir', default='refs', help='who: where references cut from the clips are saved')
    import paths
    ap.add_argument('--pose', default=os.path.join(paths.MODELS, 'yolo', 'yolov8s-pose.pt'),
                    help='who: the person detector for CLIP:SECONDS:X')
    ap.add_argument('--margin', type=float, default=WHO_MARGIN, help='who: lead over the next person')
    ap.add_argument('--text', action='append', default=None, help='subject: the main character in words (repeatable)')
    ap.add_argument('--not', dest='nots', action='append', default=[],
                    help='subject: who the main character is not, in words (repeatable)')
    ap.add_argument('--thr', type=float, default=0.5,
                    help='subject: how sure CLIP must be of a person (0.8 when the others look alike: kids vs coach)')
    ap.add_argument('--tag', default='', help='subject: name of this main character (file _subject_NAME.npy)')
    ap.add_argument('--clip-cache', default=None, help='where the CLIP model is kept, e.g. models/clip')
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--weights', default=None, help='pose model, e.g. models/yolo/yolov8s-pose.pt')
    ap.add_argument('--objects', default=None, help='object model to spot the radio, e.g. models/yolo/yolov8s.pt')
    ap.add_argument('--every', type=int, default=25, help='at 50 fps; scaled to the clip rate')
    ap.add_argument('--clusters', type=int, default=6)
    ap.add_argument('--sheet', default=None)
    a = ap.parse_args()
    renders = sorted({os.path.normcase(p): p for r in a.renders for p in glob.glob(r)}.values())
    if a.step in ('detect', 'passes'):
        gpu.device()
        from ultralytics import YOLO
        model = YOLO(a.weights)
        obj_model = YOLO(a.objects) if a.objects and a.step == 'detect' else None
        for r in renders:
            t0 = time.time()
            if a.step == 'detect':
                out = os.path.splitext(r)[0] + '_people.npz'
                if os.path.exists(out):
                    continue
                fps = float(av.open(r).streams.video[0].average_rate)
                np.savez(out, **detect(model, r, max(1, int(round(a.every * fps / 50.0))), obj_model))
                print(f'{os.path.basename(r)}: poses in {time.time() - t0:.0f}s', flush=True)
            elif not os.path.exists(os.path.splitext(r)[0] + '_passes.npz'):
                ev = passes(model, r)
                print(f'{os.path.basename(r)}: {len(ev)} close passes checked in {time.time() - t0:.0f}s', flush=True)
        if a.step == 'passes' and a.sheet:
            passes_sheet(renders, a.sheet)
    elif a.step == 'pilot':
        find_pilot(renders, a.clusters, a.sheet)
    elif a.step == 'pilotclip':
        clip_scores(renders, a.clip_cache)
        find_pilot_clip(renders, a.sheet)
    elif a.step == 'subject':
        if not a.text:
            raise SystemExit('subject: describe the main character with --text')
        find_subject(renders, a.clip_cache, a.text, row_thr=a.thr, sheet=a.sheet, tag=a.tag, nots=a.nots)
    elif a.step == 'who':
        refs = {}
        originals = [x.strip() for x in open(a.originals, encoding='utf-8') if x.strip()] if a.originals else []
        for spec in a.ref:
            n, path = spec.split('=', 1)
            parts = path.rsplit(':', 2)
            if not os.path.exists(path) and not path.startswith('@') and len(parts) == 3:
                # CLIP:SECONDS:X -- the person nearest to X (0..1 across) at that moment of that clip
                stem = os.path.splitext(os.path.basename(parts[0]))[0].lower()
                src = next((o for o in originals if os.path.splitext(os.path.basename(o))[0].lower() == stem), None)
                if src is None:
                    raise SystemExit(f'{spec}: give the clip list with --originals')
                os.makedirs(a.ref_dir, exist_ok=True)
                path = cut_ref(src, float(parts[1]), float(parts[2]),
                               os.path.join(a.ref_dir, f'{n}_{stem}_{parts[1]}.jpg'), a.weights or a.pose)
                print(f'reference {n}: {path}')
            refs.setdefault(n, []).append(path)
        if not refs:
            raise SystemExit('who: give each person with --ref NAME=IMAGE')
        find_who(renders, a.clip_cache, refs, sheet=a.sheet, sim=a.sim, margin=a.margin)
        return
    elif a.step == 'split':
        split_looks(renders, a.tag, a.k, a.sheet, None if a.keep is None else {int(v) for v in a.keep.split(',')})
        return
    summary(renders, a.tag)


if __name__ == '__main__':
    main()
