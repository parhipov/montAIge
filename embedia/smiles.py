"""Smiles at the camera: the local vision-language model of captions.py asked about candidate heads only.

    python smiles.py RENDER.mp4 [...] [--min-area 0.06] [--step 1.0] [--top 20] [--sheet OUT.jpg]
        -> <render>_smiles.npz (frame, head square, asked, answer, smile) next to each; prints the longest
           stretches of smiles as CLIP:START-END, ready for edit.py --must

Few and right rather than many: a smile counts only when a model said yes to a sharp, big enough head and to
another candidate of the same clip within CONFIRM_S.

Candidates come from people.py detect: a person facing the camera (its face flag) covering MIN_AREA of the frame
or more, not the pilot, at most one per STEP seconds of a clip (the biggest such person). The model sees a square
around the head (nose, eyes, ears; at least the shoulders' width) scaled to HEAD_PX, asked PROMPT; a head of under
MIN_HEAD_PX in the render or one blurred (Laplacian variance at 96 px under MIN_SHARP) is not asked. Not run by
montage.py: a tool for a shoot that asks for it (a promo ending on faces, kids at a party), with captions.py's model
(models/vlm, Qwen3-VL 2B).

Measured on a boxing gym (40 min of hand camera, 18 min of GoPros, 22 of drone, an RTX 5060 Ti). Whole 640 px
frames with people from 2 % of the frame: 3713 candidates, 328 "smiling", nearly all far away in the gym (faces of
10-20 px: the model guessed), 475 s. Heads of people from 6 %: 2522 candidates, 400 smiling in 129 s (batches of 8,
~0.05 s a head), and the false ones were of one kind: smeared or covered heads (a GoPro inside the ring, a whip of
the camera; sharpness 16-230 against 240-3100 for faces), heads of 32-36 px, and single yeses without a neighbour
(nearly every unconfirmed one on a sheet of 40 across the sharpness range was wrong). The model loads in ~8 s.
"""
import argparse
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

PROMPT = 'Is this person smiling? Answer yes or no.'
MIN_AREA, STEP_S, BATCH, MAX_NEW = 0.06, 1.0, 8, 16
KP_OK, HEAD_PX = 0.35, 336      # people.py's keypoint confidence; the head crop is scaled to this
MIN_HEAD_PX, MIN_SHARP = 48, 250.0
CONFIRM_S = 1.6        # a yes counts with another yes of the clip this close
GAP_S = 2.5            # counted smiles this close are one stretch


def candidates(render, min_area=MIN_AREA, step=STEP_S):
    """(frames, head squares (cx, cy, side) in frame fractions, side in fractions of the height, fps): per STEP
    seconds the biggest person facing the camera (not the pilot) of MIN_AREA or more, from people.py's files beside
    the render; empty without them."""
    base = os.path.splitext(render)[0]
    if not os.path.exists(base + '_people.npz'):
        return np.zeros(0, int), np.zeros((0, 3)), 0.0
    z = np.load(base + '_people.npz')
    fps = float(z['fps'])
    pil = np.load(base + '_pilot.npy') if os.path.exists(base + '_pilot.npy') else np.zeros(len(z['frame']))
    notp = np.load(base + '_notperson.npy') if os.path.exists(base + '_notperson.npy') else np.zeros(len(z['frame']))
    area = z['box'][:, 2] * z['box'][:, 3]
    ok = (z['face'] > 0.5) & (area >= min_area) & (pil < 0.5) & (notp < 0.5)
    best = {}
    for r in np.flatnonzero(ok):
        k = int(z['frame'][r] / fps / step)
        if k not in best or area[r] > area[best[k]]:
            best[k] = r
    rows = sorted(best.values(), key=lambda r: z['frame'][r])
    with av.open(render) as c:
        cc = c.streams.video[0].codec_context
        aspect = cc.width / cc.height
    return (np.array([z['frame'][r] for r in rows], int),
            np.array([head_square(z['kp'][r].reshape(17, 3), z['box'][r], aspect) for r in rows]).reshape(-1, 3), fps)


def head_square(kp, box, aspect):
    """(cx, cy, side) of a square around the head: the face points' centre, side 2.5 x their height spread, at
    least the shoulders' width and a fifth of the person's height (frame fractions; side in fractions of the
    frame's height, `aspect` its width over its height)."""
    head = kp[:5][kp[:5, 2] > KP_OK]
    cx, cy = head[:, :2].mean(0)
    side = max(2.5 * np.ptp(head[:, 1]) if len(head) > 1 else 0.0, 0.2 * box[3])
    if kp[5, 2] > KP_OK and kp[6, 2] > KP_OK:
        side = max(side, abs(kp[5, 0] - kp[6, 0]) * aspect)
    return cx, cy, side


def head_crop(img, sq):
    """The head square of a PIL frame, as it is in the render (its side in pixels: crop.size[0])."""
    W, H = img.size
    cx, cy, side = sq
    s = min(max(int(side * H), 16), W, H)
    x0 = int(np.clip(cx * W - s / 2, 0, W - s))
    y0 = int(np.clip(cy * H - s * 0.45, 0, H - s))
    return img.crop((x0, y0, x0 + s, y0 + s))


def sharpness(crop):
    """Laplacian variance of a head crop at 96 px grey: a smeared or covered head reads under MIN_SHARP."""
    g = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA), cv2.CV_64F).var())


def frames_at(render, wanted):
    """PIL images of the wanted frame numbers (sorted), decoded in one pass, and the numbers found."""
    from PIL import Image
    out, want = {}, set(int(w) for w in wanted)
    with av.open(render) as c:
        st = c.streams.video[0]
        st.thread_type = 'AUTO'
        fps = float(st.average_rate)
        last = max(want) if want else -1
        for fr in c.decode(st):
            i = int(round(float(fr.pts * st.time_base) * fps))
            if i in want:
                out[i] = Image.fromarray(pix.ndarray(fr, format='rgb24'))
            if i >= last:
                break
    return [out[w] for w in wanted if w in out], [w for w in wanted if w in out]


def confirmed(frames, yes, fps):
    """The yeses with another yes of the clip within CONFIRM_S."""
    t = frames / fps
    return np.array([y and bool((yes & (np.abs(t - t[k]) <= CONFIRM_S) & (np.arange(len(t)) != k)).any())
                     for k, y in enumerate(yes)], bool)


def smiles(render, vlm, min_area=MIN_AREA, step=STEP_S, batch=BATCH, force=False):
    """(frames, smile, head squares, fps) of a render's candidates, cached in <render>_smiles.npz: `smile` is a
    confirmed yes to a sharp head of MIN_HEAD_PX or more."""
    p = os.path.splitext(render)[0] + '_smiles.npz'
    if os.path.exists(p) and not force:
        z = np.load(p, allow_pickle=False)
        return z['frame'], z['smile'], z['head'], float(z['fps'])
    cand, sq, fps = candidates(render, min_area, step)
    imgs, got = frames_at(render, cand) if cand.size else ([], [])
    keep = set(got)
    sq = np.array([s for f, s in zip(cand, sq) if f in keep]).reshape(-1, 3)
    crops = [head_crop(im, s) for im, s in zip(imgs, sq)]
    asked = np.array([c.size[0] >= MIN_HEAD_PX and sharpness(c) >= MIN_SHARP for c in crops], bool)
    todo = [c.resize((HEAD_PX, HEAD_PX)) for c, a in zip(crops, asked) if a]
    answers = []
    for k in range(0, len(todo), batch):
        answers += vlm.ask(todo[k:k + batch], PROMPT, MAX_NEW)
    text = np.full(len(crops), '', dtype=object)
    text[asked] = answers
    yes = np.array([t.lower().lstrip(' *"\'').startswith('yes') for t in text], bool)
    frames = np.array(got, np.int64)
    smile = confirmed(frames, yes, fps)
    np.savez(p, frame=frames, head=sq, asked=asked, text=text.astype(str), yes=yes, smile=smile, fps=fps,
             min_area=min_area, step=step)
    return frames, smile, sq, fps


def stretches(frames, smile, fps):
    """(start s, end s, smiles) of runs of smiles (GAP_S bridged), the most smiles first."""
    t = frames[smile] / fps if len(frames) else np.zeros(0)
    out, k = [], 0
    while k < len(t):
        j = k
        while j + 1 < len(t) and t[j + 1] - t[j] <= GAP_S:
            j += 1
        out.append((max(t[k] - 0.5, 0.0), t[j] + 0.5, j - k + 1))
        k = j + 1
    return sorted(out, key=lambda s: (-s[2], s[0]))


def sheet(items, path, per_row=8, most=32):
    """The heads counted as smiling, spread over all of them, with their clip and second."""
    tiles = []
    for render, f, fps, sq in items[::max(1, -(-len(items) // most))]:
        crop = head_crop(frames_at(render, [f])[0][0], sq).resize((200, 200))
        a = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
        cv2.putText(a, f'{short(render)[-8:]} {f / fps:.1f}s', (3, 14), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(a)
    if not tiles:
        return
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[k:k + per_row]) for k in range(0, len(tiles), per_row)]))


def short(render):
    """The clip's name as edit.py takes it (without the render's _stabilized)."""
    s = os.path.splitext(os.path.basename(render))[0]
    return s[:-len('_stabilized')] if s.endswith('_stabilized') else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--min-area', type=float, default=MIN_AREA, help='a candidate person covers this much or more')
    ap.add_argument('--step', type=float, default=STEP_S, help='at most one candidate per this many seconds')
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--force', action='store_true', help='ask again even where an answer file exists')
    ap.add_argument('--top', type=int, default=20, help='stretches printed')
    ap.add_argument('--sheet', default=None, help='a picture of the heads counted as smiling')
    a = ap.parse_args()
    from .captions import VLM
    vlm, t0, n, found, shown = VLM(), time.time(), 0, [], []
    for r in a.renders:
        fr, sm, hd, fps = smiles(r, vlm, a.min_area, a.step, a.batch, a.force)
        n += len(fr)
        found += [(k, short(r), s0, s1) for s0, s1, k in stretches(fr, sm, fps)]
        shown += [(r, int(f), fps, h) for f, s, h in zip(fr, sm, hd) if s]
    print(f'{len(a.renders)} renders, {n} candidates, {len(shown)} smiles in {len(found)} stretches, '
          f'{time.time() - t0:.0f}s; the longest:')
    for k, name, s0, s1 in sorted(found, key=lambda x: -x[0])[:a.top]:
        print(f'  {name}:{s0:.1f}-{s1:.1f}  ({k} smiles)')
    if a.sheet:
        sheet(shown, a.sheet)


if __name__ == '__main__':
    main()
