"""How beautiful the view is, per half second, with beauty in the job's own words: an option for edits that want
beautiful views (edit.py --beauty W; off by default). Mountains for one job, a beach for another.

    python -m embedia.analyzers.beauty RENDER.mp4 [...] --text "mountains" --text "a canyon with a river"
                     [--plain "snow up close" ...] [--clip-cache models/clip] [--sheet beauty.jpg]
                                                                          -> <render>_beauty.npz next to each

Two numbers from the CLIP ViT-B/32 frame embeddings places.py keeps (every 0.5 s, computed when missing):
  aes     the LAION aesthetic predictor (a linear head on CLIP trained on people's 1-10 ratings of photos;
          models/aesthetic/sa_0_4_vit_b_32_linear.pth from github.com/LAION-AI/aesthetic-predictor): a good
          picture whatever it shows (light, composition, not blurred, not a fence up close)
  scenic  CLIP zero-shot: the probability of the --text views against MUNDANE (what else drone footage shows)
          and the job's own plain views (--plain: what fills this footage without being a view, e.g. snow and
          skis up close, a queue at a ski lift), softmax over both lists; without --text, 0
What flying makes a view beautiful (smooth, far, the view opening up) is the cinematic piloting score in
edit.py; this is only what is in the picture. Prints per clip the median aesthetic, the share of scenic
views and the best stretches; --sheet: the best and the worst half seconds of all clips, to check by eye.
"""
import argparse
import os

import numpy as np

from .. import paths
from . import places

AES_HEAD = os.path.join(paths.MODELS, 'aesthetic', 'sa_0_4_vit_b_32_linear.pth')
MUNDANE = ['a parking lot', 'a dirt yard', 'grass up close', 'the ground up close', 'a road with cars',
           'a wall up close', 'a blurry picture', 'the sky', 'a room indoors', 'people standing close to the camera',
           'a construction site', 'a garage', 'a fence', 'a car up close', 'bushes up close']
SCENIC_HIT = 0.5        # a scenic half second


def aesthetic_head():
    """The head's weight (512) and bias, kept as .npz beside the .pth after the first read (torch not needed)."""
    npz = os.path.splitext(AES_HEAD)[0] + '.npz'
    if not os.path.exists(npz):
        import torch
        sd = torch.load(AES_HEAD, map_location='cpu', weights_only=True)
        np.savez(npz, weight=sd['weight'].numpy().ravel().astype(np.float32), bias=float(sd['bias']))
    z = np.load(npz)
    return z['weight'], float(z['bias'].ravel()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--clip-cache', default=os.path.join(paths.MODELS, 'clip'))
    ap.add_argument('--text', action='append', default=[], help='a beautiful view of this job, in words (repeatable)')
    ap.add_argument('--plain', action='append', default=[],
                    help='a plain view of this job, in words (repeatable): added to MUNDANE')
    ap.add_argument('--sheet', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if not os.path.exists(AES_HEAD):
        raise SystemExit(f'missing {AES_HEAD}: download sa_0_4_vit_b_32_linear.pth from '
                         'https://github.com/LAION-AI/aesthetic-predictor')
    w, b = aesthetic_head()
    mdl = places.model_of(a.clip_cache)
    scenic = a.text
    temb = places.text_emb(mdl, scenic + MUNDANE + a.plain) if scenic else None
    rows = []
    for r in a.renders:
        fr, fps, emb = places.embed(r, mdl)
        aes = emb @ w + b
        sc = places.probs(emb, temb, len(scenic)).sum(1) if scenic else np.zeros(len(aes))
        np.savez(os.path.splitext(r)[0] + '_beauty.npz', frame=fr, fps=fps, aes=aes.astype(np.float32),
                 scenic=sc.astype(np.float32))
        t = fr / fps
        best = sorted(places.stretches(sc, t), key=lambda s: -(s[1] - s[0]) * s[2])[:3]
        print(f'{places.short(r)}: aesthetic median {np.median(aes):.2f} (p90 {np.percentile(aes, 90):.2f}), '
              f'scenic {np.mean(sc > SCENIC_HIT) * 100:.0f} % of the time; '
              + ', '.join(f'{s0:.0f}-{s1:.0f}s' for s0, s1, _ in best), flush=True)
        rows += [(r, float(tt), float(x), float(y)) for tt, x, y in zip(t, aes, sc)]
    if a.sheet and rows:
        # one number to sort by for the check: aesthetic rank and scenic probability together
        aes_all = np.sort([x[2] for x in rows])
        key = [0.5 * np.searchsorted(aes_all, x[2]) / len(aes_all) + 0.5 * x[3] for x in rows]
        order = np.argsort(key)[::-1]
        picked, seen = [], []
        for i in order:                                          # the best, at least 10 s apart in a clip
            r, tt = rows[i][0], rows[i][1]
            if all(r != s[0] or abs(tt - s[1]) > 10 for s in seen):
                seen.append((r, tt))
                picked.append(i)
            if len(picked) == 24:
                break
        worst = [i for i in order[::-1][:200:25]]
        items = [(rows[i][0], rows[i][1], f'{places.short(rows[i][0])} {rows[i][1]:.0f}s a{rows[i][2]:.1f} s{rows[i][3]:.2f}')
                 for i in picked] + [(None, 0, '')] * (-len(picked) % 8) + \
                [(rows[i][0], rows[i][1], f'WORST {places.short(rows[i][0])} {rows[i][1]:.0f}s a{rows[i][2]:.1f}')
                 for i in worst]
        places.sheet_of(items, a.sheet)
        print(f'sheet: {a.sheet} (24 best, at least 10 s apart in a clip; then 8 of the worst)')


if __name__ == '__main__':
    main()
