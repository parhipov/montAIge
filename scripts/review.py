"""A look at the edit's shots by the vision-language model before the previews go out (the reviewer of CutClaw,
2026): per shot the start, the middle and the end side by side, a sentence on what it shows and three yes-or-no
questions.

    python review.py EDIT_edl.json [--out EDIT_review.txt]

The model (captions.VLM) says in a sentence who or what the shot is about and what happens, then answers three
questions one at a time with one word each: is the subject clearly visible (VISIBLE / HIDDEN: hidden, cut off or
blurred), does anything happen (ACTION / NOTHING), is it pleasant to look at (PLEASANT / DULL: empty water,
ground or sky). Asked for a list of problems from a menu, the 2B model copied the whole menu for 17 of 18 shots;
asked one question at a time it answers. Each shot gets a line in <edit>_review.txt; a shot answered HIDDEN,
NOTHING or DULL is a WARN line, which montage.py's qc step prints with qc.py's. The numbers of the plan pass a shot
of a man just standing there; the picture does not. It is the model's opinion of three stills: read the line
before swapping the scene (edl_edit.py --replace K). A shot under a phrase of speech (a say: bite) is not NOTHING:
its action is what is said, which stills do not show.
"""
import argparse
import json
import os
import re
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import captions  # noqa: E402
import pix  # noqa: E402

TILE_W = 448
INTRO = 'These are the start, the middle and the end of one shot of a video, side by side. '
DESCRIBE = INTRO + ('In one short sentence about the shot itself, not about the panels: who or what is it about, '
                    'and what happens in it?')
QUESTIONS = [
    ('visible', INTRO + 'Is the main subject (the people, or the place if there are none) clearly visible in the '
                'frames, or is it hidden, cut off by the frame edge or blurred? Answer with one word: VISIBLE or HIDDEN.',
     'HIDDEN', 'subject hidden, cut off or blurred'),
    ('action', INTRO + 'Does anything happen in this shot (people moving, an action, the camera flying past '
               'something), or is it a still view where nothing happens? Answer with one word: ACTION or NOTHING.',
     'NOTHING', 'nothing happening'),
    ('view', INTRO + 'Is this shot pleasant to look at, or dull: mostly empty water, ground or sky with nothing '
             'to see? Answer with one word: PLEASANT or DULL.', 'DULL', 'dull view'),
]


def strip_of(render, fps, frames, w=TILE_W):
    """The frames side by side, each w wide."""
    out = []
    with av.open(render) as c:
        st = c.streams.video[0]
        H, W = st.codec_context.height, st.codec_context.width
        h = int(round(w * H / W / 2)) * 2
        for fr_i in frames:
            c.seek(int(max(fr_i - 2, 0) / fps / st.time_base), stream=st, backward=True)
            im = None
            for fr in c.decode(st):
                im = pix.ndarray(fr, format='rgb24', width=w, height=h)
                if int(round(float(fr.pts * st.time_base) * fps)) >= fr_i:
                    break
            out.append(im if im is not None else np.zeros((h, w, 3), np.uint8))
    return np.hstack(out)


def said(answer, word):
    """The one-word answer is `word` (the first word that matters in the reply)."""
    m = re.search(r'\b(VISIBLE|HIDDEN|ACTION|NOTHING|PLEASANT|DULL)\b', answer.upper())
    return bool(m) and m.group(1) == word


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('edl')
    ap.add_argument('--out', default=None)
    ap.add_argument('--batch', type=int, default=6)
    a = ap.parse_args()
    e = json.load(open(a.edl, encoding='utf-8'))
    out = a.out or os.path.splitext(a.edl)[0].replace('_edl', '') + '_review.txt'
    flights = {g['idx']: g for g in e['flights']}
    shots = [(s, x) for s in e['scenes'] for x in s['shots']]
    talk = [(b['t0'], b['t1']) for b in e.get('bites') or [] if b.get('talk')]
    from PIL import Image
    vlm = captions.VLM()
    lines, flagged = [], {}
    for b0 in range(0, len(shots), a.batch):
        batch = shots[b0:b0 + a.batch]
        imgs = []
        for s, x in batch:
            g = flights[x['flight']]
            fps = float(g['fps'])
            a0, a1 = x['src_frame'], x['src_frame'] + x['src_frames'] - 1
            k = int(0.3 * fps)
            imgs.append(Image.fromarray(strip_of(g['render'], fps, [min(a0 + k, a1), (a0 + a1) // 2, max(a1 - k, a0)])))
        about = vlm.ask(imgs, DESCRIBE, 64)
        verdicts = {key: vlm.ask(imgs, q, 8) for key, q, _, _ in QUESTIONS}
        for j, (s, x) in enumerate(batch):
            probs = [what for key, _, bad, what in QUESTIONS if said(verdicts[key][j], bad)
                     and not (key == 'action' and any(t0 < x['t1'] and t1 > x['t0'] for t0, t1 in talk))]
            name = os.path.splitext(os.path.basename(flights[x['flight']]['original']))[0]
            head = f'shot {b0 + j} (scene {s["idx"]}, {x["t0"]:.1f}-{x["t1"]:.1f} s, {name} {x["src_t0"]:.1f} s)'
            lines.append(f'{"WARN" if probs else "OK  "} {head}: {about[j].rstrip(".")} | '
                         f'{", ".join(probs) if probs else "fine"}')
            for p in probs:
                flagged[p] = flagged.get(p, 0) + 1
    summary = (f'review: {sum(1 for l in lines if l.startswith("WARN"))} of {len(shots)} shots flagged'
               + (': ' + ', '.join(f'{p} x{n}' for p, n in sorted(flagged.items(), key=lambda v: -v[1])) if flagged else ''))
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines + [summary]) + '\n')
    print('\n'.join(l for l in lines if l.startswith('WARN')))
    print(summary)


if __name__ == '__main__':
    main()
