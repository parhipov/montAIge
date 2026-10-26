"""Quality report of a finished edit in about a dozen lines, so nobody has to watch it to find faults.

    python qc.py EDIT_edl.json EDIT.mp4

From the EDL: length, shot lengths, chronology, repeated movement kinds, people, the main character of
an edit about one person (--focus), piloting (shots of people up close and editor's pieces apart),
activities filling many scenes (with places.py activities), flights.
From the video (every frame at 96x54): audio vs video length, dark frames, frozen / repeated
frames (stutter), brightness jumps at cuts. Each line ends in OK or WARN. Propellers are not
checked: after stabilization they must not be in view, and if they are, that is Gyroflow's zoom
setting, not something the edit can fix.
"""
import argparse
import json
import os

import av
import cv2
import numpy as np


def line(name, value, ok):
    print(f'{"OK  " if ok else "WARN"} {name}: {value}')


def activity_scenes(e, hit=0.4):
    """{activity: scenes showing it} from places.py activities (<render>_activities.npz), {} without them.
    A shot shows the activity its frames score highest on average, if that is `hit` or more; the pilot is
    left out (qc counts the pilot on its own line)."""
    data = {}
    for g in e['flights']:
        p = os.path.splitext(g.get('render') or '')[0] + '_activities.npz'
        if os.path.exists(p):
            data[g['idx']] = np.load(p)
    if not data:
        return {}
    out = {}
    for s in e['scenes']:
        seen = set()
        for x in s['shots']:
            z = data.get(x['flight'])
            if z is None:
                continue
            i, j = np.searchsorted(z['frame'], [x['src_frame'], x['src_frame'] + x['src_frames']])
            m = z['p'][min(i, len(z['frame']) - 1):max(j, i + 1)].astype(np.float32).mean(0)
            k = int(np.argmax(m))
            if m[k] >= hit and 'FPV goggles' not in str(z['texts'][k]):
                seen.add(str(z['texts'][k]))
        for t in seen:
            out[t] = out.get(t, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('edl')
    ap.add_argument('video')
    ap.add_argument('--min-shot', type=float, default=2.5)
    a = ap.parse_args()
    e = json.load(open(a.edl, encoding='utf-8'))
    shots = [dict(x, day=s.get('day'), flight=s['flight'], must=s.get('must', False)) for s in e['scenes']
             for x in s['shots']]
    L = np.array([x['t1'] - x['t0'] for x in shots])
    dur = shots[-1]['t1']

    # ---------------------------------------------------------------- the plan
    line('length', f'{dur:.1f} s, {len(shots)} shots', True)
    line('shot length', f'min {L.min():.1f} / median {np.median(L):.1f} / max {L.max():.1f} s, '
         f'{np.sum(L < a.min_shot)} under {a.min_shot} s', np.sum(L < a.min_shot) <= 1)
    days = [x['day'] for x in shots if x.get('day')]
    if days:
        back = sum(1 for p, q in zip(days, days[1:]) if q < p)
        line('chronology', f'{len(set(days))} days, {back} steps back in time', back == 0)
    mv = [x.get('move') for x in shots if x.get('move')]
    if mv:
        run = best = 1
        for p, q in zip(mv, mv[1:]):
            run = run + 1 if p == q else 1
            best = max(best, run)
        top = max(set(mv), key=mv.count)
        line('movement variety', f'{len(set(mv))} kinds, most common {top} x{mv.count(top)}, longest run {best}',
             best <= 3 and mv.count(top) <= 0.4 * len(mv))
    if any('people' in x for x in shots):
        pp = np.array([x.get('people', 0) for x in shots])
        line('people', f'in {np.mean(pp > 0.15) * 100:.0f} % of shots, {np.sum(L[pp > 0.15]):.0f} s of {dur:.0f} s', True)
    if any('pilot_alone' in x for x in shots):
        pa = np.array([x.get('pilot_alone', 0) for x in shots])
        line('pilot leads the picture', f'{np.sum(pa > 0.3)} shots, {np.sum(L * pa):.0f} s of {dur:.0f} s',
             np.sum(L * pa) <= 0.15 * dur + 0.5)
    if e.get('focus') and any('subject' in x for x in shots):
        # an edit about one person: the opening / editor's pieces may be without them, most shots not
        sb = np.array([x.get('subject', 0) for x in shots])
        line(f'main character ({e.get("subject")})', f'in {np.sum(sb > 0.3)} of {len(shots)} shots, '
             f'{np.sum(L[sb > 0.3]):.0f} s of {dur:.0f} s', np.sum(L[sb > 0.3]) >= 0.6 * dur)
        si = np.array([x.get('stand_in', 0) for x in shots])
        line('look-alikes standing in for them', f'{np.sum(si > 0.3)} shots', np.sum(si > 0.3) == 0)
    elif e.get('subject') and any('subject' in x for x in shots):   # main characters among other people: a count
        sb = np.array([x.get('subject', 0) for x in shots])
        line(f'main characters ({e.get("subject")})', f'in {np.sum(sb > 0.3)} of {len(shots)} shots, '
             f'{np.sum(L[sb > 0.3]):.0f} s of {dur:.0f} s', True)
    sk = [x['skill'] for x in shots if 'skill' in x]
    if sk:
        # hovering is the point of a shot of people up close (a person at 2 % of the frame or more counts 1),
        # and an editor's piece is there whatever it scores (a slow climb from height scores ~0.1)
        meant = [x.get('people', 0) >= 1.0 or x.get('must') for x in shots if 'skill' in x]
        rest = [v for v, m in zip(sk, meant) if not m]
        line('piloting', f'median {np.median(sk):.2f}, worst {min(sk):.2f}; without {sum(meant)} shots of people up '
             f'close or editor\'s pieces: worst {min(rest) if rest else 1.0:.2f}', not rest or min(rest) >= 0.3)
    acts = activity_scenes(e)
    if acts:
        # one activity all through the edit (village: the old man on the shed roof in 5 of 8 shots) reads as a
        # repeat however different the angles are; `edit.py --theme` limits it
        top = sorted(acts.items(), key=lambda x: -x[1])
        allowed = max(2, len(e['scenes']) // 3)
        line('activities (scenes)', ', '.join(f'{t} x{k}' for t, k in top[:4]), top[0][1] <= allowed)
    fl = [x['flight'] for x in shots]
    adj = sum(1 for s, t in zip(e['scenes'], e['scenes'][1:]) if s['flight'] == t['flight'])
    line('flights', f'{len(set(fl))} different, same flight in neighbouring scenes x{adj}', adj <= 1)

    # ---------------------------------------------------------------- the picture
    c = av.open(a.video)
    vs, au = c.streams.video[0], (c.streams.audio[0] if c.streams.audio else None)
    fps = float(vs.average_rate)
    vdur = vs.frames / fps if vs.frames else c.duration / 1e6
    adur = float(au.duration * au.time_base) if au is not None and au.duration else 0.0
    line('audio vs video', f'{adur:.2f} s / {vdur:.2f} s', au is not None and abs(adur - vdur) < 0.15)
    vs.thread_type = 'AUTO'
    means, diffs = [], []
    prev = None
    for fr in c.decode(vs):
        im = fr.to_ndarray(format='bgr24', width=96, height=54)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        means.append(g.mean())
        diffs.append(np.abs(g - prev).mean() if prev is not None else 99.0)
        prev = g
    c.close()
    means, diffs = np.array(means), np.array(diffs)
    t = np.arange(len(means)) / fps
    inner = (t > 1.0) & (t < dur - 3.0)                          # outside the fades
    line('dark frames', f'{np.mean(means[inner] < 25) * 100:.1f} %', np.mean(means[inner] < 25) < 0.02)
    frozen = np.flatnonzero(inner & (diffs < 0.25))
    line('frozen / repeated frames', f'{frozen.size} ({frozen.size / max(inner.sum(), 1) * 100:.1f} %)',
         frozen.size / max(inner.sum(), 1) < 0.01)
    cuts = [int(round(x['t0'] * fps)) for x in shots[1:]]
    jumps = [abs(means[k] - means[k - 1]) for k in cuts if 0 < k < len(means)]
    line('brightness jumps at cuts', f'max {max(jumps) if jumps else 0:.0f} (0-255), '
         f'{sum(1 for j in jumps if j > 50)} over 50', sum(1 for j in jumps if j > 50) <= 1)


if __name__ == '__main__':
    main()
