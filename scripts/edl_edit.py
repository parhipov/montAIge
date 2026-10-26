"""Change a finished edit without planning it again: the music stays as it is, and so does every shot not touched.

    python edl_edit.py EDIT_edl.json --end-from K -o OUT.mp4
    python edl_edit.py EDIT_edl.json --replace K [--replace K ...] --job out/JOB/job.toml --edit NAME -o OUT.mp4
                       [--clip CLIP] [--seed N] [-- EDIT.PY OPTIONS]
    python edl_edit.py EDIT_edl.json --put K.J CLIP:SEC -o OUT.mp4
    python edl_edit.py EDIT_edl.json --match CLIP -o OUT.mp4

--end-from K: the ending takes over from scene K (0-based, as in edl_sheet.py): scenes K .. the one before the last
leave, and the last shot starts at scene K's start, earlier in its own clip, at 1x, up to the same last frame. If
the clip has not that much usable footage before it, the edit ends earlier (the track fades out there). Cutting
a scene out of the track instead moved every later scene off its beats.

--replace K: scene K gets other footage for the same stretch of music, found the way edit.py finds a scene (its
scoring, with the edit's own options from the job file, or given after --), against the rest of the edit as it
stands: the footage of the other scenes stays out, and so does the scene's own; their looks, movements and themes
count as placed. Its recording day first, as in edit.py's chapters. --clip keeps the search to one clip; --seed N
gives another candidate.

--put K.J CLIP:SEC: shot J of scene K shows CLIP from SEC on, at 1x, for its own stretch of music: a moment the
owner named ("the fly-through from the other edit, here"); its grade is found again.

--match CLIP: every other clip's colour matched to CLIP's (edit.py --match: two cameras of one shoot); every shot's
grade is found again.

Writes OUT_edl.json (what final.py takes; OUT_music.wav too when the edit ends earlier) and the preview OUT.mp4
like edit.py's (proxies, 25 fps, the EDL's LUT, grade and zoom).
"""
import argparse
import json
import os
import sys
import types

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import edit  # noqa: E402

USABLE_FROM_S = 3.0     # a clip's first seconds are not usable (edit.py trims them too)


def read_audio(path):
    with av.open(path) as c:
        st = c.streams.audio[0]
        rate = st.rate
        rs = av.AudioResampler(format='fltp', layout='stereo', rate=rate)
        chunks = [o.to_ndarray() for fr in c.decode(st) for o in rs.resample(fr)]
        chunks += [o.to_ndarray() for o in rs.resample(None)]
    return np.concatenate(chunks, axis=1), rate


def write_wav(path, x, rate):
    import wave
    pcm = (np.clip(x.T, -1, 1) * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(x.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def end_from(e, k):
    """The EDL with the last shot taking over from scene k; -> (new EDL, new end in seconds or None)."""
    scenes = [dict(s, shots=[dict(x) for x in s['shots']]) for s in e['scenes']]
    last = scenes[-1]
    x = last['shots'][0]
    if len(last['shots']) > 1:                                   # the last scene as one shot, ending as it did
        y = last['shots'][-1]
        x.update(t1=y['t1'], anchors=[x['anchors'][0], y['anchors'][-1]],
                 anchor_t=[0.0, y['t0'] - x['t0'] + y['anchor_t'][-1]])
        last['shots'] = [x]
    f = next(g for g in e['flights'] if g['idx'] == x['flight'])
    fps = f['fps']
    T = scenes[k]['shots'][0]['t0']
    a_end, t_end = x['anchors'][-1], x['anchor_t'][-1]          # the last frame and when (from the shot start)
    L = t_end + (x['t0'] - T)                                    # seconds from the new start to that frame, at 1x
    a0 = a_end - L * fps
    lo = USABLE_FROM_S * fps
    end = None
    if a0 < lo:                                                  # not enough before it: start at the usable start
        L = (a_end - lo) / fps
        a0 = lo
        end = T + L + (x['t1'] - x['t0'] - t_end)                # and end that much earlier
    x.update(t0=T, anchors=[float(a0), float(a_end)], anchor_t=[0.0, float(L)], speed=1.0,
             src_frame=int(a0), src_frames=int(a_end - a0), src_t0=round(a0 / fps, 4),
             b0=int(round(T * e['bpm'] / 60.0)))
    if end is not None:
        x['t1'] = end
    out = scenes[:k] + [dict(last, idx=k)]
    return dict(e, scenes=out), end


def edit_options(a, extra, e):
    """edit.py's options of the edit: the job file's (as montage.py passes them) and any given after --."""
    opts = []
    if a.job:
        import tomllib
        import montage
        with open(a.job, 'rb') as fh:
            job = tomllib.load(fh)
        job['_out'] = os.path.dirname(os.path.abspath(a.job))      # out/<job>/job.toml: cameras.json beside it
        ed = next((x for x in job.get('edit', []) if x['name'] == a.edit), None)
        if ed is None:
            raise SystemExit(f'{a.job}: no edit named {a.edit}')
        opts = montage.edit_args(job, ed)
    return edit.make_parser().parse_args([e['music'], '-o', 'x.mp4'] + opts + extra)


def new_speeds(x, f, scenes, o):
    """Anchors of a new shot: ramped against the shots around it in the same clip (with --ramps; without, onto
    the musical hits inside it, as edit.ramp_all does), else 1x."""
    hits_only = not o.ramps
    if not o.no_ramps and (o.ramps or (edit.HIT_ON[0] and x['hit'][1:].any())):
        lst = [y for s in scenes for y in s['shots'] if y['flight'] == f.idx and y is not x]
        mg = int(0.3 * f.fps)
        lo = max([y['src_frame'] + y['src_frames'] + mg for y in lst if y['src_frame'] < x['src_frame']], default=0)
        hi = min([y['src_frame'] - mg for y in lst if y['src_frame'] > x['src_frame']], default=f.n - 1)
        anchors = edit.snap(x, f, lo, hi, hits_only)
        if anchors is not None:
            x['anchors'] = [float(v) for v in anchors]
            x['anchor_t'] = np.linspace(0, x['t1'] - x['t0'], len(anchors)).tolist()
            x['src_frame'], x['src_frames'] = int(np.floor(anchors[0])), int(np.ceil(anchors[-1] - anchors[0]))
            return
    edit.constant_speed(x)


def load_flights(e, cache):
    """edit.py's flights of the EDL's footage, each with the EDL's colour match."""
    fl = sorted(e['flights'], key=lambda f: f['idx'])
    flights = edit.find_flights_list([f['original'] for f in fl], cache, os.path.dirname(fl[0]['render']))
    if [os.path.basename(f.src) for f in flights] != [os.path.basename(f['original']) for f in fl]:
        raise SystemExit('the footage is not the one the EDL was made from')
    for f, g in zip(flights, fl):
        f.match = g.get('match')
    return flights


def lut_of(e):
    return edit.CubeLUT(e['lut']) if e.get('lut') else None


def put(e, k, j, spec, cache):
    """The EDL with shot j of scene k showing CLIP:SEC on at 1x (see the module notes)."""
    flights = load_flights(e, cache)
    stem, sec = spec.rsplit(':', 1)
    f = next((f for f in flights if f.name.lower() == os.path.splitext(os.path.basename(stem))[0].lower()), None)
    if f is None:
        raise SystemExit(f'--put {spec}: no such clip in the EDL')
    scenes = [dict(s, shots=[dict(x) for x in s['shots']]) for s in e['scenes']]
    if not (0 <= k < len(scenes) and 0 <= j < len(scenes[k]['shots'])):
        raise SystemExit(f'--put {k}.{j}: no such shot')
    x = scenes[k]['shots'][j]
    a, n = int(round(float(sec) * f.fps)), int(round((x['t1'] - x['t0']) * f.fps))
    if a < 0 or a + n > f.n:
        raise SystemExit(f'--put {spec}: the clip ends before the shot does')
    was = f'{flights[x["flight"]].name} {x["src_frame"] / flights[x["flight"]].fps:.1f}s'
    for y in (y for s in scenes for y in s['shots'] if y is not x and y['flight'] == f.idx):
        if a < y['src_frame'] + y['src_frames'] and y['src_frame'] < a + n:
            print(f'  note: the shot at {y["t0"]:.1f}s shows some of the same footage')
    x.update(flight=f.idx, src_frame=a, src_frames=n, speed=1.0, hero_ramp=False, src_t0=a / f.fps)
    for key in ('theme', 'fly_kind', 'hero_kind', 'hero_frame'):
        x.pop(key, None)
    edit.constant_speed(x)
    if x.get('grade'):
        x['grade'] = edit.shot_grade(f, a, n, lut_of(e))
    if all(y['flight'] == f.idx for y in scenes[k]['shots']):
        scenes[k]['flight'] = f.idx
    print(f'scene {k} shot {j} ({x["t0"]:.1f}-{x["t1"]:.1f}s): {was} -> {f.name} {a / f.fps:.1f}-{(a + n) / f.fps:.1f}s')
    return dict(e, scenes=scenes)


def match(e, ref, cache):
    """The EDL with every clip's colour matched to ref's, every shot graded again (see the module notes)."""
    flights = load_flights(e, cache)
    lut = lut_of(e)
    edit.colour_match(flights, ref, lut)
    scenes = [dict(s, shots=[dict(x) for x in s['shots']]) for s in e['scenes']]
    if any(x.get('grade') for s in scenes for x in s['shots']):
        edit.grade_all(scenes, flights, lut)
    fl = sorted(e['flights'], key=lambda g: g['idx'])
    return dict(e, scenes=scenes, flights=[dict(g, match=f.match) for g, f in zip(fl, flights)])


def replace(e, ks, o, cache, clip=None):
    """The EDL with scenes ks found again (see the module notes)."""
    rng = edit.apply_options(o)
    flights = load_flights(e, cache)
    edit.prepare_flights(flights, o)
    m = edit.analyse_music(e['music'], e['bpm'])
    acc, d0 = m['acc'], m['downbeat']
    scenes = []
    for s in e['scenes']:
        s = dict(s, shots=[dict(x) for x in s['shots']])
        for x in s['shots']:                                     # what the EDL leaves out (edit.plan_scenes)
            a, n = x['b0'], x['nbeats']
            x['bpm'] = m['bpm']
            x['beat_w'] = acc[a:a + n] / (acc.mean() + 1e-9) + 0.5 * ((np.arange(a, a + n) - d0) % 4 == 0)
            x['hit'] = m['hit'][a:a + n]
        scenes.append(s)
    pools = edit.Pools(flights)
    used = {f.idx: [] for f in flights}
    seen, moves, hero_used, shown = [], [], set(), [0] * len(edit.THEMES)
    edit.PILOT_USED[:] = [0.0, m['duration']]
    for i, s in enumerate(scenes):
        for x in s['shots']:
            f = flights[x['flight']]
            used[f.idx].append((x['src_frame'], x['src_frame'] + x['src_frames']))   # the scene's own too
            if i in ks:
                continue
            moves.append((edit.window_moves(f, np.array([x['src_frame']]), x['src_frames'])[0], x['t0']))
            lk = edit.shot_look(f, x['src_frame'], x['src_frames'])
            if lk is not None:
                seen.append(lk)
            hero_used.update(k for k in (x.get('fly_kind'), x.get('hero_kind')) if k)
            edit.PILOT_USED[0] += x.get('pilot_alone', 0.0) * (x['t1'] - x['t0'])
        if i not in ks:
            for t, (text, _) in enumerate(edit.THEMES):
                shown[t] += any(x.get('theme') == text for x in s['shots'])
    days = sorted({f.day for f in flights})
    for i in sorted(ks):
        sc = scenes[i]
        was = sorted({flights[x['flight']].name for x in sc['shots']})
        sc['must'] = None
        for x in sc['shots']:
            x['speed'] = edit.BASE_SPEED[int(x['cls'])]
            for k in ('theme', 'fly_kind', 'hero_kind', 'grade'):
                x.pop(k, None)
        nbr = [scenes[j]['flight'] for j in (i - 1, i + 1) if 0 <= j < len(scenes)]
        d = days.index(sc['day']) if sc.get('day') in days else None
        tries = [sc.get('day')] + ([days[j] for j in (d + 1, d - 1, d + 2, d - 2) if 0 <= j < len(days)]
                                   if d is not None else [])
        best = None
        for day in tries:
            for f in flights:
                if (day and f.day != day) or (clip and f.name.lower() != clip.lower()):
                    continue
                r = edit.scene_dp(sc, f, pools, edit.rec_used(f, used, flights), rng, edit.look_penalty(f, seen),
                                  moves, hero_used, edit.theme_penalty(f, shown),
                                  (edit.cut_side(scenes, i - 1, flights, True),
                                   edit.cut_side(scenes, i + 1, flights, False)))
                if r is None:
                    continue
                score = r[0] - edit.W_SAME_SCENE * sum(1 for x in nbr if flights[x].rec == f.rec)
                if best is None or score > best[0]:
                    best = (score, f, r[1], r[2])
            if best is not None:
                break
        if best is None:
            raise SystemExit(f'scene {i}: nothing else fits' + (f' in {clip}' if clip else ''))
        score, f, starts, ns = best
        sc['score'], sc['day'] = float(score), f.day
        edit.take(sc, f, starts, ns, pools, used, seen, moves, hero_used, shown)
        for x in sc['shots']:
            new_speeds(x, f, scenes, o)
        if o.grade == 'auto':
            edit.grade_all([sc], flights, edit.CubeLUT(o.lut) if o.lut else None)
        print(f'scene {i} ({sc["shots"][0]["t0"]:.1f}-{sc["shots"][-1]["t1"]:.1f}s): {", ".join(was)} -> {f.name} '
              + ', '.join(f'{x["src_frame"] / f.fps:.1f}-{(x["src_frame"] + x["src_frames"]) / f.fps:.1f}s'
                          for x in sc['shots'])
              + f' (piloting {np.mean([x["skill"] for x in sc["shots"]]):.2f}, '
              f'people {np.mean([x.get("people", 0.0) for x in sc["shots"]]):.2f})')
    out = []
    for s in scenes:
        s = dict(s, must=bool(s.get('must')))
        s['shots'] = [{k: v for k, v in x.items() if k not in ('beat_w', 'bpm', 'hit')}
                      | dict(src_t0=x['src_frame'] / flights[x['flight']].fps) for x in s['shots']]
        out.append(s)
    return dict(e, scenes=out)


def render_edl(e, out, height=480, fps=25):
    """edit.py's preview of an EDL: the proxies the EDL names, its music, LUT, grade and zoom."""
    flights = {}
    for f in e['flights']:
        p = f.get('proxy') or f['render']
        n, _ = edit.frame_count(p)
        flights[f['idx']] = types.SimpleNamespace(idx=f['idx'], proxy=p, stab=f['render'], fps=f['fps'], n=n,
                                                  match=f.get('match'))
    m = edit.analyse_music(e['music'], e['bpm'])
    m['path'] = e['music']
    lut = edit.CubeLUT(e['lut']) if e.get('lut') else None
    edit.render(e['scenes'], flights, m, out, height, fps, lut=lut, zoom=e.get('zoom') or 1.0)


def main():
    import paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('edl')
    ap.add_argument('--end-from', type=int, default=None, help='scene the ending takes over from (0-based)')
    ap.add_argument('--replace', type=int, action='append', default=[], help='scene to find again (0-based), repeatable')
    ap.add_argument('--job', default=None, help="--replace: the job file with the edit's options")
    ap.add_argument('--edit', default=None, help='--replace: the edit in the job file')
    ap.add_argument('--clip', default=None, help='--replace: look in this clip only')
    ap.add_argument('--put', nargs=2, metavar=('K.J', 'CLIP:SEC'), default=None,
                    help='shot J of scene K (0-based) shows CLIP from SEC at 1x')
    ap.add_argument('--match', default=None, metavar='CLIP', help="every other clip's colour matched to CLIP's")
    ap.add_argument('--seed', type=int, default=None, help='--replace: another candidate')
    ap.add_argument('--cache', default=os.path.join(paths.ROOT, 'out', 'diagnostics', '.telemetry_cache'))
    ap.add_argument('-o', '--out', required=True, help='the preview; _edl.json next to it')
    ap.add_argument('--height', type=int, default=None,
                    help='preview height; default that of the edit\'s own preview beside the EDL (else 480)')
    a, extra = ap.parse_known_args()
    if a.height is None:                                         # as montage.py made it (640 px wide)
        own = a.edl[:-len('_edl.json')] + '.mp4' if a.edl.endswith('_edl.json') else ''
        a.height = 480
        if os.path.exists(own):
            with av.open(own) as c:
                a.height = c.streams.video[0].codec_context.height
    extra = [x for x in extra if x != '--']
    with open(a.edl, encoding='utf-8') as fh:
        e = json.load(fh)
    if sum((a.end_from is not None, bool(a.replace), a.put is not None, a.match is not None)) != 1:
        raise SystemExit('one change at a time: --end-from K, --replace K, --put K.J CLIP:SEC or --match CLIP')
    base = os.path.splitext(os.path.abspath(a.out))[0]
    os.makedirs(os.path.dirname(base), exist_ok=True)
    if a.put:
        k, j = (int(v) for v in a.put[0].split('.'))
        e2 = put(e, k, j, a.put[1], a.cache)
        e2['edl_edit'] = dict(from_edl=a.edl, put=a.put)
    elif a.match:
        e2 = match(e, a.match, a.cache)
        e2['edl_edit'] = dict(from_edl=a.edl, match=a.match)
    elif a.replace:
        bad = [k for k in a.replace if not 0 <= k < len(e['scenes'])]
        if bad:
            raise SystemExit(f'--replace {bad}: scenes 0..{len(e["scenes"]) - 1}')
        o = edit_options(a, extra, e)
        if a.seed is not None:
            o.seed = a.seed
        e2 = replace(e, set(a.replace), o, a.cache, a.clip)
        e2['edl_edit'] = dict(from_edl=a.edl, replace=a.replace, clip=a.clip, seed=a.seed)
    else:
        k = a.end_from
        if not 0 < k < len(e['scenes']) - 1:
            raise SystemExit(f'--end-from {k}: scenes 1..{len(e["scenes"]) - 2}')
        e2, end = end_from(e, k)
        if end is not None:                                      # the track ends with the picture, faded out
            x, rate = read_audio(e['music'])
            n = int(round(end * rate))
            x = x[:, :n].copy()
            fl = int(edit.FADE_OUT_S * rate)
            x[:, -fl:] *= np.linspace(1, 0, fl)
            write_wav(base + '_music.wav', x, rate)
            e2['music'] = base + '_music.wav'
        e2['edl_edit'] = dict(from_edl=a.edl, end_from=k)
        x = e2['scenes'][-1]['shots'][0]
        fps = next(g for g in e2['flights'] if g['idx'] == x['flight'])['fps']
        print(f'the ending from {x["t0"]:.2f}s: {x["anchors"][0] / fps:.1f}-{x["anchors"][1] / fps:.1f}s of its clip'
              + (f', the edit ends at {end:.1f}s' if end is not None else ', the track as it was'))
    with open(base + '_edl.json', 'w', encoding='utf-8') as fh:
        json.dump(e2, fh, ensure_ascii=False, indent=1)
    render_edl(e2, a.out, a.height)
    print(f'preview: {a.out}')


if __name__ == '__main__':
    main()
