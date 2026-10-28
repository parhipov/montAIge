"""Where the camera is, in words: CLIP on whole frames, searched by text, and routes through places.

    python places.py embed RENDER.mp4 [...] --clip-cache models/clip            -> <render>_clip.npz
    python places.py search RENDER.mp4 [...] --clip-cache models/clip --text "a staircase" [--text ...]
                    [--not "a tennis court" ...] [--top 8] [--sheet out.jpg]
    python places.py route RENDER.mp4 [...] --clip-cache models/clip --text "a lobby" --text "a staircase"
                    --text "a tennis court" [--not ...] [--within 20] [--sheet out.jpg]
    python places.py locations RENDER.mp4 [...] [--sheet locations.jpg]
    python places.py activities RENDER.mp4 [...] [--text "an extra activity" ...] [--only] [--sheet activities.jpg]

embed   Every half second, the CLIP ViT-B/32 embedding of the whole frame (cached; `search` and `route`
        compute it when missing).
search  Each frame scores softmax over the --text places and the --not alternatives (what else the footage
        shows; generic places when none are given). Prints the clips' best stretches for each text: where
        the camera is in a doorway, on the stairs, over the water... A moment the user names in words is
        found here instead of by watching.
route   The texts are places passed in that order: a stretch of the first, then of the second within
        --within seconds, and so on (from the lobby through the door, up the stairs, onto the court). Prints
        the routes found with their times, best first. A route is a flight to somewhere: an editor's moment
        (edit.py --must CLIP:START-END).
locations  Which clips were shot at the same place (an archive of several trips: one edit per place). Two
        clips are alike when the views of each are found in the other (median over its frames of the best
        cosine to the other clip's frames, both ways); clips are joined while the average likeness of two
        groups is LOCATION_SIM or more. Prints the groups with their recording times; --sheet: a row of
        frames per group.
activities What happens in the footage, from a general list (ACTIVITIES, plus --text; with --only the --text
        list alone: a shoot's own trades, "a crane operator", "a welder", "a painter"): seconds and clips per
        activity, best first, and <render>_activities.npz per render (qc.py reports an activity filling
        many scenes). The edit's --theme comes from here: an activity that fills the footage gets one scene.
--sheet saves frames of what was found, to confirm by eye once.
"""
import argparse
import os
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run as a file
from embedia import paths
from embedia import pix

EVERY_S = 0.5
TEXT_CACHE = os.path.join(paths.MODELS, 'clip', 'texts.npz')     # text embeddings computed so far
GENERIC_NOT = ['an open field outdoors', 'a large indoor sports hall', 'a room with people', 'the sky',
               'the ground up close', 'a blurry picture', 'a wall up close']
HIT, MIN_S = 0.5, 1.0         # a stretch: probability over HIT for at least MIN_S (gaps of one sample allowed)
ARRIVE_S = 3.0                # a route ends this far into its last place
# clips of one place: views found in each other (see locations). Village + tennis hall + Koprino (28 clips):
# pairs of one place 0.847..0.96 (median 0.92), of different places 0.80 at most (median 0.64)
LOCATION_SIM = 0.83
# activities: the default, a compromise; a shoot with its own trades replaces it (activities --text ... --only).
# The first part from the footage of past jobs (village: "a man working on a roof" caught the shed roof from every
# angle, "on a ladder repairing the roof" missed it from above); the rest a selection of Kinetics-400 (DeepMind's
# 400 human actions), phrased the same way: what one frame shows from a distance, one phrase per sport or kind.
# Kinetics-400 whole (175 clips of past jobs): 22 % of half seconds over ACT_HIT against 48 % for the first part
# alone, "disc golfing" on forests seen from above, "dancing macarena" on a dance; the first part alone put
# "a person wearing FPV goggles flying a drone" on drone shots of most jobs. This list: 33 %, ice hockey 0.98 where the
# first part said FPV goggles 0.51; the background below takes the views from above and the empty factory hall
# ("a person welding" 566 s -> 104 s of a factory, the hall without sparks gone). Zero-shot on one frame stays a
# first answer: "people skydiving" on a plane seen from above, "people riding a jet ski" on a motorboat (dropped).
ACTIVITIES = ['a man working on a roof', 'a person on a ladder', 'a person mowing the grass',
              'people sitting on a garden swing', 'people sitting at a table outdoors', 'people having a barbecue',
              'children playing', 'a child riding a bicycle', 'people walking on a path', 'people swimming',
              'people playing ball', 'people playing tennis', 'people dancing', 'a band playing on a stage',
              'people working in a vegetable garden', 'a person chopping wood', 'a person washing a car',
              'a person walking a dog', 'a person pushing a baby stroller', 'people skiing',
              'people riding a chairlift', 'people snowboarding', 'people standing in a queue',
              'people taking a selfie', 'people sitting in a cafe',
              'people sledding in the snow', 'people fishing', 'people on a boat', 'people having a picnic',
              'people waving at the camera', 'workers at a construction site', 'a car driving on a road',
              # Kinetics-400, selected
              'people playing ice hockey', 'people ice skating', 'people playing football', 'people playing basketball',
              'people playing volleyball', 'people playing badminton', 'people playing golf', 'people bowling',
              'people riding bicycles', 'people running', 'people rock climbing', 'people doing yoga',
              'people exercising in a gym', 'people riding horses', 'people skateboarding', 'people riding motorcycles',
              'people jumping on a trampoline', 'people canoeing or kayaking', 'people sailing', 'people surfing',
              'people windsurfing', 'people jumping into the water', 'people diving underwater',
              'people paddleboarding', 'people skiing cross-country', 'people riding snowmobiles',
              'people making a snowman', 'people shoveling snow', 'people hiking in the mountains',
              'people paragliding', 'people skydiving', 'people flying a kite', 'people cooking in a kitchen',
              'people digging with a shovel', 'people picking fruit', 'a person driving a tractor', 'a person welding',
              'people hugging', 'people shaking hands', 'people eating at a table indoors',
              'people drinking and toasting with glasses', 'people singing', 'a person playing the guitar',
              'people playing a board game', 'people clapping and cheering', 'people opening presents',
              'people blowing out birthday candles', 'children on a playground', 'a person carrying a baby',
              'people feeding animals', 'people marching in a parade', 'a person driving a car']
NOT_ACTIVITY = GENERIC_NOT + ['an empty garden with a house', 'a village seen from above', 'trees and bushes up close',
                              'a field and a forest', 'a building up close', 'a forest seen from above',
                              'a river seen from above', 'a road seen from above', 'the inside of a factory hall']
ACT_HIT = 0.4                 # an activity is on screen where its probability is this or more


class Lazy:
    """The CLIP model (model, processor, device), loaded the first time it is unpacked: a script whose frame
    embeddings are cached already and whose texts are in TEXT_CACHE never imports torch (8-10 s a process)."""
    def __init__(self, cache):
        self.cache, self.parts = cache, None

    def load(self):
        if self.parts is None:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
            from embedia import gpu
            dev = gpu.device()
            name = 'openai/clip-vit-base-patch32'
            self.parts = (CLIPModel.from_pretrained(name, cache_dir=self.cache).to(dev).eval(),
                          CLIPProcessor.from_pretrained(name, cache_dir=self.cache), dev)
        return self.parts

    def __iter__(self):
        return iter(self.load())


def model_of(cache):
    return Lazy(cache)


def feats(out):
    import torch
    out = out if torch.is_tensor(out) else getattr(out, 'pooler_output', out[0])
    return out / out.norm(dim=-1, keepdim=True)


def embed(render, mdl):
    """-> (frame numbers, fps, unit embeddings n x 512), cached in <render>_clip.npz."""
    path = os.path.splitext(render)[0] + '_clip.npz'
    if os.path.exists(path):
        z = np.load(path)
        return z['frame'], float(z['fps']), z['emb'].astype(np.float32)
    import torch
    from PIL import Image
    model, proc, dev = mdl
    c = av.open(render)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    every = max(1, int(round(EVERY_S * fps)))
    frames, embs, batch = [], [], []

    def flush():
        if batch:
            with torch.no_grad():
                inp = proc(images=batch, return_tensors='pt').to(dev)
                embs.append(feats(model.get_image_features(**inp)).cpu().numpy())
            batch.clear()
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i % every:
            continue
        frames.append(i)
        batch.append(Image.fromarray(pix.ndarray(fr, format='rgb24')))
        if len(batch) == 64:
            flush()
    flush()
    c.close()
    emb = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
    np.savez(path, frame=np.array(frames), fps=fps, emb=emb.astype(np.float16))
    return np.array(frames), fps, emb


def text_emb(mdl, texts):
    """Unit embeddings of the texts (n x 512), kept in TEXT_CACHE: only texts not seen before go through the
    model, so beauty, the timeline, an edit's find:TEXT, --want and --theme do not load it for a few words."""
    known = {}
    if os.path.exists(TEXT_CACHE):
        with np.load(TEXT_CACHE) as z:                           # closed before the file is replaced below
            known = dict(zip(z['text'].tolist(), z['emb']))
    new = [t for t in dict.fromkeys(texts) if t not in known]
    if new:
        import torch
        model, proc, dev = mdl
        with torch.no_grad():
            e = feats(model.get_text_features(**proc(text=new, return_tensors='pt', padding=True).to(dev))).cpu().numpy()
        known.update(zip(new, e.astype(np.float32)))
        os.makedirs(os.path.dirname(TEXT_CACHE), exist_ok=True)
        np.savez(TEXT_CACHE + '.tmp.npz', text=np.array(list(known), dtype=str),
                 emb=np.stack(list(known.values())).astype(np.float32))
        os.replace(TEXT_CACHE + '.tmp.npz', TEXT_CACHE)
    return np.stack([known[t] for t in texts]).astype(np.float32)


def probs(emb, temb, n_q):
    """Softmax over all texts (CLIP's scale 100), the first n_q columns: the places asked for."""
    s = 100.0 * emb @ temb.T
    s = np.exp(s - s.max(1, keepdims=True))
    return (s / s.sum(1, keepdims=True))[:, :n_q]


def stretches(p, t):
    """(start s, end s, mean prob) of runs over HIT, one-sample gaps bridged, at least MIN_S long."""
    on = p > HIT
    for k in range(1, len(on) - 1):
        on[k] |= on[k - 1] and on[k + 1]
    out, k = [], 0
    while k < len(on):
        if on[k]:
            j = k
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if t[j] - t[k] + EVERY_S >= MIN_S:
                out.append((float(t[k]), float(t[j] + EVERY_S), float(p[k:j + 1].mean())))
            k = j + 1
        else:
            k += 1
    return out


def sheet_of(items, path, per_row=8):
    """items: (render, seconds, label) -> one tile each (render None: an empty tile, to start a new row; an
    empty one too where the proxy is not kept)."""
    tiles = []
    for render, sec, label in items:
        if render is None or not os.path.exists(render):       # a new row, or a proxy that is not kept
            tiles.append(np.zeros((150, 200, 3), np.uint8))
            continue
        c = av.open(render)
        st = c.streams.video[0]
        c.seek(int(max(sec - 0.5, 0) / st.time_base), stream=st, backward=True)
        img = None
        for fr in c.decode(st):
            if float(fr.pts * st.time_base) >= sec:
                img = cv2.resize(pix.ndarray(fr, format='bgr24'), (200, 150))
                break
        c.close()
        if img is None:
            continue
        cv2.putText(img, label[:34], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % per_row)
    cv2.imwrite(path, np.vstack([np.hstack(tiles[k:k + per_row]) for k in range(0, len(tiles), per_row)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def short(render):
    return os.path.basename(render).replace('_stabilized.mp4', '')


def found_stretches(data, temb, texts):
    """{text: [(mean prob, render, start s, end s)]} over all renders."""
    found = {q: [] for q in texts}
    for r, (fr, fps, emb) in data.items():
        t, p = fr / fps, probs(emb, temb, len(texts))
        for qi, q in enumerate(texts):
            found[q] += [(m, r, s, e) for s, e, m in stretches(p[:, qi], t)]
    return found


def routes_of(found, texts, renders, within):
    """Routes through the places in order, best first: (mean prob, render, [(start, end, prob) per place]).
    A route ends on arrival: ARRIVE_S into the last place (which may last the rest of the flight)."""
    routes = []
    for r in renders:
        legs = [sorted((s, e, m) for m, rr, s, e in found[q] if rr == r) for q in texts]

        def extend(path, k):
            if k == len(legs):
                routes.append((float(np.mean([m for _, _, m in path])), r, path))
                return
            for s, e, m in legs[k]:
                if path[-1][1] - 2 * EVERY_S <= s <= path[-1][1] + within:
                    extend(path + [(s, e, m)], k + 1)
        for leg in legs[0]:
            extend([leg], 1)
    routes.sort(key=lambda x: (-x[0], x[2][-1][0] - x[2][0][0]))
    return routes


def find(renders, text, cache=None, nots=(), within=20.0, longest=8.0):
    """The best piece for words: "a staircase" (the best stretch) or "a corridor -> a staircase -> a tennis
    court" (a route; the piece runs from the last-but-one place to the arrival). At most `longest` seconds,
    so one shot can show it. -> (render, start s, end s, prob) or None."""
    texts = [t.strip() for t in text.split('->') if t.strip()]
    mdl = model_of(cache)
    data = {r: embed(r, mdl) for r in renders}
    found = found_stretches(data, text_emb(mdl, texts + (list(nots) or GENERIC_NOT)), texts)
    if len(texts) == 1:
        if not found[texts[0]]:
            return None
        m, r, s, e = max(found[texts[0]])
        c = (s + e) / 2
        return r, max(s, c - longest / 2), min(e, c + longest / 2), m
    routes = routes_of(found, texts, renders, within)
    if not routes:
        return None
    m, r, path = routes[0]
    end = min(path[-1][1], path[-1][0] + ARRIVE_S)
    return r, max(path[-2][0], end - longest), end, m


def recorded(render):
    """Recording time of a render: from a DJI name (DJI_YYYYMMDDhhmmss_...), else the file time."""
    import datetime
    import re
    m = re.search(r'(\d{14})', os.path.basename(render))
    if m:
        return datetime.datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
    return datetime.datetime.fromtimestamp(os.path.getmtime(render))


def locations(data, sheet=None):
    """Clips grouped by place (see the module notes) -> [[render, ...], ...], biggest first."""
    rs = list(data)
    emb = [data[r][2] for r in rs]
    n = len(rs)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = emb[i] @ emb[j].T
            sim[i, j] = sim[j, i] = (np.median(s.max(1)) + np.median(s.max(0))) / 2
    groups = [[i] for i in range(n)]
    while len(groups) > 1:                                       # average linkage
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                v = sim[np.ix_(groups[a], groups[b])].mean()
                if v > best:
                    best, pair = v, (a, b)
        if best < LOCATION_SIM:
            break
        a, b = pair
        groups[a] += groups.pop(b)
    groups.sort(key=lambda g: (-len(g), min(recorded(rs[i]) for i in g)))
    items = []
    for k, g in enumerate(groups):
        g.sort(key=lambda i: recorded(rs[i]))
        t = [recorded(rs[i]) for i in g]
        secs = sum(len(data[rs[i]][0]) * EVERY_S for i in g)
        inside = sim[np.ix_(g, g)][~np.eye(len(g), dtype=bool)].mean() if len(g) > 1 else 1.0
        other = [j for j in range(n) if j not in g]
        out = sim[np.ix_(g, other)].mean() if other else 0.0
        print(f'location {k + 1}: {len(g)} clips, {secs / 60:.0f} min, {t[0]:%Y-%m-%d %H:%M}..{t[-1]:%H:%M}, '
              f'alike inside {inside:.2f}, to the rest {out:.2f}')
        print('   ' + ' '.join(short(rs[i]) for i in g))
        for i in g[:8]:
            fr, fps, _ = data[rs[i]]
            items += [(rs[i], float(fr[len(fr) // 2]) / fps, f'{k + 1}: {short(rs[i])[-10:]}')]
        items += [(None, 0, '')] * (-len(items) % 8)
    if sheet and items:
        sheet_of(items, sheet)
    return [[rs[i] for i in g] for g in groups]


def activities(data, mdl, extra=(), sheet=None, only=False):
    """What happens in the footage (see the module notes); saves <render>_activities.npz. only: the extra texts
    replace the general list."""
    texts = list(extra) + ([] if only else [t for t in ACTIVITIES if t not in extra])
    if not texts:
        raise SystemExit('activities --only: give the activities with --text')
    temb = text_emb(mdl, texts + NOT_ACTIVITY)
    total = {t: [0.0, set(), (0.0, None, 0.0)] for t in texts}
    for r, (fr, fps, emb) in data.items():
        p = probs(emb, temb, len(texts))
        np.savez(os.path.splitext(r)[0] + '_activities.npz', frame=fr, fps=fps, texts=np.array(texts),
                 p=p.astype(np.float16))
        for k, t in enumerate(texts):
            on = p[:, k] >= ACT_HIT
            if on.any():
                total[t][0] += on.sum() * EVERY_S
                total[t][1].add(r)
                b = int(np.argmax(p[:, k]))
                if p[b, k] > total[t][2][0]:
                    total[t][2] = (float(p[b, k]), r, float(fr[b]) / fps)
    secs = sum(len(d[0]) * EVERY_S for d in data.values())
    print(f'activities in {secs / 60:.0f} min of footage (probability {ACT_HIT} or more):')
    items = []
    for t, (s, rr, (pm, rb, tb)) in sorted(total.items(), key=lambda x: -x[1][0]):
        if s < 2.0:
            continue
        print(f'  {s:6.0f} s {s / secs * 100:4.0f} %  {len(rr):2d} clips  {t}  (best {short(rb)} {tb:.1f}s)')
        items.append((rb, tb, f'{t[:22]} {s:.0f}s'))
    if sheet and items:
        sheet_of(items[:16], sheet)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('embed', 'search', 'route', 'locations', 'activities'))
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--clip-cache', default=None)
    ap.add_argument('--text', action='append', default=[])
    ap.add_argument('--not', dest='nots', action='append', default=[])
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--within', type=float, default=20.0, help='route: seconds from one place to the next')
    ap.add_argument('--only', action='store_true', help='activities: the --text list replaces the general one')
    ap.add_argument('--sheet', default=None)
    a = ap.parse_args()
    renders = paths.expand(a.renders)
    mdl = model_of(a.clip_cache)
    data = {r: embed(r, mdl) for r in renders}
    if a.step == 'embed':
        print(f'{len(renders)} renders embedded, {sum(len(d[0]) for d in data.values())} frames')
        return
    if a.step == 'locations':
        locations(data, a.sheet)
        return
    if a.step == 'activities':
        activities(data, mdl, a.text, a.sheet, a.only)
        return
    if not a.text:
        raise SystemExit('describe the place(s) with --text')
    found = found_stretches(data, text_emb(mdl, a.text + (a.nots or GENERIC_NOT)), a.text)
    items = []
    if a.step == 'search':
        for q in a.text:
            hits = sorted(found[q], reverse=True)[:a.top]
            print(f'"{q}": {len(found[q])} stretches' + ''.join(
                f'\n   {short(r)} {s:6.1f}-{e:6.1f}s  p {m:.2f}' for m, r, s, e in hits))
            items += [(r, (s + e) / 2, f'{q} {short(r)[-6:]} {s:.0f}s') for m, r, s, e in hits]
    else:
        routes = routes_of(found, a.text, renders, a.within)
        print(f'route {" -> ".join(a.text)} (each next within {a.within:.0f}s): {len(routes)} found')
        for m, r, path in routes[:a.top]:
            end = min(path[-1][1], path[-1][0] + ARRIVE_S)
            print(f'   {short(r)} {path[0][0]:6.1f}-{end:6.1f}s  p {m:.2f}  ' +
                  ' -> '.join(f'{s:.1f}-{e:.1f}' for s, e, _ in path[:-1]) + f' -> arrive {path[-1][0]:.1f}')
            items += [(r, (s + e) / 2 if k < len(path) - 1 else s + 1.0, f'{q} {short(r)[-6:]} {s:.0f}s')
                      for k, (q, (s, e, _)) in enumerate(zip(a.text, path))]
    if a.sheet and items:
        sheet_of(items, a.sheet, per_row=len(a.text) if a.step == 'route' else 8)


if __name__ == '__main__':
    main()
