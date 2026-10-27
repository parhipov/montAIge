"""One table of everything the features know, for questions across them: out/<job>/timeline.sqlite.

    python timeline.py out/JOB                          build (montage.py does it after `features`)
    python timeline.py out/JOB --sql "SELECT ..."       ask; prints at most --limit rows

The features stay where they are (npz per render, the source); this is an index built from them, rebuilt in
seconds. A question like "where is marina in view while we go fast with a beautiful view behind" is one query:

    SELECT clip, t, speed, marina, scenic FROM t WHERE marina > 0.01 AND speed > 20 ORDER BY scenic DESC

Tables:
  clips   one row per clip: clip, path, render, start (ISO, chapters of a split recording included), day,
          seconds, fps, gps (share of time with a fix), lat, lon, alt_min, alt_max, speed_p50 (km/h)
  t       one row per clip per 0.5 s (the CLIP samples): clip, t (s), frame; speed (km/h), alt (m), climb (m/s,
          + up) from the GPS (NULL without a fix); rot (visible camera rotation, deg/s), flow, parallax (resid90),
          bright, sharp from analyze.py; people (count), people_area (the biggest, share of the frame), facing,
          waving; one column per main character (people.py who / subject: their area, 0 when not in view); aes and
          scenic (beauty.py); activity (the likeliest of places.ACTIVITIES, or 'none') and activity_p; caption
          (captions.py: what the model said of the 3 s chunk this moment falls in, NULL without captions)
  people  one row per person row of people.py detect: clip, t, frame, track (people.py track, else -1), x, y, w,
          h (centre and size, shares of the frame), area, facing, waving, name (main character or NULL)
  captions  one row per captioned chunk: clip, t0, t1, text; captions_fts (FTS5) searches the words:
          SELECT clip, t0, text FROM captions_fts JOIN captions ON captions.rowid = captions_fts.rowid
          WHERE captions_fts MATCH 'jump*'
"""
import argparse
import glob
import os
import sqlite3
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

GRID_S = 0.5


def subject_names(base):
    """Main characters labelled for a render: <render>_subject_NAME.npy, without the _not look-alike files."""
    out = []
    for p in glob.glob(glob.escape(base) + '_subject_*.npy'):
        n = os.path.basename(p)[len(os.path.basename(base)) + len('_subject_'):-4]
        if n and not n.endswith('_not') and n.replace('_', '').isalnum():
            out.append(n)
    return sorted(out)


def activity_labels(renders, cache):
    """{render: (frames, label, p)} on the CLIP samples: from <render>_activities.npz where places.py activities
    ran (it carries the job's own --text activities: "welding with sparks" in a factory), else from the general
    list on the embeddings (places.py embed); {} without either."""
    import places
    out, todo = {}, []
    for r in renders:
        a = os.path.splitext(r)[0] + '_activities.npz'
        if os.path.exists(a):
            z = np.load(a)
            texts, p = z['texts'].tolist(), z['p'].astype(np.float32)
            k = np.argmax(p, 1)
            best = p[np.arange(len(k)), k]
            out[r] = (z['frame'], np.where(best >= places.ACT_HIT, np.array(texts, object)[k], 'none'), best)
        elif os.path.exists(os.path.splitext(r)[0] + '_clip.npz'):
            todo.append(r)
    if not todo:
        return out
    mdl = places.model_of(cache)
    texts = list(places.ACTIVITIES)
    temb = places.text_emb(mdl, texts + places.NOT_ACTIVITY)
    for r in todo:
        z = np.load(os.path.splitext(r)[0] + '_clip.npz')
        p = places.probs(z['emb'].astype(np.float32), temb, len(texts))
        k = np.argmax(p, 1)
        best = p[np.arange(len(k)), k]
        lab = np.where(best >= places.ACT_HIT, np.array(texts, object)[k], 'none')
        out[r] = (z['frame'], lab, best)
    return out


def build(job_dir, proxy, cache, clip_cache):
    import av
    import edit
    with open(os.path.join(job_dir, 'originals.txt'), encoding='utf-8') as fh:
        srcs = [x.strip() for x in fh if x.strip()]
    renders = [os.path.join(proxy, os.path.splitext(os.path.basename(s))[0] + '_stabilized.mp4') for s in srcs]
    pairs = [(s, r) for s, r in zip(srcs, renders) if os.path.exists(r)]
    names = sorted({n for _, r in pairs for n in subject_names(os.path.splitext(r)[0])})
    acts = activity_labels([r for _, r in pairs], clip_cache)
    dst = os.path.join(job_dir, 'timeline.sqlite')
    tmp = dst + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    db = sqlite3.connect(tmp)
    db.execute('CREATE TABLE clips (clip TEXT PRIMARY KEY, path TEXT, render TEXT, start TEXT, day TEXT, seconds REAL, '
               'fps REAL, gps REAL, lat REAL, lon REAL, alt_min REAL, alt_max REAL, speed_p50 REAL)')
    cols = ['clip TEXT', 't REAL', 'frame INTEGER', 'speed REAL', 'alt REAL', 'climb REAL', 'rot REAL', 'flow REAL',
            'parallax REAL', 'bright REAL', 'sharp REAL', 'people INTEGER', 'people_area REAL', 'facing INTEGER',
            'waving INTEGER', 'aes REAL', 'scenic REAL', 'activity TEXT', 'activity_p REAL', 'caption TEXT'] + \
        [f'"{n}" REAL' for n in names]
    db.execute(f'CREATE TABLE t ({", ".join(cols)})')
    db.execute('CREATE TABLE people (clip TEXT, t REAL, frame INTEGER, track INTEGER, x REAL, y REAL, w REAL, h REAL, '
               'area REAL, facing INTEGER, waving INTEGER, name TEXT)')
    db.execute('CREATE TABLE captions (clip TEXT, t0 REAL, t1 REAL, text TEXT)')
    try:                                                         # word search: SELECT ... FROM captions_fts WHERE
        db.execute('CREATE VIRTUAL TABLE captions_fts USING fts5(clip, text, content=captions)')   # captions_fts MATCH 'jump*'
        fts = True
    except sqlite3.OperationalError:
        fts = False
    for src, r in pairs:
        clip = os.path.splitext(os.path.basename(src))[0]
        base = os.path.splitext(r)[0]
        with av.open(r) as c:
            st = c.streams.video[0]
            n, fps = st.frames, float(st.average_rate)
        grid = np.arange(0, n, max(int(round(GRID_S * fps)), 1))
        tt = grid / fps
        row = {'clip': [clip] * len(grid), 't': np.round(tt, 2), 'frame': grid}
        gps = os.path.join(proxy, clip + '_stabilized_gps.npz')
        lat = lon = amin = amax = sp50 = None
        share = 0.0
        if os.path.exists(gps):
            z = np.load(gps)
            ok = z['fix'] > 0
            if ok.sum() > 1:
                tg = z['t'][ok]
                known = (tt >= tg[0] - 1) & (tt <= tg[-1] + 1)
                for k, scale in (('speed', 3.6), ('alt', 1.0), ('climb', 1.0)):
                    row[k] = np.where(known, np.interp(tt, tg, z[k][ok]) * scale, np.nan)
                lat, lon = float(np.median(z['lat'][ok])), float(np.median(z['lon'][ok]))
                amin, amax = float(z['alt'][ok].min()), float(z['alt'][ok].max())
                sp50 = float(np.median(z['speed'][ok]) * 3.6)
                share = float(known.mean())
        try:
            q, mfps, _ = edit.load_motion(src, cache, r)
            if len(q) >= n:
                vis = np.linalg.norm(edit.rates_from_quats(q[:n], mfps), axis=1)
                k = max(int(GRID_S * fps), 1)
                vis = np.convolve(vis, np.ones(k) / k, 'same')
                row['rot'] = vis[grid]
        except (SystemExit, OSError, ValueError, KeyError):
            pass
        if os.path.exists(base + '_flow.npz'):
            z = np.load(base + '_flow.npz')
            for k, key in (('flow', 'flow'), ('parallax', 'resid90'), ('bright', 'bright'), ('sharp', 'sharp')):
                if key in z.files:
                    row[k] = np.interp(grid, z['frame'], z[key])
        if os.path.exists(base + '_people.npz'):
            z = np.load(base + '_people.npz')
            area = z['box'][:, 2] * z['box'][:, 3]
            g = np.clip(np.searchsorted(grid, z['frame']), 0, len(grid) - 1)
            cnt, big, fac, wav = (np.zeros(len(grid)) for _ in range(4))
            np.add.at(cnt, g, 1)
            np.maximum.at(big, g, area)
            np.add.at(fac, g, z['face'] > 0)
            np.add.at(wav, g, z['wave'] > 0)
            row.update(people=cnt, people_area=big, facing=fac, waving=wav)
            tid = np.load(base + '_tid.npy') if os.path.exists(base + '_tid.npy') else -np.ones(len(area), int)
            who = np.full(len(area), None, object)
            for nm in names:
                sp = base + f'_subject_{nm}.npy'
                col = np.zeros(len(grid))
                if os.path.exists(sp):
                    m = np.load(sp).astype(bool)
                    if len(m) == len(area):
                        np.maximum.at(col, g[m], area[m])
                        who[m & (who == None)] = nm  # noqa: E711
                row[nm] = col
            db.executemany('INSERT INTO people VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', [
                (clip, round(float(f / fps), 2), int(f), int(t_), float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                 float(a), int(fc > 0), int(wv > 0), w) for f, t_, b, a, fc, wv, w in
                zip(z['frame'], tid, z['box'], area, z['face'], z['wave'], who)])
        if os.path.exists(base + '_beauty.npz'):
            z = np.load(base + '_beauty.npz')
            row['aes'] = np.interp(grid, z['frame'], z['aes'])
            row['scenic'] = np.interp(grid, z['frame'], z['scenic'])
        if r in acts:
            fr, lab, p = acts[r]
            k = np.clip(np.searchsorted(fr, grid), 0, len(fr) - 1)
            row['activity'], row['activity_p'] = lab[k], p[k]
        if os.path.exists(base + '_captions.npz'):              # captions.py: the chunk each half second falls in
            z = np.load(base + '_captions.npz', allow_pickle=False)
            cf, ct, chunk = z['frame'], z['text'], float(z['chunk_s'])
            if cf.size:                                            # a clip shorter than a chunk has no captions
                k = np.clip(np.searchsorted(cf, grid, side='right') - 1, 0, len(cf) - 1)
                row['caption'] = np.where(grid < cf[k] + chunk * fps, ct[k], None)
            db.executemany('INSERT INTO captions VALUES (?,?,?,?)',
                           [(clip, round(float(f / fps), 2), round(float(f / fps + chunk), 2), str(t)) for f, t in zip(cf, ct)])
        keys = [c.split()[0].strip('"') for c in cols]
        data = []
        for i in range(len(grid)):
            vals = []
            for k in keys:
                v = row.get(k)
                v = None if v is None else v[i]
                if isinstance(v, (np.floating, float)) and not np.isfinite(v):
                    v = None
                elif isinstance(v, np.generic):
                    v = v.item()
                vals.append(v)
            data.append(vals)
        db.executemany(f'INSERT INTO t VALUES ({",".join("?" * len(keys))})', data)
        when = edit.creation_time(src)
        db.execute('INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (clip, src, r, when.isoformat(), when.date().isoformat(), n / fps, fps, share, lat, lon, amin, amax,
                    sp50))
    db.execute('CREATE INDEX t_clip ON t (clip, t)')
    db.execute('CREATE INDEX people_clip ON people (clip, t)')
    if fts:
        db.execute("INSERT INTO captions_fts(captions_fts) VALUES ('rebuild')")
    db.commit()
    nt = db.execute('SELECT COUNT(*) FROM t').fetchone()[0]
    npl = db.execute('SELECT COUNT(*) FROM people').fetchone()[0]
    ncap = db.execute('SELECT COUNT(*) FROM captions').fetchone()[0]
    db.close()
    os.replace(tmp, dst)
    print(f'timeline: {len(pairs)} clips, {nt} half seconds, {npl} people rows, {ncap} captions, main characters: '
          f'{", ".join(names) or "none"} -> {dst}')


def ask(job_dir, sql, limit):
    db = sqlite3.connect(os.path.join(job_dir, 'timeline.sqlite'))
    cur = db.execute(sql)
    head = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(limit + 1)
    fmt = [[(f'{v:.3g}' if isinstance(v, float) else str(v)) for v in r] for r in rows[:limit]]
    w = [max([len(h)] + [len(r[i]) for r in fmt]) for i, h in enumerate(head)]
    print('  '.join(h.ljust(w[i]) for i, h in enumerate(head)))
    for r in fmt:
        print('  '.join(v.ljust(w[i]) for i, v in enumerate(r)))
    if len(rows) > limit:
        print(f'... (more than {limit} rows)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('job_dir', help='out/<job>')
    ap.add_argument('--sql', default=None)
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--proxy-dir', default=None, help='default: out/diagnostics/proxy_<job>')
    ap.add_argument('--cache', default=None, help='telemetry cache, default out/diagnostics/.telemetry_cache')
    ap.add_argument('--clip-cache', default=os.path.join(paths.MODELS, 'clip'))
    a = ap.parse_args()
    if a.sql:
        ask(a.job_dir, a.sql, a.limit)
        return
    job = os.path.basename(os.path.normpath(a.job_dir))
    root = os.path.dirname(os.path.normpath(a.job_dir))
    build(a.job_dir, a.proxy_dir or os.path.join(root, 'diagnostics', 'proxy_' + job),
          a.cache or os.path.join(root, 'diagnostics', '.telemetry_cache'), a.clip_cache)


if __name__ == '__main__':
    main()
