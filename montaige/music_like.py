"""Tracks that sound like one the owner liked: CLAP's music model on the audio, over catalogue candidates.

    python music_like.py LIKED.mp3 [LIKED2.mp3 ...] [--not DISLIKED.mp3 ...] [--top 12]
                         [--bpm-tol 0.12] [--feel 2] [--files CAND.mp3 ...]
                         [--bpm 128 --feels "Driving, Intense"]      a liked track not in the catalogue

Picking tracks by the catalogue's words went wrong twice in three on one job (an orchestral "epic, uplifting" and
a "driving" synth track: "the music does not fit at all"), while a happy steel-drum piece fitted. So once the owner
has liked a track, the next ones are found by how they sound. Candidates: the incompetech catalogue
(models/music/pieces.json) within --bpm-tol of the liked track's catalogue BPM and sharing --feel of its feel words
(or the --files given), downloaded once into models/music/catalogue/. Each track is cut into WIN_S windows (CLAP's
input length), embedded by laion/larger_clap_music (models/clap fills itself from Hugging Face), and averaged
into one unit vector (cached beside the mp3). A candidate scores its cosine to the mean of the liked tracks, minus
DISLIKE_W x its cosine to the closest disliked one. Prints the best with BPM, feel and instruments; check the
tempo grid with music_check.py before using one.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository: embedia, montaige (run as a file)
from embedia import paths  # noqa: E402
from embedia import sounds  # noqa: E402

NAME = 'laion/larger_clap_music'
WIN_S, BATCH = 10.0, 16
DISLIKE_W = 0.5
CATALOGUE = os.path.join(paths.MODELS, 'music', 'pieces.json')
CACHE = os.path.join(paths.MODELS, 'music', 'catalogue')
URL = 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/'


def model_of(cache=None):
    import torch  # noqa: F401
    from transformers import ClapModel, ClapProcessor
    from embedia import gpu
    dev = gpu.device()
    cache = cache or os.path.join(paths.MODELS, 'clap')
    m = ClapModel.from_pretrained(NAME, cache_dir=cache).to(dev).eval()
    if dev != 'cpu':
        m = m.half()
    return m, ClapProcessor.from_pretrained(NAME, cache_dir=cache), dev


def track_emb(path, mdl):
    """Unit mean of the track's window embeddings, cached in <path>.clapm.npy."""
    import torch
    p = path + '.clapm.npy'
    if os.path.exists(p):
        return np.load(p)
    y = sounds.audio(path)
    model, proc, dev = mdl
    W = int(WIN_S * sounds.SR)
    starts = np.arange(0, max(y.size - W, 0) + 1, W)
    out = []
    for b in range(0, starts.size, BATCH):
        x = proc(audio=[y[s:s + W] for s in starts[b:b + BATCH]], sampling_rate=sounds.SR, return_tensors='pt').to(dev)
        with torch.no_grad():
            out.append(sounds.unit(model.get_audio_features(input_features=x['input_features'].to(model.dtype),
                                                            is_longer=x.get('is_longer'))).cpu().numpy())
    e = np.concatenate(out).astype(np.float32).mean(0)
    e /= np.linalg.norm(e)
    np.save(p, e)
    return e


def bpm_of(p):
    try:
        return float(p['bpm'])
    except (TypeError, ValueError):
        return 0.0


def feel_of(p):
    return {w.strip().lower() for w in (p.get('feel') or '').split(',') if w.strip()}


def same(path, piece):
    """Is this file the catalogue piece (by file name, spaces or underscores)?"""
    n = os.path.basename(path).lower().replace('_', ' ')
    return n == piece['filename'].lower().replace('_', ' ')


def candidates(liked, cat, tol, feel, bpm=None, feels=None):
    """Catalogue pieces near the liked ones in tempo and sharing `feel` of their feel words, downloaded. A liked
    track from outside the catalogue (the owner's own) is described by `bpm` and `feels` instead."""
    ref = [next((p for p in cat if same(f, p)), None) for f in liked]
    ref = [p for p in ref if p]
    if bpm:
        ref.append({'bpm': bpm, 'feel': feels or ''})
    if not ref:
        raise SystemExit('the liked tracks are not in the catalogue by file name: give --bpm and --feels, or '
                         'candidates with --files')
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for p in cat:
        if any(p is r for r in ref) or not p.get('filename'):
            continue
        b = bpm_of(p)
        if not any(abs(b - bpm_of(r)) <= tol * bpm_of(r) and len(feel_of(p) & feel_of(r)) >= feel for r in ref):
            continue
        dst = os.path.join(CACHE, p['filename'].replace(' ', '_'))
        if not os.path.exists(dst):
            try:
                urllib.request.urlretrieve(URL + urllib.parse.quote(p['filename']), dst + '.tmp')
                os.replace(dst + '.tmp', dst)
            except OSError as e:
                print(f'  {p["title"]}: {e}')
                continue
        out.append((dst, p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('liked', nargs='+')
    ap.add_argument('--not', dest='disliked', action='append', default=[], help='a track that did not fit')
    ap.add_argument('--files', nargs='*', default=None, help='candidates instead of the catalogue')
    ap.add_argument('--bpm-tol', type=float, default=0.12)
    ap.add_argument('--feel', type=int, default=2, help='feel words shared with a liked track')
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--bpm', type=float, default=None, help='a liked track not in the catalogue: its BPM')
    ap.add_argument('--feels', default=None, help='... and its feel in the catalogue\'s words: "Driving, Intense"')
    a = ap.parse_args()
    with open(CATALOGUE, encoding='utf-8') as fh:
        cat = json.load(fh)
    cands = ([(f, next((p for p in cat if same(f, p)), {})) for f in a.files] if a.files is not None
             else candidates(a.liked, cat, a.bpm_tol, a.feel, a.bpm, a.feels))
    print(f'{len(cands)} candidates')
    mdl = model_of()
    embs = {f: track_emb(f, mdl) for f in list(a.liked) + a.disliked + [f for f, _ in cands]}
    # every track's mean embedding is ~0.98 like any other's (music against all sounds): centred on the mean of
    # all tracks embedded so far (the catalogue cache and these), Paradise Found against At Launch went 0.98 -> -0.47
    seen = [np.load(os.path.join(CACHE, n)) for n in os.listdir(CACHE) if n.endswith('.clapm.npy')] if \
        os.path.isdir(CACHE) else []
    mu = np.mean(seen + list(embs.values()), 0)
    for f, e in embs.items():
        embs[f] = (e - mu) / np.linalg.norm(e - mu)
    like = np.mean([embs[f] for f in a.liked], 0)
    like /= np.linalg.norm(like)
    bad = np.array([embs[f] for f in a.disliked]) if a.disliked else np.zeros((0, like.size))
    rows = []
    for f, p in cands:
        e = embs[f]
        s_like = float(e @ like)
        s_bad = float((bad @ e).max()) if len(bad) else 0.0
        rows.append((s_like - DISLIKE_W * s_bad, s_like, s_bad, f, p))
    rows.sort(key=lambda r: -r[0])
    print('score  like  dislike  bpm  track | feel | instruments')
    for sc, sl, sb, f, p in rows[:a.top]:
        print(f'{sc:5.2f} {sl:5.2f} {sb:6.2f} {p.get("bpm", "?"):>5}  {os.path.relpath(f, paths.ROOT)} | '
              f'{p.get("feel", "")} | {(p.get("instruments") or "")[:50]}')


if __name__ == '__main__':
    main()
