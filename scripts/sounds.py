"""The footage's own sound in words (CLAP, the audio twin of CLIP): laughter, a shout of joy, a splash.

    python sounds.py embed --list originals.txt --proxy-dir DIR          -> DIR/<clip>_stabilized_clap.npz
    python sounds.py search "people laughing" [...] --list originals.txt --proxy-dir DIR    the best moments

Windows of WIN_S every HOP_S of the clip's sound (48 kHz mono), embedded by CLAP (laion/larger_clap_general,
general sounds, not music; models/clap fills itself from Hugging Face). A sound in words is told from what else
footage hears (ALTERNATIVES), as places.py tells a place from other places. edit.py takes a sound as an editor's
piece: --must "hear:people laughing" (the picture of that moment: whoever laughs is usually in it).

A carried camera hears people; a drone hears its propellers. Arkhyz (a helmet GoPro on the slopes, 66 min): wind
in the microphone 1535 s, skis on snow 624 s, a humming machine 833 s, a shout of joy 110 s, laughter 11 s,
talking 6 s; the best laugh fell inside the stretch of Marina up close the owner liked most. 8 min for the 66 min,
read from a busy disk.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

NAME = 'laion/larger_clap_general'
SR, WIN_S, HOP_S, BATCH = 48000, 2.0, 1.0, 32
ALTERNATIVES = ['wind blowing into a microphone', 'skis scraping on snow', 'a machine humming', 'quiet',
                'footsteps', 'traffic noise', 'water splashing', 'music playing', 'people talking', 'people laughing',
                'a person shouting with joy', 'propellers buzzing']
PIECE_S = 3.0           # a sound found is shown as a piece this long around it


TEXT_CACHE = os.path.join(paths.MODELS, 'clap', 'texts.npz')     # text embeddings computed so far


class Lazy:
    """The CLAP model (model, processor, device), loaded the first time it is unpacked: a script whose sound
    embeddings are cached and whose texts are in TEXT_CACHE never imports torch (edit.py --bites)."""
    def __init__(self, cache=None):
        self.cache, self.parts = cache or os.path.join(paths.MODELS, 'clap'), None

    def load(self):
        if self.parts is None:
            import torch  # noqa: F401
            from transformers import ClapModel, ClapProcessor
            import gpu
            dev = gpu.device()
            m = ClapModel.from_pretrained(NAME, cache_dir=self.cache).to(dev).eval()
            if dev != 'cpu':
                m = m.half()
            self.parts = (m, ClapProcessor.from_pretrained(NAME, cache_dir=self.cache), dev)
        return self.parts

    def __iter__(self):
        return iter(self.load())


def model_of(cache=None):
    return Lazy(cache)


def unit(out):
    import torch
    out = (out if torch.is_tensor(out) else getattr(out, 'pooler_output', out[0])).float()
    return out / out.norm(dim=-1, keepdim=True)


def text_emb(mdl, texts):
    """Unit embeddings of the texts, kept in TEXT_CACHE: only texts not seen before go through the model."""
    known = {}
    if os.path.exists(TEXT_CACHE):
        with np.load(TEXT_CACHE) as z:
            known = dict(zip(z['text'].tolist(), z['emb']))
    new = [t for t in dict.fromkeys(texts) if t not in known]
    if new:
        import torch
        model, proc, dev = mdl
        with torch.no_grad():
            e = unit(model.get_text_features(**proc(text=new, return_tensors='pt', padding=True).to(dev))).cpu().numpy()
        known.update(zip(new, e.astype(np.float32)))
        os.makedirs(os.path.dirname(TEXT_CACHE), exist_ok=True)
        np.savez(TEXT_CACHE + '.tmp.npz', text=np.array(list(known), dtype=str),
                 emb=np.stack(list(known.values())).astype(np.float32))
        os.replace(TEXT_CACHE + '.tmp.npz', TEXT_CACHE)
    return np.stack([known[t] for t in texts]).astype(np.float32)


def audio(path):
    """The clip's sound, mono at SR, or None without a sound track."""
    import av
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        rs = av.AudioResampler(format='flt', layout='mono', rate=SR)
        ch = [o.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for o in rs.resample(fr)]
    return np.concatenate(ch) if ch else None


def cache_path(render):
    return os.path.splitext(render)[0] + '_clap.npz'


def embed(original, render, mdl):
    """-> (window centres s, unit embeddings), cached in <render>_clap.npz; None without sound."""
    import torch
    p = cache_path(render)
    if os.path.exists(p):
        z = np.load(p)
        return (z['t'], z['emb'].astype(np.float32)) if z['t'].size else None
    y = audio(original)
    t, emb = np.zeros(0), np.zeros((0, 512), np.float16)
    if y is not None and y.size >= WIN_S * SR:
        model, proc, dev = mdl
        W, H = int(WIN_S * SR), int(HOP_S * SR)
        starts = np.arange(0, y.size - W + 1, H)
        out = []
        for b in range(0, starts.size, BATCH):
            x = proc(audio=[y[s:s + W] for s in starts[b:b + BATCH]], sampling_rate=SR, return_tensors='pt').to(dev)
            with torch.no_grad():
                out.append(unit(model.get_audio_features(input_features=x['input_features'].to(model.dtype),
                                                         is_longer=x.get('is_longer'))).cpu().numpy())
        t, emb = (starts + W / 2) / SR, np.concatenate(out).astype(np.float16)
    np.savez(p, t=t, emb=emb)
    return (t, emb.astype(np.float32)) if t.size else None


def probs(emb, temb, n_q):
    """Softmax over all texts (CLAP's scale 100), the first n_q columns: the sounds asked for."""
    s = 100.0 * emb @ temb.T
    s = np.exp(s - s.max(1, keepdims=True))
    return (s / s.sum(1, keepdims=True))[:, :n_q]


def moments(pairs, text, cache=None, top=10):
    """The best moments of a sound in words over (original, render) pairs: [(prob, render, second)], strongest
    first, at least 5 s apart within a clip."""
    from scipy.ndimage import uniform_filter1d
    mdl = model_of(cache)
    temb = text_emb(mdl, [text] + [x for x in ALTERNATIVES if x != text])
    found = []
    for o, r in pairs:
        e = embed(o, r, mdl)
        if e is None:
            continue
        p = uniform_filter1d(probs(e[1], temb, 1)[:, 0], 3)
        for i in np.argsort(-p):
            if p[i] < 0.3 or len([1 for _, rr, s in found if rr == r]) >= top:
                break
            if all(abs(s - e[0][i]) >= 5.0 for _, rr, s in found if rr == r):
                found.append((float(p[i]), r, float(e[0][i])))
    return sorted(found, reverse=True)[:top]


def find(pairs, text, cache=None):
    """The best moment of a sound: (render, start s, end s, prob) of a PIECE_S piece around it, or None."""
    m = moments(pairs, text, cache, top=1)
    if not m:
        return None
    p, r, s = m[0]
    return r, max(s - PIECE_S / 2, 0.0), s + PIECE_S / 2, p


def pairs_of(listfile, proxy_dir):
    with open(listfile, encoding='utf-8') as fh:
        clips = [x.strip() for x in fh if x.strip()]
    out = []
    for c in clips:
        r = os.path.join(proxy_dir, os.path.splitext(os.path.basename(c))[0] + '_stabilized.mp4')
        if os.path.exists(r):
            out.append((c, r))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('embed', 'search'))
    ap.add_argument('texts', nargs='*')
    ap.add_argument('--list', required=True, help='originals.txt: one clip per line')
    ap.add_argument('--proxy-dir', required=True)
    ap.add_argument('--cache', default=None, help='CLAP model folder (default models/clap)')
    ap.add_argument('--top', type=int, default=8)
    a = ap.parse_args()
    pairs = pairs_of(a.list, a.proxy_dir)
    if a.step == 'embed':
        mdl = None
        for o, r in pairs:
            if os.path.exists(cache_path(r)):
                continue
            mdl = mdl or model_of(a.cache)
            e = embed(o, r, mdl)
            print(f'{os.path.basename(o)}: ' + (f'{e[0].size} windows' if e else 'no sound'), flush=True)
        return
    for text in a.texts:
        m = moments(pairs, text, a.cache, a.top)
        print(f'{text}: ' + (', '.join(f'{os.path.basename(r).replace("_stabilized.mp4", "")} {s:.0f}s ({p:.2f})'
                                      for p, r, s in m) or 'nothing'))


if __name__ == '__main__':
    main()
