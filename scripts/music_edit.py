"""Shorten a studio track to a target length by cutting whole 16-beat phrases.

The path through the phrases keeps the intro and the real ending, keeps at
least one quiet phrase (the breakdown) when the track has one, and jumps only
where the music would have sounded the same: jumping from phrase i to j is
seamless when j resembles i+1 or i resembles j-1 (bar-aligned log spectra).
Joins sit exactly on phrase starts (downbeats), so the beat grid stays intact.

    python music_edit.py TRACK.mp3 OUT.wav --target 100
"""
import argparse
import itertools
import json
import wave

import av
import numpy as np
from scipy.signal import stft

import music as mus


SECTION_SIM = 0.5      # two phrases are one section above this, on similarity rescaled from the track's median pair


def phrase_features(path, beats, period, starts):
    """Per phrase, its 4 bars as log spectra (48 bands, 40 Hz-10 kHz), mean-centred, unit length."""
    y = mus.load_audio(path)
    f, t, Z = stft(y, fs=mus.SR, nperseg=4096, noverlap=4096 - 1024)
    S = np.abs(Z) ** 2
    edges = np.geomspace(40, 10000, 49)
    L = np.log10(np.array([S[(f >= a) & (f < b)].sum(0) for a, b in zip(edges[:-1], edges[1:])]) + 1e-9)

    def bar(b0):
        v = L[:, (t >= beats[b0]) & (t < beats[b0] + 4 * period)].mean(1)
        v = v - v.mean()
        return v / (np.linalg.norm(v) + 1e-12)
    return np.array([[bar(p + 4 * k) for k in range(4)] for p in starts])


def phrase_similarity(path, beats, period, starts):
    F = phrase_features(path, beats, period, starts)
    return np.einsum('ikd,jkd->ij', F, F) / 4


def sections(F, energy):
    """A section label per phrase (A, B, ... in order of appearance) and the chorus: the label that comes back
    with the highest mean energy (None if nothing comes back). Phrases are joined by average linkage on their
    similarity rescaled per track, the median pair to 0 and a copy to 1: the median sat at 0.46 (Shiny Tech) to
    0.94 (Morgana Rides), repeats at 0.93-0.99."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    n = len(F)
    if n < 2:
        return ['A'] * n, None
    S = np.einsum('ikd,jkd->ij', F, F) / 4
    med = np.median(S[~np.eye(n, dtype=bool)])
    D = np.clip(1 - (S - med) / max(1 - med, 1e-6), 0, None)
    np.fill_diagonal(D, 0)
    names, labels = {}, []
    for c in fcluster(linkage(squareform(D, checks=False), 'average'), 1 - SECTION_SIM, 'distance'):
        labels.append(names.setdefault(c, chr(65 + len(names) % 26)))
    again = {l for l in labels if labels.count(l) > 1}
    chorus = max(again, key=lambda l: np.mean([e for e, x in zip(energy, labels) if x == l]), default=None)
    return labels, chorus


def like(F, F_ref, labels_ref):
    """The label of the most similar reference phrase, per phrase (a shortened track against the whole one)."""
    return [labels_ref[int(np.argmax(np.einsum('kd,jkd->j', f, F_ref)))] for f in F]


def path_cost(seq, sim, dur, quiet, fixed_pre, fixed_post, target, cls=None):
    """The cost of one phrase sequence (the search's own formula), and its total length and jump cost."""
    n = len(dur)
    total = fixed_pre + fixed_post + sum(dur[i] for i in seq)
    k = sum(1 for a, b in zip(seq, seq[1:]) if b != a + 1)
    jump = 0.0
    for i, j in zip(seq, seq[1:]):
        if j != i + 1:
            jump += 1.0 - max(sim[i + 1, j] if i + 1 < n else -1, sim[i, j - 1])
    has_quiet = any(quiet[i] for i in seq) or not any(quiet)
    cost = abs(total - target) / 10.0 + 4.0 * jump + 0.05 * k + (0 if has_quiet else 5.0)
    if cls is not None:
        drop = any(b == a + 1 and cls[b] - cls[a] >= 2 for a, b in zip(seq, seq[1:]))
        cost += 1.5 * (1 - np.mean([cls[i] for i in seq]) / 3.0) - (0.5 if drop else 0.0)
    return cost, total, jump


def best_path(sim, dur, quiet, fixed_pre, fixed_post, target, max_jumps=3, cls=None):
    """Phrase index sequence (increasing, starts at 0, ends at the last) with <= max_jumps jumps.

    With energy classes (0 quiet .. 3 biggest) the path also prefers loud phrases and keeping
    at least one drop (a phrase at least 2 classes above the one before it, as in the original).

    The cost of a path is its jump cost plus terms of its count of phrases, jumps, energy sum and two flags
    (a quiet phrase kept, a drop kept), so a dynamic programme over (phrase, jumps, count, flags, energy sum)
    needs only the least jump cost per such bucket, and the best bucket is the search's optimum: the same
    path as trying every cut (`best_path_search`, which took 2 min on a 40-phrase track), in under a second.
    Ties are common (jumping i -> j and i+1 -> j+1 cost the same when j resembles i+1), and the search took the
    first set of cuts it tried (fewest jumps, then the earliest cuts): the programme runs from the end, so the
    path is read forwards choosing the earliest jump among the paths that are as good."""
    n = len(dur)
    if n < 2 or np.ptp(dur) > 1e-9:                              # phrases of unequal length: the plain search
        return best_path_search(sim, dur, quiet, fixed_pre, fixed_post, target, max_jumps, cls)
    K = max_jumps
    cl = [int(c) for c in cls] if cls is not None else [0] * n
    S = sum(cl) + 1
    drop_in = [False] + [cl[b] - cl[b - 1] >= 2 for b in range(1, n)]
    jump_w = {(i, j): 1.0 - max(sim[i + 1, j] if i + 1 < n else -1, sim[i, j - 1])
              for i in range(n) for j in range(i + 2, n)}
    INF = np.inf
    shape = (K + 1, n + 1, 2, 2, S)                              # jumps, phrases, quiet?, drop?, energy sum
    V = [np.full(shape, INF) for _ in range(n)]                  # least jump cost of a path from phrase i to the end
    V[n - 1][0, 1, int(quiet[n - 1]), 0, cl[n - 1]] = 0.0
    for i in range(n - 2, -1, -1):
        qi, ci = int(quiet[i]), cl[i]
        for j in range(i + 1, n):
            Vj = V[j]
            if not np.isfinite(Vj).any():
                continue
            jumped = j > i + 1
            w = jump_w[(i, j)] if jumped else 0.0
            di = int(drop_in[j] and not jumped)                  # the drop between i and i+1, kept
            ks, kd = (slice(0, K), slice(1, K + 1)) if jumped else (slice(0, K + 1), slice(0, K + 1))
            for q in (0, 1):
                for d in (0, 1):
                    src = Vj[ks, :n, q, d, :S - ci] + w
                    dst = V[i][kd, 1:, q | qi, d | di, ci:]
                    np.minimum(dst, src, out=dst)
    fixed = fixed_pre + fixed_post
    best, starts = None, []
    for k, c, q, d, s in zip(*np.nonzero(np.isfinite(V[0]))):
        total = fixed + c * dur[0]
        cost = abs(total - target) / 10.0 + 4.0 * float(V[0][k, c, q, d, s]) + 0.05 * k \
            + (0 if q or not any(quiet) else 5.0)
        if cls is not None:
            cost += 1.5 * (1 - s / c / 3.0) - (0.5 if d else 0.0)
        key = (round(cost, 9), int(k))
        if best is None or key < best:
            best, starts = key, []
        if key == best:
            starts.append((int(k), int(c), int(q), int(d), int(s)))
    # read the path forwards: the states on paths as good as the best, the earliest jump first (the search's order)
    seq, i, states = [0], 0, set(starts)
    while i < n - 1:
        moves = {}                                               # next phrase -> states there
        for k, c, q, d, s in states:
            v = V[i][k, c, q, d, s]
            for j in range(i + 1, n):
                jumped = j > i + 1
                if jumped and k == 0:
                    continue
                w = jump_w[(i, j)] if jumped else 0.0
                di = int(drop_in[j] and not jumped)
                for q2 in (0, 1):
                    for d2 in (0, 1):
                        if (q2 | int(quiet[i])) != q or (d2 | di) != d:
                            continue
                        sub = (k - jumped, c - 1, q2, d2, s - cl[i])
                        if V[j][sub] + w == v:
                            moves.setdefault(j, set()).add(sub)
        jumps = [j for j in moves if j > i + 1]
        i = min(jumps) if jumps else i + 1
        states = moves[i]
        seq.append(i)
    cost, total, jump = path_cost(seq, sim, dur, quiet, fixed_pre, fixed_post, target, cls)
    return cost, seq, total, jump


def best_path_search(sim, dur, quiet, fixed_pre, fixed_post, target, max_jumps=3, cls=None):
    """best_path by trying every set of cuts: exponential in the number of phrases (2 min at 40)."""
    n = len(dur)
    best = None
    for k in range(max_jumps + 1):
        for cuts in itertools.combinations(range(1, n), 2 * k):
            # cuts = (end1, start2, end2, start3, ...): play [0, end1) then [start2, end2) ...
            segs, a, ok = [], 0, True
            for e, s in zip(cuts[0::2], cuts[1::2]):
                if s <= e:
                    ok = False
                    break
                segs.append((a, e))
                a = s
            if not ok:
                continue
            segs.append((a, n))
            seq = [i for a, b in segs for i in range(a, b)]
            total = fixed_pre + fixed_post + sum(dur[i] for i in seq)
            jump = 0.0
            for (a1, b1), (a2, b2) in zip(segs, segs[1:]):
                i, j = b1 - 1, a2
                jump += 1.0 - max(sim[i + 1, j] if i + 1 < n else -1, sim[i, j - 1])
            has_quiet = any(quiet[i] for i in seq) or not any(quiet)
            cost = abs(total - target) / 10.0 + 4.0 * jump + 0.05 * k + (0 if has_quiet else 5.0)
            if cls is not None:
                drop = any(b == a + 1 and cls[b] - cls[a] >= 2 for a, b in zip(seq, seq[1:]))
                cost += 1.5 * (1 - np.mean([cls[i] for i in seq]) / 3.0) - (0.5 if drop else 0.0)
            if best is None or cost < best[0]:
                best = (cost, seq, total, jump)
    return best


def read_stereo(path, sr=44100):
    with av.open(path) as c:
        rs = av.AudioResampler(format='fltp', layout='stereo', rate=sr)
        ch = []
        for fr in c.decode(audio=0):
            ch += [o.to_ndarray() for o in rs.resample(fr)]
        ch += [o.to_ndarray() for o in rs.resample(None)]
    return np.concatenate(ch, axis=1).astype(np.float64)


def splice(x, sr, segs, xfade_s=0.02):
    """Concatenate [t0, t1) pieces of x (2, N); equal-power crossfade ending on each join."""
    out = None
    k = int(xfade_s * sr)
    for t0, t1 in segs:
        a, b = int(round(t0 * sr)), int(round(t1 * sr))
        piece = x[:, a:b]
        if out is None:
            out = piece.copy()
            continue
        # fade the outgoing tail over the last k samples before the join, the incoming
        # piece starts at full level so its downbeat transient is untouched
        pre = x[:, a - k:a] if a >= k else np.zeros((2, k))
        g = np.sin(np.linspace(0, np.pi / 2, k)) ** 2
        out[:, -k:] = out[:, -k:] * (1 - g) + pre * g
        out = np.concatenate([out, piece], axis=1)
    return out


def write_wav(path, x, sr):
    pcm = (np.clip(x.T, -1, 1) * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def shorten(path, out_wav, m, phrases, target, F=None):
    """phrases: (first beat, n beats, quiet?[, energy class]) of each 16-beat phrase from the planner; F: their
    phrase_features, when the caller has them already."""
    beats, P = m['beats'], m['period']
    starts = [p[0] for p in phrases]
    cls = [p[3] for p in phrases] if all(len(p) > 3 for p in phrases) else None
    if F is None:
        F = phrase_features(path, beats, P, starts)
    sim = np.einsum('ikd,jkd->ij', F, F) / 4
    dur = [16 * P] * len(starts)
    quiet = [p[2] for p in phrases]
    pre = beats[starts[0]]                                        # pickup: 0 .. first phrase
    end_last = starts[-1] + 16
    post = m['duration'] - beats[end_last] if end_last < len(beats) else m['duration'] - (beats[-1] + P)
    cost, seq, total, jump = best_path(sim, dur, quiet, pre, post, target, cls=cls)
    t_end = lambda i: beats[starts[i] + 16] if starts[i] + 16 < len(beats) else beats[starts[i]] + 16 * P
    segs, run = [], [seq[0]]
    for i in seq[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            segs.append(run)
            run = [i]
    segs.append(run)
    pieces = []
    for r in segs:
        t0 = 0.0 if r[0] == 0 else beats[starts[r[0]]]
        t1 = m['duration'] if r[-1] == len(starts) - 1 else t_end(r[-1])
        pieces.append((float(t0), float(t1)))
    sr = 44100
    x = splice(read_stereo(path, sr), sr, pieces)
    write_wav(out_wav, x, sr)
    return dict(source=path, phrases=seq, pieces=pieces, seconds=x.shape[1] / sr, jump_cost=float(jump))


if __name__ == '__main__':
    import sys
    import edit as B
    ap = argparse.ArgumentParser()
    ap.add_argument('track')
    ap.add_argument('out')
    ap.add_argument('--target', type=float, required=True)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    m = B.analyse_music(a.track)
    sc = B.plan_scenes(m, None)
    ph = [(s['shots'][0]['b0'], 16, s['cls'] == 0) for s in sc if s['kind'] == 'phrase']
    r = shorten(a.track, a.out, m, ph, a.target)
    print(json.dumps(r, indent=1))
