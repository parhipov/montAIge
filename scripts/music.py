"""Deterministic beat / bar / energy analysis (numpy + scipy only)."""
import numpy as np
import av
from scipy.signal import stft, butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d

SR = 22050
HOP = 256          # ~86 onset frames per second


def load_audio(path, sr=SR):
    with av.open(path) as c:
        rs = av.AudioResampler(format='flt', layout='mono', rate=sr)
        chunks = []
        for fr in c.decode(audio=0):
            for o in rs.resample(fr):
                chunks.append(o.to_ndarray().reshape(-1))
        for o in rs.resample(None):
            chunks.append(o.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float64)


def onset_envelope(y, sr=SR):
    f, t, Z = stft(y, fs=sr, nperseg=2048, noverlap=2048 - HOP, boundary=None, padded=False)
    S = np.log1p(100 * np.abs(Z))
    flux = np.maximum(0.0, np.diff(S, axis=1))
    low = flux[f < 200].sum(0)                     # kick / bass
    full = flux.sum(0)
    def norm(x):
        x = x - uniform_filter1d(x, 43)            # remove slow trend (~0.5 s)
        x = np.maximum(x, 0)
        return x / (x.std() + 1e-9)
    rate = sr / HOP
    times = (np.arange(full.size) + 1) * HOP / sr + 1024 / sr
    return norm(full), norm(low), times, rate


def tempo(env, rate, lo=70, hi=180, center=115):
    x = env - env.mean()
    ac = np.correlate(x, x, 'full')[x.size - 1:]
    lags = np.arange(int(rate * 60 / hi), int(rate * 60 / lo) + 1)
    bpm = 60 * rate / lags
    prior = np.exp(-0.5 * (np.log2(bpm / center) / 0.9) ** 2)
    score = ac[lags] * prior
    # refine with sub-lag parabola
    i = int(np.argmax(score))
    if 0 < i < len(score) - 1:
        a, b, c = score[i - 1], score[i], score[i + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        d = 0.0
    period = lags[i] + d
    return 60 * rate / period, period


def track_beats(env, period, tightness=100.0):
    """Ellis 2007 dynamic-programming beat tracker; period in onset frames."""
    n = env.size
    score = env.copy()
    back = -np.ones(n, dtype=int)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for t in range(n):
        a, b = t - hi, t - lo
        if b < 0:
            continue
        prev = np.arange(max(a, 0), b + 1)
        pen = -tightness * np.log((t - prev) / period) ** 2
        cand = score[prev] + pen
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[t] = env[t] + cand[j]
            back[t] = prev[j]
    # start from the best-scoring frame in the last period
    t = n - hi + int(np.argmax(score[n - hi:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1])


def analyse(path):
    y = load_audio(path)
    env, low, times, rate = onset_envelope(y)
    bpm, period = tempo(env, rate)
    b = track_beats(env, period)
    bt = times[b]
    # trim leading/trailing beats in silence
    rms = np.sqrt(uniform_filter1d(y ** 2, SR // 10))
    loud = rms[np.clip((bt * SR).astype(int), 0, y.size - 1)]
    keep = loud > 0.05 * loud.max()
    bt, b = bt[keep], b[keep]
    # downbeat phase: which of 4 positions carries the most kick energy
    win = lambda e, i: e[max(i - 2, 0):i + 3].max()
    kick = np.array([win(low, i) for i in b])
    phase = int(np.argmax([kick[p::4].mean() for p in range(4)]))
    # loudness per beat interval (dB)
    edges = np.append(bt, bt[-1] + 60 / bpm)
    e = np.array([np.sqrt(np.mean(y[int(edges[k] * SR):int(edges[k + 1] * SR)] ** 2) + 1e-12) for k in range(bt.size)])
    e_db = 20 * np.log10(e)
    # per-beat onset density (busy-ness): mean full-band onset between beats
    tb = np.searchsorted(times, edges)
    busy = np.array([env[tb[k]:max(tb[k + 1], tb[k] + 1)].mean() for k in range(bt.size)])
    acc = np.array([win(env, i) for i in b])            # accent strength on each beat
    return dict(path=path, duration=y.size / SR, bpm=bpm, beats=bt, downbeat_phase=phase,
                energy_db=e_db, busy=busy, accent=acc, kick=kick)


if __name__ == '__main__':
    import sys, os
    for p in sys.argv[1:]:
        r = analyse(p)
        bt = r['beats']; ibi = np.diff(bt)
        e = r['energy_db']
        nb = bt.size // 4
        bars = e[r['downbeat_phase']:][:(bt.size - r['downbeat_phase']) // 4 * 4].reshape(-1, 4).mean(1)
        print(f"{os.path.basename(p)}: {r['duration']:.1f}s, {r['bpm']:.1f} BPM, {bt.size} beats {bt[0]:.2f}..{bt[-1]:.2f}s, "
              f"IBI std {ibi.std()*1000:.1f} ms, downbeat phase {r['downbeat_phase']}")
        print('   bar loudness dB:', ' '.join(f'{v:.0f}' for v in bars))
