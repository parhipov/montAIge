"""Is a track fit for edit.py? Constant-tempo grid fit and the energy classes of its phrases.

    python music_check.py TRACK.mp3 BPM [TRACK.mp3 BPM ...]      (BPM from the incompetech catalogue)

Per track: the fitted tempo, how far the strongest onsets sit from the beat grid (median, ms; a
studio track sits within ~10 ms, a played or swung one 50+ ms and should be dropped), how many of
them land on a beat or an eighth, how sure the downbeat is (the kick on it over the next strongest
position of four; a kick on 1 and 3 alike flips it between the whole track and its cut, so the
phrases and every scene cut land half a bar off: take another track), and the energy class of every
16-beat phrase (0 quiet .. 3 biggest), which tells what music_edit.py can keep. A phrase is the unit
the track is shortened by, so edit lengths come in steps of 16 beats (10 s at 96 BPM).
"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository: embedia, montaige (run as a file)
from montaige import edit as bf  # noqa: E402

DOWNBEAT_SURE = 10   # %: a track with the kick 4 % stronger on 1 than on 3 had its cut's downbeat on 3

if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
    raise SystemExit(__doc__)
for path, bpm in zip(sys.argv[1::2], sys.argv[2::2]):
    m = bf.analyse_music(path, float(bpm))
    env, t = m['env'], m['env_t']
    pk, _ = find_peaks(env, height=np.percentile(env, 95), distance=5)
    d = np.abs(t[pk][:, None] - m['beats'][None, :]).min(1) * 1000
    half = np.abs(t[pk][:, None] - (m['beats'][None, :] + m['period'] / 2)).min(1) * 1000
    _, cls, _, _, _ = bf.phrase_layout(m)
    off = np.median(np.minimum(d, half))
    kick = np.sort([m['kick'][p::4].mean() for p in range(4)])
    sure = (1 - kick[-2] / max(kick[-1], 1e-9)) * 100
    print(f"{os.path.basename(path):26} {m['duration']:6.1f}s {m['bpm']:7.2f} BPM, phrase {16 * m['period']:4.1f}s | "
          f"strong onsets {pk.size}: median off-grid {off:.0f} ms ({'OK' if off < 20 else 'DROP'}), "
          f"on beat {np.mean(d < 40) * 100:.0f}%, on an eighth {np.mean(np.minimum(d, half) < 40) * 100:.0f}% | "
          f"downbeat +{sure:.0f}% kick ({'OK' if sure >= DOWNBEAT_SURE else 'UNSURE'}) | phrases {''.join(map(str, cls))}")
