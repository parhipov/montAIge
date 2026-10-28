"""Is a track fit for edit.py? Constant-tempo grid fit and the energy classes of its phrases.

    python music_check.py TRACK.mp3 BPM [TRACK.mp3 BPM ...]      (BPM from the incompetech catalogue)

Per track: the fitted tempo, how far the strongest onsets sit from the beat grid (median, ms; a
studio track sits within ~10 ms, a played or swung one 50+ ms and should be dropped), how many of
them land on a beat or an eighth, how sure the downbeat is (the kick on it over the next strongest
position of four; a kick on 1 and 3 alike flips it between the whole track and its cut, so the
phrases and every scene cut land half a bar off: take another track), whether the grid is on the beat (the
sub-bass on the half-beats against the beats: a bass under the half-beats means the grid sits on an oom-pah's
chord, half a beat off; the onsets there say how even the eighths are, for reading only), what beats.py's tracker
moved (the grid onto the beats and the downbeat a listener hears; the checks above are measured after it), and the
energy class of every 16-beat phrase (0 quiet .. 3 biggest), which tells what music_edit.py can keep. A phrase is the unit
the track is shortened by, so edit lengths come in steps of 16 beats (10 s at 96 BPM).
"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks, stft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository: embedia, montaige (run as a file)
from montaige import edit as bf  # noqa: E402
from montaige import beats as heard_beats  # noqa: E402
from montaige import music as mus  # noqa: E402

DOWNBEAT_SURE = 10   # %: a track with the kick 4 % stronger on 1 than on 3 had its cut's downbeat on 3
BASS_OFF = 2.0       # sub-bass on the half-beats over the beats: country, bluegrass and polka 2.8-5.7 (the strummed
                     # chord is the strongest onset, the bass falls between the grid's beats; the chords then change on its
                     # "4 and"); 13 others 0.2-1.5


def beat_phase(path, m):
    """The onsets (all bands) and the sub-bass flux (BOOM_BAND) on the half-beats over those on the beats."""
    f, t, Z = stft(mus.load_audio(path), fs=mus.SR, nperseg=2048, noverlap=2048 - 256)
    flux, t = np.maximum(0.0, np.diff(np.log1p(100 * np.abs(Z)), axis=1)), t[1:]
    sub = flux[(f >= bf.BOOM_BAND[0]) & (f < bf.BOOM_BAND[1])].sum(0)

    def at(e, x):
        return np.mean([e[j - 2:j + 3].max() for j in np.clip(np.searchsorted(t, x), 2, t.size - 3)])
    b, h = m['beats'], m['beats'] + m['period'] / 2
    return at(flux.sum(0), h) / at(flux.sum(0), b), at(sub, h) / at(sub, b)

if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
    raise SystemExit(__doc__)
heard_beats.ensure(sys.argv[1::2])
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
    flat, bass = beat_phase(path, m)
    beat = 'DROP' if bass >= BASS_OFF else 'OK'
    print(f"{os.path.basename(path):26} {m['duration']:6.1f}s {m['bpm']:7.2f} BPM, phrase {16 * m['period']:4.1f}s | "
          f"strong onsets {pk.size}: median off-grid {off:.0f} ms ({'OK' if off < 20 else 'DROP'}), "
          f"on beat {np.mean(d < 40) * 100:.0f}%, on an eighth {np.mean(np.minimum(d, half) < 40) * 100:.0f}% | "
          f"downbeat +{sure:.0f}% kick ({'OK' if sure >= DOWNBEAT_SURE else 'UNSURE'}) | "
          f"half-beats {flat:.2f} of the onsets, bass {bass:.1f}x ({beat}) | phrases {''.join(map(str, cls))}"
          + ('' if not m['heard'] else f" | beats.py: grid {m['heard']['shift']:+.2f} beat, downbeat "
             + ('unsure' if m['heard']['told'] is None else 'as the kick' if m['heard']['told'] == m['heard']['kick']
                else f"{m['heard']['told']}, not the kick's {m['heard']['kick']}")))
