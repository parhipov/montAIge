"""What people say in the footage and who says it: speech in words (Whisper) and voices told apart (pyannote).

    models/venvs/speech/Scripts/python speech.py run --list originals.txt --proxy-dir DIR --out out/JOB [--language ru]
    python speech.py search "words" [...] --list originals.txt --proxy-dir DIR     the phrases that say it
    python speech.py text --list originals.txt --proxy-dir DIR                     the whole transcript

Per clip, cached in <render>_speech.json: pyannote's speaker-diarization-community-1 (models/diarization:
segmentation, WeSpeaker embeddings, VBx clustering) says who speaks when in the clip; Whisper (large-v3-turbo,
models/whisper) writes the words of that speech only (no text over music or silence: Whisper's hallucinations
live there), with their times, and a phrase it looped on is dropped (looped()); the words become phrases (a
sentence's end, a pause of PAUSE_S, another speaker, a comma after PHRASE_SOFT_S, PHRASE_MAX_S at most); CLAP
(sounds.py) tells singing from speech per phrase (`sung`: the best singing text's cosine over the best speaking
one's; sung over SUNG_OVER, since music under a host's words lifts it a little). A phrase also has its level
(dBFS), its clearness over what is heard around it without speech (`snr`, dB: background()) and the share of it
where two people speak at once (`overlap`).

Over the clips of a job, the clips' speakers are joined into voices by their embeddings (average linkage, cosine
SAME or more): V1 talks most. -> out/JOB/voices.json, out/JOB/voices/V1.wav (PORTRAIT_S of the voice, to
listen), out/JOB/transcript.txt (a line per phrase: clip, seconds, voice, words; sung ones marked).

A voice is nearly a person in view, but not quite: whoever holds the camera talks loudest and is never in the
picture (`level` of a voice). `run` needs a venv of its own (models/venvs/speech, or SPEECH_PY: torch with CUDA,
pyannote.audio 4, transformers, av): pyannote wants numpy 2. Reading the json (edit.py's say: pieces and --talk, timeline.py,
search) needs none of it.
"""
import argparse
import json
import os
import re
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run as a file
from embedia import paths  # noqa: E402

SR = 16000                                                      # what pyannote and Whisper run at
DIARIZATION = os.path.join(paths.MODELS, 'diarization', 'speaker-diarization-community-1')
WHISPER = 'openai/whisper-large-v3-turbo'
REGION_GAP_S, REGION_MAX_S = 0.5, 28.0          # speech regions Whisper hears: turns this close joined, this long
PAUSE_S, PHRASE_SOFT_S, PHRASE_MAX_S = 0.6, 6.0, 12.0   # a phrase ends at a pause, at a comma once this long, here
SAME = 0.5                                      # cosine of two clips' speakers for one voice
MIN_TALK_S = 1.0                                # a clip's speaker with less speech has no embedding worth joining
PORTRAIT_S = 20.0
SUNG = ['people singing', 'children singing a song', 'a choir singing']
SPOKEN = ['a person speaking', 'people talking', 'a child speaking']
CLAP_MAX_S = 10.0
LOOP_RATIO, LOOP_CPS = 2.4, 30.0                # a phrase Whisper looped on (looped())
SUNG_OVER = 0.08                                # a phrase is sung over this margin (sung())
AROUND_S = 5.0                                  # a phrase's clearness: over what is heard this near it
BATCH = 1                                       # Whisper pieces at once: word times keep the encoder's attention
#                                                 of all 32 layers, ~2.9 GB a piece (8 spilled 30 GB on an 8 GB card)
# Whisper's well-known lines on noise (subtitle credits, "thanks for watching"), in any language it writes them
HALLUCINATIONS = re.compile(r'субтитр|продолжение следует|спасибо за просмотр|подписывайтесь на (наш )?канал|'
                            r'amara\.org|thanks for watching|subtitles by', re.I)


def cache_path(render):
    return os.path.splitext(render)[0] + '_speech.json'


def load(render):
    """The clip's speech json, or None (no speech step, or no sound)."""
    p = cache_path(render)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def pairs_of(listfile, proxy_dir):
    from embedia import sounds
    return sounds.pairs_of(listfile, proxy_dir)


def clip_name(render):
    return os.path.basename(render).replace('_stabilized.mp4', '')


# ------------------------------------------------------------------------------------------- one clip
def regions(turns, dur, y):
    """The speech of a clip as pieces Whisper can take whole: turns closer than REGION_GAP_S joined; a longer
    piece cut at its quietest tenth of a second near every REGION_MAX_S."""
    out = []
    for t0, t1, _ in sorted(turns):
        if out and t0 - out[-1][1] < REGION_GAP_S:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1])
    rms = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(SR // 10) / (SR // 10), 'same'))
    res = []
    for t0, t1 in out:
        t0, t1 = max(t0 - 0.2, 0.0), min(t1 + 0.2, dur)
        while t1 - t0 > REGION_MAX_S:
            lo, hi = int((t0 + REGION_MAX_S - 6) * SR), int((t0 + REGION_MAX_S) * SR)
            cut = (lo + int(np.argmin(rms[lo:hi]))) / SR
            res.append((t0, cut))
            t0 = cut
        res.append((t0, t1))
    return res


def speaker_at(t0, t1, excl):
    """The speaker of the exclusive diarization overlapping (t0, t1) most, or None."""
    best, who = 0.0, None
    for a, b, s in excl:
        o = min(b, t1) - max(a, t0)
        if o > best:
            best, who = o, s
    return who


def phrases_of(words, excl, turns):
    """Words [(t0, t1, text)] -> phrases: a break at a sentence's end, a pause of PAUSE_S, another speaker, a comma
    after PHRASE_SOFT_S (a host's verse has commas only), or PHRASE_MAX_S."""
    out, cur = [], []
    for w in words:
        spk = speaker_at(w[0], w[1], excl)
        if cur and (w[0] - cur[-1][1] > PAUSE_S or re.search(r'[.!?…]$', cur[-1][2].strip())
                    or (spk is not None and cur[-1][3] is not None and spk != cur[-1][3])
                    or w[1] - cur[0][0] > PHRASE_MAX_S
                    or (cur[-1][1] - cur[0][0] >= PHRASE_SOFT_S and re.search(r'[,;:]$', cur[-1][2].strip()))):
            out.append(cur)
            cur = []
        cur.append((w[0], w[1], w[2], spk))
    if cur:
        out.append(cur)
    res = []
    for ws in out:
        text = ''.join(w[2] for w in ws).strip()
        t0, t1 = ws[0][0], ws[-1][1]
        if not text or HALLUCINATIONS.search(text) or looped(text, t1 - t0):
            continue
        spk = {}
        for w in ws:
            if w[3] is not None:
                spk[w[3]] = spk.get(w[3], 0.0) + w[1] - w[0]
        grid = np.arange(t0, t1, 0.05)
        many = np.zeros(len(grid))
        for a, b, _ in turns:
            many += (grid >= a) & (grid < b)
        res.append(dict(t0=round(t0, 2), t1=round(t1, 2), text=text,
                        speaker=max(spk, key=spk.get) if spk else None,
                        overlap=round(float((many >= 2).mean()) if len(grid) else 0.0, 2),
                        words=[[round(w[0], 2), round(w[1], 2), w[2].strip()] for w in ws]))
    return res


def sung(ph):
    """Is the phrase sung? A children's choir: 50 of 60 phrases over SUNG_OVER (median margin 0.16); a host
    talking with music under her: 3 of 62 (median 0.01; at 0, 37 of them were "sung")."""
    return ph.get('sung', -1.0) > SUNG_OVER


def looped(text, dur):
    """Whisper caught in a loop, as it is on singing: "листок, листок, ..." sixty times in 6 s, "Ооооо..." of 400
    letters in half a second. The text packs more than LOOP_RATIO (zlib, Whisper's own threshold), or comes
    faster than LOOP_CPS letters a second (speech is 10-20)."""
    import zlib
    b = text.encode('utf-8')
    return (len(b) > 40 and len(b) / len(zlib.compress(b)) > LOOP_RATIO) or len(text) > LOOP_CPS * max(dur, 0.5)


def levels(y, turns):
    """The clip's 0.1 s levels (dBFS), which of them are speech, and the quiet floor: the 10th percentile of
    the levels outside speech (of all of them when the clip talks all the time)."""
    lev = 20 * np.log10(np.sqrt(np.mean(y[:y.size // 1600 * 1600].reshape(-1, 1600).astype(np.float64) ** 2, 1))
                        + 1e-9)
    talk = np.zeros(len(lev), bool)
    for a, b, _ in turns:
        talk[int(a * 10):int(b * 10) + 1] = True
    return lev, talk, float(np.percentile(lev[~talk] if (~talk).sum() > 50 else lev, 10))


def background(ph, lev, talk, floor):
    """What is heard around a phrase without speech (dBFS): the median 0.1 s level outside speech within
    AROUND_S of it (music, a hall's noise), the clip's quiet floor when there is no such second. A phrase's snr
    is its level over this: over the clip's floor instead, a clip with a quiet minute ranked all its phrases
    first, whatever played under them."""
    a, b = max(int((ph['t0'] - AROUND_S) * 10), 0), min(int((ph['t1'] + AROUND_S) * 10), len(lev))
    bg = lev[a:b][~talk[a:b]]
    return float(np.median(bg)) if bg.size >= 10 else floor


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-9))


def analyse(original, render, models, language):
    """-> the clip's speech dict, cached in <render>_speech.json."""
    import torch
    from embedia import sounds
    from scipy.signal import resample_poly
    p = cache_path(render)
    if os.path.exists(p):
        return load(render)
    y48 = sounds.audio(original)
    res = dict(clip=clip_name(render), language=language, duration=0.0, turns=[], speakers={}, phrases=[])
    if y48 is None or y48.size < SR:
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, ensure_ascii=False)
        return res
    y = resample_poly(y48, 1, sounds.SR // SR).astype(np.float32)
    dur = y.size / SR
    res['duration'] = round(dur, 2)
    pipe, asr, clap = models(diarize=True)
    d = pipe({'waveform': torch.from_numpy(y)[None], 'sample_rate': SR})
    turns = [(s.start, s.end, k) for s, _, k in d.speaker_diarization.itertracks(yield_label=True)]
    excl = [(s.start, s.end, k) for s, _, k in d.exclusive_speaker_diarization.itertracks(yield_label=True)]
    labels = list(d.speaker_diarization.labels())
    emb = d.speaker_embeddings if d.speaker_embeddings is not None else np.zeros((0, 256))
    lev, talk, floor = levels(y, turns)
    words = []
    if turns:
        _, asr, _ = models(asr=True)
        regs = regions(turns, dur, y)
        kw = {'task': 'transcribe'}
        if language:
            kw['language'] = language
        outs = asr([{'raw': y[int(a * SR):int(b * SR)], 'sampling_rate': SR} for a, b in regs], batch_size=BATCH,
                   return_timestamps='word', generate_kwargs=kw)
        for (a, b), o in zip(regs, outs):
            for c in o.get('chunks', []):
                w0, w1 = c['timestamp']
                if w0 is None:
                    continue
                w1 = w0 + 0.3 if w1 is None else w1
                words.append((a + w0, min(a + w1, b), c['text']))
    phrases = phrases_of(sorted(words), excl, turns)
    if phrases:
        _, _, clap = models(clap=True)
        model, proc, dev = clap
        temb = sounds.text_emb(clap, SUNG + SPOKEN)
        for ph in phrases:
            a, b = int(ph['t0'] * sounds.SR), int(ph['t1'] * sounds.SR)
            mid, half = (a + b) // 2, int(CLAP_MAX_S * sounds.SR / 2)
            seg = y48[max(a, mid - half):min(b, mid + half)]
            if seg.size < sounds.SR // 2:
                seg = y48[max(mid - sounds.SR // 4, 0):mid + sounds.SR // 4]
            x = proc(audio=[seg], sampling_rate=sounds.SR, return_tensors='pt').to(dev)
            with torch.no_grad():
                e = sounds.unit(model.get_audio_features(input_features=x['input_features'].to(model.dtype),
                                                        is_longer=x.get('is_longer'))).cpu().numpy()[0]
            s = temb @ e
            ph['sung'] = round(float(s[:len(SUNG)].max() - s[len(SUNG):].max()), 3)
            ph['level'] = round(db(y[int(ph['t0'] * SR):int(ph['t1'] * SR)]), 1)
            ph['snr'] = round(ph['level'] - background(ph, lev, talk, floor), 1)
    for i, k in enumerate(labels):
        mine = [(a, b) for a, b, s in excl if s == k]
        e = np.asarray(emb[i], np.float64) if i < len(emb) else np.full(256, np.nan)
        ok = np.isfinite(e).all() and np.linalg.norm(e) > 0
        res['speakers'][k] = dict(talk=round(sum(b - a for a, b in mine), 2),
                                  level=round(float(np.median([db(y[int(a * SR):int(b * SR)]) for a, b in mine
                                                               if b - a > 0.3] or [-90.0])), 1),
                                  emb=[round(float(v), 5) for v in e / np.linalg.norm(e)] if ok else None,
                                  voice=None)
    res.update(floor=round(floor, 1), turns=[[round(a, 2), round(b, 2), k] for a, b, k in turns],
               exclusive=[[round(a, 2), round(b, 2), k] for a, b, k in excl], phrases=phrases)
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False)
    return res


class Models:
    """pyannote, Whisper and CLAP, each loaded the first time a clip needs it."""
    def __init__(self):
        self.pipe = self.asr = self.clap = None

    def __call__(self, diarize=False, asr=False, clap=False):
        import torch
        from embedia import gpu
        dev = gpu.device()
        if diarize and self.pipe is None:
            from pyannote.audio import Pipeline
            self.pipe = Pipeline.from_pretrained(DIARIZATION)
            self.pipe.to(torch.device(dev))
        if asr and self.asr is None:
            from transformers import pipeline
            self.asr = pipeline('automatic-speech-recognition', model=WHISPER, device=dev,
                                dtype=torch.float16 if dev == 'cuda' else torch.float32,
                                model_kwargs={'cache_dir': os.path.join(paths.MODELS, 'whisper')})
        if clap and self.clap is None:
            from embedia import sounds
            self.clap = sounds.Lazy()
            self.clap.load()
        return self.pipe, self.asr, self.clap


# ------------------------------------------------------------------------------------------- the job's voices
def speakers_of(data):
    """The clips' speakers with an embedding worth joining: [(render, label, unit embedding, talk s)]."""
    return [(r, k, np.array(s['emb']), s['talk']) for _, r, d in data for k, s in d['speakers'].items()
            if s.get('emb') and s['talk'] >= MIN_TALK_S]


def join(items, same=SAME):
    """Speakers [(render, label, embedding, talk)] -> {(render, label): 'V1'..} by average linkage on cosine
    (`same` or more), V1 talking most."""
    from scipy.cluster.hierarchy import fcluster, linkage
    groups = []
    if len(items) == 1:
        groups = [0]
    elif items:
        e = np.stack([x[2] for x in items])
        groups = list(fcluster(linkage(e, 'average', metric='cosine'), 1 - same, 'distance') - 1)
    talk = {}
    for (r, k, _, t), g in zip(items, groups):
        talk[g] = talk.get(g, 0.0) + t
    name = {g: f'V{i + 1}' for i, g in enumerate(sorted(talk, key=lambda g: -talk[g]))}
    return {(r, k): name[g] for (r, k, _, _), g in zip(items, groups)}


def voices(pairs, out, same=SAME):
    """The clips' speakers joined into voices (join), written into every <render>_speech.json
    (speakers[k]['voice'], phrases' 'voice') and out/voices.json, with a portrait of each voice in out/voices/."""
    data = [(o, r, load(r)) for o, r in pairs]
    data = [(o, r, d) for o, r, d in data if d is not None]
    of = join(speakers_of(data), same)
    summary = {}
    for _, r, d in data:
        for k, s in d['speakers'].items():
            s['voice'] = of.get((r, k))
            if s['voice']:
                v = summary.setdefault(s['voice'], dict(talk=0.0, clips=[], levels=[], phrases=0, sung=0))
                v['talk'] += s['talk']
                v['clips'].append(d['clip'])
                v['levels'].append((s['level'], s['talk']))
        for ph in d['phrases']:
            ph['voice'] = d['speakers'].get(ph['speaker'], {}).get('voice') if ph['speaker'] else None
            if ph['voice']:
                summary[ph['voice']]['phrases'] += 1
                summary[ph['voice']]['sung'] += sung(ph)
        with open(cache_path(r), 'w', encoding='utf-8') as fh:
            json.dump(d, fh, ensure_ascii=False)
    for v in summary.values():
        lv = sorted(v.pop('levels'))
        w = np.cumsum([t for _, t in lv])
        v['level'] = lv[int(np.searchsorted(w, w[-1] / 2))][0]
        v['talk'] = round(v['talk'], 1)
        v['clips'] = sorted(set(v['clips']))
    names = sorted(summary, key=lambda v: int(v[1:]))
    with open(os.path.join(out, 'voices.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(same=same, voices={v: summary[v] for v in names}), fh, ensure_ascii=False, indent=1)
    portraits(data, os.path.join(out, 'voices'), names)
    with open(os.path.join(out, 'transcript.txt'), 'w', encoding='utf-8') as fh:
        fh.write(transcript(data))
    print(f'voices: {len(names)} in {len(data)} clips (cosine >= {same}): ' + ', '.join(
        f'{v} {summary[v]["talk"]:.0f}s in {len(summary[v]["clips"])} clips at {summary[v]["level"]:.0f} dB'
        + (f' ({summary[v]["sung"]}/{summary[v]["phrases"]} phrases sung)' if summary[v]['sung'] else '')
        for v in names[:12]) + (' ...' if len(names) > 12 else ''))
    return summary


def portraits(data, folder, names):
    """PORTRAIT_S of each voice from its longest turns spoken alone -> folder/V1.wav (16 kHz)."""
    from embedia import sounds
    from scipy.signal import resample_poly
    os.makedirs(folder, exist_ok=True)
    turns, cache = {}, {}
    for o, _, d in data:
        for a, b, k in d.get('exclusive', []):
            v = d['speakers'].get(k, {}).get('voice')
            if v and b - a >= 1.0:
                turns.setdefault(v, []).append((b - a, a, b, o))
    for v in names:
        pick, tot = [], 0.0
        for n, a, b, src in sorted(turns.get(v, []), key=lambda x: -x[0]):
            pick.append((a, b, src))
            tot += n
            if tot >= PORTRAIT_S:
                break
        parts = []
        for a, b, src in sorted(pick, key=lambda x: (x[2], x[0])):
            if src not in cache:
                y = sounds.audio(src)
                cache[src] = None if y is None else resample_poly(y, 1, sounds.SR // SR)
            if cache[src] is not None:
                parts += [cache[src][int(a * SR):int(min(b, a + PORTRAIT_S) * SR)], np.zeros(SR // 4)]
        if parts:
            x = np.clip(np.concatenate(parts), -1, 1)
            with wave.open(os.path.join(folder, v + '.wav'), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((x * 32767).astype('<i2').tobytes())


def transcript(data):
    lines = []
    for _, _, d in data:
        for ph in d['phrases']:
            lines.append(f'{d["clip"]} {ph["t0"]:7.1f}-{ph["t1"]:7.1f}  {ph.get("voice") or "?":4s}'
                         f'{" ♪" if sung(ph) else "  "} {ph["text"]}')
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------------------------------- search
def norm_words(text):
    return re.findall(r'\w+', text.lower().replace('ё', 'е'))


def word_match(q, w):
    """A word of the query found in a phrase: the same word, or one with the same stem (Russian endings)."""
    k = max(len(q) - 2, 4)
    return w == q or (len(q) >= 4 and w[:k] == q[:k])


def find(renders, text, top=1):
    """The phrases that say `text` best (its words found, in runs of up to 3 phrases of a clip):
    [(render, t0, t1, text, voice, share of the words found)], best first; empty when under half of the words."""
    q = norm_words(text)
    if not q:
        return []
    found = []
    for r in renders:
        d = load(r)
        if not d:
            continue
        ph = d['phrases']
        for i in range(len(ph)):
            for j in range(i, min(i + 3, len(ph))):
                if j > i and ph[j]['t0'] - ph[j - 1]['t1'] > 1.5:
                    break
                ws = [w for p in ph[i:j + 1] for w in norm_words(p['text'])]
                share = sum(any(word_match(x, w) for w in ws) for x in q) / len(q)
                score = share - 0.02 * (j - i) - 0.002 * len(ws)
                found.append((score, share, r, ph[i]['t0'], ph[j]['t1'], ' '.join(p['text'] for p in ph[i:j + 1]),
                              ph[i].get('voice')))
    found = [x for x in sorted(found, key=lambda x: -x[0]) if x[1] >= 0.5]
    out = []
    for x in found:
        if all(x[2] != o[0] or x[4] <= o[1] or x[3] >= o[2] for o in out):
            out.append((x[2], x[3], x[4], x[5], x[6], x[1]))
        if len(out) >= top:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('run', 'voices', 'search', 'text'))
    ap.add_argument('texts', nargs='*')
    ap.add_argument('--list', required=True, help='originals.txt: one clip per line')
    ap.add_argument('--proxy-dir', required=True)
    ap.add_argument('--out', help='the index folder: voices.json, voices/, transcript.txt')
    ap.add_argument('--language', default=None, help='the footage\'s language for Whisper (ru, en, ...); '
                    'default: Whisper finds it per piece of speech')
    ap.add_argument('--same', type=float, default=SAME, help='cosine of two speakers for one voice')
    ap.add_argument('--top', type=int, default=5)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    pairs = pairs_of(a.list, a.proxy_dir)
    if a.step == 'run':
        import warnings
        warnings.filterwarnings('ignore', message='.*TensorFloat-32.*')
        warnings.filterwarnings('ignore', message='.*degrees of freedom is <= 0.*')
        models = Models()
        for o, r in pairs:
            if os.path.exists(cache_path(r)):
                continue
            d = analyse(o, r, models, a.language)
            ns = sum(sung(p) for p in d['phrases'])
            print(f'{d["clip"]}: {len(d["speakers"])} speakers, {sum(b - a_ for a_, b, _ in d["turns"]):.0f}s of '
                  f'speech, {len(d["phrases"])} phrases ({ns} sung)', flush=True)
    if a.step in ('run', 'voices'):
        if not a.out:
            raise SystemExit('--out: the index folder')
        voices(pairs, a.out, a.same)
        return
    if a.step == 'text':
        print(transcript([(o, r, load(r)) for o, r in pairs if load(r)]), end='')
        return
    for text in a.texts:
        hits = find([r for _, r in pairs], text, a.top)
        print(f'{text}: ' + ('; '.join(f'{clip_name(r)}:{t0:.1f}-{t1:.1f} {v or "?"} ({s:.2f}) "{tx}"'
                                       for r, t0, t1, tx, v, s in hits) or 'nothing'))


if __name__ == '__main__':
    main()
