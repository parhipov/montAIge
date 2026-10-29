"""What happens in the footage, in words: a local vision-language model describes every CHUNK_S seconds of a
render as a short video clip (FPS_IN frames a second, CHUNK_W px wide).

    python -m embedia.analyzers.captions RENDER.mp4 [...] [--chunk 3] [--batch 6]   -> <render>_captions.npz
    python -m embedia.analyzers.captions --search "jumping into the water" RENDER.mp4 [...]   the stretches
                                                                                   whose captions say so
    python -m embedia.analyzers.captions --ask "What is the person doing?" IMAGE.jpg [...]   one question about
                                                                                   pictures (review.py)

The model (MODEL, Qwen3-VL 2B, from Hugging Face into models/vlm; ~4 GB of GPU memory, 0.5 s a chunk on an RTX
3050 in batches of 6) answers one question per chunk in a sentence: who is there and what they are doing, or what
the place is. A caption is a searchable record of the moment: "the jump off the roof", "someone waving", "where
does the boy dance" are found by words in the captions (search, edit.py --must "cap:TEXT", timeline.py's caption
column) instead of by looking at frame sheets. Single frames every 2 s missed a jump that a 3 s clip of the same
footage described ("one person is jumping into the water"); CLIP on whole frames had missed a small person in the
air, and zero-shot action words on person crops fired on a third of all rows. A 2B model is still a 2B model: on a
yacht it called a boy's dance "jumping"; the captions are a first answer, the frame sheet the proof.

A word of the query matches a caption word by its stem (dancing, dances, danced -> danc); every word of the query
has to match. Stretches: matching chunks with gaps of one chunk bridged, the longest first.
"""
import argparse
import os
import re
import time

import av
import numpy as np

from .. import paths
from ..media import pix

MODEL = 'Qwen/Qwen3-VL-2B-Instruct'
CACHE = os.path.join(paths.MODELS, 'vlm')
CHUNK_S, FPS_IN, CHUNK_W = 3.0, 4, 448
# no list of actions in the question: given one, the model named "jumping into the water" in 21 of 55 chunks of a
# clip with one jump; asked as a careful observer it did in 9, the jump itself among them
PROMPT = ('In one sentence, say what the people in this clip are doing, as a careful observer would: name only an '
          'action you actually see happen (someone in the air for a jump, arms up and moving for a dance, an arm '
          'waving), otherwise say they are standing, sitting, swimming or walking. If there are no people or '
          'animals, say what the place is in a few words.')
MAX_NEW = 48


class VLM:
    """The model and its processor, loaded on first use."""
    def __init__(self, name=MODEL, cache=CACHE):
        self.name, self.cache, self.parts = name, cache, None

    def load(self):
        if self.parts is None:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from ..media import gpu
            dev = gpu.device()
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.name, cache_dir=self.cache, dtype=torch.float16,
                                                                    device_map=dev, attn_implementation='sdpa').eval()
            proc = AutoProcessor.from_pretrained(self.name, cache_dir=self.cache)
            proc.tokenizer.padding_side = 'left'
            self.parts = (model, proc, dev)
        return self.parts

    def _answer(self, inp, max_new):
        import torch
        model, proc, dev = self.load()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        out = out[:, inp['input_ids'].shape[1]:]
        return [t.strip().replace('\n', ' ') for t in proc.batch_decode(out, skip_special_tokens=True)]

    def ask(self, images, prompt=PROMPT, max_new=MAX_NEW):
        """One answer per PIL image (one batch), the same question to each."""
        model, proc, dev = self.load()
        msgs = [[{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]] * len(images)
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        return self._answer(proc(text=texts, images=list(images), return_tensors='pt', padding=True).to(dev), max_new)

    def ask_video(self, chunks, prompt=PROMPT, max_new=MAX_NEW):
        """One answer per chunk (an array of frames T x H x W x 3 at FPS_IN), the same question to each."""
        model, proc, dev = self.load()
        msgs = [[{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt}]}]] * len(chunks)
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        meta = [dict(fps=float(FPS_IN), total_num_frames=len(c), duration=len(c) / FPS_IN,
                     frames_indices=list(range(len(c)))) for c in chunks]      # the frames' timestamps for the model
        try:
            inp = proc(text=texts, videos=list(chunks), video_metadata=meta, return_tensors='pt', padding=True,
                       do_sample_frames=False)
        except (TypeError, ValueError):                          # another transformers: without the metadata
            inp = proc(text=texts, videos=list(chunks), return_tensors='pt', padding=True, do_sample_frames=False)
        return self._answer(inp.to(dev), max_new)


def cache_path(render):
    return os.path.splitext(render)[0] + '_captions.npz'


def caption(render, vlm, chunk_s=CHUNK_S, batch=6, prompt=PROMPT):
    """-> (first frame of each chunk, its caption) of a render, cached in <render>_captions.npz."""
    p = cache_path(render)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return z['frame'], z['text']
    c = av.open(render)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    W, H = st.codec_context.width, st.codec_context.height
    w = min(CHUNK_W, W)
    h = int(round(w * H / W / 2)) * 2
    every = max(1, int(round(fps / FPS_IN)))
    per = int(round(chunk_s * FPS_IN))
    frames, texts, chunks, starts, cur = [], [], [], [], []

    def flush():
        if chunks:
            texts.extend(vlm.ask_video(chunks, prompt))
            frames.extend(starts)
            chunks.clear()
            starts.clear()
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i % every:
            continue
        if not cur:
            start = i
        cur.append(pix.ndarray(fr, format='rgb24', width=w, height=h))
        if len(cur) == per:
            chunks.append(np.stack(cur))
            starts.append(start)
            cur = []
            if len(chunks) == batch:
                flush()
    if len(cur) >= per // 2:                                     # the last, shorter chunk
        chunks.append(np.stack(cur))
        starts.append(start)
    flush()
    c.close()
    np.savez(p, frame=np.array(frames, np.int64), text=np.array(texts, dtype=str), fps=fps, chunk_s=chunk_s)
    return np.array(frames), np.array(texts)


def stem(w):
    w = re.sub(r'[^a-z]', '', w.lower())
    for suf in ('ing', 'ed', 'es', 's'):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def matches(text, query):
    """Every word of the query matches a word of the caption by its stem."""
    words = {stem(w) for w in re.findall(r'[A-Za-z]+', text)}
    return all(stem(q) in words for q in re.findall(r'[A-Za-z]+', query) if stem(q))


def stretches(frames, texts, fps, query, chunk_s=CHUNK_S):
    """(start s, end s, matching chunks) of runs of captions matching the query, one-chunk gaps bridged, the
    longest first."""
    hit = np.array([matches(t, query) for t in texts], bool)
    out, k = [], 0
    while k < len(hit):
        if not hit[k]:
            k += 1
            continue
        j, n = k, 0
        while j < len(hit) and (hit[j] or (j + 1 < len(hit) and hit[j + 1])):
            n += int(hit[j])
            j += 1
        out.append((frames[k] / fps, frames[j - 1] / fps + chunk_s, n))
        k = j
    return sorted(out, key=lambda s: (-s[2], s[0]))


def find(renders, query):
    """The best stretch of all renders: (render, start s, end s, matching chunks), or None."""
    best = None
    for r in renders:
        p = cache_path(r)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        for s in stretches(z['frame'], z['text'], float(z['fps']), query, float(z['chunk_s'])):
            if best is None or (s[2], -s[0]) > (best[3], -best[1]):
                best = (r, s[0], s[1], s[2])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--chunk', type=float, default=CHUNK_S, help='seconds per captioned clip')
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--search', default=None, help='words to find in the captions')
    ap.add_argument('--ask', default=None, help='a question about the pictures given instead of renders')
    ap.add_argument('--top', type=int, default=6)
    ap.add_argument('--prompt', default=PROMPT, help='the question put to the model about every chunk')
    a = ap.parse_args()
    vlm = VLM(a.model)
    if a.ask:
        from PIL import Image
        for p, t in zip(a.renders, vlm.ask([Image.open(p).convert('RGB') for p in a.renders], a.ask, 96)):
            print(f'{os.path.basename(p)}: {t}')
        return
    if a.search:
        for r in a.renders:
            if not os.path.exists(cache_path(r)):
                print(f'{os.path.basename(r)}: no captions yet')
                continue
            z = np.load(cache_path(r), allow_pickle=False)
            st = stretches(z['frame'], z['text'], float(z['fps']), a.search, float(z['chunk_s']))
            print(f'{os.path.basename(r)}: {len(st)} stretches' + ''.join(
                f'\n   {s0:6.1f}-{s1:6.1f}s  {n} chunks' for s0, s1, n in st[:a.top]))
        return
    for r in a.renders:
        t0 = time.time()
        done = os.path.exists(cache_path(r))
        fr, tx = caption(r, vlm, a.chunk, a.batch, a.prompt)
        people = sum(1 for t in tx if re.search(r'\b(man|woman|person|people|boy|girl|child|children|kids?)\b', t, re.I))
        print(f'{os.path.basename(r)}: {len(fr)} captions' + (' (done before)' if done else f' in {time.time() - t0:.0f}s')
              + f', people in {people}', flush=True)


if __name__ == '__main__':
    main()
