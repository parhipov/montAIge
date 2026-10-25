# Koprino: a family film

**Footage:** 5 GoPro clips (50 fps) at a snowy resort: a mother walking a pram, the pilot in a yellow jacket,
a cat. Calm flying: one rotation trick in 20 min. The owner's old export in the folder was named after the
highlights ("Townhouse in the forest. And the cat!").
**Task:** a family film about the mother with the pram; calm flying.

**What happened:**
- The mother pushing the pram was taken for the pilot (241 of 373 people rows): both hands in front of the
  body fire the radio pose and "holding a drone radio". Adding look-alikes with busy hands (a pram, a phone,
  a bag, a child's hand) to the texts CLIP chooses from fixed it: 70 pilot rows, all the man in the yellow
  jacket. The owner on the first fix (a special rule for prams): "не пытайся системный промпт так прописывать
  — туда тока общее" ("don't write it like that, only general things go in"), "мы же делаем универсальную
  систему" ("we are building a universal system").
- The main character in words: `people.py subject --text "a woman pushing a baby stroller"`, all 24 picks on
  the check sheet right.
- Animals count like a person; the cat running towards the quad and, as the last shot, the cat peeking into
  the camera on the ground (54.5 s) went in as editor's moments. The owner loves the cat moments.
- A descent from height onto the houses (a new manoeuvre kind; it found exactly the three in this footage)
  went in with `--must`: the piloting score undervalues it, the owner liked it.
- Flying scenes (`--fly`) and `W_EMPTY` for shots with nobody in them.

**Edits:** Carefree (96 BPM) was the owner's pick, then "no flights to nowhere, shorter". With little people
footage the answer was a shorter edit: *Carefree* v2, 45 s, 9 shots, the mother in 6, the cat in 2, the
descent as one 10 s shot (the track cannot be cut to 55 s with good joins; 45 s it is: "ок"). Daily Beetle
and Life of Riley stayed first cuts. 4K final: 45 s, 384 MB, 9.5 min. *Carefree* v2 is one of the two clips
at the top of the README.

## Prompt and comments

> **Prompt:** A sweet family video: a mother walks a pram at a resort. A minute or a minute and a half; music
> your choice, 2-3 variants. If there is good piloting, it must be in.

Comments on the previews:
- "Too many moments of just flying to nowhere: cut them, make it shorter."
- "You left out the descent onto the houses from height, and it's beautiful."
- "The cat is fire. The cat at the end too, but carefully: it's at the edge of the frame."
- "45 s is fine."
