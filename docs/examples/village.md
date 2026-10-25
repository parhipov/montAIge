# Village: one place, three ideas

**Footage:** an archive shoot, 16 Aug 2025: 11 DJI O4 Pro clips (4K 4:3, 50 fps, 10-bit, normal colour,
~15 min) of one plot in a village: a yellow wooden house, a vegetable garden, an old man repairing the roof of
a shed, a kid with an orange lawn mower, granny and the kid on a swing, the pilot. One clip was the 1.2 s tail
of a split recording and was left out rather than joined. Two `_stabilized` files in the folder were the
owner's old exports.
**Task:** "archive recordings of a village: check the locations, one not too long edit per location; people
and good piloting; interesting to watch; 2-3 variants with different ideas and music; 640×480 is enough".

**Location:** one (a contact sheet, later `places.py locations`), so one edit in three versions:

| | leto | drive | sad |
|---|---|---|---|
| idea | summer in the village, people first | fast, piloting first | a magic garden, long shots |
| track | Still Pickin, 100 BPM | Hillbilly Swing, 145 BPM | Almost New, 80 BPM |
| length | 58 s | 48 s | 59 s |
| opening / arc | between the pilot's legs; take-off by the house, 1.67x | a vertical look down at the beds, 1.51x | the highest arc, 1.7x |

**The owner's comments and what they led to:**
- On whole-clip stabilization before planning: "а зачем? не проще сначала сформировать план и стабить куски?"
  ("why not plan first and stabilize pieces?"). The plan needs all the footage seen, decoding 4K is the cost
  either way, and Gyroflow decodes on the GPU faster than PyAV on the CPU; "нет ты прав, согласен" ("no, you
  are right").
- "длог там нет" ("there is no D-Log"): no LUT.
- The first cuts had the old man on the shed roof in 5 of 8 shots: `--theme`, then `places.py activities`.
- "добавь полетов вверх — интересно" ("add flights up, they are interesting") and "кроп сделай чуть больше —
  там на краях кадра видны края защиты линзы" ("crop a bit more, the lens guard's edges show"): the `ascent`
  kind from the picture, `--want`, `--zoom 1.16`.
- "ищи высокий подъем и спуск — красиво" ("look for a high climb and the way down, beautiful"): up-and-down
  arcs, pieces longer than a phrase played faster, `--arcs`.
- "сгенери 4к — запусти через час — 4к в папку 4к" ("render 4K, start in an hour, into a 4k folder"): a
  waiting script; 3840×2880, 50 fps, ~23 min per edit.
- "учитывай что нам не все надо каждый раз — по сути мы делаем инструментарий" ("not everything is needed every
  time: we are building a toolkit"), "мелочки иногда проще пилить на месте" ("small things are sometimes easier
  done on the spot"): new abilities are options and tools, not steps of every run.

**Also found:** the pickup before the first phrase took the opening from `--start`; the pilot standing in
front with someone far behind slipped past the pilot budget; YOLO animals were all false (cabbage beds,
clothes, a stool); `people.py who` could not carry the pilot by look at 640 px; a hero took the footage next
to an editor's piece; with the zoom the 4K pieces are wider than H.264 on NVENC allows.
