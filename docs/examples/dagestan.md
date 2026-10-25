# Dagestan: a holiday report

**Footage:** 14 GoPro HERO8 clips (4K 30p, SuperView) from four days in September 2022, ~47 min, 31 GB. Five
of them, the owner's favourites, existed only as his old Gyroflow exports (`_stabilized`, no gyro left); their
originals were nowhere on the machine. The footage sat on a USB SSD that read these old files at 4-19 MB/s.
**Task:** a holiday report, 1-2 minutes, an overview. Dynamic flying; keep "us" in when there are people;
beautiful flybys over beautiful places too. The owner did not remember where they had been: "that's why I
want the video".

**What happened:**
- **Motion from the picture** (`vis_motion.py`): the five exports had no gyro, so the camera rotation comes
  from corners tracked between frames of the proxy, measured against the gyro of the clips that have one (see
  notes, Telemetry): 0.95-0.99 correlation on the three axes, flips up to 700°/s caught; tilt is unreliable low
  over a slope, so tilt manoeuvres count only where the rotation fits. The toolset takes such a clip as
  footage (`proxy.py` scales it down, finals cut it directly).
- **Beautiful views** (`beauty.py`, `--beauty 3`): an option, beauty in the job's own words (mountains, a canyon,
  a turquoise lake...) plus the LAION aesthetic score. The check sheet found the places by itself: the
  turquoise Sulak canyon (5 Sep), the abandoned stone village of Gamsutl on its peak (8 Sep); the ekranoplan on
  the Caspian beach (7 and 9 Sep) came from a search in words, and the aesthetic score put it among the worst
  views: a landmark is a theme or a moment.
- **Chapters** by day share scenes by good flying and, with `--beauty`, beautiful views (the Gamsutl day had
  71 good seconds of flying, 285 with the views); an editor's moment goes to a scene of its own day.
- **Landings** are marked (the 10 s before a touchdown) and kept out of all but the last scene: every clip ended
  in a slow approach and a flop that read as a roll.
- **Changing a finished edit** (`edl_edit.py`): planning again after a small comment moved other scenes; cutting
  a phrase out of the track moved every later scene off its beats ("the music stays, the picture is cut"). Now
  `--end-from K` lets the ending take over and `--replace K` finds one scene again, the rest frame for frame.

**Edits:** three first cuts (Living Voyage 122 BPM, Tabuk 125, Nonstop 128). Tabuk was dropped ("the music does
not fit"), Nonstop set aside. *Living Voyage* became the one, after four rounds of comments: 1:31, 12 scenes, in
date order: the Sulak canyon with people waving on the rocks, the owner in a yellow raincoat by the ekranoplan in
the rain, a pass under its engines, Gamsutl from above and the village on its peak, a dive over the surf line,
and the dive onto the ekranoplan seen whole from above as the ending. Piloting median 0.8; the owner: "it came
out great".

## Prompt and comments

> **Prompt:** There are many locations: I want a holiday report. Pick cool, dynamic flights; if there are people
> (us), try to keep them, but beautiful flybys over beautiful places are needed too. How will you understand
> that? A riddle. Maybe add a new "beauty" tracker? The final video: a minute, two at most. I don't want to
> listen to the music in advance: just make 2-3 videos; previews at 640 px for now.

The first previews:
- "Voyage sounds best, but 0:47 is a dull landing, not the point; the same moment is in the other video too.
  Tabuk is out: the music doesn't fit."
- "Voyage is the main one, it's fine; I only don't like its ending. The last 10 seconds: better put something
  else there."

Finishing Voyage:
- "1:04: a bad end of the shot with the houses on the mountain."
- "1:17-1:25 is rubbish, just me standing: it can go, or put something else instead. The plane at the end is
  fine as it is, and the music ends in silence anyway."
