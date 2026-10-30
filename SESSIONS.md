# Moving a job to a new session

Every step of a session re-reads the whole conversation (the prompt cache makes it cheaper, not free): in a long
session most of the cost is the history, pictures included, read again on each step. A job done late in a long
session cost 3.5 times what it would on its own. `/compact` is no way out: its summary loses the exact seconds and
the reasons. So when a job reaches a resting point, the next part starts in a new session from a short file.

## When

- The owner accepted a preview, and what is left is mechanical: a 4K final, porting a one-off script into the
  repository, a batch of renders.
- A new job, or a new idea for the same job (another edit, another track).
- The session has run for hours and the next step does not need its details.

Not in the middle of tuning: while the owner comments preview after preview, the history is the context.

## How

1. The session asks the owner, in one line, at such a point: "a new session from here?" A no costs nothing;
   on a yes it writes `out/<job>/handoff.md` (a page of output):
   - **Accepted**: which preview, the owner's words, the exact commands and files that made it (EDL, options,
     LUTs, scripts in the job folder).
   - **Running / to do**: background jobs with their command lines and logs, the next steps in order, what to check
     first (a few seconds before the whole).
   - **Don't**: what the owner rejected, with the reason, so the next session does not offer it again.
   The job's write-up (`out/<job>/example.md`) and the memory stay as they are; the handoff points at them.
2. Background jobs (renders, analysis) belong to the session that started them and die when it is closed. Either
   wait for them, or leave that session open (idle costs nothing) and start the new one alongside; the handoff
   names their logs, and a rerun skips what is done.
3. The owner starts a new session with one line:

   ```
   продолжи по out/<job>/handoff.md
   ```

   The new session reads that file, then only what a step needs (a function's range, not a whole script).
4. On another machine: the handoff travels with the job folder (`out/<job>/`, copied there).

## What the handoff is not

Not the job's story (that is `example.md`), not a transcript, not a summary of every step: only what the next
session acts on. If a line would not change what it does, it does not go in.
