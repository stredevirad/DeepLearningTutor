# Deep Learning Tutor - working rules

A Streamlit workbench for learning deep learning: load a dataset, build a Keras model, train it,
inspect the results (test set, confusion matrix, Grad-CAM), export a script, and ask a Gemini
tutor. A student project: tidy, honest, measured. Every change must be easy to explain.

## Every session
1. Read `memory/roadmap.md` before any work. Check `git status -sb` and `gh run list -L 1`.
2. Resume at the first item that is not DONE or SKIPPED. Never redo a DONE item, never skip one.
3. One roadmap item per session (two only when both are size S), then stop with a three-line
   status: what was done, the commit, the next item. Nothing may depend on chat history.
4. If CI is red at the start of a session, fixing it is that session's whole job.

## Checkpoint (end of every item, in this order)
1. Targeted tests pass, then the full suite passes and pyflakes is silent.
2. Update `memory/roadmap.md`: status, what changed, files, test count, before/after numbers,
   anything undone, "Exact next action".
3. One commit with the code and the roadmap together. Privacy check. `git push origin main`.
4. `git status -sb` shows main level with origin/main.
5. The item's commit hash and CI result go into the roadmap at the start of the next item.

## Rules
- No `git clean`, no `git reset --hard`, no force push, no history rewriting. Drop unfinished work
  with `git stash` or `git restore`. Work on main; it is the only remote branch.
- One item = one commit = one push. Never mix two items.
- Tests stay hermetic (no network, no `.env`). Never delete or weaken a test to get green.
  A red full suite is never committed.
- `.env` holds real API keys: never print, log or commit it. Run the privacy check script (kept
  in the run folder, outside the repo) before every push.
- This file and the roadmap are public: no machine paths, personal details or key material.
- Install and upgrade nothing. No Ollama or Docker. Real Gemini calls are counted in the roadmap.
- Library facts come from current documentation (`npx ctx7@latest library|docs ...`), not memory.
- Every published number is measured. An optimisation with no measured gain is undone and
  recorded as "tried, no gain".
- Save tokens: no subagents; read only the files an item names; never read `app.py` or
  `config/translations.py` top to bottom; send test output to a log and read its last 30 lines.

## Running the tests (project environment, Python 3.13)
- Point TEMP/TMP and the pip cache at the run folder first; give every command a timeout.
- One file: `python -m pytest tests/test_codegen.py -q`. Full: `python -m pytest -q`, `python -m pyflakes .`
