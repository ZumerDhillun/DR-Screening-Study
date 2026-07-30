# Git Workflow (two-person team)

The whole point of this file is to prevent two failure modes: (1) you
both silently diverge and spend a day reconciling conflicting results,
and (2) someone accidentally commits a 10GB dataset folder and breaks
the repo for both of you. Follow this exactly, it's short.

## 0. One-time repo setup (whoever creates the repo does this)

```bash
git init retinavision
cd retinavision
# add README.md, .gitignore, requirements.txt, etc.
git add .
git commit -m "chore: initial project skeleton"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

The other teammate then clones it:
```bash
git clone <your-github-repo-url>
cd retinavision
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Branching model — keep it simple

- `main` — always working, always the version either of you would run
  right now. Nothing broken ever sits on `main`.
- One short-lived branch per step or per fix, named like:
  - `step1-hash-dedup`
  - `step2-binarization`
  - `fix-convnext-normalization`

Never work directly on `main`. Even solo changes go through a branch +
pull request — it costs 30 seconds and gives you both a review trail,
which matters a lot when you're writing the methods section later and
need to explain exactly what changed and when.

```bash
git checkout main
git pull                      # always pull before branching
git checkout -b step2-binarization
# ... do work ...
git add .
git commit -m "feat: add ICDR-to-binary label mapping"
git push -u origin step2-binarization
```

Then open a Pull Request on GitHub, tag your teammate, and **do not
merge your own PR** — the other person reviews and merges. This is the
single easiest habit that catches silent mistakes (wrong threshold,
swapped label, hardcoded path) before they contaminate a training run.

## 2. Commit message convention

Prefix every commit with one of:
- `feat:` — new capability (e.g. a new script, a new metric)
- `fix:` — bug fix
- `chore:` — setup, config, non-functional changes
- `docs:` — README/protocol updates
- `data:` — changes to configs describing data (never actual data files)
- `results:` — committing a report/CSV/figure output

Example: `results: step1 dedup report — 0 exact dupes, 3 near-dupes flagged`

This makes `git log --oneline` double as a lightweight lab notebook —
useful when you're reconstructing your methods timeline for the paper.

## 3. What never gets committed

Already covered by `.gitignore`, but know why:
- **Raw or processed image data.** Too large, and DDR/DeepDRiD/APTOS all
  have their own license/redistribution terms — committing them to a
  (even private) GitHub repo can violate those terms.
- **Model checkpoints (`.pt`, `.pth`, `.ckpt`).** Too large for git.
  Share these via a cloud drive link pasted in your team chat, or use
  Git LFS / DVC if you want them version-controlled (optional, only set
  this up if you're both comfortable with it — don't let tooling eat
  time you should spend on Step 2 onward).
- **`.venv/`, `__pycache__/`, `.ipynb_checkpoints/`** — environment/cache
  junk, machine-specific.

What **does** get committed: scripts, configs (with placeholder paths,
not your personal machine's paths), small CSV/JSON reports, README/
protocol updates, and figures under ~a few MB.

## 4. Staying in sync day to day

- **Pull before you start work, every session**, even if you think
  nothing changed: `git checkout main && git pull`.
- If you're both about to touch the same script, say so in your team
  chat first. Merge conflicts in preprocessing code are the kind of bug
  that silently changes your numbers if resolved carelessly — worth 10
  seconds of coordination to avoid.
- **Update the checklist in `README.md`** in the same PR that completes
  a step, so "what's actually done" is always visible in one place
  instead of living in someone's memory or a side chat.
- End of each work session: push your branch even if it's unfinished
  (`git push -u origin <branch-name>`) — never leave work sitting only
  on your local machine overnight.

## 5. When results disagree between your two machines

If you run the same script and get different numbers, in this order,
check: (1) did `environment_check.py` output match? (2) same `configs/`
values? (3) same random seed set in the script? Don't just re-run it
and hope — silently divergent environments are exactly the kind of
thing that undermines a "fair comparison" study like this one.
