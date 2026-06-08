# Contributing to Ramanujan-24

Thank you for your interest in contributing. R24 is a small research project
and contributions are welcome — but please read this first so your work lands cleanly.

---

## What We're Looking For

**High priority:**
- Cross-encoder reranker as Stage 4 (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Evaluation on new datasets (especially multilingual / non-English corpora)
- Configurable hash width (currently fixed at 24 dimensions)
- Stronger Brahmagupta latent chain (Track B — image pipeline)
- Abstract/Warli style prompt tuning for `r24_story.py`

**Also welcome:**
- Bug reports with a minimal reproducer
- Benchmark corrections (if you find an error in the numbers, open an issue)
- Documentation improvements
- New story presets for `r24_story.py`

**Not in scope (for now):**
- Replacing the C hash with a Python implementation (speed is the point)
- Cloud API integrations (the project is intentionally local/offline)
- Changing the mathematician naming scheme

---

## How to Contribute

### 1. Open an issue first

For anything beyond a typo fix, open an issue before writing code.
Describe what you want to change and why. This avoids duplicate work.

### 2. Fork and branch

```bash
git clone https://github.com/wild-coder-ux/ramanujan24.git
cd ramanujan24
git checkout -b your-feature-name
```

### 3. Make your changes

- Keep C code in `src/` compatible with `gcc -O2 -shared -fPIC`
- Python code should run on Python 3.9+
- No new dependencies without discussion — the project runs on a single `pip install`
- If you change the Pingala hash, re-run all three benchmarks and include the numbers

### 4. Test before submitting

```bash
# Rebuild the hash
gcc -O2 -shared -fPIC -o src/libr24_pingala.so src/r24_pingala.c

# Run benchmarks
python src/r24_bm25_squad.py
python src/r24_bm25_d2l.py
python src/r24_bm25.py
```

Include benchmark output in your pull request description.

### 5. Open a pull request

- Title: one line describing what changed
- Body: what you changed, why, and benchmark results if applicable
- Keep PRs focused — one change per PR

---

## Code Style

**C (`r24_pingala.c`):**
- ANSI C, no external dependencies
- Comments explain the mathematical principle, not just the code
- Variable names should reflect the mathematical concept where possible

**Python:**
- No formatter enforced, but be consistent with the existing style
- Functions get a one-line docstring
- Magic numbers get a comment

---

## Reporting Bugs

Open a GitHub issue with:
- Your OS and Python version
- The exact command you ran
- The full error output
- Which dataset you were using

---

## Questions

Open a GitHub Discussion or file an issue tagged `question`.

---

## A Note on the Naming

The mathematician names (Pingala, Madhava, Aryabhata, etc.) are not decorative.
Each name maps to a specific mathematical principle used in that stage.
If you add a new stage or component, please follow this convention —
find the historical precedent and name it accordingly.
If you're unsure, mention it in the issue and we'll work it out together.
