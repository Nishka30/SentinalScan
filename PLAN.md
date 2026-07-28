# Sentinel — Deployment Risk Analyzer (Build Plan for Claude Code)

## How to use this file
1. Create an empty folder, run `git init`, and save this file as `PLAN.md` in the root.
2. Open Claude Code in that folder.
3. Paste the kickoff line at the very bottom of this file.
4. Claude Code will build **one phase at a time and stop**. Review, then tell it to continue.

---

## 1. What we are building

A command-line tool (`sentinel`) that reads a git repository and answers one question:

> **"Is this change safe to deploy?"**

It gives a risk score, plain-English reasons, and a recommendation.
Every feature exists only to answer that one question better.

The important idea: the risk score is **learned from the repository's own
bug history**, not guessed by an AI. An AI is used only at the very end to
explain the score in plain language.

---

## 2. Golden rules for you, Claude Code

Read these carefully. They matter more than speed.

1. **Build one phase at a time. Stop after each phase and wait for me to review.**
   Do not jump ahead. Do not build Phase 3 while I'm reviewing Phase 1.
2. **Do not add scope.** If a feature is not in this plan, do not build it.
   If you think something is missing, ask me first — do not add it silently.
3. **Ask before adding any new dependency** that is not in the locked stack below.
4. **Keep it simple.** Prefer boring, readable code over clever code. This is a
   portfolio project that must be explainable in an interview, line by line.
5. **Write a short test for each real piece of logic.** Not 100% coverage — just
   enough that the core functions are proven to work.
6. **Commit at the end of each phase** with a clear message (e.g.
   `feat: phase 1 - CLI, git reader, rule-based risk`).
7. **Never hardcode secrets.** All keys come from environment variables.
8. At the end of each phase, print a short summary of what you built and how to run it.

---

## 3. Locked tech stack (do not change without asking)

- Language: **Python 3.11+**
- CLI: **typer**
- Terminal output: **rich**
- Git mining: **PyDriller** (falls back to **GitPython** only if needed)
- Code complexity: **lizard**
- Dependency graph: **networkx** (in memory — NO graph database)
- ML model: **lightgbm**
- Model explanations: **shap**
- LLM calls: **openai** client, pointed at the NVIDIA endpoint (see section 6)
- Config: **pydantic-settings** + a `.env` file
- Testing: **pytest**

Package/dependency manager: use `uv` if available, otherwise `pip` + `venv`.

---

## 4. Out of scope — do NOT build these

These were considered and deliberately cut. Do not build them:

- No vector database
- No Redis
- No Neo4j or any external database (keep everything file-based or in memory)
- No multiple "AI agents" — there is exactly one LLM call, for explanation only
- No web dashboard / React frontend
- No authentication system
- No microservices — this is a single CLI package

If any of these seem needed, STOP and ask me.

---

## 5. Project structure

Create this structure. Keep modules small and single-purpose.

```
sentinel/
  __init__.py
  cli.py                # typer commands: scan, diff, explain
  config.py             # settings loaded from env / .env
  git_reader.py         # reads git log/blame/diff via PyDriller
  static_analysis.py    # complexity + test-file detection via lizard
  risk_rules.py         # rule-based scoring (Phase 1)
  history_mining.py     # SZZ: label past commits buggy/clean (Phase 2)
  features.py           # turn a change into a feature vector (Phase 2)
  model.py              # train + predict with lightgbm (Phase 2)
  evaluation.py         # time-based split + metrics (Phase 2)
  blast_radius.py       # dependency graph + impacted files (Phase 3)
  explain.py            # one NVIDIA LLM call to explain the score (Phase 4)
  report.py             # format the final terminal output
tests/
  ...
github-action/          # composite action + example workflow (Phase 5)
mcp_server/             # single MCP server exposing get_deployment_risk (Phase 5)
PLAN.md
README.md
pyproject.toml
.env.example
.gitignore
```

---

## 6. NVIDIA build model integration (Phase 4 only)

Use NVIDIA's hosted, OpenAI-compatible API.

- Endpoint (base URL): `https://integrate.api.nvidia.com/v1`
- Auth: an API key from build.nvidia.com (starts with `nvapi-`), read from
  the environment variable `NVIDIA_API_KEY`. Never hardcode it.
- Use the standard `openai` Python client, just with a custom `base_url`:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
resp = client.chat.completions.create(
    model=settings.llm_model,      # configurable, see below
    messages=[...],
    temperature=0.2,
    max_tokens=800,
)
```

- **Model must be configurable**, not hardcoded. Read it from env var
  `SENTINEL_LLM_MODEL`, defaulting to `meta/llama-3.3-70b-instruct`.
  (I can swap this for another catalog model like `openai/gpt-oss-120b`
  later without touching code.)
- Some NVIDIA models return a `reasoning_content` field in addition to the
  normal content. Read `message.content` for the answer; ignore or log
  `reasoning_content`.
- Handle failures gracefully: if `NVIDIA_API_KEY` is missing, the request
  errors, or a rate limit is hit, the tool must **still work** and just skip
  the AI explanation with a clear note. The LLM layer is optional polish —
  the score and reasons come from the math, not the AI.

Put `NVIDIA_API_KEY` and `SENTINEL_LLM_MODEL` in `.env.example` (with empty/
placeholder values) so I know what to fill in.

---

## 7. Two concepts you MUST get right

These are the heart of the project. Getting them wrong makes the whole thing fake.

**A. SZZ labeling (Phase 2).**
To know which changes were "risky," mine the repo's own history:
1. Find bug-fixing commits — commit messages containing words like `fix`,
   `bug`, `patch`, `hotfix`, or referencing an issue number. Make the keyword
   list configurable.
2. For each bug-fixing commit, look at the lines it changed and use
   `git blame` on the version **before** the fix to find the commit(s) that
   originally introduced those lines.
3. Label those introducing commits as "bug-inducing" (positive). All other
   commits are "clean" (negative).
This gives real training labels. PyDriller has helpers for this — use them.

**B. Time-based evaluation (Phase 2).**
When you train and test the model, split by **time**, never randomly:
train on older commits, test on newer commits. A random split lets the model
peek at the future and reports fake-good accuracy. This rule is non-negotiable.
Report metrics that suit rare events: **ROC-AUC and PR-AUC**, plus a simple
baseline (predict everything using only "lines changed") so we can show the
model beats the baseline.

---

## 8. Build phases (do these in order, stop after each)

### Phase 0 — Scaffold
- Create the folder structure, `pyproject.toml`, `.gitignore`, `.env.example`,
  a stub `README.md`, and set up the test runner.
- `sentinel --help` should run and list the (currently empty) commands.
- Commit. Stop for review.

### Phase 1 — Working tool with rule-based scoring
Goal: a genuinely useful tool, no ML yet.
- `sentinel scan` analyzes the whole repo; `sentinel diff` analyzes only
  uncommitted/changed files.
- `git_reader.py`: for each file, compute change count, number of past
  bug-fix commits touching it, and the author's familiarity with it.
- `static_analysis.py`: cyclomatic complexity of changed files; detect whether
  a matching test file exists / changed.
- `risk_rules.py`: a transparent points-based score (hot file, low tests, high
  complexity, author new to file, risky timing, large change). Each rule
  contributes a labeled amount so the reasons are obvious.
- `report.py`: clean `rich` output showing score, the reasons that fired, and
  a simple recommendation.
- Tests for the rule engine and git reader.
- Commit. Stop for review.

### Phase 2 — Learn from history (the spine)
- `history_mining.py`: implement SZZ labeling (section 7A).
- `features.py`: turn any change into a numeric feature vector (size,
  diffusion across folders, file history, bug-fix history, ownership,
  complexity, test signal, day/hour).
- `model.py`: train a lightgbm classifier on the labeled history; save/load
  the model to a file in the repo (e.g. `.sentinel/model.txt`).
- `evaluation.py`: time-based split, report ROC-AUC + PR-AUC vs the
  lines-changed baseline (section 7B). Add a `sentinel train` command and a
  `sentinel evaluate` command.
- After training, `sentinel scan` uses the model's probability as the score,
  and uses SHAP to produce the top reasons.
- Tests for the labeling logic and the feature extractor.
- Commit. Stop for review.

### Phase 3 — Blast radius (the standout feature)
- `blast_radius.py`: build an in-memory dependency graph with networkx by
  parsing imports/references (start with Python and Java imports; keep it
  extensible). For a changed file, report which other files/modules depend on
  it, directly and indirectly.
- Show impacted areas in the report ("this change affects: X, Y, Z").
- Tests on a small sample graph.
- Commit. Stop for review.

### Phase 4 — AI explanation (NVIDIA)
- `explain.py`: take the score, the top SHAP reasons, and the blast-radius
  summary, and make ONE call to the NVIDIA endpoint (section 6) to produce a
  short plain-English explanation + recommended rollout + rollback trigger.
- The AI only explains numbers it is given. It must not compute or change the
  score. If the LLM is unavailable, skip this section cleanly.
- Add a `sentinel explain` command that prints the full explanation.
- Commit. Stop for review.

### Phase 5 — Ship it
- `github-action/`: a composite action + an example workflow that runs
  `sentinel diff` on every pull request and posts the score and reasons as a
  PR comment. Use a non-zero exit code when the score crosses a threshold.
- `mcp_server/`: a single MCP server exposing one tool, `get_deployment_risk`,
  which runs the analysis on a given repo/diff and returns the score, reasons,
  and blast radius as structured JSON. So Cursor/Claude can call it.
- Update `README.md` with install, usage, a screenshot placeholder, the
  evaluation results, and a short "how it works" section.
- Commit. Stop for review.

---

## 9. Coding standards

- Type hints on all public functions. Docstrings that say *why*, not *what*.
- Small functions. No file over ~300 lines; split if it grows.
- No silent failures — log or surface errors.
- Deterministic where possible (set random seeds for the model).
- `README.md` must let a stranger clone, install, and run `sentinel scan` in
  under 5 minutes.

---

## 10. KICKOFF — paste this line into Claude Code

> Read PLAN.md fully. Then do **Phase 0 only**: scaffold the project exactly as
> described, get `sentinel --help` working, set up pytest, and commit. Do not
> start Phase 1. When done, show me the file tree, how to run it, and wait for
> my review.
