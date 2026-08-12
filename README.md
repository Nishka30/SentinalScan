<div align="center">

# 🛡️ Sentinel — Deployment Risk Analyzer

**One question, one answer: _is this change safe to deploy?_**

Sentinel answers from your repository's own bug history — not a language model's guess.

[![PyPI version](https://img.shields.io/pypi/v/sentinel-risk.svg?style=flat-square)](https://pypi.org/project/sentinel-risk/)
[![npm version](https://img.shields.io/npm/v/sentinel-risk-mcp.svg?style=flat-square)](https://www.npmjs.com/package/sentinel-risk-mcp)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square)
[![MCP](https://img.shields.io/badge/MCP-ready-8A2BE2?style=flat-square)](#-mcp-server--use-sentinel-from-claude--cursor)

</div>

```
┌─────────── Sentinel — deployment risk ────────────────────────┐
│   Score: 67/100   ██████████████████░░░░░   HIGH RISK         │
│   3 file(s) changed — since origin/main                       │
│   Scored by: trained model (4,854 commits)                    │
│                                                               │
│   Top reasons:                                                │
│    • auth/session.py — fixed 14 times in the last 2 years     │
│    • Author has never touched this file before                │
│    • Change spans 6 folders — high blast radius               │
│                                                               │
│   Blast radius: 12 files directly affected, 38 transitively   │
│                                                               │
│   → Hold this, or ship it behind a flag with a rollback ready.│
└───────────────────────────────────────────────────────────────┘
```

---

## 📑 Table of Contents

- [Why Sentinel Exists](#-why-sentinel-exists)
- [What Makes It Different](#-what-makes-it-different)
- [Install](#-install)
- [Quickstart (60 seconds)](#-quickstart-60-seconds)
- [Mental Model: Two Phases](#-mental-model-two-phases)
- [The CLI](#-the-cli)
- [How the Score Is Computed](#-how-the-score-is-computed)
  - [The rule engine (default)](#the-rule-engine-default)
  - [Relative thresholds](#relative-thresholds-how-sentinel-adapts-to-your-repo)
  - [Score bands](#score-bands)
  - [How the headline number is assembled](#how-the-headline-number-is-assembled)
- [Learning From History: SZZ + the Model](#-learning-from-history-szz--the-model)
  - [The 20-feature vector](#the-20-feature-vector)
  - [How the model is trained](#how-the-model-is-trained)
  - [Honest, time-based evaluation](#honest-time-based-evaluation)
  - [No data leakage: the guarantees](#no-data-leakage-the-guarantees)
- [Blast Radius](#-blast-radius)
- [The AI Layer — Explanation Only](#-the-ai-layer--explanation-only)
- [MCP Server — Use Sentinel from Claude / Cursor](#-mcp-server--use-sentinel-from-claude--cursor)
- [GitHub Action — Score Every PR](#-github-action--score-every-pr)
- [JSON Output (for scripting & CI)](#-json-output-for-scripting--ci)
- [Configuration Reference](#-configuration-reference)
- [Exit Codes](#-exit-codes)
- [Architecture](#-architecture)
- [Non-Goals (Deliberate Simplicity)](#-non-goals-deliberate-simplicity)
- [FAQ](#-faq)
- [Troubleshooting](#-troubleshooting)
- [Glossary](#-glossary)
- [Author & License](#-author--license)

---

## 🔍 Why Sentinel Exists

Every team has files they are quietly afraid of — the ones that broke production twice last year, that only one person truly understands, that nothing has tests for. You can feel that instinct during code review, but you can't put a number on it.

**Sentinel puts a number on it.**

It reads your repository's full git history, finds which past commits later had to be *fixed*, traces the bug-inducing commits with a line-level blame algorithm (**SZZ**), and builds a lightweight gradient-boosted model trained entirely on your own data. It then scores any incoming change on a **0–100 risk scale** — using that model, or a transparent rule engine when no model exists yet.

> **The score is always deterministic.** A language model can *optionally* narrate the findings in plain English, but it structurally cannot affect the number. Remove the API key and the score is identical — only the paragraph disappears.

---

## 💡 What Makes It Different

| Tool | How it decides risk |
| --- | --- |
| Linters | Style rules — know nothing about *history* |
| Coverage tools | Tell you what's tested — not what has *historically broken* |
| PR-review bots | Usually an LLM *is the judge* — opaque and unrepeatable |
| **Sentinel** | **Trains on your repo's own bug record. The score is a probability from your history, not an opinion.** |

Design principles baked into the code:

- **No lookahead in training.** A historical commit's features are built only from data available *before* that commit. The running history tally is applied *after* featurising, never before.
- **Relative thresholds.** Eight past bug fixes is remarkable in a young service and unremarkable in a fifteen-year-old library. Sentinel uses your repo's own percentile distribution as thresholds, not hardcoded absolutes.
- **Blast radius is separate from the score.** The dependency graph tells you *what else can break*, but it is deliberately **not** fed to the model (doing so honestly would require re-parsing the whole tree at every historical commit).
- **The LLM is additive only.** The AI layer can *read* the score. It cannot *write* to it — and that's enforced by the type system, not a prompt.

---

## 📦 Install

**Requirements:** Python **3.11+**, and `git` on your PATH.

### Option A — pip (recommended)

```bash
pip install sentinel-risk
```

Installs the `sentinel` CLI (`scan`, `diff`, `train`, `evaluate`, `explain`), the `sentinel-mcp` server binary, and all dependencies (`lightgbm`, `pydriller`, `GitPython`, `lizard`, `networkx`, `shap`, `rich`, `fastmcp`, …). **This is the direct route to the `sentinel scan` command.**

📦 [`sentinel-risk` on PyPI](https://pypi.org/project/sentinel-risk/)

### Option B — npm (for the MCP server)

For wiring Sentinel into Claude Desktop or Cursor without setting up Python yourself. The published package is **`sentinel-risk-mcp`**.

Zero-install — always runs the latest:

```bash
npx -y sentinel-risk-mcp
```

Or install it globally and run the command:

```bash
npm install -g sentinel-risk-mcp
sentinel-risk-mcp
```

The Node wrapper locates Python and installs the underlying `sentinel-risk` package via `pip` behind the scenes, then starts the MCP server. That bootstrap also places the `sentinel` CLI on your PATH — so `sentinel scan` works afterwards too — but if the CLI is all you want, `pip install sentinel-risk` (Option A) is the cleaner path.

📦 [`sentinel-risk-mcp` on npm](https://www.npmjs.com/package/sentinel-risk-mcp)

---

## 🚀 Quickstart (60 seconds)

```bash
# 1. Install
pip install sentinel-risk

# 2. Go to any git repo you work in
cd /path/to/your/project

# 3. Score committed changes vs the default branch
sentinel scan

# 4. Or score what you haven't committed yet
sentinel diff
```

That's it — **no configuration needed for the core score.** Want plain-English explanations too? Run `sentinel configure` once, then add `--explain`.

---

## 🧠 Mental Model: Two Phases

Sentinel runs in two clearly separated phases. **Training is optional** — the rule engine works out of the box; training simply makes the score sharper for your codebase.

### Phase 1 — Train (learn from history) · optional, run once per repo

```
1. Read git log                  → all commits in the window (default: 1,500)
2. Detect bug-fix commits        → keyword match on subjects ("fix", "hotfix", …)
3. SZZ blame                     → for each fix, blame the deleted lines against the
                                    parent to find the commits that introduced the bug
4. Label commits                 → blame-identified = 1 (risky), everything else = 0
5. Build feature vectors         → 20 features per commit, from info available *at commit time*
6. Train LightGBM + early stop   → ≤ 300 rounds, stops when held-out AP stops improving
7. Save model + metadata         → .sentinel/model.txt + .sentinel/model.meta.json
```

### Phase 2 — Score (assess a change) · every `scan` / `diff` / MCP call

```
1. Identify changed files        → git diff vs base (scan) or working tree (diff)
2. Read file histories           → bug-fix counts, churn, per-author ownership
3. Static analysis               → cyclomatic complexity via lizard
4. Build feature vector          → the SAME 20-feature function used in training
5. Score                         → trained model if present; rule engine otherwise
6. Blast radius                  → networkx dependency graph → direct + transitive impact
7. Report                        → rich terminal table, or JSON for CI
8. Explain (optional)            → one LLM call for a plain-English narrative
```

> **Phase 2 never touches the network.** Steps 1–7 are fully offline. Only step 8 — opt-in, via `--explain` — makes an API call.

---

## 🖥️ The CLI

| Command | What it does |
| --- | --- |
| `sentinel scan` | Score **committed** changes vs the default branch (or `--since <ref>`) |
| `sentinel diff` | Score **uncommitted** work in the working tree (staged + unstaged) |
| `sentinel explain` | Score **and** narrate in plain English (shorthand for scan + `--explain`) |
| `sentinel train` | Mine bug history (SZZ) and train the LightGBM model into `.sentinel/` |
| `sentinel evaluate` | Time-based train/test split; compare the model to a baseline |
| `sentinel configure` | Interactive wizard to set up an LLM provider and write `.env` |
| `sentinel version` | Print the installed version |

### `sentinel scan` — committed changes

```bash
sentinel scan                        # vs auto-detected default branch
sentinel scan --since origin/main    # vs a specific ref
sentinel scan --since HEAD~10        # last 10 commits
sentinel scan --all                  # rank every tracked file by inherent risk
sentinel scan --explain              # also request the LLM narrative
sentinel scan --json                 # machine-readable JSON
sentinel scan --repo /other/project  # analyze a different repository
```

### `sentinel diff` — uncommitted work

```bash
sentinel diff
sentinel diff --explain
sentinel diff --json
```

### `sentinel explain` — score and explain

```bash
sentinel explain                     # explains committed changes
sentinel explain --diff              # explains uncommitted changes
sentinel explain --since HEAD~5
sentinel explain --json
```

### `sentinel train` — build a model on your history

```bash
sentinel train                       # mine up to 1,500 commits (default)
sentinel train --max-commits 5000    # deeper history (slower — SZZ blame is the slow half)
sentinel train --repo /other/project
```

After training, every `scan` / `diff` automatically uses the model. Re-run it periodically (e.g. on a schedule in CI) as history grows.

### `sentinel evaluate` — is the model actually good?

```bash
sentinel evaluate
sentinel evaluate --max-commits 3000
```

Example output:

```
Training rows:   3,621   Positives: 312 (8.6%)
Test rows:       1,207   Positives: 104 (8.6%)

Rule / baseline    PR-AUC: 0.271
LightGBM model     PR-AUC: 0.438   ✓ beats baseline
```

> The split is **always by time** — newest commits in the test set, never a random shuffle. See [Honest, time-based evaluation](#honest-time-based-evaluation).

---

## 📐 How the Score Is Computed

### The rule engine (default)

With no trained model, Sentinel uses a **transparent, points-based** engine. Every rule is a small pure function that either stays silent or contributes points *with its own evidence string*. The report isn't a separate rendering of the score — it **is** the list of rules that fired.

**File-level rules** (evaluated per changed file):

| Rule | Trigger (default) | Points |
| --- | --- | --- |
| **Hot file** (many past fixes) | ≥ 8 bug-fix commits | **+35** |
| Hot file | ≥ 3 bug-fix commits | +20 |
| **Large edit** | ≥ 300 lines changed | +25 |
| Sizeable edit | ≥ 80 lines changed | +12 |
| **Missing tests** | no test file found | +20 |
| Stale tests | tests exist but weren't touched | +10 |
| **Very high complexity** | peak cyclomatic complexity ≥ 20 | +22 |
| High complexity | peak CCN ≥ 10 | +12 |
| **Author new to file** | 0 prior commits by this author | +15 |
| Low ownership | author made < 25% of past commits | +8 |
| Churn | ≥ 25 commits ever touched it | +10 |

**Change-level rules** (evaluated once for the whole change; skipped for `--all`):

| Rule | Trigger (default) | Points |
| --- | --- | --- |
| Broad change (files) | ≥ 10 files touched | +10 |
| Broad change (lines) | ≥ 600 lines total | +10 |
| Weekend deploy | committed Sat/Sun | +10 |
| Outside working hours | committed ≥ 22:00 or < 06:00 | +12 |

Every threshold and point value lives in one place (`config.py` → `RiskWeights`) and is overridable via `SENTINEL_RULES` — so you can re-tune policy without editing scoring logic.

### Relative thresholds (how Sentinel adapts to your repo)

Absolute counts don't travel between repositories. When your repo has enough history (≥ 20 files with history), Sentinel **raises** the hot-file and churn thresholds to your repo's own percentiles:

| Signal | Percentile used |
| --- | --- |
| "Very hot" file (bug fixes) | 90th |
| "Hot" file (bug fixes) | 75th |
| High churn | 90th |

The configured absolutes act as **floors** (`max(config, percentile)`), so a young repo whose 90th percentile is a single fix never hands out the top tier for one fix, and a threshold can never get *looser* than your written policy. Disable with `SENTINEL_DISTRIBUTION='{"enabled": false}'`.

### Score bands

| Band | Score | Meaning |
| --- | --- | --- |
| 🟢 **low** | 0 – 34 | Safe to deploy with a normal review. |
| 🟡 **medium** | 35 – 64 | Deployable, but review properly and watch it after release. |
| 🔴 **high** | 65 – 100 | Hold it, or ship behind a flag with a rollback ready. |

### How the headline number is assembled

The headline is **the riskiest file's score, plus change-level context penalties** — *not* a sum across all files. Summing would make any large refactor hit 100 and say nothing. A change is dangerous mainly because of the worst thing in it:

```
score = min(100,  riskiest_file_score  +  Σ change-level penalties)
```

`scan --all` is different: there's no "change", so it ranks every tracked file by *inherent* riskiness and reports the worst — change-level and timing rules are switched off, because "it's Saturday" describes the clock, not a deploy.

---

## 🧬 Learning From History: SZZ + the Model

This is what turns Sentinel from a linter into a predictor. The only honest source of "what a risky change looks like" is your repo's own record of what it later had to fix.

**The SZZ pipeline:**

1. Find commits whose subject reads like a bug fix.
2. For each, **blame the lines that the fix *deleted*** against the parent commit — the commits that last touched those lines are the ones that introduced the bug.
3. Those commits are labelled bug-inducing (**1**); everything else is clean (**0**).

PyDriller's `get_commits_last_modified_lines` is exactly this line-level blame, already written and tested — which is why it earns its place in the stack alongside the faster `git log --numstat` walk used for aggregate counts.

**Bug-fix detection** matches these keywords on word boundaries (so `fix` never matches `prefix`):

```
fix · fixes · fixed · fixing · bug · bugs · bugfix ·
patch · hotfix · defect · regression · crash · broken
```

Issue references only count when a *fixing verb* is attached (`closes #123`, `fixes ABC-45`). A bare `#1234` is usually just a squash-merge PR number — treating those as bug evidence mislabelled **51%** of commits on the `requests` repo, versus **16%** with the stricter pattern. Over-labelling here doesn't just add noise; it corrupts every downstream training label.

**Two honest caveats** (both designed around by the time-based evaluation):

- Keyword-matched fixes are a *proxy*: a fix that says "tidy up" is missed; a feature commit that says "fix" is a false positive.
- Recent commits look cleaner than they are — nobody's found their bugs *yet*. That's exactly why the newest commits belong in the **test** set, never the training set.

### The 20-feature vector

Training and prediction call **one** `vector()` function, so the two can never drift apart. Feature order is part of the saved model's contract (append-only, never reordered).

| # | Feature | Reads as |
| --- | --- | --- |
| 1 | `lines_added` | Lines added |
| 2 | `lines_deleted` | Lines deleted |
| 3 | `lines_changed` | Size of the change |
| 4 | `files_changed` | Number of files touched |
| 5 | `folders_touched` | Spread across folders |
| 6 | `max_file_commits` | Change history of the busiest file |
| 7 | `mean_file_commits` | Avg change history of files touched |
| 8 | `max_file_bugfixes` | Bug-fix history of the worst file |
| 9 | `mean_file_bugfixes` | Avg bug-fix history of files touched |
| 10 | `new_files` | Brand-new files |
| 11 | `min_author_ownership` | Familiarity with the least-known file |
| 12 | `mean_author_ownership` | Familiarity with these files |
| 13 | `files_author_never_touched` | Files the author never changed |
| 14 | `max_complexity` | Complexity of the most complex file |
| 15 | `mean_complexity` | Avg complexity of files touched |
| 16 | `code_files_changed` | Source files touched |
| 17 | `test_files_changed` | Test files touched |
| 18 | `hour_of_day` | Hour of day |
| 19 | `day_of_week` | Day of week |
| 20 | `is_weekend` | Weekend |

Every input is knowable *at the moment the change was made* — no future commits, no current file contents, no blast radius (see [No data leakage](#no-data-leakage-the-guarantees)).

### How the model is trained

Capacity is deliberately small — a repo yields a few hundred labelled commits with a few dozen positives, and a wide model just memorises the training period.

| Setting | Default | Why |
| --- | --- | --- |
| `num_leaves` | 7 | Small tree — avoids memorising |
| `learning_rate` | 0.05 | Gentle boosting |
| `num_rounds` | 300 | *Ceiling*, not target — early stopping decides |
| `early_stopping_rounds` | 30 | Stop when held-out score plateaus |
| `early_stopping_metric` | `average_precision` | ROC-AUC is dominated by the huge true-negative pool |
| `min_data_in_leaf` | 30 | Prevents tiny, overfit leaves |
| `train_fraction` | 0.75 | Oldest 75% train, newest 25% test |
| `holdout_fraction` | 0.85 | Slice used to fit before early stopping |
| `seed` | 42 | Reproducible runs |

Bug-inducing commits are a small minority, so the positive class is **weighted** by `negatives / positives`. Unweighted, LightGBM maximises accuracy by predicting "clean" for everything — a great-looking, useless model.

> **Measured example** (`requests` repo): stopping on ROC-AUC halted at 17 rounds → 0.407 held-out PR-AUC; stopping on average precision ran to 43 rounds → **0.438**. That's why the metric choice is a first-class setting.

The trained model is a plain text file plus a small JSON sidecar recording the exact feature list. Load a model built against a *different* feature set and the columns still line up numerically but the predictions are nonsense — so the sidecar lets Sentinel **detect** that, warn, and fall back to the rule engine instead of silently scoring garbage.

### Honest, time-based evaluation

`sentinel evaluate` enforces two non-negotiable rules:

- **Split by time, never at random.** Train on older commits, test on newer. A random split lets the model peek at commits that came *after* the ones it's tested on — the single easiest way to fake a good result.
- **Report rare-event metrics against a baseline.** With positives at a few percent, plain accuracy is ~95% for a model that always says "clean". Sentinel reports **ROC-AUC** and **PR-AUC** next to a **lines-changed-only baseline** — a model that can't beat "big diffs are risky" hasn't earned its complexity.

Both metrics are implemented from scratch (ROC-AUC via the Mann-Whitney rank identity, with tie handling) and unit-tested against hand-computed values — no scikit-learn dependency, and you can explain exactly how your headline number was produced.

### No data leakage: the guarantees

- Historical features are built from data available *before* the commit; the running history tally is applied *after* featurising.
- **Blast radius is excluded from the model on purpose.** Feeding it in honestly would need the import graph *as it stood at each past commit* — every source file re-parsed at every commit. Using today's graph for a 2015 commit would leak present-day structure into a past feature — exactly what the time-based split exists to catch.
- Nothing derived from the future (later commits, current file contents) ever enters a feature.

---

## 💥 Blast Radius

Sentinel builds an **in-memory** import graph (networkx) and walks it *backwards* from the changed files — an edge points from importer to imported, so a change's dependents are its ancestors in the graph. In memory on purpose: a graph database would be permanent infrastructure for something rebuilt from scratch in under a second.

Three real-world messes are handled deliberately:

- **Cycles** (`a` imports `b` imports `a`) terminate via a visited set, and files sitting in a cycle are flagged.
- **Hubs** — a settings/types module imported by everything. Reporting "affects all 400 files" is true and useless, so depth is capped, listings are truncated (full counts kept), and hubs are *named as hubs* (≥ 20 direct dependents).
- **Direct vs transitive** dependents are never merged — importing the changed file directly is a different proposition from being five hops away.

**Languages parsed for the dependency graph:** Python (via `ast`, so relative/aliased/multi-line imports are handled correctly) and Java. Adding a language means adding one `LanguageParser` — the graph and traversal code never changes.

| Blast setting | Default |
| --- | --- |
| `max_depth` (hops followed) | 3 |
| `max_listed` (names per list) | 12 |
| `hub_dependents` (hub cutoff) | 20 |
| `max_analyzed_files` (per-file impact) | 40 |
| `max_files` (graph size guard) | 5,000 |
| `max_bytes_per_file` | 400,000 |

---

## 🤖 The AI Layer — Explanation Only

The LLM's job is to **translate** a finished analysis into human guidance. It is *not* a judge.

**Why it structurally cannot move the score:**

- `explain()` receives an **already-frozen** `ChangeRisk`, computed before it ran.
- It returns an `Explanation` object that has **no field for a score, band, or reason**.
- The result is rebuilt immutably, so the analysis that was scored is the analysis that's reported.

That's a guarantee from the type system, not a promise in a prompt — because prompts aren't a security boundary. **Remove the API key and the score is byte-for-byte identical; only the narrative disappears.**

**Graceful by design.** No key, a timeout, a rate limit, a malformed response — every failure degrades to the deterministic report *with a note*. A risk tool that can't answer without a third-party API is worse than one with no AI at all.

**Provider-agnostic** (any OpenAI-compatible endpoint):

```bash
# NVIDIA NIM (default)
NVIDIA_API_KEY=nvapi-xxxx
SENTINEL_LLM_MODEL=meta/llama-3.3-70b-instruct
SENTINEL_LLM_BASE_URL=https://integrate.api.nvidia.com/v1

# OpenAI
OPENAI_API_KEY=sk-...
SENTINEL_LLM_BASE_URL=https://api.openai.com/v1
SENTINEL_LLM_MODEL=gpt-4o

# Local Ollama
SENTINEL_LLM_API_KEY=anything
SENTINEL_LLM_BASE_URL=http://localhost:11434/v1
SENTINEL_LLM_MODEL=llama3.2

# DeepSeek
SENTINEL_LLM_API_KEY=sk-...
SENTINEL_LLM_BASE_URL=https://api.deepseek.com/v1
SENTINEL_LLM_MODEL=deepseek-chat
```

Key resolution order: `SENTINEL_LLM_API_KEY` → `NVIDIA_API_KEY` → `OPENAI_API_KEY`. Set any one with `sentinel configure` (interactive) or in `.env`.

The narrative is returned as four grounded sections — **summary**, **rollout**, **rollback trigger**, **monitoring** — and the model is instructed to use *only* the facts it's given (real file names and numbers), so it can't invent history.

---

## 🔌 MCP Server — Use Sentinel from Claude / Cursor

Sentinel exposes **one** MCP tool, `get_deployment_risk`, over stdio (the transport Claude Desktop and Cursor use). Built on **FastMCP**, so it also speaks SSE/HTTP for remote use.

**Tool signature:**

```
get_deployment_risk(
    repo_path: str,           # absolute path to the git repo
    scope: str = "scan",      # "scan" | "diff" | "all"
    since: str | None = None, # e.g. "main", "HEAD~20" (for scope="scan")
    explain: bool = False,    # add the LLM narrative (needs a key)
) -> dict                     # the full risk report as JSON
```

> The score is computed from your repo's history — never by a language model. `explain` defaults to `false`, so the tool is offline unless you opt in.

### Claude Desktop

Edit `claude_desktop_config.json`
(Windows: `%APPDATA%\Claude\claude_desktop_config.json` · macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "sentinel": {
      "command": "sentinel-mcp",
      "env": { "NVIDIA_API_KEY": "nvapi-your-key-here" }
    }
  }
}
```

**Zero-install (npx):**

```json
{
  "mcpServers": {
    "sentinel": { "command": "npx", "args": ["-y", "sentinel-risk-mcp"] }
  }
}
```

> If Claude can't find `sentinel-mcp`, GUI apps may not inherit your shell PATH — use the absolute path (Windows: `...\Scripts\sentinel-mcp.exe`; macOS/Linux: `~/.local/bin/sentinel-mcp`).

### Cursor

Settings → MCP → Add: name `sentinel`, command `sentinel-mcp`. Status turns green when connected.

### Remote HTTP / SSE (teams)

```bash
fastmcp run mcp_server/server.py --transport sse --port 8000
```

```json
{ "mcpServers": { "sentinel": { "url": "http://localhost:8000/sse" } } }
```

### Test it with the Inspector

```bash
npx @modelcontextprotocol/inspector sentinel-mcp
```

Open the printed URL, pick the `sentinel` server, and call `get_deployment_risk` interactively.

---

## ⚙️ GitHub Action — Score Every PR

A composite action posts (and **updates**) a single sticky comment on each PR with the score, reasons, and blast radius.

```yaml
# .github/workflows/deployment-risk.yml
name: Deployment risk
on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write   # needed to post the comment

jobs:
  risk:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0        # REQUIRED — the score comes from history
      - uses: Nishka30/SentinalScan@main
        with:
          threshold: "65"
          fail-over-threshold: "false"   # comment first, gate later
          # explain: "true"
          # nvidia-api-key: ${{ secrets.NVIDIA_API_KEY }}
```

> **`fetch-depth: 0` is not tidiness — it's required.** A shallow clone hides commit history, so every file looks brand new and every score collapses.

**Inputs:** `threshold` (default `65`), `fail-over-threshold` (default `false`), `base-ref`, `comment` (default `true`), `explain` (default `false`), `nvidia-api-key`, `llm-model`, `python-version` (default `3.12`), `github-token`.
**Outputs:** `score`, `band`, `json-path`.

**Two upgrades worth knowing:**

1. **Train a model in CI** and cache it — the model beats the rule baseline. Do it on a schedule, not every PR:
   ```yaml
   - uses: actions/cache@v4
     with: { path: .sentinel, key: sentinel-model-${{ github.ref_name }} }
   - run: sentinel train --max-commits 5000
     if: steps.cache.outputs.cache-hit != 'true'
   ```
2. **Gate merges** by setting `fail-over-threshold: "true"` and adding the check to branch protection. Start with a high threshold and lower it as the team trusts it — a gate that fires on everything gets ignored, and an ignored gate is worse than none.

---

## 🧾 JSON Output (for scripting & CI)

`--json` (or the MCP tool) returns a structured payload:

```jsonc
{
  "score": 67,
  "band": "high",
  "recommendation": "Hold this, or ship it behind a flag with a rollback ready. …",
  "scope": { "mode": "scan", "author": "…", "files_analyzed": 3,
             "base_ref": "origin/main", "commits_walked": 4854, "model": { … } },
  "reasons": [ { "rule": "hot_file", "label": "…", "points": 35,
                 "detail": "14 past bug-fix commits", "path": "auth/session.py" } ],
  "files":   [ { "path": "…", "score": 62, "band": "high", "reasons": [ … ] } ],
  "blast":   { "direct_count": 12, "transitive_count": 38, "hubs": [ … ] },
  "explanation": { "available": false }   // populated only with --explain
}
```

Pipe it straight into a gate:

```bash
pip install sentinel-risk
sentinel scan --json > risk.json
SCORE=$(jq '.score' risk.json)
[ "$SCORE" -gt 80 ] && { echo "::error::Risk $SCORE exceeds 80"; exit 1; } || true
```

---

## 🎛️ Configuration Reference

Everything is read from environment variables or a `.env` file. **No module other than `config.py` touches the environment**, so "never hardcode secrets" is checkable by reading one file.

### LLM

| Variable | Description | Default |
| --- | --- | --- |
| `NVIDIA_API_KEY` | NVIDIA NIM key | — |
| `OPENAI_API_KEY` | OpenAI key | — |
| `SENTINEL_LLM_API_KEY` | Key for any other provider | — |
| `SENTINEL_LLM_BASE_URL` | OpenAI-compatible base URL | `https://integrate.api.nvidia.com/v1` |
| `SENTINEL_LLM_MODEL` | Model for explanations | `meta/llama-3.3-70b-instruct` |
| `SENTINEL_LLM_TIMEOUT` | Seconds to wait for prose | `60` |
| `SENTINEL_LLM_MAX_RETRIES` | Retries (scan should fail fast) | `1` |
| `SENTINEL_LLM_MAX_TOKENS` | Max tokens in the narrative | `800` |
| `SENTINEL_LLM_TEMPERATURE` | Sampling temperature | `0.2` |

### Scoring, distribution, blast, training

| Variable | Overrides | Example |
| --- | --- | --- |
| `SENTINEL_RULES` | any `RiskWeights` field | `{"file_large_lines": 150, "file_large_points": 30}` |
| `SENTINEL_DISTRIBUTION` | percentile settings | `{"enabled": false}` |
| `SENTINEL_BLAST` | blast-radius limits | `{"max_depth": 2}` |
| `SENTINEL_TRAINING` | any `TrainingSettings` field | `{"max_commits": 5000}` |
| `SENTINEL_BUGFIX` | bug-fix keyword/issue patterns | — |
| `SENTINEL_REPORT_TOP_FILES` | how many files the report lists | `10` |

All values are JSON objects, validated by pydantic — a typo fails loudly instead of being silently ignored.

---

## 🚦 Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success (a score was produced, or there was nothing in scope) |
| `2` | Repository error (not a git repo, bad ref, `--all` combined with `--since`) |
| `3` | Model error (e.g. no bug-inducing commits found to train on) |

Stable codes make Sentinel safe to wire into a pipeline — a crash is never mistaken for a passing check.

---

## 🏗️ Architecture

The core library imports **neither** `typer` nor `rich` and **never** touches the network. Three callers share it — the CLI, the MCP server, and the GitHub Action — and only one has a terminal, so anything that must be *said* rather than *returned* comes back as `AnalysisResult.warnings`.

```
sentinalScan/
├── sentinel/                  # Core library — no terminal, no network
│   ├── analysis.py            # Orchestrator → AnalysisResult
│   ├── git_reader.py          # Low-level git (GitPython); default-branch + merge-base
│   ├── commit_log.py          # git log --numstat walk (+ rename tracking)
│   ├── history_mining.py      # SZZ labelling (PyDriller blame)
│   ├── features.py            # Data model + the 20-feature vector()
│   ├── risk_rules.py          # Transparent points-based engine
│   ├── model.py               # LightGBM: train / predict / SHAP explain
│   ├── blast_radius.py        # networkx dependency graph + impact walk
│   ├── static_analysis.py     # Cyclomatic complexity (lizard) + test detection
│   ├── explain.py             # LLM narrative (OpenAI-compatible) — read-only
│   ├── evaluation.py          # Time-based split, ROC-AUC / PR-AUC vs baseline
│   ├── report.py              # Rich terminal rendering
│   ├── pr_comment.py          # Renders the GitHub PR comment + threshold gate
│   ├── serialization.py       # JSON for --json and MCP
│   ├── results.py             # AnalysisResult / Scope / Explanation
│   └── config.py              # Every setting + tunable number (only env reader)
├── mcp_server/server.py       # FastMCP wrapper — one tool, delegates to sentinel/
├── github-action/             # Composite action + example workflow
├── bin/sentinel-mcp.js        # npm wrapper (auto-installs sentinel-risk via pip)
├── tests/                     # One test file per module
├── pyproject.toml             # Package metadata + entry points
└── package.json               # npm wrapper metadata
```

**Data flow (scan):**

```
git diff → git_reader → FileChange[]
                            ↓
          git log → FileHistory (per file)
                            ↓
          lizard → ComplexityInfo (per file)
                            ↓
                     features.vector()        ← same fn used in training
                            ↓
              ┌── model.predict() ──────────── .sentinel/model.txt exists?
              │                                       yes ↗     no ↘
              │                           LightGBM score       rule engine
              │                                       ↘        ↗
              └────────────── ChangeRisk (score, band, reasons)
                                        ↓
                             blast_radius.compute()   ← networkx walk
                                        ↓
                                  AnalysisResult
                                        ↓
                          report.render() / serialization.to_dict()
                                        ↓ (optional, opt-in)
                          explain.explain() → LLM API → narrative
```

---

## 🚫 Non-Goals (Deliberate Simplicity)

Sentinel is intentionally small. Things it does **not** use, and why:

| Not used | Why |
| --- | --- |
| Vector database | Nothing needs embedding or semantic search |
| Redis / cache | A local scan is fast enough to recompute |
| Neo4j (graph server) | The blast graph is built in memory in < 1s |
| "AI agents" | The pipeline steps are just functions |
| Many MCP servers | Sentinel exposes **one** server with **one** clear tool |

Each cut removes months of maintenance and zero real value. That restraint is a feature.

---

## ❓ FAQ

**Does `sentinel scan` call an LLM?**
No. `scan` and `diff` are fully offline — the score comes from the LightGBM model (if trained) or the rule engine, both pure math. The LLM is only ever contacted with `--explain` or `sentinel explain`. Run with no API key set and the score is identical.

**Wait — so what's the "model" in `sentinel scan`?**
The *LightGBM* model — a gradient-boosted tree trained on your repo's own bug history. That's deterministic math, not a language model. The only LLM in the project is the optional explanation layer.

**Why not just ask an LLM "is this risky"?**
A language model doesn't know your repo's history, defect patterns, ownership, or dependency graph — and it isn't repeatable. Sentinel measures those signals directly and gives the same answer twice.

**What if my repo has almost no history?**
The rule engine handles it out of the box. Training needs at least some bug-fixing commits; with none, `train` exits with a clear message (code `3`).

**Does it support my language?**
Complexity and test detection cover many languages (Python, Java/Kotlin/Scala/Groovy, JS/TS, Go, Ruby, Rust, C#, PHP, Swift, C/C++, …). The *dependency graph* (blast radius) currently parses **Python and Java**; other languages still get history, complexity, and test signals.

**Will it slow my CI?**
Scoring is seconds. Training is minutes (SZZ blame is the slow half) — so train on a schedule and cache `.sentinel/` rather than on every PR.

**Can the AI ever change the number?**
No. It receives a frozen result and returns prose with no score field. It's a structural guarantee, not a prompt instruction.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
| --- | --- |
| Every file scores as if brand new (in CI) | Add `fetch-depth: 0` to `actions/checkout` — a shallow clone hides history |
| `warning: the trained model was built from a different feature set …` | Sentinel updated its features; re-run `sentinel train` |
| `error: no bug-inducing commits were found` | Increase `--max-commits`, or check that commit messages mention fixes |
| Claude/Cursor can't find `sentinel-mcp` | GUI apps may not inherit PATH — use the absolute path to the binary |
| Explanation missing but score present | No API key / timeout / rate limit — expected; the score is unaffected |
| LLM call is slow | Swap `SENTINEL_LLM_MODEL` to a smaller/faster model; don't just raise the timeout |

---

## 📖 Glossary

- **SZZ** — the algorithm that links a bug-fixing commit back to the commit that introduced the bug, via line-level blame.
- **Blast radius** — the set of files that (transitively) depend on what you changed.
- **Hot file** — a file with an unusually high count of past bug fixes for this repo.
- **Churn** — how often a file changes, independent of bugs.
- **Ownership** — an author's share of a file's past commits.
- **Cyclomatic complexity (CCN)** — the number of independent paths through a function; more paths, more places to hide a bug.
- **PR-AUC** — area under the precision-recall curve; the right metric when positives are rare.
- **Band** — the low/medium/high bucket a score falls into.

---

## 👩‍💻 Author & License

**Nishka Shrimali** — [@Nishka30](https://github.com/Nishka30)

Licensed under the **MIT License** — see [LICENSE](LICENSE).

<div align="center">

**If Sentinel saves you from a bad deploy, leave a ⭐ on GitHub.**

</div>