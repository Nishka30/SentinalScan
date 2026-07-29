# 🛡️ Sentinel — Deployment Risk Analyzer

[![PyPI version](https://img.shields.io/pypi/v/sentinel-risk.svg?style=flat-square)](https://pypi.org/project/sentinel-risk/)
[![NPM version](https://img.shields.io/npm/v/sentinel-risk-mcp.svg?style=flat-square)](https://www.npmjs.com/package/sentinel-risk-mcp)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-296%20passing-brightgreen?style=flat-square)]()

> **One question, one answer: is this change safe to deploy?**
>
> Sentinel answers from your repository's own bug history — not a language model's guess.

```
┌─────────── Sentinel — deployment risk ────────────────────────┐
│   Score: 67/100   ██████████████████░░░░░  HIGH RISK          │
│   3 file(s) changed — since origin/main                       │
│   Scored by: trained model (4,854 commits)                    │
│                                                               │
│   Top reasons:                                                │
│    • auth/session.py  — fixed 14 times in the last 2 years   │
│    • Author has never touched this file before                │
│    • Change spans 6 folders — high blast radius               │
│                                                               │
│   Blast radius: 12 files directly affected, 38 transitively   │
└───────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

- [What Is Sentinel?](#-what-is-sentinel)
- [How It Works — The Core Pipeline](#-how-it-works--the-core-pipeline)
- [What Makes It Different](#-what-makes-it-different)
- [Installation](#-installation)
- [Quickstart: First Scan in 60 Seconds](#-quickstart-first-scan-in-60-seconds)
- [Full CLI Reference](#-full-cli-reference)
- [Enabling AI Explanations](#-enabling-ai-explanations-optional)
- [Training a Custom Model on Your Repository](#-training-a-custom-model-on-your-repository)
- [MCP Server — For Claude Desktop & Cursor](#-mcp-server--for-claude-desktop--cursor)
- [Configuration Reference](#-configuration-reference)
- [Architecture Overview](#-architecture-overview)
- [Running Tests](#-running-tests)
- [Publishing](#-publishing)
- [Author](#-author)

---

## 🔍 What Is Sentinel?

Every team has files they are quietly afraid of. The ones that broke production twice last year, that only one person truly understands, that nothing has tests for. You can feel that instinct during code review — but you cannot put a number on it.

Sentinel puts a number on it.

It works by reading your repository's full git history, finding which past commits later had to be *fixed*, tracing those bug-inducing commits using a line-level blame algorithm (called SZZ), and building a lightweight gradient-boosted model trained entirely on your own data. It then uses that model — or its transparent rule engine when no model exists — to score any incoming change on a 0–100 risk scale.

**The score is always deterministic.** A language model can optionally be called to narrate the findings in plain English, but it structurally cannot affect the numeric score. Remove the API key and the score is unchanged — only the paragraph disappears.

---

## ⚙️ How It Works — The Core Pipeline

Sentinel runs in two distinct phases:

### Phase 1 — Training (learn from history)

> Run once per repository with `sentinel train`. The result is a model file saved in `.sentinel/`.

```
1. Read git log                     → All commits in the configured window (default: 1,500)
2. Detect bug-fix commits           → Keyword match on commit subjects ("fix", "hotfix", ...)
3. SZZ blame                        → For each bug fix, blame the deleted lines against the
                                      parent commit to find which earlier commits introduced it
4. Label commits                    → Commits identified by blame = 1 (risky), others = 0
5. Build feature vectors            → 20 features per commit, computed from info available
                                      *at commit time* (no lookahead)
6. Train LightGBM with early stop   → Max 300 rounds, stops when held-out AP stops improving
7. Save model + metadata            → .sentinel/model.txt + .sentinel/model.meta.json
```

### Phase 2 — Scoring (assess a new change)

> Runs every time you call `sentinel scan`, `sentinel diff`, or the MCP tool.

```
1. Identify changed files           → git diff vs. base branch (scan) or working tree (diff)
2. Read file histories              → Bug-fix counts, churn, author ownership per file
3. Static analysis                  → Cyclomatic complexity via lizard
4. Build feature vector             → Same 20-feature function used in training
5. Score                            → Trained model if available; rule engine otherwise
6. Blast radius                     → networkx dependency graph → direct + transitive impact
7. Report                           → Rich terminal table, or JSON for CI pipelines
8. Explain (optional)               → Send risk summary to configured LLM for plain-English narration
```

---

## 💡 What Makes It Different

| Tool | How it decides risk |
|---|---|
| Most linters | Checks style rules — knows nothing about *history* |
| Code coverage tools | Tells you what is tested — not what has *historically* broken |
| PR review bots | Often LLM-based — the model is the judge |
| **Sentinel** | **Trains on your repo's own bug record. The score is a probability from your history, not an opinion.** |

Key design principles baked into the code:

- **No lookahead in training.** Features for a historical commit are built from data available *before* that commit was made. The running history tally is applied *after* featurising, not before.
- **Relative thresholds.** Eight past bug fixes is remarkable in a new service and unremarkable in a fifteen-year-old library. Sentinel uses the repo's own distribution percentiles as thresholds, not hardcoded absolute numbers.
- **Blast radius is separate from the score.** The dependency graph tells you *what else can break*. It is computed from today's import graph and shown in the report, but it is deliberately not fed into the model (doing so honestly would require re-parsing the entire tree at every historical commit).
- **LLM is additive only.** The AI explanation layer can read the score. It cannot write to it.

---

## 📦 Installation

**Requirements:** Python 3.11+

### Option A — pip (recommended)

```bash
pip install sentinel-risk
```

This installs the `sentinel` CLI, the `sentinel-mcp` server binary, and all dependencies (`lightgbm`, `pydriller`, `GitPython`, `rich`, `fastmcp`, etc.).

### Option B — npx (zero-install MCP server only)

```bash
npx sentinel-risk-mcp
```

Runs the MCP server without any Python setup. Useful for connecting Claude Desktop to Sentinel without installing anything globally.

### Option C — from source

```bash
git clone https://github.com/Nishka30/SentinalScan.git
cd SentinalScan

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
```

---

## 🚀 Quickstart: First Scan in 60 Seconds

```bash
# 1. Install
pip install sentinel-risk

# 2. Go to any git repository you work in
cd /path/to/your/project

# 3. Scan changes since main
sentinel scan

# 4. Or score uncommitted (staged + unstaged) changes
sentinel diff
```

That is it. No configuration required for the core score.

**Want plain-English explanations too?** Run the interactive setup wizard:

```bash
sentinel configure
```

This walks you through choosing your LLM provider and saves the key to a local `.env` file. Once done:

```bash
sentinel scan --explain
# or
sentinel diff --explain
```

---

## 📖 Full CLI Reference

### `sentinel scan` — Score committed changes

Scores the diff between `HEAD` and the default branch (or a specified ref).

```bash
sentinel scan                        # vs. auto-detected default branch
sentinel scan --since origin/main    # vs. a specific ref
sentinel scan --since HEAD~10        # last 10 commits
sentinel scan --all                  # rank every tracked file by inherent risk
sentinel scan --explain              # also request LLM narrative
sentinel scan --json                 # machine-readable JSON output
sentinel scan --repo /other/project  # analyze a different repository
```

### `sentinel diff` — Score uncommitted work

Scores the files currently modified in your working tree (staged and unstaged).

```bash
sentinel diff
sentinel diff --explain
sentinel diff --json
```

### `sentinel explain` — Score and explain

Shorthand for scan + explain in one step.

```bash
sentinel explain                     # explains committed changes
sentinel explain --diff              # explains uncommitted changes
sentinel explain --since HEAD~5
sentinel explain --json
```

### `sentinel train` — Train the model on your history

Mines the repository's git history using the SZZ algorithm and trains a LightGBM model. The trained model is saved to `.sentinel/` in the repository root.

```bash
sentinel train                       # mine up to 1,500 commits (default)
sentinel train --max-commits 5000    # mine deeper history (slower)
sentinel train --repo /other/project
```

After training, all subsequent `scan` / `diff` calls use the model instead of the rule engine.

### `sentinel evaluate` — Measure model quality

Mines history and runs a time-based train/test split to measure how well the model beats a naive "lines changed" baseline. Always uses the newest commits as the test set — never a random split.

```bash
sentinel evaluate
sentinel evaluate --max-commits 3000
```

Example output:

```
  Training rows:   3,621   Positives: 312 (8.6%)
  Test rows:       1,207   Positives: 104 (8.6%)

  Rule engine        PR-AUC: 0.271
  LightGBM model     PR-AUC: 0.438   ✓ beats baseline
```

### `sentinel configure` — Set up LLM integration

Interactive wizard that saves your LLM settings to a local `.env` file.

```bash
sentinel configure
```

Prompts:
1. **Provider** — `NVIDIA`, `OpenAI`, or `Custom` (any OpenAI-compatible endpoint)
2. **Base URL** — auto-filled for known providers, or enter your own
3. **Model name** — auto-filled for known providers, or enter your own
4. **API key** — entered securely (hidden input)

Supported out of the box:
- **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`) — default
- **OpenAI** (`https://api.openai.com/v1`)
- **Any custom endpoint** — Ollama, DeepSeek, Together AI, Azure OpenAI, etc.

### `sentinel version` — Print version

```bash
sentinel version
# sentinel 0.1.1
```

---

## 🤖 Enabling AI Explanations (Optional)

Explanations are completely optional. The risk score is identical with or without them.

### Step 1 — Configure

```bash
sentinel configure
```

Or set environment variables manually:

```bash
# .env  (copy from .env.example)
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
SENTINEL_LLM_MODEL=meta/llama-3.3-70b-instruct
SENTINEL_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
```

### Step 2 — Run with `--explain`

```bash
sentinel scan --explain
```

The explanation appears below the risk table in the terminal output. It summarizes *why* the score is what it is, citing specific files, rules, and history signals.

### Using a different LLM provider

```bash
# OpenAI GPT-4o
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

---

## 🧠 Training a Custom Model on Your Repository

The default rule engine works out of the box, but training a model on your own history significantly improves accuracy — especially for repositories with complex, high-churn areas that the rules cannot distinguish.

```bash
# Step 1: Train (takes 2–10 minutes depending on history size)
sentinel train

# Step 2: Verify the model beats the baseline
sentinel evaluate

# Step 3: Scan normally — the model is used automatically
sentinel scan
```

**What gets saved:**

```
your-project/
└── .sentinel/
    ├── model.txt           # The LightGBM booster
    └── model.meta.json     # Training metadata (rows, positives, feature list)
```

The `.sentinel/` directory is auto-added to `.gitignore` so you do not accidentally commit your model. If you *want* to commit and share it, remove it from `.gitignore`.

**If the feature set changes** (Sentinel updates), the old model is detected as incompatible and the rule engine is used as fallback. A warning is printed. Re-run `sentinel train` to build a fresh model.

---

## 🔌 MCP Server — For Claude Desktop & Cursor

Sentinel exposes a single MCP tool — `get_deployment_risk` — that allows any MCP-compatible AI client to analyze a repository's deployment safety on demand.

### Tool Signature

```python
get_deployment_risk(
    repo_path: str,          # Absolute path to the git repository
    scope: str = "scan",     # "scan" | "diff" | "all"
    since: str | None = None,# e.g. "main", "HEAD~20" (for scope="scan")
    explain: bool = False,   # Request LLM narrative (needs API key)
) -> dict                    # Full risk report as structured JSON
```

### Claude Desktop Setup

Edit `claude_desktop_config.json`:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sentinel": {
      "command": "sentinel-mcp",
      "env": {
        "NVIDIA_API_KEY": "nvapi-your-key-here"
      }
    }
  }
}
```

> **Tip:** If Claude cannot find `sentinel-mcp`, GUI apps may not inherit your shell PATH. Use the absolute path instead:
> - Windows: `C:\Users\YourName\AppData\Local\Programs\Python\Python311\Scripts\sentinel-mcp.exe`
> - macOS: `/Users/yourname/.local/bin/sentinel-mcp`

### Cursor Setup

In Cursor settings → MCP Servers → Add:

```json
{
  "sentinel": {
    "command": "sentinel-mcp"
  }
}
```

### Zero-install via npx

```bash
# In claude_desktop_config.json
{
  "mcpServers": {
    "sentinel": {
      "command": "npx",
      "args": ["-y", "sentinel-risk-mcp"]
    }
  }
}
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector sentinel-risk-mcp
```

Open the URL shown in the terminal, select the `sentinel` server, and call `get_deployment_risk` interactively.

---

## ⚙️ Configuration Reference

All settings are read from environment variables or a `.env` file in the working directory.

### LLM Settings

| Variable | Description | Default |
|---|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM API key (from [build.nvidia.com](https://build.nvidia.com)) | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `SENTINEL_LLM_API_KEY` | API key for any other provider | — |
| `SENTINEL_LLM_BASE_URL` | OpenAI-compatible base URL | `https://integrate.api.nvidia.com/v1` |
| `SENTINEL_LLM_MODEL` | Model name to use for explanations | `meta/llama-3.3-70b-instruct` |
| `SENTINEL_LLM_TIMEOUT` | Seconds to wait for LLM response | `60` |
| `SENTINEL_LLM_MAX_TOKENS` | Max tokens in the explanation | `800` |
| `SENTINEL_LLM_TEMPERATURE` | Sampling temperature | `0.2` |

The key is resolved in order: `SENTINEL_LLM_API_KEY` → `NVIDIA_API_KEY` → `OPENAI_API_KEY`.

### Scoring Thresholds

Risk weights can be overridden if you find the defaults too lenient or strict for your team:

| Variable | Description | Default |
|---|---|---|
| `SENTINEL_RULES` | JSON object overriding any `RiskWeights` field | (see config.py) |
| `SENTINEL_DISTRIBUTION` | JSON overriding percentile settings | (see config.py) |
| `SENTINEL_BLAST` | JSON overriding blast radius limits | (see config.py) |

Example — tighten the "large edit" threshold:

```bash
SENTINEL_RULES='{"file_large_lines": 150, "file_large_points": 30}'
```

### Training Settings

| Variable | Description | Default |
|---|---|---|
| `SENTINEL_TRAINING` | JSON overriding any `TrainingSettings` field | (see config.py) |

Example — mine more history:

```bash
SENTINEL_TRAINING='{"max_commits": 5000}'
```

---

## 🏗️ Architecture Overview

```
sentinalScan/
├── sentinel/                  # Core library — no terminal, no network
│   ├── analysis.py            # Orchestrator: calls all modules, returns AnalysisResult
│   ├── git_reader.py          # Low-level git operations (GitPython)
│   ├── commit_log.py          # Reads and parses git log (numstat + rename tracking)
│   ├── history_mining.py      # SZZ labeling algorithm (PyDriller blame)
│   ├── features.py            # Data model + feature vector (20 features)
│   ├── risk_rules.py          # Rule engine: transparent, points-based scoring
│   ├── model.py               # LightGBM wrapper: train, predict, SHAP explain
│   ├── blast_radius.py        # Dependency graph (networkx) + impact walk
│   ├── static_analysis.py     # Cyclomatic complexity (lizard), test detection
│   ├── explain.py             # LLM narrative layer (OpenAI-compatible client)
│   ├── evaluation.py          # Time-based model evaluation / PR-AUC
│   ├── report.py              # Rich terminal rendering
│   ├── serialization.py       # JSON serialization for --json and MCP
│   ├── results.py             # AnalysisResult dataclass
│   └── config.py              # All settings via pydantic-settings
│
├── mcp_server/
│   └── server.py              # FastMCP wrapper — one tool, delegates to sentinel/
│
├── tests/                     # 296 tests, all passing
│   ├── conftest.py            # Shared git repo fixtures
│   ├── test_cli.py            # End-to-end CLI tests (typer test runner)
│   ├── test_blast_radius.py   # Dependency graph tests
│   ├── test_risk_rules.py     # Rule engine tests
│   ├── test_model.py          # LightGBM training + prediction tests
│   ├── test_explain.py        # LLM client tests (mocked)
│   ├── test_mcp_server.py     # MCP server tool tests
│   └── ...                    # one file per module
│
├── pyproject.toml             # Package metadata + entry points
├── package.json               # npm wrapper for npx sentinel-risk-mcp
└── .env.example               # Template for LLM configuration
```

### Data flow (scan)

```
git diff → git_reader → FileChange[]
                             ↓
           git log → FileHistory (per file)
                             ↓
           lizard → ComplexityInfo (per file)
                             ↓
                      features.vector()           ← same function used in training
                             ↓
               ┌─── model.predict() ────────────── .sentinel/model.txt exists?
               │                                               yes ↗   no ↘
               │                              LightGBM score       rule engine
               │                                           ↘      ↗
               └──────────────── ChangeRisk (score, band, reasons)
                                         ↓
                              blast_radius.compute()   ← networkx dependency walk
                                         ↓
                                   AnalysisResult
                                         ↓
                              report.render() / to_dict()
                                         ↓ (optional)
                              explain.explain() → LLM API → narrative string
```

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all 296 tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_risk_rules.py -v

# Run tests matching a keyword
pytest -k "blast_radius" -v
```

Tests use in-memory git repositories built by `conftest.py` fixtures — no real remote connections, no temporary files on disk.

---

## 🚢 Publishing

### Python Package (PyPI)

```bash
# Install build tools (once)
pip install build twine

# Build wheel + sdist
python -m build

# Upload to real PyPI
twine upload dist/*

# (Optional) Test on TestPyPI first
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ sentinel-risk==0.1.1
```

Before publishing, make sure `pyproject.toml` has the correct version and your GitHub URL:

```toml
[project.urls]
Homepage = "https://github.com/Nishka30/SentinalScan"
Repository = "https://github.com/Nishka30/SentinalScan"
```

### Node Package (npm)

```bash
# Log in to npm (once)
npm login

# Publish the zero-install wrapper
npm publish --access public
```

The npm package installs `sentinel-risk` via pip automatically and provides the `npx sentinel-risk-mcp` entrypoint. See `package.json` and `bin/sentinel-mcp.js` for details.

---

## 📝 CI / CD Integration

Use `--json` to pipe Sentinel output into your CI pipeline:

```yaml
# GitHub Actions example
- name: Risk check
  run: |
    pip install sentinel-risk
    sentinel scan --json > risk.json
    SCORE=$(jq '.risk.score' risk.json)
    echo "Risk score: $SCORE"
    if [ "$SCORE" -gt 80 ]; then
      echo "::error::Risk score $SCORE exceeds threshold of 80"
      exit 1
    fi
```

Or use the GitHub Action in `github-action/` if you want a dedicated action in your workflow (see `github-action/` for full usage).

---

## 👩‍💻 Author

**Nishka Shrimali**
- GitHub: [@Nishka30](https://github.com/Nishka30)
- Email: shrimalinishka30@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If Sentinel saves you from a bad deploy, leave a ⭐ on GitHub.**

</div>
