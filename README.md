# Sentinel

**Is this change safe to deploy?** Sentinel answers that from your repository's
own bug history — not from a language model's opinion.

```
┌───────── Sentinel — deployment risk ─────────┐
│   42/100   MEDIUM RISK                       │
│   1 file(s) — changes since origin/main      │
│   scored by: trained model (4854 commits)    │
└──────────────────────────────────────────────┘
```

## The problem

Every team has files everyone is quietly afraid of. The ones that broke
production twice last year, that only one person understands, that nothing has
tests for. That knowledge lives in people's heads, so it walks out of the door
when they do — and it is not in the pull request.

Sentinel reads it out of git. It mines which commits were bug fixes, blames the
lines those fixes changed to find what introduced the bug, and learns what risky
changes in *your* codebase look like. Then it scores the change in front of you,
shows the reasons, and names what else could break.

The score is arithmetic. An LLM is used at the very end, for one call, to phrase
it in English — and it structurally cannot alter the number.

## Quickstart

Requires Python 3.11+ and git.

```bash
git clone <this-repo> && cd sentinelScan
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"

sentinel diff                   # score your uncommitted work
sentinel scan                   # score your branch against the default branch
```

That works immediately, with transparent rule-based scoring. To learn from the
repository's own history:

```bash
sentinel train --max-commits 5000     # mine bug history, train, save the model
sentinel evaluate --max-commits 5000  # prove it beats a naive baseline
sentinel scan                         # now scored by the model
```

`train` writes `.sentinel/model.txt`. Delete that directory to go back to the
rules.

### Commands

| Command | What it does |
| --- | --- |
| `sentinel scan` | Score committed changes vs the default branch. `--since <ref>`, `--all` |
| `sentinel diff` | Score uncommitted work in the working tree |
| `sentinel train` | Mine bug history and train the model |
| `sentinel evaluate` | Time-split metrics vs a lines-changed baseline |
| `sentinel explain` | Score, then narrate it in plain English |

Add `--json` to any scan for machine-readable output, or `--explain` to append
the narrative.

## What it looks like

`sentinel scan` on a real change to `psf/requests`:

```
Why
 impact │ reason                                  │ evidence
────────┼─────────────────────────────────────────┼───────────────────────────
    +69 │ Lines added                             │ value 7; log-odds +0.687
    -39 │ Complexity of the most complex file     │ value 19; log-odds -0.386
    -16 │ Lines deleted                           │ value 0; log-odds -0.162
    +10 │ Source files touched                    │ value 1; log-odds +0.100

Impact
  5 file(s) import these directly, 8 more within 3 hops.
  direct: src/requests/adapters.py, src/requests/auth.py,
          src/requests/models.py, src/requests/sessions.py, tests/test_utils.py
  cycle: src/requests/utils.py takes part in a circular import, so some of its
         dependents are also its dependencies.

Recommendation
  Deployable, but review it properly and watch it after release. Suggested:
  consider splitting this into smaller, separately deployable changes; the most
  complex function here deserves a second reviewer.
```

> _Screenshot placeholder — the above is real terminal output, captured as text._

And `sentinel explain` on the same change:

```
┌───── AI explanation — nvidia/llama-3.3-nemotron-super-49b-v1 ─────┐
│  Why this is risky                                                │
│  This change has a medium risk score of 42, primarily driven by    │
│  the 7 lines added (+69 impact), somewhat offset by the complexity │
│  of the most complex file (-39). The complexity of the file,       │
│  despite being high, lowered the risk according to the model's     │
│  historical data.                                                 │
│                                                                   │
│  Roll back if                                                     │
│  ...unexpected errors in the affected files (e.g.,                │
│  src/requests/adapters.py, tests/test_utils.py)...                │
└───── generated prose; the score above is computed, not written ────┘
```

Every number and filename there was handed to it. Note the complexity sentence:
SHAP reported that high complexity *lowered* the score for this change, because
that is what this repository's history showed, and the model was instructed to
report such factors as measured rather than "correct" them.

## How it works

```
git log ──> SZZ labelling ──> features ──> LightGBM ──> score
                                                          │
   import graph ──> blast radius ─────────────────────────>├──> report
                                                          │
                              one NVIDIA call ────────────>┘  (optional prose)
```

**1. SZZ labelling** — find commits whose subject reads like a bug fix, then
`git blame` the lines each fix *deleted* to find the commits that introduced
them. Those are labelled bug-inducing. Real labels, mined from the repository,
not guessed.

**2. Features** — each commit becomes numbers: size, spread across folders, the
file's change and bug-fix history, how well the author knows it, complexity, day
and hour. Every feature is read from a running tally of *prior* history, before
the commit is applied, so nothing can leak from the commit it describes.

**3. The model** — LightGBM, weighting the rare positive class, with the number
of boosting rounds chosen on a held-out slice. SHAP turns each prediction back
into named reasons.

**4. Blast radius** — an in-memory networkx import graph, walked backwards from
the changed files to find their dependents, direct and transitive kept separate.

**5. The explanation** — one call to an NVIDIA-hosted model, given the finished
analysis and asked to phrase it.

Without a trained model, steps 1–3 are replaced by a transparent points-based
rule engine whose thresholds come from the repository's own percentiles. The
report always says which engine scored it.

## Evaluation

Measured on `psf/requests` — 4,875 non-merge commits, 777 labelled bug-inducing
(16.0%):

| metric | model | lines-changed baseline | verdict |
| --- | --- | --- | --- |
| ROC-AUC | **0.737** | 0.722 | model wins |
| PR-AUC | **0.250** | 0.220 | model wins |

Train on the 3,640 older commits, test on the 1,214 newest, split at 2017-05-17.
Base rate in the test set is 8.2%, so PR-AUC 0.250 is roughly 3× random.

**The margin is modest, and here is the honest reason why.** SZZ labels a commit
by blaming the lines it wrote, so a large commit has more lines available to be
blamed later — size and the label are mechanically correlated. Median
`lines_changed` is 19 for bug-inducing commits against 4 for clean ones. That
makes "bigger diffs are riskier" a genuinely strong baseline, and beating it by
0.03 PR-AUC is a real but unspectacular result. It also depends on having enough
labelled history: on the newest 1,200 commits alone (98 positives) the model
*lost* to the baseline; across the full history (777 positives) it wins.

Two rules keep the number honest:

- **The split is by time, never at random.** A random split lets the model learn
  from commits that came after the ones it is tested on. It is the easiest way to
  report an accuracy the tool will never reproduce.
- **The baseline is deliberately hard to beat.** A model that cannot beat one
  feature is not worth its complexity.

## GitHub Action

Comments on every pull request with the score, reasons and blast radius.

```yaml
name: Deployment risk
on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write

jobs:
  risk:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # required — the score comes from commit history

      - uses: your-org/sentinel@main
        with:
          threshold: "65"
          fail-over-threshold: "false" # comment first; gate once trusted
```

`fetch-depth: 0` is not tidiness. A shallow clone hides the history the score is
built from, and every file looks brand new.

The comment is rendered from `--json`, so nothing is recomputed for CI:

> ## 🔴 HIGH — deployment risk 71/100
>
> **Hold this, or ship it behind a flag with a rollback ready.**
>
> Scored by a model trained on 4,854 of this repository's commits over 13 changed file(s).
>
> \> Above the configured threshold of 65.
>
> ### Why
>
> | SHAP impact | Factor | Evidence |
> | --- | --- | --- |
> | +125 | Lines added | value 56; log-odds +1.255 |
> | -29 | Average complexity of the files touched | value 12.25; log-odds -0.287 |
>
> ### Impact
>
> **8** file(s) import these directly, **2** more within 3 hops.
>
> ⚠️ `src/requests/utils.py` is part of a circular import.

Defaults to informing rather than blocking, because a gate that fires on
everything gets ignored. Set `fail-over-threshold: "true"` to make it a required
check. The AI narrative is off in CI unless you pass `explain: "true"` with an
`nvidia-api-key` — CI should not depend on a third-party API, and the score is
identical either way. Full options in
[github-action/action.yml](github-action/action.yml); a complete example in
[github-action/example-workflow.yml](github-action/example-workflow.yml).

## Use as an MCP server

One server, one tool — `get_deployment_risk` — so Claude Desktop, Cursor, or any
MCP-compatible client can ask Sentinel about a repository directly.

The server is built with **FastMCP** and speaks the MCP protocol over stdio.

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "sentinel": { "command": "sentinel-mcp" }
  }
}
```

> **If the client cannot find `sentinel-mcp`**, GUI apps don't inherit your
> shell PATH. Use the absolute path to the venv's binary instead:
> - macOS / Linux: `/path/to/.venv/bin/sentinel-mcp`
> - Windows: `C:\path\to\.venv\Scripts\sentinel-mcp.exe`

### Optional `env` block — NVIDIA_API_KEY for the `explain` feature

```json
{
  "mcpServers": {
    "sentinel": {
      "command": "sentinel-mcp",
      "env": { "NVIDIA_API_KEY": "nvapi-..." }
    }
  }
}
```

Without the key the analysis runs normally; only the plain-English narrative is
skipped.

### Calling it from the client

Ask the model to assess the deployment risk of a repository and pass its
absolute path; the model invokes `get_deployment_risk` automatically.

Example prompt: _"What is the deployment risk of `/home/me/myapp`? Use
`sentinel`."_

The tool accepts:
- `repo_path` (required) — absolute path to the git repository
- `scope` — `scan` (default) / `diff` / `all`
- `since` — ref to compare against, e.g. `main` or `HEAD~20`
- `explain` — `true` to append the plain-English narrative (needs `NVIDIA_API_KEY`)

It returns the same JSON contract the CLI emits — score, band, reasons, blast
radius — as both a text block and `structuredContent`.

## Configuration

Copy `.env.example` to `.env`. Every value is optional — the score is computed
without an LLM, so a missing key only skips the explanation.

Sentinel supports any OpenAI-compatible LLM provider (NVIDIA, OpenAI, DeepSeek, Local/Ollama, etc.).

| Variable | Purpose | Default |
| --- | --- | --- |
| `SENTINEL_LLM_API_KEY` | API Key for the LLM provider (Falls back to `NVIDIA_API_KEY` or `OPENAI_API_KEY`) | unset (explanation skipped) |
| `SENTINEL_LLM_BASE_URL` | Base URL of the OpenAI-compatible API endpoint | `https://integrate.api.nvidia.com/v1` |
| `SENTINEL_LLM_MODEL` | The model name to query | `meta/llama-3.3-70b-instruct` |
| `SENTINEL_LLM_TIMEOUT` | Seconds to wait for the explanation | `60` |
| `SENTINEL_LLM_MAX_TOKENS` | Length cap on the narrative | `800` |
| `SENTINEL_LLM_TEMPERATURE` | Sampling temperature | `0.2` |

Every threshold, weight and training knob lives in
[config.py](sentinel/config.py) and is tunable without touching any logic.

## Design notes

**The AI cannot move the score, structurally.** `explain()` receives an
already-frozen `ChangeRisk`; `Explanation` has no field for a score, band or
reason; attaching one returns a new result rather than mutating. Prompts are not
a security boundary, so a model that decides the score "should really be 90" has
nowhere to put it. A test feeds it a reply insisting the change is *"0/100 and
perfectly safe"* and asserts the payload is byte-identical apart from the prose.

**History is measured before the change being scored.** Otherwise a change
improves its own files' track record and hides its own risk.

**History follows renames.** Without it, a repository that moved `foo/` to
`src/foo/` looks brand new — `requests/models.py` reported 9 commits and 4 bug
fixes when the real figures were 726 and 181.

**Thresholds are relative to the repository.** Eight past bug fixes is
remarkable in a young service and unremarkable in a fifteen-year-old library. On
`requests`, percentiles cut the files qualifying for the top tier from 40 (28% of
files with any fix) to 15 (10%).

**Bug-fix detection requires a fixing verb.** A bare `#1234` is a pull-request
number in any squash-merging repository; counting those labelled 51% of
`requests` as bug fixes instead of 13%.

## Limitations

Kept deliberately visible.

- **Bug-fix detection is keyword-based.** It misses a fix whose subject says
  "tidy up" and catches a feature that mentions "fix".
- **Recent commits look cleaner than they are**, because nobody has found their
  bugs yet. Inherent to SZZ, and part of why evaluation splits by time.
- **SZZ correlates with commit size**, which is why the model's margin over a
  size-only baseline is modest rather than dramatic. Stated in full above.
- **The dependency graph sees static imports only.** Java same-package
  references, `importlib.import_module(name)`, plugin registries and dependency
  injection are invisible. Blast radius is a **floor** on impact, not a ceiling.
- **Test detection is convention-based.** `src/requests/models.py` reads as
  untested even though `tests/test_requests.py` covers it, because the names do
  not correspond.
- **The AI narrative is unverified prose.** It is grounded in the facts it is
  given and cannot alter the score, but nothing checks its rollout advice is
  *good*. Treat the numbers as the finding and the narrative as a starting point.

## Performance

| Step | Cost on `psf/requests` (6,486 commits) |
| --- | --- |
| History walk (`git log --numstat`) | ~5s |
| Dependency graph | 0.13s |
| SZZ blame (`train` / `evaluate`) | ~4.5 min for 656 bug-fix commits |
| AI explanation | one call, 60s timeout, optional |

`train` and `evaluate` take `--max-commits` to bound the slow part, and always
report the window they used so a truncated run cannot be mistaken for a full one.

## Tests

```bash
pytest
```

298 tests, no network. The OpenAI client is replaced wholesale and an autouse
fixture blanks `NVIDIA_API_KEY`, so a real key in a developer's `.env` cannot
turn the suite into a live API call. The git tests build real repositories in
temp directories — including one with a bug deliberately introduced and later
fixed, so the SZZ blame is checked against a known answer — because the point of
these modules is that they agree with git, and a mocked git proves nothing.

## Project layout

```
sentinel/
  cli.py             typer commands; presentation only
  analysis.py        orchestration — no typer, no rich
  commit_log.py      parsing git's output (numstat, renames, dates)
  git_reader.py      asking the repository questions
  static_analysis.py complexity + test detection
  risk_rules.py      transparent points-based scoring
  history_mining.py  SZZ labelling
  features.py        shared data model + the feature vector
  model.py           LightGBM + SHAP
  evaluation.py      time split, ROC-AUC / PR-AUC
  blast_radius.py    import graph + impact
  explain.py         the single LLM call
  results.py         result types
  serialization.py   the versioned JSON contract
  report.py          terminal rendering
  pr_comment.py      the PR comment
mcp_server/          one MCP tool over stdio
github-action/       composite action + example workflow
```

See [PLAN.md](PLAN.md) for the build plan, phase by phase.

## License

MIT

---

## Publishing

### Python Package (PyPI)

The PyPI distribution name is **`sentinel-risk`** (the command the user types stays `sentinel`). If `sentinel-risk` is already taken on PyPI, change only the `name` field in `pyproject.toml`.

#### Step-by-step
```bash
# 1. Install build tools
pip install build twine

# 2. Build the wheel and source archive
python -m build
# → dist/sentinel_risk-0.1.0-py3-none-any.whl
# → dist/sentinel_risk-0.1.0.tar.gz

# 3. Upload to TestPyPI first
twine upload --repository testpypi dist/*

# 4. Verify on TestPyPI
pip install --index-url https://test.pypi.org/simple/ sentinel-risk

# 5. Upload to real PyPI
twine upload dist/*
```

Twine will prompt for your PyPI credentials (or read them from `~/.pypirc`).

---

### Node Package (NPM)

To make it incredibly easy for Node/JavaScript developers to use Sentinel as an MCP server without manually configuring Python environments, we also package a zero-config wrapper for NPM (`sentinel-mcp`).

#### Step-by-step
```bash
# 1. Log in to npm (once)
npm login

# 2. Publish to the npm registry
npm publish --access public
```

Once published, users can launch the MCP server with zero setup via:
```bash
npx sentinel-mcp
```
*(If Python/`sentinel-risk` is not already installed on the user's system, the npm wrapper will attempt to automatically auto-install it via `pip` on the fly).*

---

### No-publish alternative (install directly from GitHub)

```bash
pip install git+https://github.com/<user>/sentinelScan.git
```

This clones and installs the package in one step without touching PyPI, useful for private or pre-release deployments.
