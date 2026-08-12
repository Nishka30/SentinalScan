# Contributing to Sentinel

Thanks for your interest in improving Sentinel. This guide covers everything you need to work on the codebase: local setup, running the tests, the architecture rules the project holds itself to, and how a release is cut.

---

## 🛠️ Development Setup

**Requirements:** Python **3.11+**, `git`, and (for the npm wrapper) Node **18+**.

```bash
# 1. Clone
git clone https://github.com/Nishka30/SentinalScan.git
cd SentinalScan

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Upgrade pip and install in editable mode with dev extras
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install (`-e`) means your changes take effect immediately — no reinstall between edits. It also puts the `sentinel` and `sentinel-mcp` commands on your PATH.

Verify it worked:

```bash
sentinel --help
sentinel version
```

---

## 🧪 Running the Tests

```bash
pytest                            # run everything
pytest -v                         # verbose
pytest tests/test_risk_rules.py   # a single module
pytest -k blast_radius -v         # match by keyword
pytest --cov=sentinel             # with coverage (if pytest-cov is installed)
```

The suite is one file per module (~15 files, 240+ test functions) and uses **in-memory git repositories** built by `conftest.py` fixtures — no network calls, no temp files left on disk, and no dependency on any particular repository being checked out. The from-scratch metrics (ROC-AUC, PR-AUC) are checked against hand-computed values.

**Please keep the suite green.** A PR that adds behaviour should add tests for it; a PR that fixes a bug should add a test that fails without the fix.

---

## 🧭 Architecture Rules

These are the invariants that keep Sentinel honest and maintainable. A change that breaks one of them needs a very good reason.

1. **The core library stays pure.** Everything under `sentinel/` (except the CLI, `report.py`, and `explain.py`) must be free of `typer`, `rich`, and network calls. Anything that must be *said* to a user rather than *returned* comes back as `AnalysisResult.warnings`, so the CLI, MCP server, and GitHub Action all behave identically.

2. **The score never depends on the network.** `scan` / `diff` are offline. The LLM in `explain.py` receives an already-frozen `ChangeRisk` and returns prose with no score field — it can read the result, never write to it. Removing the API key must leave the score byte-for-byte identical.

3. **No lookahead in features.** Everything in `features.vector()` must be computable *at the moment the change was made*. Never use later commits, current file contents, or the present-day dependency graph for a historical commit. This is what makes the time-based evaluation meaningful.

4. **The feature vector is append-only.** Feature order is part of the saved model's contract. **Append** to `FEATURE_NAMES`; never reorder or remove — an old model file would silently read the wrong columns. Bump behaviour is guarded by the model's JSON sidecar, but don't rely on it to catch a reorder.

5. **Adding a language to blast radius** means implementing one `LanguageParser` (declare `extensions`, `module_names`, `references`) and adding it to `PARSERS` in `blast_radius.py`. The graph and traversal code must not need to know which language it's looking at.

6. **Scoring policy lives in `config.py`.** Every threshold and point value belongs in `RiskWeights` / `TrainingSettings` / etc., not inline in scoring logic. `config.py` is also the *only* module allowed to read `os.environ`.

7. **Evaluate by time, never at random.** Any change to `evaluation.py` must preserve the time-ordered train/test split and the baseline comparison.

---

## 🔀 Pull Requests

1. Fork and branch from `main`.
2. Make your change, with tests, keeping the suite green.
3. Keep commits focused; write commit messages that describe *why*, not just *what*.
4. Open a PR describing the change and linking any related issue.

Sentinel scores itself in CI, so expect a deployment-risk comment on your PR — that's the tool eating its own dog food, not a failure.

---

## 🚢 Releasing (maintainers only)

Both registries bake the description and README in at publish time and **won't accept a re-upload of the same version**, so bump the version first (`pyproject.toml` for PyPI, `package.json` for npm).

### PyPI — `sentinel-risk`

```bash
python -m pip install --upgrade build twine

# Clean previous artifacts (PowerShell)
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
# bash: rm -rf dist build *.egg-info

python -m build
python -m twine check dist/*        # must say PASSED
python -m twine upload dist/*
```

Verify: <https://pypi.org/project/sentinel-risk/>

### npm — `sentinel-risk-mcp`

```bash
npm login
npm whoami                          # confirm you're logged in
npm publish --access public
npm view sentinel-risk-mcp          # confirm it went live
```

Verify: <https://www.npmjs.com/package/sentinel-risk-mcp>

### After a release

- Tag the release in git: `git tag v0.1.2 && git push --tags`.
- Keep the two version numbers (`pyproject.toml`, `package.json`) in sync so the pip and npm packages track each other.

> **Never commit API tokens or credentials.** Keep them in your environment or a local credentials file that is git-ignored.

---

## 📜 License

By contributing, you agree that your contributions are licensed under the project's [MIT License](LICENSE).