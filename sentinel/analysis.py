"""Assemble a complete analysis.

This is the library entry point, and it deliberately imports neither typer nor
rich. Three things call it — the CLI, the MCP server and the GitHub Action — and
only one of them has a terminal. Anything that needs to be *said* rather than
returned comes back in `AnalysisResult.warnings`, so a caller with no console
still gets to hear it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from sentinel import blast_radius, explain as explain_module, git_reader, model
from sentinel.config import Settings, get_settings
from sentinel.features import Change, FileChange, vector
from sentinel.results import AnalysisResult, ModelInfo, Scope
from sentinel.risk_rules import ChangeRisk, calibrate, score_change
from sentinel.static_analysis import TestIndex, analyze_complexity

logger = logging.getLogger(__name__)

#: The scopes an analysis can cover.
MODES = ("scan", "diff", "all")


def run_analysis(
    repo_path: Path,
    *,
    mode: str = "scan",
    since: str | None = None,
    use_model: bool = True,
    settings: Settings | None = None,
) -> AnalysisResult:
    """Analyze a repository and return the risk plus what was analyzed.

    `mode` is one of:

    * ``diff`` — uncommitted work. History runs up to HEAD.
    * ``scan`` — commits since the base ref. History runs up to the *base*, so a
      change cannot improve its own files' track record and hide its own risk.
    * ``all``  — every tracked file, scored for inherent riskiness only.

    A trained model is used when one exists and its feature list still matches;
    otherwise the rules score the change. Falling back is normal operation.

    Never touches the network. The AI explanation is a separate, explicit step.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    settings = settings or get_settings()
    warnings: list[str] = []

    repo = git_reader.open_repo(repo_path)
    root = git_reader.repo_root(repo)

    base_ref: str | None = None
    base_commit: str | None = None

    if mode == "diff":
        changed = git_reader.files_in_working_tree(repo)
        author = git_reader.configured_author(repo)
        when = datetime.now()
        history_until = None
    elif mode == "all":
        changed = git_reader.all_tracked_files(repo)
        author = git_reader.configured_author(repo)
        when = datetime.now()
        history_until = None
    else:
        base_ref = since or git_reader.resolve_default_branch(repo)
        base_commit = git_reader.resolve_base(repo, base_ref)
        changed = git_reader.files_changed_since(repo, base_commit)
        author = git_reader.head_author(repo)
        when = repo.head.commit.authored_datetime
        history_until = base_commit

    # Never score our own output. `train` drops a .gitignore into the directory
    # so git normally hides it, but a model saved by an older version — or a
    # deleted .gitignore — would otherwise put a 1,500-line model file into the
    # change and let it dominate the score.
    artifacts = f"{settings.training.model_dir}/"
    changed = [c for c in changed if not c.path.startswith(artifacts)]
    changed_paths = {c.path for c in changed}

    scan = git_reader.collect_file_histories(
        repo,
        settings.bugfix,
        until=history_until,
        distribution=settings.distribution,
    )
    # Changed paths are unioned in deliberately: a test file added in this very
    # change is not tracked yet, and missing it would report "no tests" at the
    # exact moment somebody wrote them.
    tracked = git_reader.tracked_paths(repo)
    test_index = TestIndex(set(tracked) | changed_paths)

    files = tuple(
        FileChange(
            path=c.path,
            lines_added=c.lines_added,
            lines_deleted=c.lines_deleted,
            tests=test_index.signal_for(c.path, changed_paths),
            author=author,
            history=scan.histories.get(c.path),
            complexity=analyze_complexity(root / c.path),
        )
        for c in changed
    )
    change = Change(files=files, author=author, when=when)

    trained = model.load(root, settings.training) if use_model else None
    if trained is not None and not trained.is_compatible:
        message = (
            "the trained model was built from a different feature set and was "
            "ignored; re-run `sentinel train`"
        )
        logger.warning(message)
        warnings.append(message)
        trained = None

    if trained is not None and files:
        risk = score_with_model(change, trained, settings, whole_change=mode != "all")
        model_info = ModelInfo(
            trained_at=trained.trained_at,
            rows=trained.rows,
            positives=trained.positives,
            commits_considered=trained.commits_considered,
        )
    else:
        weights, relative = calibrate(
            settings.rules, scan.distribution, settings.distribution
        )
        risk = score_change(
            change,
            weights,
            include_change_rules=mode != "all",
            relative_thresholds=relative,
        )
        model_info = None

    # Impact is computed after scoring so the riskiest files are walked first —
    # which is what the per-file cap should keep, if it has to drop any.
    blast = blast_radius.compute(
        root,
        tracked,
        [f.path for f in risk.files] or sorted(changed_paths),
        settings.blast,
    )

    scope = Scope(
        mode=mode,
        repo=str(root),
        author=author,
        when=when,
        files_analyzed=len(files),
        base_ref=base_ref,
        base_commit=base_commit,
        commits_walked=scan.commits_walked,
        commits_skipped=scan.commits_skipped,
        model=model_info,
    )
    return AnalysisResult(
        risk=risk, scope=scope, blast=blast, warnings=tuple(warnings)
    )


def score_with_model(
    change: Change, trained, settings: Settings, *, whole_change: bool = True
) -> ChangeRisk:
    """Score the change as a whole, then each file on its own for the ranking.

    Per-file numbers come from asking the model about a one-file change, so a
    file's score answers "how risky would this edit be on its own" — which is
    the question the ranked table is for.

    `whole_change=False` is for `scan --all`, where the file set is the entire
    repository rather than a change. Feeding the model a 400-file "change" would
    ask it about something it has never seen; the answerable question there is
    which single file is riskiest, so the headline becomes the worst file.
    """
    per_file: list[model.FileProbability] = []
    single_vectors = [
        vector(Change(files=(f,), author=change.author, when=change.when))
        for f in change.files
    ]

    if len(change.files) > 1 or not whole_change:
        probabilities = model.predict(trained.booster, single_vectors)
        for changed_file, single, file_probability in zip(
            change.files, single_vectors, probabilities
        ):
            top = model.explain(trained.booster, single, top=1)
            per_file.append(
                model.FileProbability(
                    path=changed_file.path,
                    probability=file_probability,
                    lines_added=changed_file.lines_added,
                    lines_deleted=changed_file.lines_deleted,
                    top=top[0] if top else None,
                )
            )

    if whole_change:
        headline_vector = vector(change)
        probability = model.predict(trained.booster, [headline_vector])[0]
    else:
        worst = max(range(len(single_vectors)), key=lambda i: per_file[i].probability)
        headline_vector = single_vectors[worst]
        probability = per_file[worst].probability

    contributions = model.explain(
        trained.booster, headline_vector, top=settings.training.top_shap_reasons
    )
    return model.as_change_risk(probability, contributions, per_file, settings.rules)


def add_explanation(
    result: AnalysisResult, settings: Settings | None = None
) -> AnalysisResult:
    """Attach an AI narrative to a finished analysis.

    Separate from `run_analysis` on purpose: the scoring path never touches the
    network, and this function can only ever add prose to an already-frozen
    result.
    """
    settings = settings or get_settings()
    explanation = explain_module.explain(
        result.risk, result.scope, result.blast, settings
    )
    return result.with_explanation(explanation)
