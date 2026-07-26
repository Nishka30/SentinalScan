"""Command-line entry point.

Nothing but argument parsing and presentation. The analysis itself lives in
`analysis`, which knows nothing about typer or rich, because the MCP server and
the GitHub Action need the same orchestration without a terminal attached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from sentinel import (
    __version__,
    evaluation,
    git_reader,
    history_mining,
    model,
    report,
    serialization,
)
from sentinel.analysis import add_explanation, run_analysis
from sentinel.config import get_settings
from sentinel.git_reader import RepositoryError
from sentinel.model import ModelError
from sentinel.results import AnalysisResult

app = typer.Typer(
    name="sentinel",
    help="Answer one question: is this change safe to deploy?",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

RepoOption = Annotated[
    Path,
    typer.Option("--repo", "-r", help="Path to the git repository to analyze."),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Print the result as JSON instead of a report."),
]
MaxCommitsOption = Annotated[
    int | None,
    typer.Option("--max-commits", help="How far back to mine history. Higher is slower."),
]
ExplainOption = Annotated[
    bool,
    typer.Option(
        "--explain",
        help="Also ask the configured LLM to explain the result (needs NVIDIA_API_KEY).",
    ),
]



def _emit(result: AnalysisResult, *, as_json: bool) -> None:
    # Warnings go to stderr even in JSON mode, so stdout stays pipeable while
    # the human still hears about a stale model.
    for warning in result.warnings:
        err_console.print(f"[yellow]warning:[/] {warning}")

    if as_json:
        typer.echo(json.dumps(serialization.to_dict(result), indent=2))
        return
    report.render(result, console, top_files=get_settings().report_top_files)


def _run(
    mode: str, repo: Path, since: str | None, as_json: bool, with_ai: bool = False
) -> None:
    """Shared command body: analyze, or fail with a message and a real exit code."""
    try:
        result = run_analysis(repo, mode=mode, since=since)
    except RepositoryError as exc:
        err_console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=2) from exc

    if result.scope.files_analyzed == 0:
        if as_json:
            _emit(result, as_json=True)
        else:
            console.print(
                f"[green]Nothing to score[/] — no files in scope "
                f"({result.scope.description})."
            )
            if mode == "scan":
                console.print("[dim]On the default branch already? Try --since HEAD~10.[/]")
        return

    if with_ai:
        result = add_explanation(result)

    _emit(result, as_json=as_json)


def _mine(repo_path: Path, max_commits: int | None) -> tuple[history_mining.MiningResult, Path]:
    """Run the SZZ pipeline with a progress display attached."""
    settings = get_settings()
    repo = git_reader.open_repo(repo_path)
    root = git_reader.repo_root(repo)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        blame_task = progress.add_task("blaming bug fixes (SZZ)", total=None)
        feature_task = progress.add_task("building features", total=None)

        def on_blame(done: int, total: int, sha: str) -> None:
            progress.update(blame_task, completed=done, total=total)

        def on_features(done: int, total: int, sha: str) -> None:
            progress.update(feature_task, completed=done, total=total)

        result = history_mining.mine(
            repo,
            settings.bugfix,
            settings.training,
            max_commits=max_commits,
            on_blame=on_blame,
            on_features=on_features,
        )
    return result, root


@app.command()
def scan(
    repo: RepoOption = Path("."),
    since: Annotated[
        str | None,
        typer.Option("--since", help="Score changes since this ref (default: the default branch)."),
    ] = None,
    all_files: Annotated[
        bool,
        typer.Option("--all", help="Score every tracked file instead of a change."),
    ] = False,
    as_json: JsonOption = False,
    with_ai: ExplainOption = False,
) -> None:
    """Score the deployment risk of committed changes."""
    if all_files and since:
        err_console.print("[red]error:[/] --all and --since cannot be combined")
        raise typer.Exit(code=2)
    _run("all" if all_files else "scan", repo, since, as_json, with_ai)


@app.command()
def diff(
    repo: RepoOption = Path("."),
    as_json: JsonOption = False,
    with_ai: ExplainOption = False,
) -> None:
    """Score the deployment risk of the uncommitted changes in the working tree."""
    _run("diff", repo, None, as_json, with_ai)


@app.command()
def train(repo: RepoOption = Path("."), max_commits: MaxCommitsOption = None) -> None:
    """Mine the repository's bug history and train the risk model."""
    try:
        result, root = _mine(repo, max_commits)
        if not result.rows:
            err_console.print("[red]error:[/] no commits with changed files were found")
            raise typer.Exit(code=2)

        report.render_mining(result, console)

        booster, rounds = model.train_with_early_stopping(
            [r.features for r in result.rows],
            [r.label for r in result.rows],
            get_settings().training,
        )
        path = model.save(
            booster,
            root,
            get_settings().training,
            rows=len(result.rows),
            positives=result.positives,
            commits_considered=result.commits_considered,
        )
    except RepositoryError as exc:
        err_console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    except ModelError as exc:
        err_console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=3) from exc

    console.print()
    console.print(f"[green]Model saved[/] to {path} ({rounds} boosting rounds)")
    console.print("[dim]`sentinel scan` will now score with the model. Run `sentinel evaluate` "
                  "to see whether it beats the baseline.[/]")


@app.command()
def evaluate(repo: RepoOption = Path("."), max_commits: MaxCommitsOption = None) -> None:
    """Mine history, then measure the model on a time-based split."""
    try:
        result, _root = _mine(repo, max_commits)
        report.render_mining(result, console)
        outcome = evaluation.evaluate(list(result.rows), get_settings().training)
    except RepositoryError as exc:
        err_console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    except ModelError as exc:
        err_console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=3) from exc

    report.render_evaluation(outcome, console)


@app.command()
def explain(
    repo: RepoOption = Path("."),
    since: Annotated[
        str | None,
        typer.Option("--since", help="Explain changes since this ref instead of the diff."),
    ] = None,
    uncommitted: Annotated[
        bool,
        typer.Option("--diff", help="Explain uncommitted work instead of committed changes."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Score the change, then explain it in plain English with the configured LLM."""
    mode = "diff" if uncommitted else "scan"
    try:
        result = run_analysis(repo, mode=mode, since=since)
    except RepositoryError as exc:
        err_console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=2) from exc

    if result.scope.files_analyzed == 0:
        console.print(
            f"[green]Nothing to explain[/] — no files in scope "
            f"({result.scope.description})."
        )
        return

    with console.status("asking the model to explain the score…"):
        result = add_explanation(result)

    _emit(result, as_json=as_json)


@app.command()
def version() -> None:
    """Print the Sentinel version."""
    console.print(f"sentinel {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
