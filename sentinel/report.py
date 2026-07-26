"""Terminal output.

All rich rendering lives here so the analysis modules stay pure functions
returning data. The machine-readable form lives in `serialization`; these two
are separate because terminal wording can be improved whenever it reads badly,
while the JSON shape is a contract other tools depend on.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sentinel.blast_radius import BlastRadius
from sentinel.evaluation import EvaluationResult
from sentinel.history_mining import MiningResult
from sentinel.results import AnalysisResult, Explanation, Scope
from sentinel.risk_rules import ChangeRisk, FileRisk

BAND_COLOURS = {"low": "green", "medium": "yellow", "high": "red"}
BAND_LABELS = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}


def render(result: AnalysisResult, console: Console, *, top_files: int = 10) -> None:
    """Print the human-facing report."""
    risk, scope = result.risk, result.scope

    console.print()
    console.print(_headline_panel(risk, scope))

    if risk.reasons:
        console.print()
        console.print(_reasons_table(risk))
    else:
        console.print()
        console.print("[green]No risk rules fired.[/]")

    if result.blast is not None and result.blast.analyzed:
        console.print()
        _render_impact(result.blast, console)

    console.print()
    console.print(Text("Recommendation", style="bold"))
    console.print(f"  {risk.recommendation}")

    ranked = [f for f in risk.files if f.score > 0]
    if len(ranked) > 1:
        console.print()
        console.print(_files_table(ranked[:top_files], total=len(ranked)))

    if result.explanation is not None:
        console.print()
        render_explanation(result.explanation, console)

    if scope.commits_skipped:
        console.print()
        console.print(
            f"[dim]note: {scope.commits_skipped} commit(s) could not be read "
            f"and were excluded from history.[/]"
        )
    console.print()


def render_explanation(explanation: Explanation, console: Console) -> None:
    """Print the AI narrative, labelled as such.

    The heading names the model on purpose. A reader must never be in doubt
    about which part of this report is measurement and which is prose.
    """
    if not explanation.available:
        console.print(
            f"[dim]AI explanation skipped: {explanation.skipped_reason}. "
            f"The score and reasons above are unaffected.[/]"
        )
        return

    body = Text()
    for heading, content in (
        ("Why this is risky", explanation.summary),
        ("Suggested rollout", explanation.rollout),
        ("Roll back if", explanation.rollback_trigger),
        ("Watch after deploy", explanation.monitoring),
    ):
        if not content:
            continue
        if len(body):
            body.append("\n\n")
        body.append(f"{heading}\n", style="bold")
        body.append(content)

    console.print(
        Panel(
            body,
            title=f"AI explanation — {explanation.generated_by}",
            subtitle="[dim]generated prose; the score above is computed, not written[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _render_impact(blast: BlastRadius, console: Console) -> None:
    """The blast radius: who else is affected, direct before transitive."""
    console.print(Text("Impact", style="bold"))

    if blast.total == 0:
        console.print(
            f"  [green]Nothing else in the repository imports these files.[/] "
            f"[dim](graph: {blast.graph_files} files, {blast.graph_edges} edges)[/]"
        )
        return

    console.print(
        f"  [red]{blast.direct_count}[/] file(s) import these directly, "
        f"[yellow]{blast.transitive_count}[/] more within {blast.depth} hops."
    )

    if blast.direct:
        console.print(f"  [bold]direct:[/] {_join(blast.direct, blast.direct_count)}")
    if blast.transitive:
        console.print(
            f"  [dim]transitive:[/] {_join(blast.transitive, blast.transitive_count)}"
        )

    for hub in blast.hubs:
        console.print(
            f"  [yellow]hub:[/] {hub} is imported by many files — "
            "the listing is a sample, not the whole set."
        )
    for path in blast.cycle_files:
        console.print(
            f"  [yellow]cycle:[/] {path} takes part in a circular import, so some "
            "of its dependents are also its dependencies."
        )
    if blast.depth_capped:
        console.print(
            f"  [dim]note: the walk stopped at {blast.depth} hops; more files "
            f"depend on this further out.[/]"
        )
    if blast.files_omitted:
        console.print(
            f"  [dim]note: impact computed for the {len(blast.files)} riskiest "
            f"files; {blast.files_omitted} more were not walked.[/]"
        )


def _join(listed: tuple[str, ...], total: int) -> str:
    """Show the sample and admit when it is one."""
    shown = ", ".join(listed)
    if total > len(listed):
        return f"{shown} [dim]… and {total - len(listed)} more[/]"
    return shown


def render_mining(result: MiningResult, console: Console) -> None:
    """Report what SZZ found, including the class balance."""
    table = Table(
        title="Labels mined from history",
        title_justify="left",
        show_edge=False,
        header_style="bold dim",
    )
    table.add_column("measure")
    table.add_column("value", justify="right")

    table.add_row("commits considered", f"{result.commits_considered:,}")
    table.add_row("bug-fixing commits (keyword match)", f"{result.bugfix_commits:,}")
    table.add_row("blames run", f"{result.blames_run:,}")
    table.add_row("training rows", f"{len(result.rows):,}")
    table.add_row("bug-inducing (label 1)", f"{result.positives:,}")
    table.add_row("clean (label 0)", f"{len(result.rows) - result.positives:,}")
    table.add_row("class balance", f"{result.positive_rate:.1%}")

    console.print()
    console.print(table)
    if result.truncated:
        console.print(
            f"[yellow]note:[/] history was capped at {result.commits_considered:,} commits. "
            "Raise --max-commits to mine further back."
        )


def render_evaluation(result: EvaluationResult, console: Console) -> None:
    """Print the time-split metrics beside the baseline."""
    table = Table(
        title="Time-split evaluation (train on older, test on newer)",
        title_justify="left",
        show_edge=False,
        header_style="bold dim",
    )
    table.add_column("metric")
    table.add_column("model", justify="right")
    table.add_column("lines-changed baseline", justify="right")
    table.add_column("verdict", justify="left")

    for name, model_value, baseline_value in (
        ("ROC-AUC", result.model.roc_auc, result.baseline.roc_auc),
        ("PR-AUC", result.model.pr_auc, result.baseline.pr_auc),
    ):
        better = model_value > baseline_value
        table.add_row(
            name,
            Text(_fmt(model_value), style="bold cyan"),
            _fmt(baseline_value),
            Text("model wins" if better else "baseline wins", style="green" if better else "red"),
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[dim]train: {result.train_rows:,} commits ({result.train_positives:,} bug-inducing)   "
        f"test: {result.test_rows:,} commits ({result.test_positives:,} bug-inducing)[/]"
    )
    if result.split_at is not None:
        console.print(f"[dim]split at: {result.split_at.date()}[/]")
    console.print(
        f"[dim]base rate in test: {result.base_rate:.1%} — this is what a random "
        f"ranker scores on PR-AUC, and 0.500 is random on ROC-AUC.[/]"
    )


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.3f}"  # NaN != NaN


def _headline_panel(risk: ChangeRisk, scope: Scope) -> Panel:
    colour = BAND_COLOURS[risk.band]
    body = Text()
    body.append(f"{risk.score}", style=f"bold {colour}")
    body.append("/100   ", style="dim")
    body.append(BAND_LABELS[risk.band], style=f"bold {colour}")
    body.append("\n\n")
    body.append(f"{scope.files_analyzed} file(s) — {scope.description}\n", style="dim")
    body.append(f"author: {scope.author}", style="dim")
    if scope.commits_walked:
        body.append(f"   history: {scope.commits_walked} commits", style="dim")

    body.append("\n")
    if risk.scoring_method == "model":
        detail = (
            f"scored by: trained model ({scope.model.rows} commits)"
            if scope.model
            else "scored by: trained model"
        )
        body.append(detail, style="cyan")
    else:
        thresholds = "repo percentiles" if risk.relative_thresholds else "absolute thresholds"
        body.append(f"scored by: rules ({thresholds})", style="dim")

    return Panel(
        body,
        title="Sentinel — deployment risk",
        border_style=colour,
        expand=False,
        padding=(1, 3),
    )


def _reasons_table(risk: ChangeRisk) -> Table:
    model_scored = risk.scoring_method == "model"
    table = Table(
        title="Why", title_justify="left", show_edge=False, header_style="bold dim"
    )
    # The model's numbers are signed SHAP impacts, not additive points, so the
    # column is named for what it actually holds.
    table.add_column("impact" if model_scored else "pts", justify="right", style="bold", width=6)
    table.add_column("reason")
    table.add_column("evidence", style="dim")

    for reason in risk.reasons:
        label = reason.label
        if reason.path:
            label = f"{label}\n[dim]{reason.path}[/]"
        style = "red" if reason.points > 0 else "green"
        table.add_row(Text(f"{reason.points:+d}", style=style), label, reason.detail)
    return table


def _files_table(files: list[FileRisk], *, total: int) -> Table:
    shown = len(files)
    title = "Riskiest files" if shown >= total else f"Riskiest files ({shown} of {total})"
    table = Table(title=title, title_justify="left", show_edge=False, header_style="bold dim")
    table.add_column("score", justify="right", width=5)
    table.add_column("file")
    table.add_column("lines", justify="right", style="dim", width=9)
    table.add_column("top reason", style="dim")

    for file_risk in files:
        colour = BAND_COLOURS[file_risk.band]
        top = file_risk.reasons[0].label if file_risk.reasons else ""
        table.add_row(
            Text(str(file_risk.score), style=colour),
            file_risk.path,
            f"+{file_risk.lines_added}/-{file_risk.lines_deleted}",
            top,
        )
    return table
