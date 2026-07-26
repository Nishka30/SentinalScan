"""The result types every consumer shares.

Extracted so `report` (terminal rendering) and `serialization` (the JSON shape)
can each depend on the data without depending on each other — and so Phase 5's
GitHub Action and MCP server can import the shape without importing rich.

Nothing here computes anything. That is the point: an `AnalysisResult` is a
finished, frozen answer, which is what makes it safe to hand to the AI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sentinel.blast_radius import BlastRadius
from sentinel.risk_rules import ChangeRisk


@dataclass(frozen=True)
class ModelInfo:
    """Which trained model produced a score, so a stale one can be spotted."""

    trained_at: str
    rows: int
    positives: int
    commits_considered: int


@dataclass(frozen=True)
class Scope:
    """What was analyzed, so a score is never reported without its context."""

    mode: str  # "scan" | "diff" | "all"
    repo: str
    author: str
    when: datetime
    files_analyzed: int
    base_ref: str | None = None
    base_commit: str | None = None
    commits_walked: int = 0
    commits_skipped: int = 0
    model: ModelInfo | None = None

    @property
    def description(self) -> str:
        """One line naming exactly what the score covers."""
        if self.mode == "diff":
            return "uncommitted changes in the working tree"
        if self.mode == "all":
            return "every tracked file (no change measured)"
        base = self.base_ref or "base"
        return f"changes since {base}"


@dataclass(frozen=True)
class Explanation:
    """Prose written by a language model about an already-finished analysis.

    Deliberately holds no score, no band and no reasons — only narrative. The
    numbers live in `ChangeRisk`, which is frozen and was computed before this
    object existed, so there is no path by which prose can alter a score.

    `available=False` carries the reason it was skipped, because "the AI was
    unavailable" and "the AI had nothing to say" are different facts.
    """

    available: bool
    generated_by: str = ""
    summary: str = ""
    rollout: str = ""
    rollback_trigger: str = ""
    monitoring: str = ""
    skipped_reason: str = ""


@dataclass(frozen=True)
class AnalysisResult:
    """A complete analysis: the score, what was analyzed, and what it can break."""

    risk: ChangeRisk
    scope: Scope
    blast: BlastRadius
    explanation: Explanation | None = None
    #: Things the caller should be told but that are not part of the result —
    #: a stale model, for instance. Carried rather than printed, because the
    #: orchestration layer has no console and two of its three callers are not
    #: attached to a terminal.
    warnings: tuple[str, ...] = ()

    def with_explanation(self, explanation: Explanation) -> AnalysisResult:
        """Attach prose without touching anything else.

        Returns a new result rather than mutating: the analysis that was scored
        is the analysis that gets reported, whatever the AI says about it.
        """
        return AnalysisResult(
            risk=self.risk,
            scope=self.scope,
            blast=self.blast,
            explanation=explanation,
            warnings=self.warnings,
        )
