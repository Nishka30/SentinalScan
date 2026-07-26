"""The JSON shape, and nothing else.

Split from `report` because these two have different audiences and different
stability requirements: terminal output can be reworded whenever it reads badly,
while this shape is a contract. Phase 5's GitHub Action and MCP server consume
it, so it is versioned and it imports no rendering code.
"""

from __future__ import annotations

from sentinel import __version__
from sentinel.blast_radius import BlastRadius
from sentinel.results import AnalysisResult, Explanation, Scope
from sentinel.risk_rules import ChangeRisk, FileRisk, Reason

#: Bumped whenever the shape changes in a way consumers must notice.
#: v2 — `scoring_method` values beyond "rules", plus `model` and
#:      `relative_thresholds` in `scope`.
#: v3 — top-level `blast_radius`.
#: v4 — top-level `ai_explanation`.
#: v5 — top-level `warnings`.
SCHEMA_VERSION = 5

#: Stated in the payload so no consumer can mistake prose for measurement.
AI_DISCLAIMER = (
    "Written by a language model from the facts in this payload. "
    "It explains the score; it does not compute or affect it."
)


def to_dict(result: AnalysisResult) -> dict:
    """The stable machine-readable form of a complete analysis."""
    return {
        "schema_version": SCHEMA_VERSION,
        "sentinel_version": __version__,
        "scoring_method": result.risk.scoring_method,
        "score": result.risk.score,
        "band": result.risk.band,
        "recommendation": result.risk.recommendation,
        "scope": _scope_dict(result.scope, result.risk),
        "reasons": [_reason_dict(r) for r in result.risk.reasons],
        "files": [_file_dict(f) for f in result.risk.files],
        "blast_radius": blast_dict(result.blast),
        "ai_explanation": explanation_dict(result.explanation),
        # Surfaced in the payload so a CI job or MCP client learns about a stale
        # model too, not only somebody watching a terminal.
        "warnings": list(result.warnings),
    }


def _scope_dict(scope: Scope, risk: ChangeRisk) -> dict:
    return {
        "mode": scope.mode,
        "description": scope.description,
        "repo": scope.repo,
        "author": scope.author,
        "analyzed_at": scope.when.isoformat(),
        "base_ref": scope.base_ref,
        "base_commit": scope.base_commit,
        "files_analyzed": scope.files_analyzed,
        "commits_walked": scope.commits_walked,
        "commits_skipped": scope.commits_skipped,
        "relative_thresholds": risk.relative_thresholds,
        "model": None
        if scope.model is None
        else {
            "trained_at": scope.model.trained_at,
            "rows": scope.model.rows,
            "positives": scope.model.positives,
            "commits_considered": scope.model.commits_considered,
        },
    }


def _reason_dict(reason: Reason) -> dict:
    return {
        "rule": reason.rule,
        "label": reason.label,
        "points": reason.points,
        "detail": reason.detail,
        "path": reason.path,
    }


def _file_dict(file_risk: FileRisk) -> dict:
    return {
        "path": file_risk.path,
        "score": file_risk.score,
        "band": file_risk.band,
        "lines_added": file_risk.lines_added,
        "lines_deleted": file_risk.lines_deleted,
        "reasons": [_reason_dict(r) for r in file_risk.reasons],
    }


def blast_dict(blast: BlastRadius | None) -> dict:
    """Impact as structured data.

    `analyzed: false` is distinct from zero dependents: "we could not look" and
    "nothing depends on this" mean very different things to a reviewer, and a
    consumer must be able to tell them apart.
    """
    if blast is None or not blast.analyzed:
        return {"analyzed": False, "direct_count": 0, "transitive_count": 0}

    return {
        "analyzed": True,
        "direct_count": blast.direct_count,
        "transitive_count": blast.transitive_count,
        "direct": list(blast.direct),
        "transitive": list(blast.transitive),
        "hubs": list(blast.hubs),
        "cycle_files": list(blast.cycle_files),
        "max_depth": blast.depth,
        "depth_capped": blast.depth_capped,
        "listed_limit": blast.listed_limit,
        "files_omitted": blast.files_omitted,
        "graph": {
            "files": blast.graph_files,
            "edges": blast.graph_edges,
            "parsers": list(blast.parsers),
        },
        "per_file": [
            {
                "path": impact.path,
                "direct_count": impact.direct_count,
                "transitive_count": impact.transitive_count,
                "direct": list(impact.direct),
                "transitive": list(impact.transitive),
                "in_cycle": impact.in_cycle,
                "depth_capped": impact.depth_capped,
                "is_hub": impact.is_hub,
            }
            for impact in blast.files
        ],
    }


def explanation_dict(explanation: Explanation | None) -> dict | None:
    """The AI narrative, or None when it was never requested.

    Three distinguishable states, because a consumer needs to tell them apart:
    `null` (not asked for), `available: false` with a reason (asked for, could
    not be produced), and `available: true` with prose.
    """
    if explanation is None:
        return None
    if not explanation.available:
        return {
            "available": False,
            "skipped_reason": explanation.skipped_reason,
            "generated_by": None,
            "disclaimer": AI_DISCLAIMER,
        }
    return {
        "available": True,
        "generated_by": explanation.generated_by,
        "disclaimer": AI_DISCLAIMER,
        "summary": explanation.summary,
        "rollout": explanation.rollout,
        "rollback_trigger": explanation.rollback_trigger,
        "monitoring": explanation.monitoring,
    }
