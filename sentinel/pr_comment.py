"""Turn a `--json` payload into a pull-request comment.

Lives in the package rather than in the action's shell script so it can be
imported and tested. The payload is the only source of truth: nothing here
recomputes a score, re-ranks a reason or re-derives a band, because two
implementations of the same arithmetic eventually disagree and the one in CI is
the one nobody checks.

Run as `python -m sentinel.pr_comment result.json --threshold 65`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BAND_BADGES = {
    "low": "🟢 LOW",
    "medium": "🟡 MEDIUM",
    "high": "🔴 HIGH",
}

#: Lets the action find and update its own previous comment instead of adding a
#: new one to every push.
MARKER = "<!-- sentinel-deployment-risk -->"


def format_comment(
    payload: dict[str, Any], *, threshold: int, max_reasons: int = 5, max_files: int = 5
) -> str:
    """Render the markdown comment.

    Deliberately compact: a PR comment competes for attention with the diff
    itself, so it leads with the number and the reasons and pushes everything
    else into a collapsed section.
    """
    score = payload.get("score", 0)
    band = str(payload.get("band", "low"))
    badge = BAND_BADGES.get(band, band.upper())
    method = payload.get("scoring_method", "rules")
    scope = payload.get("scope", {}) or {}

    lines = [
        MARKER,
        f"## {badge} — deployment risk {score}/100",
        "",
        f"**{payload.get('recommendation', '')}**",
        "",
        f"Scored by {_describe_method(method, scope)} over "
        f"{scope.get('files_analyzed', 0)} changed file(s).",
    ]

    if score >= threshold:
        lines += ["", f"> Above the configured threshold of {threshold}."]

    reasons = payload.get("reasons") or []
    if reasons:
        lines += ["", "### Why", "", _reason_table(reasons[:max_reasons], method)]
        if len(reasons) > max_reasons:
            lines.append(f"\n_{len(reasons) - max_reasons} further factor(s) not shown._")

    lines += _impact_section(payload.get("blast_radius") or {})
    lines += _files_section(payload.get("files") or [], max_files)
    lines += _ai_section(payload.get("ai_explanation"))

    for warning in payload.get("warnings") or []:
        lines += ["", f"> ⚠️ {warning}"]

    lines += [
        "",
        "<sub>Risk is learned from this repository's own bug history. "
        f"Sentinel {payload.get('sentinel_version', '')}, "
        f"schema v{payload.get('schema_version', '')}.</sub>",
    ]
    return "\n".join(lines)


def _describe_method(method: str, scope: dict[str, Any]) -> str:
    if method == "model":
        model = scope.get("model") or {}
        rows = model.get("rows")
        return (
            f"a model trained on {rows:,} of this repository's commits"
            if rows
            else "a trained model"
        )
    thresholds = (
        "this repository's own percentiles"
        if scope.get("relative_thresholds")
        else "absolute thresholds"
    )
    return f"the rule engine ({thresholds})"


def _reason_table(reasons: list[dict[str, Any]], method: str) -> str:
    header = "SHAP impact" if method == "model" else "Points"
    rows = [f"| {header} | Factor | Evidence |", "| --- | --- | --- |"]
    for reason in reasons:
        where = f"`{reason['path']}`<br>" if reason.get("path") else ""
        rows.append(
            f"| {reason.get('points', 0):+d} | {where}{reason.get('label', '')} "
            f"| {reason.get('detail', '')} |"
        )
    return "\n".join(rows)


def _impact_section(blast: dict[str, Any]) -> list[str]:
    if not blast.get("analyzed"):
        return []

    direct = blast.get("direct_count", 0)
    transitive = blast.get("transitive_count", 0)
    if direct == 0 and transitive == 0:
        return ["", "### Impact", "", "Nothing else in the repository imports these files."]

    lines = [
        "",
        "### Impact",
        "",
        f"**{direct}** file(s) import these directly, **{transitive}** more within "
        f"{blast.get('max_depth', 0)} hops.",
    ]
    if blast.get("direct"):
        lines += ["", "Direct: " + ", ".join(f"`{p}`" for p in blast["direct"])]
    for hub in blast.get("hubs") or []:
        lines.append(f"\n⚠️ `{hub}` is a hub — the listing above is a sample.")
    for path in blast.get("cycle_files") or []:
        lines.append(f"\n⚠️ `{path}` is part of a circular import.")
    return lines


def _files_section(files: list[dict[str, Any]], limit: int) -> list[str]:
    ranked = [f for f in files if f.get("score", 0) > 0]
    if len(ranked) < 2:
        return []

    lines = [
        "",
        "<details><summary>Riskiest files</summary>",
        "",
        "| Score | File | Lines |",
        "| --- | --- | --- |",
    ]
    for entry in ranked[:limit]:
        lines.append(
            f"| {entry.get('score', 0)} | `{entry.get('path', '')}` "
            f"| +{entry.get('lines_added', 0)}/-{entry.get('lines_deleted', 0)} |"
        )
    if len(ranked) > limit:
        lines.append(f"\n_{len(ranked) - limit} more not shown._")
    lines += ["", "</details>"]
    return lines


def _ai_section(explanation: dict[str, Any] | None) -> list[str]:
    """The narrative, labelled as generated and folded away.

    Collapsed on purpose: the measured reasons are the finding, and prose should
    not be the first thing a reviewer reads.
    """
    if not explanation or not explanation.get("available"):
        return []

    lines = [
        "",
        f"<details><summary>AI explanation "
        f"(generated by {explanation.get('generated_by', 'a language model')})</summary>",
        "",
        f"_{explanation.get('disclaimer', '')}_",
    ]
    for heading, key in (
        ("Why this is risky", "summary"),
        ("Suggested rollout", "rollout"),
        ("Roll back if", "rollback_trigger"),
        ("Watch after deploy", "monitoring"),
    ):
        content = explanation.get(key)
        if content:
            lines += ["", f"**{heading}** — {content}"]
    lines += ["", "</details>"]
    return lines


def exceeds_threshold(payload: dict[str, Any], threshold: int) -> bool:
    """True when the score is at or above the gate."""
    return int(payload.get("score", 0)) >= threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="Path to a `sentinel --json` file.")
    parser.add_argument("--threshold", type=int, default=65)
    parser.add_argument(
        "--fail-over-threshold",
        action="store_true",
        help="Exit 1 when the score reaches the threshold, to gate a merge.",
    )
    parser.add_argument("--output", type=Path, help="Write the comment here as well.")
    args = parser.parse_args(argv)

    # utf-8-sig, not utf-8: PowerShell's `>` redirection writes a BOM, and a
    # comment step that dies on an invisible character is a poor way to learn
    # which shell produced the file.
    payload = json.loads(args.payload.read_text(encoding="utf-8-sig"))
    comment = format_comment(payload, threshold=args.threshold)

    print(comment)
    if args.output:
        args.output.write_text(comment, encoding="utf-8")

    over = exceeds_threshold(payload, args.threshold)
    if over and args.fail_over_threshold:
        print(
            f"\nsentinel: score {payload.get('score')} reached the threshold "
            f"of {args.threshold}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
