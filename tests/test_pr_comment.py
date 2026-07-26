"""Tests for the PR comment the GitHub Action posts, and the merge gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel.pr_comment import (
    MARKER,
    exceeds_threshold,
    format_comment,
    main,
)


def payload(**overrides) -> dict:
    base = {
        "schema_version": 5,
        "sentinel_version": "0.1.0",
        "scoring_method": "model",
        "score": 77,
        "band": "high",
        "recommendation": "Hold this, or ship it behind a flag with a rollback ready.",
        "scope": {
            "mode": "scan",
            "description": "changes since origin/main",
            "files_analyzed": 3,
            "relative_thresholds": True,
            "model": {"rows": 4854, "positives": 777},
        },
        "reasons": [
            {
                "rule": "hot_file",
                "label": "File has a long history of bug fixes",
                "points": 35,
                "detail": "97 past bug-fix commits (threshold 20)",
                "path": "src/requests/utils.py",
            },
            {
                "rule": "model:max_complexity",
                "label": "Complexity of the most complex file",
                "points": -39,
                "detail": "value 19; log-odds -0.386",
                "path": None,
            },
        ],
        "files": [
            {"path": "src/requests/utils.py", "score": 72, "band": "high",
             "lines_added": 7, "lines_deleted": 0, "reasons": []},
            {"path": "src/requests/api.py", "score": 30, "band": "low",
             "lines_added": 2, "lines_deleted": 1, "reasons": []},
        ],
        "blast_radius": {
            "analyzed": True,
            "direct_count": 5,
            "transitive_count": 8,
            "direct": ["src/requests/adapters.py", "src/requests/auth.py"],
            "transitive": ["src/requests/api.py"],
            "hubs": [],
            "cycle_files": ["src/requests/utils.py"],
            "max_depth": 3,
            "depth_capped": False,
            "listed_limit": 12,
            "files_omitted": 0,
            "graph": {"files": 37, "edges": 108, "parsers": ["python"]},
            "per_file": [],
        },
        "ai_explanation": None,
        "warnings": [],
    }
    base.update(overrides)
    return base


# --- the essentials -------------------------------------------------------


def test_the_comment_leads_with_the_score_and_band() -> None:
    comment = format_comment(payload(), threshold=65)

    assert comment.startswith(MARKER)  # so the action can find and update it
    assert "🔴 HIGH — deployment risk 77/100" in comment
    assert "Hold this, or ship it behind a flag" in comment


def test_the_marker_lets_the_action_update_its_own_comment() -> None:
    """Without this the action adds a new comment on every push."""
    assert MARKER in format_comment(payload(), threshold=65)
    assert MARKER.startswith("<!--")  # invisible in rendered markdown


def test_reasons_appear_with_their_real_evidence() -> None:
    comment = format_comment(payload(), threshold=65)

    assert "97 past bug-fix commits (threshold 20)" in comment
    assert "`src/requests/utils.py`" in comment
    assert "| +35 |" in comment


def test_a_negative_shap_contribution_keeps_its_sign() -> None:
    """A factor that lowered the risk must not read as though it raised it."""
    comment = format_comment(payload(), threshold=65)
    assert "| -39 |" in comment


def test_the_column_is_named_for_what_it_holds() -> None:
    model_comment = format_comment(payload(scoring_method="model"), threshold=65)
    assert "SHAP impact" in model_comment

    rules_comment = format_comment(payload(scoring_method="rules"), threshold=65)
    assert "| Points |" in rules_comment


def test_the_scoring_method_is_stated() -> None:
    model_comment = format_comment(payload(), threshold=65)
    assert "a model trained on 4,854 of this repository's commits" in model_comment

    rules_comment = format_comment(payload(scoring_method="rules"), threshold=65)
    assert "the rule engine (this repository's own percentiles)" in rules_comment


def test_the_threshold_is_called_out_when_crossed() -> None:
    over = format_comment(payload(score=80), threshold=65)
    assert "Above the configured threshold of 65" in over

    under = format_comment(payload(score=20, band="low"), threshold=65)
    assert "Above the configured threshold" not in under
    assert "🟢 LOW" in under


# --- impact ---------------------------------------------------------------


def test_impact_reports_direct_and_transitive_separately() -> None:
    comment = format_comment(payload(), threshold=65)

    assert "**5** file(s) import these directly, **8** more within 3 hops" in comment
    assert "`src/requests/adapters.py`" in comment
    assert "part of a circular import" in comment


def test_a_hub_is_flagged() -> None:
    blast = payload()["blast_radius"] | {"hubs": ["src/requests/compat.py"]}
    comment = format_comment(payload(blast_radius=blast), threshold=65)
    assert "is a hub — the listing above is a sample" in comment


def test_no_dependents_is_stated_plainly() -> None:
    blast = payload()["blast_radius"] | {
        "direct_count": 0, "transitive_count": 0, "direct": [], "transitive": [],
        "cycle_files": [],
    }
    comment = format_comment(payload(blast_radius=blast), threshold=65)
    assert "Nothing else in the repository imports these files" in comment


def test_an_unanalyzed_graph_is_omitted_rather_than_reported_as_zero() -> None:
    comment = format_comment(
        payload(blast_radius={"analyzed": False, "direct_count": 0, "transitive_count": 0}),
        threshold=65,
    )
    assert "### Impact" not in comment


# --- the AI section -------------------------------------------------------


def test_the_narrative_is_absent_when_it_was_not_generated() -> None:
    assert "AI explanation" not in format_comment(payload(), threshold=65)


def test_the_narrative_is_folded_away_and_labelled() -> None:
    """The measured reasons are the finding; prose should not lead."""
    comment = format_comment(
        payload(
            ai_explanation={
                "available": True,
                "generated_by": "meta/llama-3.3-70b-instruct",
                "disclaimer": "Written by a language model. It does not compute the score.",
                "summary": "This file has a poor track record.",
                "rollout": "Ship behind a flag.",
                "rollback_trigger": "Error rate above baseline.",
                "monitoring": "Watch the importers.",
            }
        ),
        threshold=65,
    )

    assert "<details><summary>AI explanation (generated by meta/llama-3.3-70b-instruct)" in comment
    assert "does not compute the score" in comment
    assert "poor track record" in comment
    # The narrative comes after the measured reasons.
    assert comment.index("### Why") < comment.index("AI explanation")


def test_a_skipped_narrative_adds_nothing() -> None:
    comment = format_comment(
        payload(ai_explanation={"available": False, "skipped_reason": "no key"}),
        threshold=65,
    )
    assert "AI explanation" not in comment


# --- warnings and truncation ---------------------------------------------


def test_warnings_are_surfaced(tmp_path: Path) -> None:
    comment = format_comment(
        payload(warnings=["the trained model was built from a different feature set"]),
        threshold=65,
    )
    assert "⚠️ the trained model was built from a different feature set" in comment


def test_long_lists_are_truncated_with_an_honest_note() -> None:
    reasons = [
        {"rule": f"r{i}", "label": f"Factor {i}", "points": 10, "detail": "d", "path": None}
        for i in range(9)
    ]
    comment = format_comment(payload(reasons=reasons), threshold=65, max_reasons=3)
    assert "6 further factor(s) not shown" in comment


def test_a_single_scoring_file_does_not_get_a_table() -> None:
    """A "riskiest files" table of one file is noise."""
    one = [payload()["files"][0]]
    assert "Riskiest files" not in format_comment(payload(files=one), threshold=65)


def test_a_minimal_payload_does_not_crash() -> None:
    """CI must never fail because a key was absent from the JSON."""
    comment = format_comment({"score": 0, "band": "low"}, threshold=65)
    assert "0/100" in comment


# --- the merge gate -------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [(64, 65, False), (65, 65, True), (66, 65, True), (0, 0, True), (10, 100, False)],
)
def test_the_threshold_boundary(score: int, threshold: int, expected: bool) -> None:
    assert exceeds_threshold(payload(score=score), threshold) is expected


def test_the_cli_exits_zero_below_the_threshold(tmp_path: Path, capsys) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload(score=30)), encoding="utf-8")

    code = main([str(path), "--threshold", "65", "--fail-over-threshold"])

    assert code == 0
    assert "deployment risk 30/100" in capsys.readouterr().out


def test_the_cli_exits_one_above_the_threshold_when_gating(tmp_path: Path, capsys) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload(score=90)), encoding="utf-8")

    code = main([str(path), "--threshold", "65", "--fail-over-threshold"])

    assert code == 1
    assert "reached the threshold" in capsys.readouterr().err


def test_without_the_gate_flag_a_high_score_still_exits_zero(tmp_path: Path) -> None:
    """Default is to inform, not to block — otherwise nobody adopts it."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload(score=99)), encoding="utf-8")

    assert main([str(path), "--threshold", "65"]) == 0


def test_a_json_file_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    """PowerShell's `>` writes a BOM; the comment step must not die on it."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload(score=40)), encoding="utf-8-sig")

    assert main([str(path), "--threshold", "65"]) == 0


def test_the_comment_can_be_written_to_a_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload()), encoding="utf-8")
    out = tmp_path / "comment.md"

    main([str(path), "--threshold", "65", "--output", str(out)])

    assert MARKER in out.read_text(encoding="utf-8")
