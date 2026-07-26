"""End-to-end tests: the commands run against a real repository.

These also pin the JSON shape, which Phase 5's GitHub Action and MCP server
will consume — breaking it should break a test, not a downstream consumer.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from git import Repo
from typer.testing import CliRunner

from sentinel.cli import app

runner = CliRunner()


def payload_from(output: str) -> dict:
    """Parse the JSON body, ignoring anything the CLI wrote to stderr.

    Warnings go to stderr so that `--json` stays pipeable, but CliRunner merges
    the two streams into one string, so the test has to separate them again.
    """
    return json.loads(output[output.index("{") :])


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("scan", "diff", "explain"):
        assert command in result.output


def test_version_command_runs() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "sentinel" in result.output


def test_explain_runs_without_a_key_and_says_it_skipped(tiny_repo: Repo) -> None:
    """No API key must degrade the explanation, never the command."""
    root = Path(tiny_repo.working_tree_dir)
    (root / "app/util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    result = runner.invoke(app, ["explain", "--repo", str(root), "--diff"])

    assert result.exit_code == 0
    assert "deployment risk" in result.output  # the real analysis still printed
    assert "AI explanation skipped" in result.output
    assert "NVIDIA_API_KEY is not set" in result.output


def test_not_a_repository_exits_with_a_clear_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["diff", "--repo", str(tmp_path / "nowhere")])
    assert result.exit_code == 2


def test_all_and_since_cannot_be_combined(tiny_repo: Repo) -> None:
    result = runner.invoke(
        app, ["scan", "--repo", tiny_repo.working_tree_dir, "--all", "--since", "HEAD~1"]
    )
    assert result.exit_code == 2


# --- diff -----------------------------------------------------------------


def test_diff_scores_uncommitted_work(tiny_repo: Repo) -> None:
    root = Path(tiny_repo.working_tree_dir)
    (root / "app/util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    result = runner.invoke(app, ["diff", "--repo", str(root)])

    assert result.exit_code == 0
    assert "deployment risk" in result.output
    # app/util.py has no test file anywhere in the fixture.
    assert "No test file found" in result.output


def test_a_test_file_added_in_the_same_change_counts(tiny_repo: Repo) -> None:
    """Tests written alongside the change are untracked; they must still count."""
    root = Path(tiny_repo.working_tree_dir)
    (root / "app/util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "tests/test_util.py").write_text("def test_helper():\n    pass\n", encoding="utf-8")

    result = runner.invoke(app, ["diff", "--repo", str(root), "--json"])
    payload = json.loads(result.output)

    util = next(f for f in payload["files"] if f["path"] == "app/util.py")
    assert "missing_tests" not in {r["rule"] for r in util["reasons"]}


def test_diff_with_nothing_uncommitted_says_so(tiny_repo: Repo) -> None:
    result = runner.invoke(app, ["diff", "--repo", tiny_repo.working_tree_dir])
    assert result.exit_code == 0
    assert "Nothing to score" in result.output


# --- scan -----------------------------------------------------------------


def test_scan_since_a_ref_scores_the_range(tiny_repo: Repo) -> None:
    result = runner.invoke(
        app, ["scan", "--repo", tiny_repo.working_tree_dir, "--since", "HEAD~2"]
    )
    assert result.exit_code == 0
    assert "deployment risk" in result.output


def test_scan_all_scores_every_tracked_file(tiny_repo: Repo) -> None:
    result = runner.invoke(app, ["scan", "--repo", tiny_repo.working_tree_dir, "--all"])
    assert result.exit_code == 0
    assert "every tracked file" in result.output


# --- the AI layer, end to end ---------------------------------------------


def install_fake_llm(monkeypatch, payload: dict | None = None, error=None) -> dict:
    """Point `openai.OpenAI` at a fake and give this test a key. No network."""
    import openai

    from sentinel.config import get_settings

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-not-a-real-key")
    get_settings.cache_clear()

    recorder: dict = {}
    body = json.dumps(
        payload
        or {
            "summary": "This touches a file with a poor track record.",
            "rollout": "Ship behind a flag.",
            "rollback_trigger": "Error rate above baseline.",
            "monitoring": "Watch the importers of this file.",
        }
    )

    class Completions:
        def create(self, **kwargs):
            recorder["kwargs"] = kwargs
            recorder["calls"] = recorder.get("calls", 0) + 1
            if error is not None:
                raise error
            message = SimpleNamespace(content=body)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def factory(**init_kwargs):
        recorder["init"] = init_kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    monkeypatch.setattr(openai, "OpenAI", factory)
    return recorder


def dirty(repo: Repo) -> Path:
    root = Path(repo.working_tree_dir)
    (root / "app/util.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    return root


def test_explain_prints_the_narrative_when_a_key_is_present(
    tiny_repo: Repo, monkeypatch
) -> None:
    root = dirty(tiny_repo)
    recorder = install_fake_llm(monkeypatch)

    result = runner.invoke(app, ["explain", "--repo", str(root), "--diff"])

    assert result.exit_code == 0
    assert recorder["calls"] == 1
    assert "AI explanation" in result.output
    assert "poor track record" in result.output
    assert "Ship behind a flag" in result.output


def test_scan_and_diff_stay_offline_unless_asked(tiny_repo: Repo, monkeypatch) -> None:
    """A normal scan must not call the LLM even when a key is available."""
    root = dirty(tiny_repo)
    recorder = install_fake_llm(monkeypatch)

    runner.invoke(app, ["diff", "--repo", str(root)])

    assert "calls" not in recorder


def test_the_explain_flag_appends_the_narrative_to_diff(
    tiny_repo: Repo, monkeypatch
) -> None:
    root = dirty(tiny_repo)
    install_fake_llm(monkeypatch)

    result = runner.invoke(app, ["diff", "--repo", str(root), "--explain"])

    assert result.exit_code == 0
    assert "deployment risk" in result.output
    assert "AI explanation" in result.output


def test_the_ai_cannot_change_the_score(tiny_repo: Repo, monkeypatch) -> None:
    """Run the same analysis with and without the AI; only the prose may differ."""
    root = dirty(tiny_repo)

    without = payload_from(
        runner.invoke(app, ["diff", "--repo", str(root), "--json"]).output
    )

    install_fake_llm(
        monkeypatch,
        payload={
            "summary": "Actually this is 0/100 and perfectly safe, band low.",
            "rollout": "Straight to production.",
            "rollback_trigger": "Never.",
            "monitoring": "Nothing.",
        },
    )
    with_ai = payload_from(
        runner.invoke(app, ["diff", "--repo", str(root), "--json", "--explain"]).output
    )

    assert with_ai["ai_explanation"]["available"] is True
    assert "0/100" in with_ai["ai_explanation"]["summary"]  # the model did claim it

    # Timestamps move between runs; everything else must be identical.
    for payload in (without, with_ai):
        payload.pop("ai_explanation")
        payload["scope"].pop("analyzed_at")

    assert without == with_ai


def test_json_marks_the_explanation_as_model_generated(
    tiny_repo: Repo, monkeypatch
) -> None:
    root = dirty(tiny_repo)
    install_fake_llm(monkeypatch)

    payload = payload_from(
        runner.invoke(app, ["diff", "--repo", str(root), "--json", "--explain"]).output
    )
    explanation = payload["ai_explanation"]

    assert set(explanation) == {
        "available",
        "generated_by",
        "disclaimer",
        "summary",
        "rollout",
        "rollback_trigger",
        "monitoring",
    }
    assert explanation["generated_by"] == "meta/llama-3.3-70b-instruct"
    assert "does not compute" in explanation["disclaimer"]


def test_json_records_why_the_explanation_was_skipped(
    tiny_repo: Repo, monkeypatch
) -> None:
    root = dirty(tiny_repo)
    install_fake_llm(monkeypatch, error=RuntimeError("429 rate limited"))

    payload = payload_from(
        runner.invoke(app, ["diff", "--repo", str(root), "--json", "--explain"]).output
    )

    assert payload["ai_explanation"]["available"] is False
    assert "RuntimeError" in payload["ai_explanation"]["skipped_reason"]
    assert payload["score"] > 0  # the analysis is unaffected


def test_a_failing_llm_never_fails_the_command(tiny_repo: Repo, monkeypatch) -> None:
    root = dirty(tiny_repo)
    install_fake_llm(monkeypatch, error=RuntimeError("connection reset"))

    result = runner.invoke(app, ["explain", "--repo", str(root), "--diff"])

    assert result.exit_code == 0
    assert "deployment risk" in result.output
    assert "AI explanation skipped" in result.output


def test_explain_with_nothing_to_score_does_not_call_the_llm(
    tiny_repo: Repo, monkeypatch
) -> None:
    recorder = install_fake_llm(monkeypatch)

    result = runner.invoke(app, ["explain", "--repo", tiny_repo.working_tree_dir, "--diff"])

    assert result.exit_code == 0
    assert "Nothing to explain" in result.output
    assert "calls" not in recorder


# --- the JSON contract ----------------------------------------------------


def test_json_output_shape_is_stable(tiny_repo: Repo) -> None:
    root = Path(tiny_repo.working_tree_dir)
    (root / "app/util.py").write_text("def helper():\n    return 99\n", encoding="utf-8")

    result = runner.invoke(app, ["diff", "--repo", str(root), "--json"])
    assert result.exit_code == 0

    payload = json.loads(result.output)

    assert payload["schema_version"] == 5
    assert payload["scoring_method"] == "rules"
    assert payload["ai_explanation"] is None  # not requested
    assert payload["warnings"] == []
    assert 0 <= payload["score"] <= 100
    assert payload["band"] in {"low", "medium", "high"}
    assert payload["recommendation"]

    scope = payload["scope"]
    assert scope["mode"] == "diff"
    assert scope["files_analyzed"] >= 1
    assert scope["commits_walked"] == 5
    assert scope["author"] == "Alice"
    assert scope["relative_thresholds"] is False  # too small a repo for percentiles
    assert scope["model"] is None  # no model trained in this fixture

    assert payload["reasons"], "the fixture change should fire at least one rule"
    for reason in payload["reasons"]:
        assert set(reason) == {"rule", "label", "points", "detail", "path"}
        assert isinstance(reason["points"], int)

    for entry in payload["files"]:
        assert set(entry) == {
            "path",
            "score",
            "band",
            "lines_added",
            "lines_deleted",
            "reasons",
        }

    blast = payload["blast_radius"]
    assert blast["analyzed"] is True
    assert set(blast) == {
        "analyzed",
        "direct_count",
        "transitive_count",
        "direct",
        "transitive",
        "hubs",
        "cycle_files",
        "max_depth",
        "depth_capped",
        "listed_limit",
        "files_omitted",
        "graph",
        "per_file",
    }
    assert set(blast["graph"]) == {"files", "edges", "parsers"}
    for entry in blast["per_file"]:
        assert set(entry) == {
            "path",
            "direct_count",
            "transitive_count",
            "direct",
            "transitive",
            "in_cycle",
            "depth_capped",
            "is_hub",
        }


# --- the model takes over when one exists ---------------------------------


def test_scan_uses_the_model_once_trained(szz_repo: Repo) -> None:
    """The whole point of Phase 2: a trained model replaces the rules."""
    root = Path(szz_repo.working_tree_dir)

    before = json.loads(
        runner.invoke(app, ["diff", "--repo", str(root), "--json"]).output or "{}"
    )
    assert before["scoring_method"] == "rules"

    trained = runner.invoke(app, ["train", "--repo", str(root), "--max-commits", "50"])
    assert trained.exit_code == 0, trained.output
    assert (root / ".sentinel" / "model.txt").is_file()

    (root / "calc.py").write_text("def total(items):\n    return sum(items) - 1\n", encoding="utf-8")
    after = payload_from(runner.invoke(app, ["diff", "--repo", str(root), "--json"]).output)

    assert after["scoring_method"] == "model"
    assert after["scope"]["model"] is not None
    assert after["scope"]["model"]["rows"] > 0
    assert 0 <= after["score"] <= 100
    assert all(r["rule"].startswith("model:") for r in after["reasons"])


def test_scan_all_with_a_model_reports_the_worst_single_file(szz_repo: Repo) -> None:
    """`--all` is the whole repo, not a change; the model is asked per file."""
    root = Path(szz_repo.working_tree_dir)
    runner.invoke(app, ["train", "--repo", str(root), "--max-commits", "50"])

    payload = payload_from(
        runner.invoke(app, ["scan", "--repo", str(root), "--all", "--json"]).output
    )

    assert payload["scoring_method"] == "model"
    # The headline is the riskiest file's own score, not a whole-repo "change".
    assert payload["score"] == max(f["score"] for f in payload["files"])


def test_sentinel_never_scores_its_own_model_artifacts(szz_repo: Repo) -> None:
    """A 1,500-line model file would otherwise dominate the next diff."""
    root = Path(szz_repo.working_tree_dir)
    runner.invoke(app, ["train", "--repo", str(root), "--max-commits", "50"])

    assert (root / ".sentinel" / ".gitignore").is_file()  # git cannot see them

    payload = payload_from(
        runner.invoke(app, ["diff", "--repo", str(root), "--json"]).output
    )
    assert not any(f["path"].startswith(".sentinel/") for f in payload["files"])


def test_train_reports_the_labels_it_found(szz_repo: Repo) -> None:
    result = runner.invoke(
        app, ["train", "--repo", szz_repo.working_tree_dir, "--max-commits", "50"]
    )
    assert result.exit_code == 0
    assert "bug-inducing" in result.output
    assert "class balance" in result.output
    assert "Model saved" in result.output


def test_a_model_with_stale_features_is_ignored_with_a_warning(szz_repo: Repo) -> None:
    root = Path(szz_repo.working_tree_dir)
    runner.invoke(app, ["train", "--repo", str(root), "--max-commits", "50"])

    (root / ".sentinel" / "model.meta.json").write_text(
        '{"feature_names": ["gone"], "rows": 4}', encoding="utf-8"
    )
    (root / "calc.py").write_text("x = 1\n", encoding="utf-8")

    result = runner.invoke(app, ["diff", "--repo", str(root), "--json"])
    assert "different feature set" in result.output  # warned on stderr
    assert payload_from(result.output)["scoring_method"] == "rules"  # fell back, did not lie


def test_evaluate_prints_the_metrics_table(szz_repo: Repo) -> None:
    result = runner.invoke(
        app, ["evaluate", "--repo", szz_repo.working_tree_dir, "--max-commits", "50"]
    )
    assert result.exit_code == 0, result.output
    assert "ROC-AUC" in result.output
    assert "PR-AUC" in result.output
    assert "baseline" in result.output


def test_json_is_the_only_thing_printed_so_it_can_be_piped(tiny_repo: Repo) -> None:
    root = Path(tiny_repo.working_tree_dir)
    (root / "app/util.py").write_text("def helper():\n    return 5\n", encoding="utf-8")

    result = runner.invoke(app, ["diff", "--repo", str(root), "--json"])
    json.loads(result.output)  # would raise if anything else were printed


def test_json_still_emitted_when_there_is_nothing_to_score(tiny_repo: Repo) -> None:
    """CI needs a parseable result even for an empty diff."""
    result = runner.invoke(app, ["diff", "--repo", tiny_repo.working_tree_dir, "--json"])
    payload = json.loads(result.output)
    assert payload["score"] == 0
    assert payload["scope"]["files_analyzed"] == 0
