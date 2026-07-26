"""Tests for the AI layer.

Two things are being proved here. The first is that the explanation works and
degrades cleanly. The second matters more: that the language model cannot move
the score. No test touches the network — the client is replaced wholesale.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from sentinel.blast_radius import BlastRadius, FileImpact
from sentinel.config import Settings
from sentinel.explain import SYSTEM_PROMPT, build_facts, explain
from sentinel.results import Explanation, Scope
from sentinel.risk_rules import ChangeRisk, FileRisk, Reason

WHEN = datetime(2026, 7, 25, 22, 15)

GOOD_JSON = json.dumps(
    {
        "summary": "utils.py has 97 past bug fixes and the author has never touched it.",
        "rollout": "Ship behind a flag to 10% of traffic first.",
        "rollback_trigger": "Any rise in request error rate above baseline.",
        "monitoring": "Watch adapters.py and sessions.py call paths.",
    }
)


# --------------------------------------------------------------------------
# Fixtures: a finished analysis, and a fake OpenAI client
# --------------------------------------------------------------------------


@pytest.fixture
def risk() -> ChangeRisk:
    return ChangeRisk(
        score=77,
        band="high",
        reasons=(
            Reason(
                "hot_file",
                "File has a long history of bug fixes",
                35,
                "97 past bug-fix commits (threshold 20)",
                "src/requests/utils.py",
            ),
            Reason(
                "author_new_to_file",
                "Author has never changed this file before",
                15,
                "Demo Dev has 0 of its 290 past commits",
                "src/requests/utils.py",
            ),
        ),
        files=(
            FileRisk(
                path="src/requests/utils.py",
                score=72,
                band="high",
                reasons=(),
                lines_added=7,
                lines_deleted=0,
            ),
        ),
        recommendation="Hold this, or ship it behind a flag with a rollback ready.",
        relative_thresholds=True,
    )


@pytest.fixture
def scope() -> Scope:
    return Scope(
        mode="scan",
        repo="/tmp/requests",
        author="Demo Dev",
        when=WHEN,
        files_analyzed=1,
        base_ref="origin/main",
        commits_walked=4874,
    )


@pytest.fixture
def blast() -> BlastRadius:
    return BlastRadius(
        files=(
            FileImpact(
                path="src/requests/utils.py",
                direct=("src/requests/adapters.py",),
                transitive=("src/requests/api.py",),
                direct_count=5,
                transitive_count=8,
                in_cycle=True,
                depth_capped=False,
                is_hub=False,
            ),
        ),
        direct=("src/requests/adapters.py", "src/requests/sessions.py"),
        transitive=("src/requests/api.py",),
        direct_count=5,
        transitive_count=8,
        hubs=(),
        cycle_files=("src/requests/utils.py",),
        depth=3,
        depth_capped=False,
        listed_limit=12,
        graph_files=37,
        graph_edges=108,
        parsers=("python",),
        analyzed=True,
    )


class FakeCompletions:
    def __init__(self, recorder: dict, response=None, error: Exception | None = None):
        self._recorder = recorder
        self._response = response
        self._error = error

    def create(self, **kwargs):
        self._recorder["calls"] = self._recorder.get("calls", 0) + 1
        self._recorder["kwargs"] = kwargs
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, recorder: dict, response=None, error: Exception | None = None):
        self._recorder = recorder
        self.chat = SimpleNamespace(
            completions=FakeCompletions(recorder, response, error)
        )


def reply(content: str, reasoning: str | None = None):
    message = SimpleNamespace(content=content)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_openai(monkeypatch):
    """Install a fake `openai.OpenAI`; return the recorder of what it saw."""
    import openai

    recorder: dict = {}

    def install(response=None, error: Exception | None = None):
        def factory(**init_kwargs):
            recorder["init"] = init_kwargs
            return FakeClient(recorder, response, error)

        monkeypatch.setattr(openai, "OpenAI", factory)
        return recorder

    return install


def settings_with_key(**overrides) -> Settings:
    values = {
        "NVIDIA_API_KEY": "nvapi-not-a-real-key",
        "SENTINEL_LLM_MODEL": "meta/llama-3.3-70b-instruct",
    }
    values.update(overrides)
    return Settings(_env_file=None, **{k: v for k, v in values.items()})


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_a_grounded_explanation_is_returned(fake_openai, risk, scope, blast) -> None:
    fake_openai(response=reply(GOOD_JSON))

    explanation = explain(risk, scope, blast, settings_with_key())

    assert explanation.available
    assert explanation.generated_by == "meta/llama-3.3-70b-instruct"
    assert "97 past bug fixes" in explanation.summary
    assert explanation.rollout
    assert explanation.rollback_trigger
    assert explanation.monitoring


def test_exactly_one_llm_call_is_made(fake_openai, risk, scope, blast) -> None:
    """The design is one call for explanation, not an agent loop."""
    recorder = fake_openai(response=reply(GOOD_JSON))
    explain(risk, scope, blast, settings_with_key())
    assert recorder["calls"] == 1


def test_the_client_is_pointed_at_the_nvidia_endpoint(
    fake_openai, risk, scope, blast
) -> None:
    recorder = fake_openai(response=reply(GOOD_JSON))
    explain(risk, scope, blast, settings_with_key())

    assert recorder["init"]["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert recorder["init"]["api_key"] == "nvapi-not-a-real-key"
    assert recorder["init"]["timeout"] == 60.0

    assert recorder["kwargs"]["model"] == "meta/llama-3.3-70b-instruct"
    assert recorder["kwargs"]["temperature"] == 0.2
    assert recorder["kwargs"]["max_tokens"] == 800


def test_the_model_is_configurable(fake_openai, risk, scope, blast) -> None:
    recorder = fake_openai(response=reply(GOOD_JSON))
    explain(
        risk, scope, blast, settings_with_key(SENTINEL_LLM_MODEL="openai/gpt-oss-120b")
    )
    assert recorder["kwargs"]["model"] == "openai/gpt-oss-120b"


# --------------------------------------------------------------------------
# Grounding: what the model is allowed to see
# --------------------------------------------------------------------------


def test_the_prompt_forbids_inventing_and_forbids_rescoring() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "do not invent" in lowered
    assert "already been computed" in lowered
    assert "only the facts" in lowered


def test_the_prompt_tells_the_model_to_respect_counterintuitive_factors() -> None:
    """SHAP can report that complexity lowered risk. That must survive."""
    lowered = SYSTEM_PROMPT.lower()
    assert "counterintuitive" in lowered
    assert "as measured" in lowered
    assert "common-sense" in lowered


def test_the_facts_carry_the_real_numbers(risk, scope, blast) -> None:
    facts = build_facts(risk, scope, blast)

    assert facts["risk_score_out_of_100"] == 77
    assert facts["risk_band"] == "high"
    assert facts["author_of_the_change"] == "Demo Dev"

    factors = facts["contributing_factors"]
    assert factors[0]["factor"] == "File has a long history of bug fixes"
    assert "97 past bug-fix commits" in factors[0]["evidence"]
    assert factors[0]["file"] == "src/requests/utils.py"

    assert facts["blast_radius"]["files_importing_the_change_directly"] == 5
    assert "src/requests/utils.py" in facts["blast_radius"]["files_in_a_circular_import"]


def test_model_scored_factors_are_labelled_as_shap_impacts(scope, blast) -> None:
    model_risk = ChangeRisk(
        score=42,
        band="medium",
        reasons=(
            Reason("model:max_complexity", "Complexity of the most complex file", -39,
                   "value 19; log-odds -0.386", None),
        ),
        files=(),
        recommendation="Deployable, but review it properly.",
        scoring_method="model",
    )
    facts = build_facts(model_risk, scope, blast)

    assert "shap_impact_x100_signed" in facts["contributing_factors"][0]
    # The negative contribution survives into the facts, so the model can be
    # told that complexity lowered the score here.
    assert facts["contributing_factors"][0]["shap_impact_x100_signed"] == -39
    assert "opposite way to general intuition" in facts["how_to_read_the_factors"]


def test_the_facts_sent_are_exactly_the_facts_built(
    fake_openai, risk, scope, blast
) -> None:
    """Nothing extra leaks into the prompt behind the facts builder's back."""
    recorder = fake_openai(response=reply(GOOD_JSON))
    explain(risk, scope, blast, settings_with_key())

    user_message = recorder["kwargs"]["messages"][1]["content"]
    assert json.loads(user_message) == build_facts(risk, scope, blast)


def test_an_unavailable_blast_radius_is_stated_not_faked(risk, scope) -> None:
    facts = build_facts(risk, scope, None)
    assert facts["blast_radius"] == "not available for this repository"


# --------------------------------------------------------------------------
# Degrading cleanly
# --------------------------------------------------------------------------


def test_a_missing_key_skips_without_calling_anything(risk, scope, blast) -> None:
    explanation = explain(risk, scope, blast, Settings(_env_file=None))

    assert not explanation.available
    assert "NVIDIA_API_KEY" in explanation.skipped_reason


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Connection refused"),
        TimeoutError("timed out after 30s"),
        ValueError("429 Too Many Requests: rate limit exceeded"),
    ],
)
def test_any_api_failure_degrades_instead_of_crashing(
    fake_openai, risk, scope, blast, error
) -> None:
    fake_openai(error=error)

    explanation = explain(risk, scope, blast, settings_with_key())

    assert not explanation.available
    assert type(error).__name__ in explanation.skipped_reason


def test_the_skip_reason_stays_short(fake_openai, risk, scope, blast) -> None:
    """It ends up in JSON and CI logs, so it must not be a wall of text."""
    fake_openai(error=RuntimeError("x" * 5000))
    explanation = explain(risk, scope, blast, settings_with_key())
    assert len(explanation.skipped_reason) < 220


def test_an_empty_answer_is_treated_as_no_answer(fake_openai, risk, scope, blast) -> None:
    fake_openai(response=reply("   "))
    assert not explain(risk, scope, blast, settings_with_key()).available


def test_a_malformed_completion_does_not_crash(fake_openai, risk, scope, blast) -> None:
    fake_openai(response=SimpleNamespace(choices=[]))
    explanation = explain(risk, scope, blast, settings_with_key())
    assert not explanation.available
    assert "no choices" in explanation.skipped_reason


# --------------------------------------------------------------------------
# Reading the reply
# --------------------------------------------------------------------------


def test_reasoning_content_is_never_shown_as_the_answer(
    fake_openai, risk, scope, blast
) -> None:
    """Some NVIDIA models return their scratch work. It is not the explanation."""
    fake_openai(
        response=reply(GOOD_JSON, reasoning="Let me think... maybe the score should be 90.")
    )

    explanation = explain(risk, scope, blast, settings_with_key())

    assert explanation.available
    for section in (
        explanation.summary,
        explanation.rollout,
        explanation.rollback_trigger,
        explanation.monitoring,
    ):
        assert "Let me think" not in section
        assert "should be 90" not in section


def test_json_wrapped_in_a_markdown_fence_is_still_read(
    fake_openai, risk, scope, blast
) -> None:
    fake_openai(response=reply(f"```json\n{GOOD_JSON}\n```"))
    explanation = explain(risk, scope, blast, settings_with_key())
    assert explanation.available
    assert "97 past bug fixes" in explanation.summary


def test_json_with_chatter_around_it_is_still_read(
    fake_openai, risk, scope, blast
) -> None:
    fake_openai(response=reply(f"Sure! Here is the analysis:\n{GOOD_JSON}\nHope that helps."))
    explanation = explain(risk, scope, blast, settings_with_key())
    assert explanation.available
    assert explanation.rollout


def test_plain_prose_becomes_the_summary_rather_than_being_discarded(
    fake_openai, risk, scope, blast
) -> None:
    """An untidy explanation still beats no explanation."""
    fake_openai(response=reply("This change touches a file with a poor track record."))

    explanation = explain(risk, scope, blast, settings_with_key())

    assert explanation.available
    assert "poor track record" in explanation.summary
    assert explanation.rollout == ""


def test_missing_sections_come_back_empty_not_absent(
    fake_openai, risk, scope, blast
) -> None:
    fake_openai(response=reply(json.dumps({"summary": "Short answer."})))

    explanation = explain(risk, scope, blast, settings_with_key())

    assert explanation.available
    assert explanation.summary == "Short answer."
    assert explanation.rollout == ""
    assert explanation.monitoring == ""


def test_a_json_array_is_rejected_as_unusable(fake_openai, risk, scope, blast) -> None:
    fake_openai(response=reply('["not", "an", "object"]'))
    explanation = explain(risk, scope, blast, settings_with_key())
    # Falls back to treating the reply as prose rather than inventing sections.
    assert explanation.summary.startswith("[")


# --------------------------------------------------------------------------
# The boundary: prose cannot move the number
# --------------------------------------------------------------------------


def test_explanation_type_has_nowhere_to_put_a_score() -> None:
    """A structural guarantee, not a promise in a prompt."""
    fields = set(Explanation.__dataclass_fields__)
    assert not fields & {"score", "band", "reasons", "recommendation", "files"}


def test_attaching_an_explanation_leaves_the_analysis_identical(
    risk, scope, blast
) -> None:
    from sentinel.results import AnalysisResult
    from sentinel.serialization import to_dict

    result = AnalysisResult(risk=risk, scope=scope, blast=blast)
    before = to_dict(result)

    explained = result.with_explanation(
        Explanation(available=True, generated_by="m", summary="prose")
    )
    after = to_dict(explained)

    assert before["ai_explanation"] is None
    assert after["ai_explanation"]["available"] is True

    del before["ai_explanation"], after["ai_explanation"]
    assert before == after  # everything else byte-identical


def test_the_original_result_is_not_mutated(risk, scope, blast) -> None:
    from sentinel.results import AnalysisResult

    result = AnalysisResult(risk=risk, scope=scope, blast=blast)
    result.with_explanation(Explanation(available=True, summary="prose"))
    assert result.explanation is None


def test_a_model_claiming_a_different_score_changes_nothing(
    fake_openai, risk, scope, blast
) -> None:
    """The adversarial case: the model tries to overrule the analysis."""
    from sentinel.results import AnalysisResult
    from sentinel.serialization import to_dict

    fake_openai(
        response=reply(
            json.dumps(
                {
                    "summary": "Actually this is a 5/100 low-risk change and the band is low.",
                    "rollout": "Ship it straight to production.",
                    "rollback_trigger": "None needed.",
                    "monitoring": "Nothing.",
                }
            )
        )
    )

    result = AnalysisResult(risk=risk, scope=scope, blast=blast)
    explained = result.with_explanation(
        explain(risk, scope, blast, settings_with_key())
    )
    payload = to_dict(explained)

    assert payload["score"] == 77
    assert payload["band"] == "high"
    assert payload["recommendation"].startswith("Hold this")
    # The claim is quoted in the prose but has no effect on the numbers.
    assert "5/100" in payload["ai_explanation"]["summary"]
