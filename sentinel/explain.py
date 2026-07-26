"""One LLM call, explanation only.

The boundary this module enforces is the whole point of it:

* It receives an analysis that is **already finished**. `ChangeRisk` is frozen,
  and it was computed before this code ran.
* It returns prose. It cannot return a score, a band, or a reason, because
  `Explanation` has no fields for them.
* `AnalysisResult.with_explanation` builds a new result rather than mutating
  one, so the analysis that was scored is the analysis that gets reported.

So there is no code path by which the language model can move the number. That
is a structural guarantee, not a promise in a prompt — which matters, because
prompts are not a security boundary and a model that decides the score "should
really be 90" must not be able to act on it.

Everything here is optional by design. No key, a timeout, a rate limit, a
malformed response — all of them degrade to the deterministic report with a
note. A risk tool that cannot answer without a third-party API is worse than
one that has no AI at all.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sentinel.blast_radius import BlastRadius
from sentinel.config import Settings
from sentinel.results import Explanation, Scope
from sentinel.risk_rules import ChangeRisk

logger = logging.getLogger(__name__)

#: Keys the model is asked to return. Anything else it says is discarded.
SECTIONS = ("summary", "rollout", "rollback_trigger", "monitoring")

SYSTEM_PROMPT = """\
You are a senior release engineer helping a colleague decide whether to deploy a \
change. You are given the output of a completed, deterministic risk analysis.

Your job is to explain that analysis in plain English. It is not to judge it.

Rules you must follow:

1. The score has already been computed. Never restate it as if you calculated \
it, never argue it should be different, and never suggest your own score or \
risk level.
2. Use ONLY the facts given to you. Do not invent file names, numbers, dates, \
authors, test names, or history. If you want to mention a specific file or \
number, it must appear in the facts.
3. Refer to the actual factors listed, with their real numbers. Generic advice \
that would apply to any change is worthless here.
4. Some factors will look counterintuitive — a model may report that high \
complexity *lowered* the risk of this change, because that is what the \
repository's own history showed. Explain such factors as measured. Do not \
correct them with common-sense assumptions about what usually increases risk.
5. If the facts are thin, say so briefly rather than padding.

Reply with a single JSON object and nothing else, using exactly these keys:

  "summary"           - 2-4 sentences on why this change carries the risk it does.
  "rollout"           - how to deploy it, given these specific factors.
  "rollback_trigger"  - the concrete signal that means roll back now.
  "monitoring"        - what to watch after deploy, tied to what this change touches.

Plain prose inside the JSON values. No markdown, no bullet characters, no \
headings.
"""


def explain(
    risk: ChangeRisk,
    scope: Scope,
    blast: BlastRadius | None,
    settings: Settings,
) -> Explanation:
    """Ask the configured model to narrate a finished analysis.

    Never raises. Every failure returns `Explanation(available=False)` with a
    reason a human can act on.
    """
    if not settings.llm_enabled:
        return Explanation(
            available=False,
            skipped_reason="NVIDIA_API_KEY is not set",
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - openai is a declared dependency
        return Explanation(available=False, skipped_reason=f"openai unavailable: {exc}")

    facts = build_facts(risk, scope, blast)

    try:
        client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.nvidia_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(facts, indent=2)},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    except Exception as exc:
        # Deliberately broad: auth, connection, timeout, rate limit and the
        # provider's own 5xx all mean the same thing here — no explanation, and
        # the report carries on regardless.
        logger.warning("AI explanation unavailable: %s", exc)
        return Explanation(available=False, skipped_reason=_describe(exc))

    return _read_response(response, settings.llm_model)


def _describe(exc: Exception) -> str:
    """A short, safe description of a failure.

    Only the exception type and a truncated message: provider errors can echo
    request details, and this string ends up in JSON output and CI logs.
    """
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    label = type(exc).__name__
    return f"{label}: {message[:160]}" if message else label


def _read_response(response: Any, model_name: str) -> Explanation:
    """Pull the answer out of a chat completion.

    Some NVIDIA-hosted models return `reasoning_content` alongside `content`.
    That is the model's scratch work, not its answer: it is logged at debug and
    never shown, because presenting it as the explanation would be presenting
    deliberation as conclusion.
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        logger.warning("unexpected completion shape: %s", exc)
        return Explanation(available=False, skipped_reason="model returned no choices")

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        logger.debug("ignoring reasoning_content (%d chars)", len(str(reasoning)))

    content = (getattr(message, "content", None) or "").strip()
    if not content:
        return Explanation(available=False, skipped_reason="model returned an empty answer")

    sections = _parse_sections(content)
    if not any(sections.values()):
        return Explanation(available=False, skipped_reason="model returned no usable prose")

    return Explanation(
        available=True,
        generated_by=model_name,
        summary=sections["summary"],
        rollout=sections["rollout"],
        rollback_trigger=sections["rollback_trigger"],
        monitoring=sections["monitoring"],
    )


def _parse_sections(content: str) -> dict[str, str]:
    """Read the four sections out of the model's reply.

    Models wrap JSON in markdown fences and add prose around it often enough
    that being strict here would throw away good answers. If the JSON cannot be
    recovered at all, the whole reply becomes the summary — a slightly untidy
    explanation still beats none.
    """
    empty = {key: "" for key in SECTIONS}

    payload = _extract_json(content)
    if payload is None:
        return {**empty, "summary": content}

    return {
        key: str(payload.get(key, "")).strip() if payload.get(key) is not None else ""
        for key in SECTIONS
    }


def _extract_json(content: str) -> dict | None:
    text = content.strip()
    if text.startswith("```"):
        # ```json\n{...}\n```
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def build_facts(
    risk: ChangeRisk, scope: Scope, blast: BlastRadius | None
) -> dict[str, Any]:
    """The exact facts handed to the model — nothing more.

    Assembled explicitly rather than by dumping the whole serialized payload, so
    it is obvious what the model can and cannot see. Anything absent here cannot
    appear in a grounded explanation, which is how "do not invent" is enforced
    in practice rather than merely requested.
    """
    facts: dict[str, Any] = {
        "risk_score_out_of_100": risk.score,
        "risk_band": risk.band,
        "scored_by": (
            "a gradient-boosted model trained on this repository's own bug history"
            if risk.scoring_method == "model"
            else "a transparent points-based rule engine"
        ),
        "deterministic_recommendation": risk.recommendation,
        "what_was_analyzed": scope.description,
        "author_of_the_change": scope.author,
        "files_changed": scope.files_analyzed,
        "commits_of_history_examined": scope.commits_walked,
        "contributing_factors": [
            {
                "factor": reason.label,
                "evidence": reason.detail,
                "file": reason.path,
                (
                    "shap_impact_x100_signed"
                    if risk.scoring_method == "model"
                    else "points_added"
                ): reason.points,
            }
            for reason in risk.reasons
        ],
        "riskiest_files": [
            {
                "path": file_risk.path,
                "score": file_risk.score,
                "lines_added": file_risk.lines_added,
                "lines_deleted": file_risk.lines_deleted,
            }
            for file_risk in risk.files[:5]
        ],
    }

    if risk.scoring_method == "model":
        facts["how_to_read_the_factors"] = (
            "Each factor carries a signed SHAP impact: positive pushed the score "
            "up, negative pushed it down. These are measurements from this "
            "repository's history, so a factor may point the opposite way to "
            "general intuition. Report them as they are."
        )

    if blast is not None and blast.analyzed:
        facts["blast_radius"] = {
            "files_importing_the_change_directly": blast.direct_count,
            "files_affected_within_%d_hops" % blast.depth: blast.transitive_count,
            "direct_dependents_sample": list(blast.direct),
            "transitive_dependents_sample": list(blast.transitive),
            "hub_files_imported_by_many": list(blast.hubs),
            "files_in_a_circular_import": list(blast.cycle_files),
            "listing_truncated": blast.depth_capped
            or blast.direct_count > len(blast.direct),
        }
    else:
        facts["blast_radius"] = "not available for this repository"

    return facts
