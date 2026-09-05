from __future__ import annotations

import pytest

from errors import ReflectionError
from runtime.reflection import CritiqueDecision, parse_critique_result


def test_parse_critique_result_converts_valid_json_to_internal_model() -> None:
    result = parse_critique_result('{"decision":"revise","feedback":" Add the missing source. "}')

    assert result.decision is CritiqueDecision.REVISE
    assert result.feedback == "Add the missing source."


@pytest.mark.parametrize("content", [None, 123, [], {}])
def test_parse_critique_result_requires_json_text(content: object) -> None:
    with pytest.raises(ReflectionError, match="JSON text"):
        parse_critique_result(content)


def test_parse_critique_result_rejects_invalid_json() -> None:
    with pytest.raises(ReflectionError, match="valid JSON"):
        parse_critique_result("not json")


@pytest.mark.parametrize(
    "content",
    [
        '{"decision":"accept"}',
        '{"decision":"accept","feedback":null,"score":1}',
        '[{"decision":"accept","feedback":null}]',
    ],
)
def test_parse_critique_result_requires_exact_fields(content: str) -> None:
    with pytest.raises(ReflectionError, match="decision and feedback"):
        parse_critique_result(content)


@pytest.mark.parametrize(
    "decision",
    ["approved", "retry", "ACCEPT", 1, None, [], {}],
)
def test_parse_critique_result_rejects_unknown_decisions(decision: object) -> None:
    import json

    content = json.dumps({"decision": decision, "feedback": None})

    with pytest.raises(ReflectionError, match="decision"):
        parse_critique_result(content)


@pytest.mark.parametrize("feedback", [None, "", "   "])
def test_parse_critique_result_requires_feedback_for_revision(
    feedback: object,
) -> None:
    import json

    content = json.dumps({"decision": "revise", "feedback": feedback})

    with pytest.raises(ReflectionError, match="revision feedback"):
        parse_critique_result(content)


def test_parse_critique_result_rejects_feedback_for_acceptance() -> None:
    with pytest.raises(ReflectionError, match="accepted critique"):
        parse_critique_result('{"decision":"accept","feedback":"Revise something"}')


def test_parse_critique_result_enforces_feedback_size_limit() -> None:
    with pytest.raises(ReflectionError, match="size limit") as captured:
        parse_critique_result(
            '{"decision":"revise","feedback":"1234"}',
            max_feedback_chars=3,
        )

    assert captured.value.context == {"max_feedback_chars": 3}
