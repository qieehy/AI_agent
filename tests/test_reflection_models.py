from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from runtime.reflection import CritiqueDecision, CritiqueResult


def test_critique_result_accepts_a_reviewed_answer_without_feedback() -> None:
    result = CritiqueResult(decision=CritiqueDecision.ACCEPT)

    assert result.decision is CritiqueDecision.ACCEPT
    assert result.feedback is None
    assert result.to_dict() == {
        "decision": "accept",
        "feedback": None,
    }


def test_critique_result_preserves_actionable_revision_feedback() -> None:
    result = CritiqueResult(
        decision=CritiqueDecision.REVISE,
        feedback="The answer omits the timeout behavior.",
    )

    assert result.to_dict() == {
        "decision": "revise",
        "feedback": "The answer omits the timeout behavior.",
    }


def test_critique_result_is_immutable_after_validation() -> None:
    result = CritiqueResult(decision=CritiqueDecision.ACCEPT)

    with pytest.raises(FrozenInstanceError):
        result.feedback = "Changed"  # type: ignore[misc]
