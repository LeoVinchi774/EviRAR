"""动态证据策略：依据证据覆盖和卡片角色路由生成器下一步动作。"""

from __future__ import annotations

from typing import Any

from evirar.knowledge.evidence_classification_gate import PRIMARY_SUPPORT


class DynamicEvidenceStrategy:
    """输出生成、继续问诊、补检索或人工复核等确定性证据策略。"""

    def decide(
        self,
        decision_cards: list[dict[str, Any]],
        coverage: dict[str, Any],
        conflicts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if conflicts:
            return {
                "action": "human_review",
                "reason": "fact_or_theory_conflict",
                "next_question": None,
            }
        if coverage.get("missing_slots"):
            actions = coverage.get("recommended_actions", [])
            return {
                "action": "ask_follow_up",
                "reason": "critical_fact_missing",
                "next_question": actions[0].get("action") if actions else None,
            }
        if any(
            item["document_gate"]["role"] == PRIMARY_SUPPORT
            for item in decision_cards
        ):
            return {
                "action": "generate",
                "reason": "coverage_and_primary_support_ready",
                "next_question": None,
            }
        return {
            "action": "retrieve_more",
            "reason": "coverage_ready_but_no_primary_support",
            "next_question": None,
        }
