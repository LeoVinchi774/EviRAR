"""模块 D 的输入：检查已获取病例事实是否覆盖关键问诊槽位。"""

from __future__ import annotations

from typing import Any

from evirar.knowledge.evidence_classification_gate import PRIMARY_SUPPORT


class CandidateAwareCoverageGate:
    """将事实完整度和主支持卡数量转换为可审计覆盖判断。"""

    _QUESTIONS = {
        "tongue": "请提供患者的舌象，包括舌质和舌苔。",
        "pulse": "请提供患者的脉象特征。",
        "course": "请说明症状持续多久及加重、缓解因素。",
    }

    def analyze(
        self, profile: dict[str, Any], assessed: list[dict[str, Any]]
    ) -> dict[str, Any]:
        fields = {item["field"] for item in profile.get("facts", [])}
        expected = {
            "symptom": {"symptom", "chief_complaint"},
            "tongue": {"tongue"},
            "pulse": {"pulse"},
            "course": {"course"},
        }
        missing = [slot for slot, source in expected.items() if not fields & source]
        primary_count = sum(
            item["document_gate"]["role"] == PRIMARY_SUPPORT
            for item in assessed
        )
        return {
            "missing_slots": missing,
            "covered_slots": [slot for slot in expected if slot not in missing],
            "coverage_score": round((len(expected) - len(missing)) / len(expected), 4),
            "convergence": {
                "converged": not missing and primary_count > 0,
                "primary_support_count": primary_count,
            },
            "recommended_actions": [
                {"slot": slot, "action": self._QUESTIONS.get(slot)}
                for slot in missing
                if slot in self._QUESTIONS
            ],
        }
