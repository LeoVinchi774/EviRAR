"""证据分类门控：候选卡角色分类、优化排序、去重与上下文选择。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from evirar.knowledge.fact_profile import FactProfileBuilder

PRIMARY_SUPPORT = "primary_support"
PARTIAL_SUPPORT = "partial_support"
DIFFERENTIAL = "differential"
IRRELEVANT = "irrelevant"


@dataclass(frozen=True)
class EvidenceClassificationConfig:
    """固定、可审计的角色分类与证据选择参数。"""

    min_primary_similarity: float = 0.45
    min_anchored_similarity: float = 0.40
    min_primary_facts: int = 2
    max_decision_cards: int = 5
    max_context_cards: int = 2
    duplicate_similarity: float = 0.96


class EvidenceClassificationGate:
    """将角色判断与高价值证据选择合并为同一确定性门控。"""

    _ALIASES = (
        ("胃胀痛", "胃脘胀痛", "胃脘痞痛"),
        ("失眠", "不寐", "睡眠不佳"),
        ("烦躁", "心烦", "心烦易怒"),
        ("怕冷", "畏寒", "恶寒"),
        ("乏力", "无力", "倦怠"),
    )
    _PULSE = {"弦", "滑", "数", "沉", "细", "弱", "浮", "迟"}
    _ROLE_BONUS = {
        PRIMARY_SUPPORT: 0.10,
        PARTIAL_SUPPORT: 0.05,
        DIFFERENTIAL: 0.00,
    }
    def __init__(self, config: EvidenceClassificationConfig | None = None) -> None:
        self.config = config or EvidenceClassificationConfig()

    def classify_and_select(
        self, candidates: list[dict[str, Any]], profile: dict[str, Any]
    ) -> dict[str, Any]:
        """分类候选卡，并输出决策集合和实际生成上下文集合。"""
        assessed = [
            self._classify_one(rank, card, profile)
            for rank, card in enumerate(candidates, 1)
        ]
        classification = {
            "assessed_results": assessed,
            "role_counts": {
                role: sum(
                    item["document_gate"]["role"] == role for item in assessed
                )
                for role in (
                    PRIMARY_SUPPORT,
                    PARTIAL_SUPPORT,
                    DIFFERENTIAL,
                    IRRELEVANT,
                )
            },
            "config": asdict(self.config),
        }
        return {
            "classification": classification,
            "selection": self._select(assessed, profile),
        }

    def _classify_one(
        self, rank: int, result: dict[str, Any], profile: dict[str, Any]
    ) -> dict[str, Any]:
        card = dict(result)
        metadata = card.get("metadata", {}) or {}
        matches = self._matches(metadata, profile)
        dense_score = self._score(card.get("similarity_score", 0.0))
        logic_score = self._score(card.get("relevance_score", dense_score))
        conflicts = self._conflicts(metadata, profile.get("attributes", {}))
        direct_matches = sum(item["source"] == "metadata" for item in matches)

        if conflicts:
            role, reason = IRRELEVANT, "explicit_patient_attribute_conflict"
        elif (
            metadata.get("differential_syndromes")
            and matches
            and len(matches) < self.config.min_primary_facts
        ):
            role, reason = DIFFERENTIAL, "explicit_differential_relation"
        elif (
            dense_score >= self.config.min_primary_similarity
            and len(matches) >= self.config.min_primary_facts
        ):
            role, reason = PRIMARY_SUPPORT, "high_similarity_multiple_fact_support"
        elif (
            dense_score >= self.config.min_anchored_similarity
            and direct_matches >= self.config.min_primary_facts
        ):
            role, reason = PRIMARY_SUPPORT, "anchored_metadata_fact_support"
        elif dense_score >= 0.60 and matches:
            role, reason = PARTIAL_SUPPORT, "partial_fact_support"
        else:
            role, reason = IRRELEVANT, "insufficient_fact_support"

        card["document_gate"] = {
            "rank": rank,
            "card_id": (
                metadata.get("card_id")
                or card.get("file_path")
                or card.get("syndrome_name")
            ),
            "role": role,
            "logic_relevance_score": logic_score,
            "dense_similarity_score": dense_score,
            "support_matches": matches,
            "support_points": [item["patient_fact"] for item in matches],
            "support_deficit": max(0, self.config.min_primary_facts - len(matches)),
            "conflict_points": conflicts,
            "decision_reason": reason,
        }
        return card

    def _select(
        self, assessed: list[dict[str, Any]], profile: dict[str, Any]
    ) -> dict[str, Any]:
        """按理论相关性、病例支持和新增覆盖选择高价值证据。"""
        selected: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        covered_facts: set[str] = set()
        total_facts = max(1, len(profile.get("facts", [])))
        remaining = [
            item
            for item in assessed
            if item["document_gate"]["role"] != IRRELEVANT
        ]

        while remaining and len(selected) < self.config.max_decision_cards:
            ranked: list[tuple[float, dict[str, Any]]] = []
            for item in remaining:
                gate = item["document_gate"]
                if self._is_near_duplicate(item, selected):
                    excluded.append({"card_id": gate["card_id"], "reason": "near_duplicate"})
                    continue
                new_facts = set(gate["support_points"]) - covered_facts
                if gate["role"] == DIFFERENTIAL and selected and not new_facts:
                    excluded.append(
                        {
                            "card_id": gate["card_id"],
                            "reason": "differential_without_new_evidence",
                        }
                    )
                    continue
                support_score = min(
                    1.0, len(gate["support_points"]) / self.config.min_primary_facts
                )
                coverage_gain = len(new_facts) / total_facts
                score = (
                    0.45 * gate["logic_relevance_score"]
                    + 0.25 * gate["dense_similarity_score"]
                    + 0.15 * support_score
                    + 0.05 * coverage_gain
                    + self._ROLE_BONUS[gate["role"]]
                )
                ranked.append((score, item))

            remaining = [
                item for item in remaining if not self._is_excluded(item, excluded)
            ]
            if not ranked:
                break
            score, chosen = max(ranked, key=lambda item: item[0])
            chosen["document_gate"]["selection_score"] = round(score, 4)
            chosen["document_gate"]["new_support_points"] = list(
                set(chosen["document_gate"]["support_points"]) - covered_facts
            )
            selected.append(chosen)
            covered_facts.update(chosen["document_gate"]["support_points"])
            remaining.remove(chosen)

        context_results = selected[: self.config.max_context_cards]
        return {
            "selected_results": selected,
            "decision_results": selected,
            "context_results": context_results,
            "selected_count": len(selected),
            "context_count": len(context_results),
            "excluded_cards": excluded,
            "covered_support_points": sorted(covered_facts),
            "selection_policy": "optimized_logic_fact_coverage",
        }

    def _matches(
        self, metadata: dict[str, Any], profile: dict[str, Any]
    ) -> list[dict[str, str]]:
        evidence = [
            (str(value), "metadata", "")
            for value in metadata.get("key_symptoms", []) or []
        ]
        found: dict[str, dict[str, str]] = {}
        for fact in profile.get("facts", []):
            patient_fact = str(fact["text"])
            for card_evidence, source, section_id in evidence:
                method = self._match(patient_fact, card_evidence)
                if not method:
                    continue
                key = FactProfileBuilder.normalize(patient_fact)
                if key not in found:
                    found[key] = {
                        "patient_fact": patient_fact,
                        "card_evidence": card_evidence[:240],
                        "method": method,
                        "source": source,
                        "section_id": section_id,
                    }
        return list(found.values())

    def _match(self, fact: str, evidence: str) -> str:
        first = FactProfileBuilder.normalize(fact)
        second = FactProfileBuilder.normalize(evidence)
        if not first or not second:
            return ""
        if first in second or second in first:
            return "exact"
        if any(
            any(FactProfileBuilder.normalize(item) in first for item in group)
            and any(FactProfileBuilder.normalize(item) in second for item in group)
            for group in self._ALIASES
        ):
            return "alias"
        if "脉" in first + second and any(
            token in first and token in second for token in self._PULSE
        ):
            return "pulse_descriptor"
        return ""

    def _is_near_duplicate(
        self, item: dict[str, Any], selected: list[dict[str, Any]]
    ) -> bool:
        return any(
            SequenceMatcher(
                None, str(item.get("content", "")), str(other.get("content", ""))
            ).ratio()
            >= self.config.duplicate_similarity
            for other in selected
        )

    @staticmethod
    def _is_excluded(item: dict[str, Any], excluded: list[dict[str, str]]) -> bool:
        card_id = item["document_gate"]["card_id"]
        return any(entry["card_id"] == card_id for entry in excluded)

    @staticmethod
    def _conflicts(
        metadata: dict[str, Any], attributes: dict[str, str]
    ) -> list[str]:
        opposite = {
            "cold_heat": {"寒": "热", "热": "寒"},
            "deficiency_excess": {"虚": "实", "实": "虚"},
            "exterior_interior": {"表": "里", "里": "表"},
        }
        return [
            field
            for field, values in opposite.items()
            if attributes.get(field) in values
            and str(metadata.get(field, "")) == values[attributes[field]]
        ]

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
