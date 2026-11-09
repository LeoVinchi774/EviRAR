"""可选 A-D 检索与文档门控流水线。"""

from __future__ import annotations

from typing import Any

from evirar.knowledge import config
from evirar.knowledge.coverage_gate import CandidateAwareCoverageGate
from evirar.knowledge.evidence_classification_gate import (
    EvidenceClassificationConfig,
    EvidenceClassificationGate,
)
from evirar.knowledge.fact_profile import FactProfileBuilder
from evirar.knowledge.feature_extractor import ClinicalFeatureExtractor
from evirar.knowledge.knowledge_retriever import KnowledgeRetriever
from evirar.knowledge.logic_validator import LogicValidator
from evirar.knowledge.dynamic_evidence_strategy import DynamicEvidenceStrategy
from evirar.knowledge.syndrome_labeler import SyndromeLabeler, SyndromeLabels


class TCMKBRAGPipeline:
    """将原 A 约束检索、证据分类门控与 D 动作路由串联为流水线。"""

    def __init__(
        self,
        knowledge_base_dir: str | None = None,
        validation_threshold: float | None = None,
        max_results: int | None = None,
    ) -> None:
        self._extractor = ClinicalFeatureExtractor()
        self._labeler = SyndromeLabeler()
        self._retriever = KnowledgeRetriever(knowledge_base_dir)
        self._validator = LogicValidator(validation_threshold)
        self._max_results = max_results or config.MAX_RETURN_RESULTS
        self._facts = FactProfileBuilder()
        self._evidence_gate = EvidenceClassificationGate(
            EvidenceClassificationConfig(
                max_decision_cards=self._max_results,
                max_context_cards=min(2, self._max_results),
            )
        )
        self._coverage_gate = CandidateAwareCoverageGate()
        self._dynamic_strategy = DynamicEvidenceStrategy()

    def retrieve(
        self, dialogue_history: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """返回允许注入生成器的卡片；D 要求追问时返回空集。"""
        return self.retrieve_with_details(dialogue_history)["generation_results"]

    def retrieve_with_details(
        self, dialogue_history: list[dict[str, str]]
    ) -> dict[str, Any]:
        """执行流水线并返回各阶段的可追溯结果。"""
        features = self._extractor.extract(dialogue_history)

        # A 开启时复用原工程的标签生成和元数据预过滤；关闭时不调用它。
        labels = self._labeler.generate_labels(features) if config.GATE_ENABLE_A else None
        candidates = self._retriever.retrieve(
            features,
            top_k=config.RETRIEVAL_TOP_K,
            syndrome_labels=labels,
        )
        labels_detail = self._labels_to_dict(labels)
        constraint_trace = dict(self._retriever.last_constraint_trace)

        # A=0 只移除模型标签和元数据预过滤；保留原工程的逻辑校验，
        # 使各组仅在 A/B/C/D 的设计因素上不同。
        validation_labels = labels if labels is not None else SyndromeLabels()
        validated_candidates = self._validator.validate_and_rerank(
            candidates, features, validation_labels
        )
        logic_validation = {
            "enabled": True,
            "input_count": len(candidates),
            "validated_count": len(validated_candidates),
            "threshold": config.LOGIC_VALIDATION_THRESHOLD,
            "syndrome_direction_available": labels is not None,
        }

        if not config.GATE_ENABLE_EVIDENCE_CLASSIFICATION_GATE:
            return self._legacy_details(
                features,
                validated_candidates,
                labels_detail,
                constraint_trace,
                logic_validation,
            )

        profile = self._facts.build(features)
        evidence_gate = self._evidence_gate.classify_and_select(
            validated_candidates, profile
        )
        document_gate = evidence_gate["classification"]
        assessed = document_gate["assessed_results"]
        selection = evidence_gate["selection"]

        coverage: dict[str, Any] = {
            "missing_slots": [],
            "covered_slots": [],
            "coverage_score": 1.0,
            "convergence": {"converged": True, "primary_support_count": 0},
            "recommended_actions": [],
        }
        if (
            config.GATE_ENABLE_EVIDENCE_CLASSIFICATION_GATE
            or config.GATE_ENABLE_DYNAMIC_EVIDENCE_STRATEGY
        ):
            coverage = self._coverage_gate.analyze(profile, assessed)

        action = {
            "action": "generate",
            "reason": "module_d_disabled",
            "next_question": None,
        }
        if config.GATE_ENABLE_DYNAMIC_EVIDENCE_STRATEGY:
            action = self._dynamic_strategy.decide(
                selection["decision_results"], coverage
            )

        generation_results = (
            selection["context_results"]
            if action["action"] == "generate"
            else []
        )
        return {
            "step1_clinical_features": features,
            "fact_profile": profile,
            "step2_syndrome_labels": labels_detail,
            "constraint_retrieval": constraint_trace,
            "step3_retrieved_count": len(candidates),
            "step4_validated_count": len(validated_candidates),
            "logic_validation": logic_validation,
            "step5_document_gate_count": len(assessed),
            "document_gate": document_gate,
            "evidence_classification_gate": evidence_gate,
            "coverage": coverage,
            "evidence_selection": selection,
            "retrieval_action": action,
            # Keep gate-selected cards available for a controlled reference
            # context even when the action is ask_follow_up/retrieve_more.
            # They are deliberately separate from generation_results: cards
            # used to guide the next question must not be treated as evidence
            # sufficient for a diagnosis.
            "reference_results": selection["context_results"],
            "final_results": selection["selected_results"],
            "generation_results": generation_results,
            "gate_modules": self._gate_modules(),
        }

    def _legacy_details(
        self,
        features: dict[str, Any],
        candidates: list[dict[str, Any]],
        labels_detail: dict[str, Any],
        constraint_trace: dict[str, Any],
        logic_validation: dict[str, Any],
    ) -> dict[str, Any]:
        final = candidates[: self._max_results]
        return {
            "step1_clinical_features": features,
            "fact_profile": {},
            "step2_syndrome_labels": labels_detail,
            "constraint_retrieval": constraint_trace,
            "step3_retrieved_count": logic_validation["input_count"],
            "step4_validated_count": len(candidates),
            "logic_validation": logic_validation,
            "document_gate": {},
            "coverage": {},
            "evidence_selection": {},
            "retrieval_action": {
                "action": "generate",
                "reason": "evidence_classification_gate_disabled",
                "next_question": None,
            },
            "reference_results": final,
            "final_results": final,
            "generation_results": final,
            "gate_modules": self._gate_modules(),
        }

    @staticmethod
    def _labels_to_dict(labels: SyndromeLabels | None) -> dict[str, Any]:
        if labels is None:
            return {}
        return {
            "inclusion_labels": labels.inclusion_labels,
            "exclusion_labels": labels.exclusion_labels,
            "organs_involved": labels.organs_involved,
            "cold_heat": labels.cold_heat,
            "deficiency_excess": labels.deficiency_excess,
            "exterior_interior": labels.exterior_interior,
            "analysis_brief": labels.analysis_brief,
        }

    @staticmethod
    def _gate_modules() -> dict[str, bool]:
        return {
            "A": config.GATE_ENABLE_A,
            "EvidenceClassificationGate": config.GATE_ENABLE_EVIDENCE_CLASSIFICATION_GATE,
            "DynamicEvidenceStrategy": (
                config.GATE_ENABLE_DYNAMIC_EVIDENCE_STRATEGY
            ),
        }
