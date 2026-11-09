"""
三维评判器模块（Three-Dimensional Critics）

实现多维自省诊断精炼机制中的三个评判器：
- EvidenceConsistencyCritic: 证据一致性评判
- TCMTheoryConsistencyCritic: 中医理论一致性评判
- DifferentialDiagnosisCritic: 鉴别诊断评判

对应 TCM_BianZheng_RAE_Proposal.md 5.4.2-5.4.3 节。
"""

import json
import logging
from typing import Any

from evirar.knowledge.llm_client import call_llm
from evirar.reflection import config
from evirar.reflection.prompt_templates import (
    DIFFERENTIAL_DIAGNOSIS_CRITIC_PROMPT,
    EVIDENCE_CONSISTENCY_CRITIC_PROMPT,
    TCM_THEORY_CONSISTENCY_CRITIC_PROMPT,
)

logger = logging.getLogger(__name__)


def _parse_critic_json(response: str) -> dict[str, Any]:
    """
    从LLM响应中解析JSON。

    尝试多种方式提取JSON：先找```json代码块，再找```代码块，最后直接解析。

    Args:
        response: LLM返回的原始文本

    Returns:
        解析后的字典
    """
    text = response.strip()

    # 尝试提取JSON代码块
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("评判器JSON解析失败，使用默认值。原始响应: %s", text[:200])
        return {
            "suggestions": [],
            "score": config.DEFAULT_CRITIC_SCORE,
            "analysis": text,
        }


def _clamp_score(score: Any) -> float:
    """将评分限制在有效范围内。"""
    try:
        val = float(score)
    except (TypeError, ValueError):
        return config.DEFAULT_CRITIC_SCORE
    return max(config.SCORE_MIN, min(config.SCORE_MAX, val))


def _format_dialogue(dialogue_history: list[dict[str, str]]) -> str:
    """将对话历史格式化为可读文本。"""
    if not dialogue_history:
        return "（无对话记录）"
    lines: list[str] = []
    for turn in dialogue_history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        label = "患者" if role == "patient" else "医生"
        lines.append(f"{label}：{content}")
    return "\n".join(lines)


def _format_features(clinical_features: dict[str, Any]) -> str:
    """将临床特征格式化为可读文本。"""
    if not clinical_features:
        return "（无临床特征）"
    parts: list[str] = []
    if clinical_features.get("chief_complaint"):
        parts.append(f"主诉：{clinical_features['chief_complaint']}")
    symptoms = clinical_features.get("symptoms", [])
    if symptoms:
        parts.append(f"症状：{', '.join(symptoms)}")
    tongue = clinical_features.get("tongue", {})
    if tongue:
        quality = tongue.get("quality", "")
        coating = tongue.get("coating", "")
        if quality or coating:
            parts.append(f"舌象：舌质{quality}，苔{coating}")
    pulse = clinical_features.get("pulse", "")
    if pulse:
        parts.append(f"脉象：{pulse}")
    return "\n".join(parts) if parts else str(clinical_features)


def _format_hypothesis_list(hypothesis_list: list[dict[str, Any]]) -> str:
    """将假设列表格式化为可读文本。"""
    if not hypothesis_list:
        return "（无候选假设）"
    lines: list[str] = []
    for i, h in enumerate(hypothesis_list, 1):
        name = h.get("syndrome_name", "未知")
        conf = h.get("confidence", 0.0)
        supporting = h.get("supporting_evidence", [])
        opposing = h.get("opposing_evidence", [])
        line = f"{i}. {name}（置信度：{conf:.2f}）"
        if supporting:
            line += f"\n   支持证据：{', '.join(supporting)}"
        if opposing:
            line += f"\n   反对证据：{', '.join(opposing)}"
        lines.append(line)
    return "\n".join(lines)


# ============================================================
# 证据一致性评判器
# ============================================================


class EvidenceConsistencyCritic:
    """
    证据一致性评判器。

    检查诊断结果中的辨证依据是否与对话中获取的临床特征一致：
    - 幻觉检测：诊断中引用但对话中未提及的症状
    - 遗漏检测：对话中提及但诊断中未纳入的重要症状
    """

    def evaluate(
        self,
        diagnosis: str,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        执行证据一致性评判。

        Args:
            diagnosis: 待审核的诊断文本
            dialogue_history: 问诊对话记录
            clinical_features: 从对话中提取的临床特征

        Returns:
            评判结果字典，包含 score、suggestions、details
        """
        logger.info("[自省-维度一] 证据一致性评判...")

        prompt = EVIDENCE_CONSISTENCY_CRITIC_PROMPT.format(
            dialogue_history=_format_dialogue(dialogue_history),
            extracted_features=_format_features(clinical_features),
            diagnosis=diagnosis,
        )

        response = call_llm(
            prompt=prompt,
            system_message="你是一位严格的中医诊断质量审核专家。请以JSON格式输出。",
            temperature=config.CRITIC_TEMPERATURE,
            max_tokens=config.CRITIC_MAX_TOKENS,
        )

        result = _parse_critic_json(response)
        score = _clamp_score(result.get("score", config.DEFAULT_CRITIC_SCORE))
        suggestions = result.get("suggestions", [])

        logger.info(
            "[自省-维度一] 评分: %.1f/10, 修改建议数: %d",
            score,
            len(suggestions),
        )

        return {
            "dimension": "evidence_consistency",
            "dimension_name": "证据一致性",
            "score": score,
            "suggestions": suggestions,
            "hallucinated_symptoms": result.get("hallucinated_symptoms", []),
            "missing_symptoms": result.get("missing_symptoms", []),
            "analysis": result.get("analysis", ""),
        }


# ============================================================
# 中医理论一致性评判器
# ============================================================


class TCMTheoryConsistencyCritic:
    """
    中医理论一致性评判器。

    检查诊断结果是否符合中医辨证理论：
    - 证候名称是否规范
    - 症状-证候映射是否正确
    - 舌脉与证候是否吻合
    - 是否存在寒热矛盾、虚实矛盾等逻辑错误
    - 【新增】八纲辨证一致性校验：诊断的八纲属性是否自洽

    利用知识库中对应证候的知识卡片进行验证。
    """

    def evaluate(
        self,
        diagnosis: str,
        kb_knowledge: str,
        hypothesis_list: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        执行中医理论一致性评判（含八纲一致性校验）。

        Args:
            diagnosis: 待审核的诊断文本
            kb_knowledge: 知识库中相关证候的知识卡片内容
            hypothesis_list: 候选证候假设列表（可选，用于八纲一致性校验）

        Returns:
            评判结果字典，包含 score、suggestions、details
        """
        logger.info("[自省-维度二] 中医理论一致性评判...")

        # 构建八纲属性附加信息
        bagang_context = self._build_bagang_context(hypothesis_list)

        prompt = TCM_THEORY_CONSISTENCY_CRITIC_PROMPT.format(
            diagnosis=diagnosis,
            kb_knowledge=kb_knowledge if kb_knowledge else "（无相关知识卡片）",
            hypothesis_list=_format_hypothesis_list(hypothesis_list or []),
        )

        # 如果有八纲信息，在prompt后追加
        if bagang_context:
            prompt += f"\n\n## 补充：八纲辨证属性参考\n{bagang_context}\n\n请在评判中额外检查：\n- 诊断结果的八纲属性（表里/寒热/虚实/阴阳）是否与症状和证候一致\n- 不同证候假设之间是否存在八纲矛盾\n- 如存在兼证，八纲属性是否兼容"

        response = call_llm(
            prompt=prompt,
            system_message="你是一位精通中医辨证论治理论的专家。请以JSON格式输出。",
            temperature=config.CRITIC_TEMPERATURE,
            max_tokens=config.CRITIC_MAX_TOKENS,
        )

        result = _parse_critic_json(response)
        score = _clamp_score(result.get("score", config.DEFAULT_CRITIC_SCORE))
        suggestions = result.get("suggestions", [])

        logger.info(
            "[自省-维度二] 评分: %.1f/10, 修改建议数: %d",
            score,
            len(suggestions),
        )

        return {
            "dimension": "tcm_theory_consistency",
            "dimension_name": "中医理论一致性",
            "score": score,
            "suggestions": suggestions,
            "syndrome_name_valid": result.get("syndrome_name_valid", True),
            "contradictions": result.get("contradictions", []),
            "pathogenesis_valid": result.get("pathogenesis_valid", True),
            "bagang_consistent": result.get("bagang_consistent", True),
            "analysis": result.get("analysis", ""),
        }

    @staticmethod
    def _build_bagang_context(
        hypothesis_list: list[dict[str, Any]] | None,
    ) -> str:
        """从假设列表中提取八纲属性信息。"""
        if not hypothesis_list:
            return ""

        lines = []
        for h in hypothesis_list:
            bagang = h.get("bagang", {})
            if not bagang:
                continue
            name = h.get("syndrome_name", "未知")
            parts = []
            for dim in ["表里", "寒热", "虚实", "阴阳"]:
                val = bagang.get(dim, "未定")
                if val != "未定":
                    parts.append(f"{dim}:{val}")
            if parts:
                lines.append(f"- {name}: {'、'.join(parts)}")

        return "\n".join(lines) if lines else ""


# ============================================================
# 鉴别诊断评判器
# ============================================================


class DifferentialDiagnosisCritic:
    """
    鉴别诊断评判器。

    检查诊断结果是否充分考虑了相似证候的鉴别：
    - 排名第2-3的假设是否被合理排除
    - 排除理由是否充分有据
    - 是否考虑了兼证或变证的可能性

    利用经验库中类似但最终诊断不同的案例进行对比。
    """

    def evaluate(
        self,
        diagnosis: str,
        hypothesis_list: list[dict[str, Any]],
        differential_experiences: str,
    ) -> dict[str, Any]:
        """
        执行鉴别诊断评判。

        Args:
            diagnosis: 待审核的诊断文本
            hypothesis_list: 候选证候假设列表（含置信度）
            differential_experiences: 类似但诊断不同的参考案例文本

        Returns:
            评判结果字典，包含 score、suggestions、details
        """
        logger.info("[自省-维度三] 鉴别诊断评判...")

        prompt = DIFFERENTIAL_DIAGNOSIS_CRITIC_PROMPT.format(
            diagnosis=diagnosis,
            hypothesis_list=_format_hypothesis_list(hypothesis_list),
            differential_experiences=(
                differential_experiences
                if differential_experiences
                else "（无参考案例）"
            ),
        )

        response = call_llm(
            prompt=prompt,
            system_message="你是一位擅长鉴别诊断的中医师。请以JSON格式输出。",
            temperature=config.CRITIC_TEMPERATURE,
            max_tokens=config.CRITIC_MAX_TOKENS,
        )

        result = _parse_critic_json(response)
        score = _clamp_score(result.get("score", config.DEFAULT_CRITIC_SCORE))
        suggestions = result.get("suggestions", [])

        logger.info(
            "[自省-维度三] 评分: %.1f/10, 修改建议数: %d",
            score,
            len(suggestions),
        )

        return {
            "dimension": "differential_diagnosis",
            "dimension_name": "鉴别诊断",
            "score": score,
            "suggestions": suggestions,
            "differentiation_clear": result.get("differentiation_clear", True),
            "exclusion_reasoning_sufficient": result.get(
                "exclusion_reasoning_sufficient", True
            ),
            "concurrent_syndrome_considered": result.get(
                "concurrent_syndrome_considered", False
            ),
            "analysis": result.get("analysis", ""),
        }
