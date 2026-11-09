"""
诊断生成与精炼模块（Diagnosis Generator）

负责：
- 初步诊断D₀的生成（基于完整对话历史、假设状态、知识库和经验库）
- 诊断精炼D_{i+1}的生成（基于D_i和文本梯度修改指令）

对应 TCM_BianZheng_RAE_Proposal.md 5.4.2 节的 Step 1 和 Step 3。
"""

import logging
from typing import Any

from evirar.knowledge.llm_client import call_llm
from evirar.reflection import config
from evirar.reflection.prompt_templates import (
    DIAGNOSIS_REFINEMENT_PROMPT,
    INITIAL_DIAGNOSIS_PROMPT,
)

logger = logging.getLogger(__name__)


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
        line = f"{i}. {name}（置信度：{conf:.2f}）"
        supporting = h.get("supporting_evidence", [])
        if supporting:
            line += f"\n   支持证据：{', '.join(supporting)}"
        opposing = h.get("opposing_evidence", [])
        if opposing:
            line += f"\n   反对证据：{', '.join(opposing)}"
        lines.append(line)
    return "\n".join(lines)


def generate_initial_diagnosis(
    dialogue_history: list[dict[str, str]],
    clinical_features: dict[str, Any],
    hypothesis_list: list[dict[str, Any]],
    kb_knowledge: str = "",
    experience_knowledge: str = "",
) -> str:
    """
    生成初步诊断 D₀。

    基于完整对话历史、临床特征、假设状态、知识库和经验库信息，
    调用LLM生成初步诊断结果。

    Args:
        dialogue_history: 完整的问诊对话历史
        clinical_features: 提取的临床特征字典
        hypothesis_list: 来自推理树的假设列表（含置信度）
        kb_knowledge: 知识库检索文本
        experience_knowledge: 经验库检索文本

    Returns:
        初步诊断文本 D₀
    """
    logger.info("[自省-Step1] 生成初步诊断 D₀...")

    prompt = INITIAL_DIAGNOSIS_PROMPT.format(
        dialogue_history=_format_dialogue(dialogue_history),
        extracted_features=_format_features(clinical_features),
        hypothesis_list=_format_hypothesis_list(hypothesis_list),
        kb_knowledge=kb_knowledge if kb_knowledge else "（无参考知识）",
        experience_knowledge=experience_knowledge if experience_knowledge else "（无参考经验）",
    )

    diagnosis = call_llm(
        prompt=prompt,
        system_message="你是一位经验丰富的中医诊断专家。请生成完整、规范的辨证诊断结果。",
        temperature=config.DIAGNOSIS_TEMPERATURE,
        max_tokens=config.DIAGNOSIS_MAX_TOKENS,
    )

    logger.info("[自省-Step1] 初步诊断生成完成（长度: %d字符）", len(diagnosis))
    return diagnosis


def refine_diagnosis(
    original_diagnosis: str,
    text_gradient: str,
    dialogue_history: list[dict[str, str]],
    kb_knowledge: str = "",
    experience_knowledge: str = "",
    differential_experiences: str = "",
    hypothesis_list: list[dict[str, Any]] | None = None,
) -> str:
    """
    根据文本梯度修改指令精炼诊断。

    基于原始诊断和三维评判器的综合修改建议（文本梯度），
    调用LLM生成精炼后的诊断结果。

    Args:
        original_diagnosis: 当前诊断文本 D_i
        text_gradient: 综合修改指令（由gradient_merger生成）
        dialogue_history: 问诊对话记录
        kb_knowledge: 知识库检索文本
        experience_knowledge: 普通医案经验参考文本
        differential_experiences: 鉴别医案参考文本
        hypothesis_list: 候选证候假设列表

    Returns:
        精炼后的诊断文本 D_{i+1}
    """
    logger.info("[自省-Step3] 根据文本梯度精炼诊断...")

    prompt = DIAGNOSIS_REFINEMENT_PROMPT.format(
        original_diagnosis=original_diagnosis,
        text_gradient=text_gradient,
        dialogue_history=_format_dialogue(dialogue_history),
        kb_knowledge=kb_knowledge if kb_knowledge else "（无参考知识）",
        experience_knowledge=(
            experience_knowledge if experience_knowledge else "（无参考经验）"
        ),
        differential_experiences=(
            differential_experiences if differential_experiences else "（无鉴别医案）"
        ),
        hypothesis_list=_format_hypothesis_list(hypothesis_list or []),
    )

    refined = call_llm(
        prompt=prompt,
        system_message="你是一位经验丰富的中医诊断专家。请根据修改建议精炼诊断。",
        temperature=config.DIAGNOSIS_TEMPERATURE,
        max_tokens=config.DIAGNOSIS_MAX_TOKENS,
    )

    logger.info("[自省-Step3] 诊断精炼完成（长度: %d字符）", len(refined))
    return refined
