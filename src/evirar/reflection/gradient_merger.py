"""
文本梯度合并模块（Gradient Merger）

综合三个维度评判器的修改建议，生成统一的"文本梯度"修改指令。

对应 TCM_BianZheng_RAE_Proposal.md 5.4.2 节的 Step 3。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def merge_critic_results(
    critic_results: list[dict[str, Any]],
) -> str:
    """
    合并三个维度评判器的结果，生成文本梯度修改指令。

    文本梯度是一组自然语言的修改指令，指导LLM对诊断进行精炼。

    Args:
        critic_results: 三个评判器返回的结果列表，每个包含
            dimension_name、score、suggestions、analysis等字段

    Returns:
        格式化的文本梯度修改指令
    """
    if not critic_results:
        return "无修改建议。"

    sections: list[str] = []
    suggestion_index = 1

    for result in critic_results:
        dimension_name = result.get("dimension_name", "未知维度")
        score = result.get("score", 0)
        suggestions = result.get("suggestions", [])
        analysis = result.get("analysis", "")

        if not suggestions and score >= 9:
            # 高分且无建议，跳过
            continue

        section_lines: list[str] = []
        section_lines.append(f"【{dimension_name}评判】（评分：{score}/10）")

        if analysis:
            section_lines.append(f"  分析：{analysis}")

        # 维度特定的详细反馈
        hallucinated = result.get("hallucinated_symptoms", [])
        if hallucinated:
            section_lines.append(
                f"  幻觉症状（诊断中有但对话中无）：{', '.join(hallucinated)}"
            )

        missing = result.get("missing_symptoms", [])
        if missing:
            section_lines.append(
                f"  遗漏症状（对话中有但诊断中未考虑）：{', '.join(missing)}"
            )

        contradictions = result.get("contradictions", [])
        if contradictions:
            section_lines.append(
                f"  理论矛盾：{', '.join(contradictions)}"
            )

        # 编号的修改建议
        for suggestion in suggestions:
            section_lines.append(f"  {suggestion_index}. {suggestion}")
            suggestion_index += 1

        sections.append("\n".join(section_lines))

    if not sections:
        return "三个维度评判均通过，无需修改。"

    header = "修改建议："
    return header + "\n\n" + "\n\n".join(sections)


def compute_aggregate_scores(
    critic_results: list[dict[str, Any]],
) -> dict[str, float]:
    """
    计算评判结果的汇总评分。

    Args:
        critic_results: 三个评判器返回的结果列表

    Returns:
        包含各维度评分和平均评分的字典
    """
    scores: dict[str, float] = {}
    total = 0.0

    for result in critic_results:
        dimension = result.get("dimension", "unknown")
        score = float(result.get("score", 0))
        scores[dimension] = score
        total += score

    count = len(critic_results) if critic_results else 1
    min_score = min(scores.values()) if scores else 0.0
    scores["average"] = total / count
    scores["min"] = min_score

    return scores
