"""
经验Prompt格式化模块（Prompt Formatter）

将检索到的经验记录格式化为结构化Prompt文本，
注入到医生模型的提示中。

对应 TCM_BianZheng_RAE_Proposal.md 5.2.2 节的经验使用方式。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def format_experience_prompt(
    retrieved_results: list[dict[str, Any]],
    stage: str = "initial",
    max_experiences: int = 3,
) -> str:
    """
    将检索到的经验记录格式化为Prompt文本。

    Args:
        retrieved_results: 检索结果列表（每项含 "experience" 键）
        stage: 检索阶段 ("initial"/"strategy"/"diagnostic")
        max_experiences: 最大展示经验数量

    Returns:
        格式化后的经验参考Prompt文本
    """
    if not retrieved_results:
        return ""

    results = retrieved_results[:max_experiences]

    lines = [
        "[经验参考]",
        "以下是与当前患者类似的高质量诊断案例，供参考：",
        "",
    ]

    for idx, result in enumerate(results, 1):
        exp = result.get("experience", result)
        exp_text = _format_single_experience(exp, idx, stage)
        lines.append(exp_text)
        lines.append("")

    lines.append("请结合以上经验和您的专业判断进行问诊。")

    return "\n".join(lines)


def _format_single_experience(
    experience: dict[str, Any],
    index: int,
    stage: str,
) -> str:
    """
    格式化单条经验记录。

    Args:
        experience: 经验记录字典
        index: 序号
        stage: 检索阶段

    Returns:
        格式化文本
    """
    reward_scores = experience.get("reward_scores", {})
    total_score = reward_scores.get("total_score", 0.0)
    profile = experience.get("patient_profile", {})
    diagnosis = experience.get("diagnosis_result", {})

    lines = [f"经验{index}（评分: {total_score:.1f}/10）："]

    # 患者主诉
    chief_complaint = profile.get("chief_complaint", "")
    if chief_complaint:
        lines.append(f"  - 患者主诉：{chief_complaint}")

    if stage == "initial":
        # 初期阶段：展示关键症状和科室
        key_symptoms = profile.get("key_symptoms", [])
        if key_symptoms:
            lines.append(f"  - 关键症状：{'、'.join(key_symptoms[:5])}")

        department = profile.get("department", "")
        if department:
            lines.append(f"  - 科室：{department}")

    elif stage == "strategy":
        # 中期阶段：展示关键问诊动作
        key_actions = experience.get("key_inquiry_actions", [])
        if key_actions:
            for action in key_actions[:3]:
                turn = action.get("turn", "?")
                action_type = action.get("action_type", "问诊")
                info_gain = action.get("info_gain", "")
                gain_text = f"（信息增益: {info_gain}）" if info_gain else ""
                lines.append(
                    f"  - 关键问诊动作：第{turn}轮询问了{action_type}，获取了关键信息{gain_text}"
                )
        else:
            # 无标注的关键动作时，展示对话摘要
            dialogue = experience.get("dialogue_trajectory", [])
            _append_dialogue_summary(lines, dialogue)

    elif stage == "diagnostic":
        # 后期阶段：展示诊断结果和推理过程
        syndrome = diagnosis.get("syndrome", "")
        if syndrome:
            lines.append(f"  - 最终诊断：{syndrome}")

        reasoning = diagnosis.get("reasoning", "")
        if reasoning:
            # 截取前200字
            reasoning_text = reasoning[:200]
            if len(reasoning) > 200:
                reasoning_text += "..."
            lines.append(f"  - 诊断依据：{reasoning_text}")

        # 展示关键症状
        key_symptoms = profile.get("key_symptoms", [])
        if key_symptoms:
            lines.append(f"  - 关键症状：{'、'.join(key_symptoms[:5])}")
    else:
        # 默认：展示关键信息
        syndrome = diagnosis.get("syndrome", "")
        if syndrome:
            lines.append(f"  - 最终诊断：{syndrome}")

    return "\n".join(lines)


def _append_dialogue_summary(
    lines: list[str], dialogue: list[dict[str, Any]]
) -> None:
    """添加对话摘要到输出行。"""
    if not dialogue:
        return

    # 展示前几轮的医生问诊内容
    doctor_questions = []
    for turn_data in dialogue[:5]:
        doctor_text = turn_data.get("doctor", "")
        if doctor_text:
            turn_num = turn_data.get("turn", "?")
            # 截取前50字
            short_text = doctor_text[:50]
            if len(doctor_text) > 50:
                short_text += "..."
            doctor_questions.append(f"第{turn_num}轮: {short_text}")

    if doctor_questions:
        lines.append("  - 问诊策略摘要：")
        for q in doctor_questions:
            lines.append(f"    · {q}")


def format_strategy_prompt(
    retrieved_results: list[dict[str, Any]],
    current_features: dict[str, Any] | None = None,
) -> str:
    """
    格式化策略参考Prompt（专用于中期阶段）。

    Args:
        retrieved_results: 阶段二检索结果
        current_features: 当前已获取的临床特征

    Returns:
        策略参考Prompt文本
    """
    if not retrieved_results:
        return ""

    lines = [
        "[策略参考]",
        "根据当前已收集的临床信息，以下高分经验中的问诊策略值得参考：",
        "",
    ]

    for idx, result in enumerate(retrieved_results[:3], 1):
        exp = result.get("experience", result)
        dialogue = exp.get("dialogue_trajectory", [])
        reward_scores = exp.get("reward_scores", {})
        total_score = reward_scores.get("total_score", 0.0)

        lines.append(f"参考经验{idx}（评分: {total_score:.1f}/10）：")

        # 展示对话中医生的问诊顺序
        for turn_data in dialogue:
            doctor_text = turn_data.get("doctor", "")
            if doctor_text:
                turn_num = turn_data.get("turn", "?")
                short_text = doctor_text[:80]
                if len(doctor_text) > 80:
                    short_text += "..."
                lines.append(f"  第{turn_num}轮医生问：{short_text}")

        lines.append("")

    lines.append("请参考以上经验的问诊策略，继续进行有针对性的问诊。")

    return "\n".join(lines)


def format_diagnostic_prompt(
    retrieved_results: list[dict[str, Any]],
) -> str:
    """
    格式化诊断参考Prompt（专用于后期阶段）。

    Args:
        retrieved_results: 阶段三检索结果

    Returns:
        诊断参考Prompt文本
    """
    if not retrieved_results:
        return ""

    lines = [
        "[诊断参考]",
        "以下是与当前病例相似的高分诊断案例，供诊断推理参考：",
        "",
    ]

    for idx, result in enumerate(retrieved_results[:3], 1):
        exp = result.get("experience", result)
        profile = exp.get("patient_profile", {})
        diagnosis = exp.get("diagnosis_result", {})
        reward_scores = exp.get("reward_scores", {})
        total_score = reward_scores.get("total_score", 0.0)

        lines.append(f"参考病例{idx}（评分: {total_score:.1f}/10）：")

        chief_complaint = profile.get("chief_complaint", "")
        if chief_complaint:
            lines.append(f"  - 患者主诉：{chief_complaint}")

        key_symptoms = profile.get("key_symptoms", [])
        if key_symptoms:
            lines.append(f"  - 关键症状：{'、'.join(key_symptoms[:8])}")

        tongue = profile.get("tongue", "")
        if tongue:
            lines.append(f"  - 舌象：{tongue}")

        pulse = profile.get("pulse", "")
        if pulse:
            lines.append(f"  - 脉象：{pulse}")

        syndrome = diagnosis.get("syndrome", "")
        if syndrome:
            lines.append(f"  - 辨证结论：{syndrome}")

        reasoning = diagnosis.get("reasoning", "")
        if reasoning:
            reasoning_text = reasoning[:300]
            if len(reasoning) > 300:
                reasoning_text += "..."
            lines.append(f"  - 辨证思路：{reasoning_text}")

        lines.append("")

    lines.append("请综合当前患者的临床表现和以上参考经验，进行辨证论治。")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# MD 格式医案经验 Prompt 格式化
# ══════════════════════════════════════════════════════════════


def format_md_experience_prompt(
    md_content: str,
    stage: str = "inquiry",
) -> str:
    """
    将 MD 格式的医案经验格式化为注入到医生模型的 Prompt 文本。

    根据问诊阶段（inquiry / diagnostic），提取不同的参考内容：
    - inquiry 阶段：提取问诊路径和关键询问要点，指导问诊方向
    - diagnostic 阶段：提取辨证逻辑链和鉴别诊断要点，辅助最终诊断

    Args:
        md_content: MD 文件的完整文本内容
        stage: 当前阶段 ("inquiry" 或 "diagnostic")

    Returns:
        格式化的经验参考 Prompt 文本
    """
    if not md_content or not md_content.strip():
        return ""

    if stage == "diagnostic":
        return _format_md_diagnostic_stage(md_content)
    return _format_md_inquiry_stage(md_content)


def _format_md_inquiry_stage(md_content: str) -> str:
    """格式化问诊阶段的 MD 经验参考 Prompt。"""
    lines = [
        "[医案经验参考——问诊策略]",
        "以下是与当前患者症状高度相关的医案经验中的问诊策略，请参考其问诊思路指导你的问诊方向：",
        "",
    ]

    # 提取"模拟问诊路径"部分的各层关键信息
    inquiry_sections = _extract_inquiry_sections(md_content)
    if inquiry_sections:
        for section_title, content_items in inquiry_sections:
            lines.append(f"**{section_title}**")
            for item in content_items:
                lines.append(f"  - {item}")
            lines.append("")
    else:
        # 回退：直接截取第一部分的内容
        section1 = _extract_section(md_content, "一")
        if section1:
            # 限制长度
            if len(section1) > 1000:
                section1 = section1[:1000] + "..."
            lines.append(section1)
            lines.append("")

    lines.append(
        "请参考以上问诊策略，结合当前已获取的信息，有针对性地继续问诊。\n"
        "注意：以上仅为参考，实际问诊应根据患者具体表现灵活调整。"
    )

    return "\n".join(lines)


def _format_md_diagnostic_stage(md_content: str) -> str:
    """格式化诊断阶段的 MD 经验参考 Prompt。"""
    lines = [
        "[医案经验参考——辨证诊断]",
        "以下是与当前患者症状高度相关的医案经验中的辨证思路，供诊断参考：",
        "",
    ]

    # 提取"辨证诊断逻辑链"部分
    diagnostic_info = _extract_diagnostic_logic(md_content)
    if diagnostic_info:
        if diagnostic_info.get("core_location"):
            lines.append(f"**核心病位**: {diagnostic_info['core_location']}")
        if diagnostic_info.get("core_nature"):
            lines.append(f"**核心病性**: {diagnostic_info['core_nature']}")
        if diagnostic_info.get("reasoning_chain"):
            lines.append("**推理过程**:")
            for step in diagnostic_info["reasoning_chain"]:
                lines.append(f"  - {step}")
        if diagnostic_info.get("conclusion"):
            lines.append(f"**辨证结论**: {diagnostic_info['conclusion']}")
        if diagnostic_info.get("differential"):
            lines.append("**鉴别要点**:")
            for diff in diagnostic_info["differential"]:
                lines.append(f"  - {diff}")
        lines.append("")
    else:
        # 回退：直接截取第二部分内容
        section2 = _extract_section(md_content, "二")
        if section2:
            if len(section2) > 1200:
                section2 = section2[:1200] + "..."
            lines.append(section2)
            lines.append("")

    lines.append(
        "请参考以上辨证思路，结合当前患者的具体临床表现进行辨证论治。\n"
        "注意：以上仅为参考案例的辨证分析，最终诊断应基于当前患者的实际情况。"
    )

    return "\n".join(lines)


def _extract_inquiry_sections(
    md_content: str,
) -> list[tuple[str, list[str]]]:
    """
    从 MD 内容中提取各层问诊策略。

    Returns:
        [(层级标题, [要点列表]), ...]
    """
    results: list[tuple[str, list[str]]] = []
    current_title = ""
    current_items: list[str] = []
    in_section1 = False

    for line in md_content.split("\n"):
        stripped = line.strip()

        # 检测第一大节开始
        if "模拟问诊路径" in stripped or ("一" in stripped and "问诊" in stripped):
            in_section1 = True
            continue

        # 检测第二大节开始（退出第一大节）
        if in_section1 and (
            "辨证诊断逻辑链" in stripped
            or (stripped.startswith("###") and "二" in stripped)
        ):
            if current_title and current_items:
                results.append((current_title, current_items))
            break

        if not in_section1:
            continue

        # 检测层级标题
        if stripped.startswith("**第") and "层" in stripped:
            if current_title and current_items:
                results.append((current_title, current_items))
            # 提取标题文本
            title = stripped.strip("*").strip()
            current_title = title
            current_items = []
            continue

        # 提取核心询问语句和医案特征提取
        if "核心询问语句" in stripped or "医案特征提取" in stripped:
            if "：" in stripped:
                content = stripped.split("：", 1)[1].strip().strip('"""')
            elif ":" in stripped:
                content = stripped.split(":", 1)[1].strip().strip('"""')
            else:
                content = stripped
            content = content.replace("**", "").strip()
            if content:
                current_items.append(content)
        elif "必查舌脉" in stripped or "关联病史追问" in stripped:
            content = stripped.lstrip("- ").replace("**", "").strip()
            if content:
                current_items.append(content)

    # 追加最后一个段落
    if current_title and current_items:
        results.append((current_title, current_items))

    return results


def _extract_diagnostic_logic(md_content: str) -> dict[str, Any]:
    """
    从 MD 内容中提取辨证诊断逻辑链的结构化信息。

    Returns:
        {
            "core_location": str,
            "core_nature": str,
            "reasoning_chain": [str],
            "conclusion": str,
            "differential": [str]
        }
    """
    result: dict[str, Any] = {}
    reasoning_chain: list[str] = []
    differential: list[str] = []
    in_section2 = False
    in_differential = False

    for line in md_content.split("\n"):
        stripped = line.strip()

        # 检测第二大节
        if "辨证诊断逻辑链" in stripped or (
            stripped.startswith("###") and "二" in stripped and "辨证" in stripped
        ):
            in_section2 = True
            continue

        if not in_section2:
            continue

        clean = stripped.lstrip("- ").replace("**", "").strip()

        if "核心病位" in clean:
            if "：" in clean:
                result["core_location"] = clean.split("：", 1)[1].strip()
            elif ":" in clean:
                result["core_location"] = clean.split(":", 1)[1].strip()

        elif "核心病性" in clean:
            if "：" in clean:
                result["core_nature"] = clean.split("：", 1)[1].strip()
            elif ":" in clean:
                result["core_nature"] = clean.split(":", 1)[1].strip()

        elif "推理公式" in clean:
            continue  # 下面的行是具体推理步骤

        elif clean.startswith("[") or "→" in clean:
            # 推理公式中的步骤
            reasoning_chain.append(clean)

        elif "辨证结论" in clean:
            if "：" in clean:
                result["conclusion"] = clean.split("：", 1)[1].strip()
            elif ":" in clean:
                result["conclusion"] = clean.split(":", 1)[1].strip()
            # 也可能在推理链中
            if "→" in clean and not result.get("conclusion"):
                result["conclusion"] = clean

        elif "鉴别要点" in clean:
            in_differential = True
            continue

        elif in_differential and clean:
            if clean.startswith("与") or "鉴别" in clean:
                # 新的鉴别项
                differential.append(clean)
            elif differential and not clean.startswith("#"):
                # 继续上一个鉴别项的内容
                differential[-1] += " " + clean

    if reasoning_chain:
        result["reasoning_chain"] = reasoning_chain
    if differential:
        result["differential"] = differential

    return result


def _extract_section(md_content: str, section_marker: str) -> str:
    """
    根据中文数字标记提取大节内容。

    Args:
        md_content: 完整 MD 内容
        section_marker: 节标记，如 "一" 或 "二"

    Returns:
        该节的文本内容
    """
    lines = md_content.split("\n")
    capturing = False
    result_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("###") and section_marker in stripped:
            capturing = True
            continue
        elif capturing and stripped.startswith("###"):
            # 遇到下一个同级标题，停止
            break
        elif capturing:
            result_lines.append(line)

    return "\n".join(result_lines).strip()


def format_md_experience_retrieval_prompt(
    retrieved_results: list[dict[str, Any]],
    stage: str = "inquiry",
    max_results: int = 2,
) -> str:
    """
    将 MD 经验检索结果格式化为 Prompt 文本。

    Args:
        retrieved_results: 检索结果列表，每项含 {"experience": {...}, "similarity": float}
        stage: 问诊阶段 ("inquiry" 或 "diagnostic")
        max_results: 最大展示结果数

    Returns:
        格式化的 Prompt 文本
    """
    if not retrieved_results:
        return ""

    results = retrieved_results[:max_results]
    parts: list[str] = []

    for idx, result in enumerate(results, 1):
        exp = result.get("experience", {})
        sim = result.get("similarity", 0.0)
        content = exp.get("content", "")

        if not content:
            continue

        header = f"--- 参考经验 {idx}（相似度: {sim:.2f}）---"
        formatted = format_md_experience_prompt(content, stage=stage)

        if formatted:
            parts.append(f"{header}\n{formatted}")

    if not parts:
        return ""

    return "\n\n".join(parts)
