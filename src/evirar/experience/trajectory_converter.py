"""
轨迹转换模块（Trajectory Converter）

将原始对话轨迹JSON（GRPO训练产物）转换为经验库所需的结构化经验记录格式。

原始格式包含 sample_id, department, ground_truth_diagnosis, trajectories 等字段；
目标格式遵循 TCM_BianZheng_RAE_Proposal.md 5.2.1 节的经验记录结构。
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from evirar.experience import config

logger = logging.getLogger(__name__)


def load_raw_trajectories(file_path: str) -> list[dict[str, Any]]:
    """
    加载原始轨迹JSON文件。

    Args:
        file_path: 原始轨迹JSON文件路径

    Returns:
        原始轨迹数据列表
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"原始轨迹文件不存在: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("原始轨迹文件应为JSON数组格式")

    logger.info("加载原始轨迹文件: %s, 共 %d 条记录", file_path, len(data))
    return data


def _extract_patient_profile(sample: dict[str, Any]) -> dict[str, Any]:
    """
    从原始样本中提取患者档案信息。

    Args:
        sample: 原始样本数据

    Returns:
        患者档案字典
    """
    chief_complaint = sample.get("patient_opening", "")
    syndrome_label = sample.get("ground_truth_diagnosis", "")

    # 从ground_truth_evidence中提取关键症状（简化处理：按顿号/逗号分割）
    evidence = sample.get("ground_truth_evidence", "")
    key_symptoms = _extract_key_symptoms_from_evidence(evidence)

    return {
        "chief_complaint": chief_complaint,
        "syndrome_label": syndrome_label,
        "key_symptoms": key_symptoms,
        "department": sample.get("department", ""),
        "tongue": "",
        "pulse": "",
    }


def _extract_key_symptoms_from_evidence(evidence: str) -> list[str]:
    """
    从ground_truth_evidence文本中提取关键症状词。

    简化处理：通过中文标点分割后取短片段作为症状关键词。
    """
    if not evidence:
        return []

    # 以中文标点（逗号、句号、分号、顿号等）分割
    segments = re.split(r"[，。；、…]+", evidence)
    symptoms = []
    for seg in segments:
        seg = seg.strip()
        # 保留长度在 2~20 的片段作为症状/证据关键词
        if 2 <= len(seg) <= 20:
            symptoms.append(seg)

    return symptoms


def _convert_dialogue_history(
    raw_dialogue: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    将原始对话历史转换为结构化的轮次格式。

    原始格式: [{"role": "user"/"assistant", "content": "..."}]
    目标格式: [{"turn": 1, "doctor": "...", "patient": "..."}]
    """
    turns: list[dict[str, Any]] = []
    turn_num = 0
    current_turn: dict[str, Any] = {}

    for msg in raw_dialogue:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            # 患者/助理的发言
            if current_turn.get("doctor"):
                # 上一轮的医生发言已存在，开始新一轮
                turns.append(current_turn)
                current_turn = {}
            turn_num += 1
            current_turn["turn"] = turn_num
            current_turn["patient"] = content

        elif role == "assistant":
            # 医生的发言
            if not current_turn.get("turn"):
                # 如果还没有患者发言，这是医生的主动发言
                turn_num += 1
                current_turn["turn"] = turn_num
                current_turn["patient"] = ""
            current_turn["doctor"] = content

    # 追加最后一轮
    if current_turn and (current_turn.get("doctor") or current_turn.get("patient")):
        current_turn.setdefault("doctor", "")
        current_turn.setdefault("patient", "")
        turns.append(current_turn)

    return turns


def _extract_diagnosis_result(
    sample: dict[str, Any], trajectory: dict[str, Any]
) -> dict[str, Any]:
    """
    从轨迹数据中提取诊断结果。

    Args:
        sample: 原始样本（含ground_truth）
        trajectory: 单条轨迹

    Returns:
        诊断结果字典
    """
    return {
        "syndrome": sample.get("ground_truth_diagnosis", ""),
        "reasoning": sample.get("ground_truth_evidence", ""),
        "final_output": trajectory.get("final_diagnosis_output", ""),
        "confidence": 0.0,  # 原始数据无此字段，默认为0
    }


def _compute_reward_scores(trajectory: dict[str, Any]) -> dict[str, float]:
    """
    从轨迹分数计算多维度奖励评分。

    原始数据中仅有一个 score 字段，将其作为总评分，
    并均匀分配到各维度（因原始数据无细粒度评分）。

    Args:
        trajectory: 单条轨迹数据

    Returns:
        奖励评分字典
    """
    total_score = float(trajectory.get("score", 0.0))

    return {
        "process_score": total_score,
        "result_score": total_score,
        "reasoning_score": total_score,
        "total_score": total_score,
    }


def _generate_syndrome_tags(sample: dict[str, Any]) -> list[str]:
    """
    从样本数据生成证候标签。

    基于科室、诊断结果等信息生成标签。
    """
    tags = []

    department = sample.get("department", "")
    if department:
        tags.append(department)

    diagnosis = sample.get("ground_truth_diagnosis", "")
    if diagnosis:
        # 从诊断文本中提取中医证候关键词
        # 常见证候关键词
        syndrome_keywords = [
            "风热", "风寒", "湿热", "痰湿", "血瘀", "气滞", "阴虚", "阳虚",
            "气虚", "血虚", "肝郁", "脾虚", "肾虚", "心火", "肺热",
            "表证", "里证", "寒证", "热证", "虚证", "实证",
            "肝经", "心经", "脾经", "肺经", "肾经",
            "毒邪", "瘀血", "痰饮", "食积", "水湿",
        ]
        for keyword in syndrome_keywords:
            if keyword in diagnosis:
                tags.append(keyword)

    return list(set(tags))  # 去重


def convert_single_trajectory(
    sample: dict[str, Any],
    trajectory: dict[str, Any],
    experience_id: str,
) -> dict[str, Any]:
    """
    将单条原始轨迹转换为结构化经验记录。

    Args:
        sample: 原始样本数据（含患者信息、ground_truth等）
        trajectory: 单条轨迹数据
        experience_id: 经验记录ID

    Returns:
        结构化经验记录字典
    """
    patient_profile = _extract_patient_profile(sample)
    raw_dialogue = trajectory.get("dialogue_history", [])
    dialogue_trajectory = _convert_dialogue_history(raw_dialogue)
    diagnosis_result = _extract_diagnosis_result(sample, trajectory)
    reward_scores = _compute_reward_scores(trajectory)
    syndrome_tags = _generate_syndrome_tags(sample)

    return {
        "experience_id": experience_id,
        "patient_profile": patient_profile,
        "dialogue_trajectory": dialogue_trajectory,
        "diagnosis_result": diagnosis_result,
        "reward_scores": reward_scores,
        "key_inquiry_actions": [],  # 需LLM分析，转换时留空
        "embedding": [],  # 需调用Embedding API生成，转换时留空
        "syndrome_tags": syndrome_tags,
    }


def convert_trajectories(
    raw_data: list[dict[str, Any]],
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    批量转换原始轨迹数据为结构化经验记录。

    仅保留总评分超过阈值θ的高分轨迹。

    Args:
        raw_data: 原始轨迹数据列表
        score_threshold: 评分阈值，默认使用配置值

    Returns:
        结构化经验记录列表
    """
    threshold = score_threshold if score_threshold is not None else config.SCORE_THRESHOLD
    experiences: list[dict[str, Any]] = []
    total_trajectories = 0
    filtered_count = 0

    for sample in raw_data:
        sample_id = sample.get("sample_id", "unknown")
        trajectories = sample.get("trajectories", [])

        for traj in trajectories:
            total_trajectories += 1
            traj_id = traj.get("trajectory_id", 0)
            score = float(traj.get("score", 0.0))

            # 评分筛选
            if score < threshold:
                filtered_count += 1
                logger.debug(
                    "跳过低分轨迹: sample_id=%s, trajectory_id=%d, score=%.1f (阈值=%.1f)",
                    sample_id, traj_id, score, threshold,
                )
                continue

            experience_id = f"exp_{sample_id}_{traj_id}"
            experience = convert_single_trajectory(sample, traj, experience_id)
            experiences.append(experience)

    logger.info(
        "轨迹转换完成: 总计 %d 条轨迹, 通过筛选 %d 条 (过滤 %d 条, 阈值=%.1f)",
        total_trajectories, len(experiences), filtered_count, threshold,
    )

    return experiences


def convert_trajectory_file(
    input_path: str,
    output_path: str | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    转换原始轨迹文件为经验库格式并保存。

    Args:
        input_path: 输入的原始轨迹JSON文件路径
        output_path: 输出的经验库JSON文件路径（可选，默认自动生成）
        score_threshold: 评分阈值

    Returns:
        转换后的经验记录列表
    """
    raw_data = load_raw_trajectories(input_path)
    experiences = convert_trajectories(raw_data, score_threshold)

    if output_path is None:
        input_file = Path(input_path)
        output_path = str(
            input_file.parent / f"{input_file.stem}_experiences.json"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experiences, f, ensure_ascii=False, indent=2)

    logger.info("经验库文件已保存: %s (共 %d 条经验)", output_path, len(experiences))
    return experiences
