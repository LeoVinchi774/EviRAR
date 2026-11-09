"""
共享工具函数模块。

提供跨模块复用的工具函数，避免代码重复。
"""

from typing import Any


def format_clinical_features(features: dict[str, Any]) -> str:
    """
    将临床特征字典格式化为文本描述。

    供 syndrome_labeler、logic_validator、knowledge_retriever 等模块共用。

    Args:
        features: 临床特征字典（由ClinicalFeatureExtractor输出）

    Returns:
        格式化后的临床特征文本
    """
    lines = []

    if features.get("chief_complaint") and features["chief_complaint"] != "未提及":
        lines.append(f"主诉：{features['chief_complaint']}")

    if features.get("symptoms"):
        lines.append(f"症状：{', '.join(features['symptoms'])}")

    tongue = features.get("tongue", {})
    if tongue.get("quality") and tongue["quality"] != "未提及":
        lines.append(f"舌质：{tongue['quality']}")
    if tongue.get("coating") and tongue["coating"] != "未提及":
        lines.append(f"舌苔：{tongue['coating']}")

    if features.get("pulse") and features["pulse"] != "未提及":
        lines.append(f"脉象：{features['pulse']}")

    if features.get("inspection"):
        lines.append(f"望诊：{', '.join(features['inspection'])}")

    if features.get("inquiry"):
        lines.append(f"问诊：{', '.join(features['inquiry'])}")

    if features.get("disease_course") and features["disease_course"] != "未提及":
        lines.append(f"病程：{features['disease_course']}")

    if features.get("constitution") and features["constitution"] != "未提及":
        lines.append(f"体质与既往史：{features['constitution']}")

    return "\n".join(lines) if lines else "（临床特征信息不足）"
