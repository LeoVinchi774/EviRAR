"""
医案经验 Markdown 文件语义检索模块（Md Experience Retriever）

基于 Embedding 向量进行语义相似度检索，
从 MD 格式的医案经验库中检索与当前问诊最相关的经验。
"""

import logging
import math
from typing import Any

from evirar.experience.md_experience_store import MdExperienceStore
from evirar.knowledge.llm_client import get_embedding

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class MdExperienceRetriever:
    """
    MD 医案经验语义检索器。

    根据当前对话历史（主诉、症状等）构建查询向量，
    在 MD 经验库中检索最相关的经验。
    """

    def __init__(self, store: MdExperienceStore):
        self._store = store

    def retrieve(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        语义检索：根据查询文本检索最相关的 MD 经验。

        Args:
            query_text: 查询文本（通常为主诉 + 当前症状摘要）
            top_k: 返回的最大结果数

        Returns:
            按相似度降序排列的经验列表，每项包含
            {"experience": {...}, "similarity": float}
        """
        experiences = self._store.get_all_experiences()
        if not experiences:
            logger.warning("MD 经验库为空，无法检索")
            return []

        try:
            query_embedding = get_embedding(query_text)
        except Exception:
            logger.exception("查询文本 Embedding 生成失败")
            return []

        scored: list[dict[str, Any]] = []
        for exp in experiences:
            emb = exp.get("embedding", [])
            if not emb:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            scored.append({"experience": exp, "similarity": sim})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        results = scored[:top_k]

        if results:
            logger.info(
                "MD 经验检索完成: 返回 Top-%d (最高相似度: %.3f)",
                len(results), results[0]["similarity"],
            )

        return results

    def retrieve_from_dialogue(
        self,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        从对话历史构建查询并检索经验。

        Args:
            dialogue_history: 对话历史
            clinical_features: 已提取的临床特征
            top_k: 返回的最大结果数

        Returns:
            检索结果列表
        """
        query = self._build_query_from_dialogue(
            dialogue_history, clinical_features
        )
        if not query:
            return []
        return self.retrieve(query, top_k)

    @staticmethod
    def _build_query_from_dialogue(
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any] | None = None,
    ) -> str:
        """从对话历史和临床特征构建检索查询文本。"""
        parts: list[str] = []

        # 提取患者发言（主诉和症状描述）
        for msg in dialogue_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("patient", "user") and content:
                parts.append(content)

        # 补充已提取的临床特征
        if clinical_features:
            if clinical_features.get("chief_complaint"):
                parts.append(clinical_features["chief_complaint"])
            for symptom in clinical_features.get("symptoms", []):
                parts.append(symptom)
            tongue = clinical_features.get("tongue", {})
            if isinstance(tongue, dict):
                for k in ("quality", "coating"):
                    if tongue.get(k):
                        parts.append(tongue[k])
            elif isinstance(tongue, str) and tongue:
                parts.append(tongue)
            if clinical_features.get("pulse"):
                parts.append(clinical_features["pulse"])

        return "，".join(parts) if parts else ""
