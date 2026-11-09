"""
Step 3：结构化过滤 + 语义检索模块（Knowledge Retriever）

实现两层检索机制：
- 3a. 元数据标签过滤（辨证方向约束）：根据辨证标签过滤候选知识子集
- 3b. 语义相似度检索：在候选范围内计算向量相似度，返回Top-K

对应 TCM_BianZheng_RAE_Proposal.md 中5.1.2节 Step 3。
"""

import hashlib
import json
import logging
import math
import os
from typing import Any

from evirar.knowledge import config
from evirar.cache import cache_file
from evirar.knowledge.knowledge_card_parser import KnowledgeCard, KnowledgeCardParser
from evirar.knowledge.llm_client import get_embedding, get_embeddings_batch
from evirar.knowledge.syndrome_labeler import SyndromeLabels

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度（纯Python实现，无需numpy）。"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class KnowledgeRetriever:
    """知识检索器：实现辨证约束过滤 + 语义相似度检索。"""

    # 向量缓存文件名
    _CACHE_FILENAME = ".kb_embedding_cache.json"

    def __init__(self, knowledge_base_dir: str | None = None):
        """
        初始化知识检索器。

        Args:
            knowledge_base_dir: 知识库目录路径，默认使用配置值
        """
        self._kb_dir = knowledge_base_dir or config.KNOWLEDGE_BASE_DIR
        self._parser = KnowledgeCardParser()
        self._cards: list[KnowledgeCard] = []
        self._loaded = False
        self.last_constraint_trace: dict[str, Any] = {}

    def load_knowledge_base(self) -> None:
        """加载并索引知识库中的所有知识卡片。支持向量缓存。"""
        logger.info("正在加载知识库: %s", self._kb_dir)
        self._cards = self._parser.parse_directory(self._kb_dir)

        if not self._cards:
            logger.warning("知识库为空，请先添加证候知识卡片到 %s", self._kb_dir)
            return

        # 计算知识库内容哈希
        content_hash = self._compute_content_hash()

        # 尝试从缓存加载向量
        cached_embeddings = self._load_embedding_cache(content_hash)
        if cached_embeddings is not None:
            loaded_count = 0
            for card in self._cards:
                emb = cached_embeddings.get(card.file_path)
                if emb:
                    card.embedding = emb
                    loaded_count += 1
            if loaded_count == len(self._cards):
                logger.info(
                    "从缓存加载了 %d 张知识卡片的向量嵌入", loaded_count
                )
                self._loaded = True
                return
            logger.info(
                "缓存部分命中 (%d/%d)，重新生成缺失的向量嵌入",
                loaded_count, len(self._cards),
            )

        # 缓存未命中或不完整，生成向量嵌入
        texts = [card.content for card in self._cards]
        logger.info("正在为 %d 张知识卡片生成向量嵌入...", len(texts))
        embeddings = get_embeddings_batch(texts)
        for card, emb in zip(self._cards, embeddings):
            card.embedding = emb

        # 保存向量缓存
        self._save_embedding_cache(content_hash)

        self._loaded = True
        logger.info("知识库加载完成，共 %d 张知识卡片", len(self._cards))

    def _compute_content_hash(self) -> str:
        """计算知识库所有卡片内容的SHA256哈希值，用于检测内容变化。"""
        hasher = hashlib.sha256()
        for card in sorted(self._cards, key=lambda c: c.file_path):
            hasher.update(card.file_path.encode("utf-8"))
            hasher.update(card.content.encode("utf-8"))
        return hasher.hexdigest()

    def _get_cache_path(self) -> str:
        """获取向量缓存文件路径。"""
        return str(cache_file(self._CACHE_FILENAME))

    def _load_embedding_cache(self, content_hash: str) -> dict[str, list[float]] | None:
        """
        加载向量缓存。

        仅当缓存文件存在、内容哈希匹配且嵌入模型一致时返回缓存数据。

        Returns:
            file_path -> embedding 的映射字典，若缓存无效则返回 None
        """
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            logger.debug("向量缓存文件不存在: %s", cache_path)
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("向量缓存文件读取失败，将重新生成")
            return None

        if cache_data.get("content_hash") != content_hash:
            logger.info("知识库内容已变更，缓存失效")
            return None

        if cache_data.get("embedding_model") != config.EMBEDDING_MODEL:
            logger.info("嵌入模型已变更，缓存失效")
            return None

        logger.debug("向量缓存验证通过")
        return cache_data.get("embeddings", {})

    def _save_embedding_cache(self, content_hash: str) -> None:
        """将当前知识卡片的向量嵌入保存为缓存文件。"""
        cache_data = {
            "content_hash": content_hash,
            "embedding_model": config.EMBEDDING_MODEL,
            "embeddings": {
                card.file_path: card.embedding
                for card in self._cards
                if card.embedding
            },
        }

        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
            logger.info("向量缓存已保存: %s", cache_path)
        except OSError:
            logger.warning("向量缓存保存失败: %s", cache_path)

    def retrieve(
        self,
        clinical_features: dict[str, Any],
        syndrome_labels: SyndromeLabels | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        执行辨证约束检索：元数据过滤 + 语义相似度排序。

        Args:
            clinical_features: 临床特征字典（Step 1输出）
            syndrome_labels: 辨证标签（Step 2输出）
            top_k: 返回的Top-K数量，默认使用配置值

        Returns:
            排序后的知识条目列表，每项包含知识卡片信息和相似度评分
        """
        if not self._loaded:
            self.load_knowledge_base()

        if not self._cards:
            logger.warning("知识库为空，无法执行检索")
            return []

        top_k = top_k or config.RETRIEVAL_TOP_K

        candidates = self._cards
        used_global_fallback = False
        if syndrome_labels is not None:
            candidates = self._metadata_filter(syndrome_labels)
            if not candidates:
                logger.warning("元数据过滤结果为空，回退到全量语义检索")
                candidates = self._cards
                used_global_fallback = True
        self.last_constraint_trace = {
            "enabled": syndrome_labels is not None,
            "total_card_count": len(self._cards),
            "metadata_candidate_count": len(candidates),
            "retrieval_candidate_count": len(candidates),
            "used_global_fallback": used_global_fallback,
        }

        # Step 3b: 语义相似度检索
        results = self._semantic_retrieval(clinical_features, candidates, top_k)
        logger.info("语义检索完成，返回 Top-%d 结果", len(results))

        return results

    def _metadata_filter(
        self, syndrome_labels: SyndromeLabels
    ) -> list[KnowledgeCard]:
        """
        Step 3a: 元数据标签过滤（辨证方向约束 + 八纲属性约束）。

        过滤逻辑：
        1. 仅保留辨证标签L覆盖的证候知识卡片
        2. 排除L_exclude标签对应的卡片
        3. 按涉及脏腑进一步筛选
        4. 【新增】八纲属性一致性约束：排除与已判定八纲属性矛盾的卡片
        """
        candidates = []
        inclusion_set = set(syndrome_labels.inclusion_labels)
        exclusion_set = set(syndrome_labels.exclusion_labels)
        organs_set = set(syndrome_labels.organs_involved)

        for card in self._cards:
            # 排除检查：如果卡片名称或标签命中排除标签，跳过
            if self._matches_any_label(card, exclusion_set):
                logger.debug("排除卡片（命中排除标签）: %s", card.syndrome_name)
                continue

            # 八纲属性矛盾排除：如果已判定的八纲属性与卡片属性矛盾，跳过
            if self._has_bagang_conflict(card, syndrome_labels):
                logger.debug("排除卡片（八纲属性矛盾）: %s", card.syndrome_name)
                continue

            # 纳入检查：卡片需要匹配纳入标签或脏腑
            if self._matches_any_label(card, inclusion_set):
                candidates.append(card)
                continue

            # 脏腑匹配：如果卡片涉及的脏腑与标签脏腑有交集
            if organs_set and set(card.organs_involved) & organs_set:
                candidates.append(card)
                continue

        return candidates

    @staticmethod
    def _has_bagang_conflict(
        card: KnowledgeCard, labels: SyndromeLabels
    ) -> bool:
        """
        检查知识卡片的八纲属性是否与已判定的辨证方向矛盾。

        只有当辨证标签中的八纲属性已明确判定（非"待定"）时才进行排除。
        混合状态（如"寒热错杂"）不产生矛盾。
        """
        # 寒热矛盾检查
        if labels.cold_heat not in ("待定", "寒热错杂") and card.cold_heat:
            if (labels.cold_heat == "寒" and card.cold_heat == "热") or (
                labels.cold_heat == "热" and card.cold_heat == "寒"
            ):
                return True

        # 虚实矛盾检查
        if labels.deficiency_excess not in ("待定", "虚实夹杂") and card.deficiency_excess:
            if (labels.deficiency_excess == "虚" and card.deficiency_excess == "实") or (
                labels.deficiency_excess == "实" and card.deficiency_excess == "虚"
            ):
                return True

        # 表里矛盾检查
        if labels.exterior_interior not in ("待定", "半表半里") and card.exterior_interior:
            if (labels.exterior_interior == "表" and card.exterior_interior == "里") or (
                labels.exterior_interior == "里" and card.exterior_interior == "表"
            ):
                return True

        return False

    def _matches_any_label(
        self, card: KnowledgeCard, label_set: set[str]
    ) -> bool:
        """检查知识卡片是否匹配任一标签。"""
        if not label_set:
            return False

        # 检查证候名称是否包含在标签中（或标签包含在名称中）
        for label in label_set:
            if card.syndrome_name and (label in card.syndrome_name or card.syndrome_name in label):
                return True

        # 检查辨证体系标签
        for system in card.diagnosis_systems:
            if system in label_set:
                return True

        # 检查关键症状标签的交集
        if set(card.key_symptoms) & label_set:
            return True

        # 检查寒热虚实属性
        if card.cold_heat and card.cold_heat in label_set:
            return True
        if card.deficiency_excess and card.deficiency_excess in label_set:
            return True

        return False

    def _semantic_retrieval(
        self,
        clinical_features: dict[str, Any],
        candidates: list[KnowledgeCard],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Step 3b: 语义相似度检索。

        在候选知识子集中，计算当前临床特征嵌入与知识条目嵌入的余弦相似度，
        返回Top-K结果。
        """
        # 构建临床特征查询文本
        query_text = self._build_query_text(clinical_features)

        # 获取查询向量
        query_embedding = get_embedding(query_text)

        # 计算相似度并排序
        scored_results = []
        for card in candidates:
            if not card.embedding:
                continue
            similarity = _cosine_similarity(query_embedding, card.embedding)
            scored_results.append(
                {
                    "syndrome_name": card.syndrome_name,
                    "content": card.content,
                    "file_path": card.file_path,
                    "similarity_score": similarity,
                    "metadata": card.to_dict(),
                }
            )

        # 按相似度降序排序
        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return scored_results[:top_k]

    def _build_query_text(self, features: dict[str, Any]) -> str:
        """将临床特征构建为检索查询文本。"""
        parts = []
        if features.get("chief_complaint") and features["chief_complaint"] != "未提及":
            parts.append(features["chief_complaint"])

        if features.get("symptoms"):
            parts.append("症状：" + "、".join(features["symptoms"]))

        tongue = features.get("tongue", {})
        if tongue.get("quality") and tongue["quality"] != "未提及":
            parts.append(f"舌质{tongue['quality']}")
        if tongue.get("coating") and tongue["coating"] != "未提及":
            parts.append(f"苔{tongue['coating']}")

        if features.get("pulse") and features["pulse"] != "未提及":
            parts.append(f"脉{features['pulse']}")

        return "，".join(parts) if parts else "中医证候"
