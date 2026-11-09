"""
多阶段自适应经验检索模块（Experience Retriever）

实现 TCM_BianZheng_RAE_Proposal.md 5.2.2 节描述的三阶段检索策略：
  阶段一（初始匹配）：基于主诉余弦相似度 + 证候标签交集约束
  阶段二（策略参考）：基于临床特征重叠度 + 关键信息增益
  阶段三（诊断参考）：精细化语义匹配 + 新颖性过滤 + 奖励分数加权
"""

import hashlib
import json
import logging
import math
import os
import re
from typing import Any

from evirar.experience import config
from evirar.experience.experience_store import ExperienceStore
from evirar.knowledge.llm_client import get_embedding, get_embeddings_batch
from evirar.cache import cache_file

# 标记未提及信息的占位文本
_NOT_MENTIONED = "未提及"

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


class ExperienceRetriever:
    """
    多阶段自适应经验检索器。

    根据当前问诊阶段（初期/中期/后期），使用不同的检索策略
    从经验库中检索最相关的高分轨迹经验。
    """

    # 向量缓存文件名
    _CACHE_FILENAME = ".exp_embedding_cache.json"

    def __init__(self, store: ExperienceStore | None = None):
        """
        初始化经验检索器。

        Args:
            store: 经验库存储实例，默认创建新实例
        """
        self._store = store or ExperienceStore()
        self._embeddings_precomputed = False

    # ------------------------------------------------------------------
    # 向量缓存相关方法
    # ------------------------------------------------------------------

    def _get_cache_path(self) -> str:
        """获取向量缓存文件路径。"""
        return str(cache_file(self._CACHE_FILENAME))

    def _compute_content_hash(self, experiences: list[dict[str, Any]]) -> str:
        """计算经验记录的内容哈希，用于检测经验库变化。"""
        hasher = hashlib.sha256()
        for exp in sorted(experiences, key=lambda e: e.get("experience_id", "")):
            exp_id = exp.get("experience_id", "")
            hasher.update(exp_id.encode("utf-8"))
            # 使用诊断相关文本作为内容指纹
            text = self._build_experience_diagnostic_text(exp)
            hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def _load_embedding_cache(
        self, content_hash: str
    ) -> dict[str, list[float]] | None:
        """
        加载经验向量缓存。

        仅当缓存文件存在、内容哈希匹配且嵌入模型一致时返回缓存数据。

        Returns:
            experience_id -> embedding 的映射字典，若缓存无效则返回 None
        """
        from evirar.knowledge import config as kb_config

        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            logger.debug("经验向量缓存文件不存在: %s", cache_path)
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("经验向量缓存文件读取失败，将重新生成")
            return None

        if cache_data.get("content_hash") != content_hash:
            logger.info("经验库内容已变更，缓存失效")
            return None

        if cache_data.get("embedding_model") != kb_config.EMBEDDING_MODEL:
            logger.info("嵌入模型已变更，缓存失效")
            return None

        logger.debug("经验向量缓存验证通过")
        return cache_data.get("embeddings", {})

    def _save_embedding_cache(
        self,
        content_hash: str,
        embeddings_map: dict[str, list[float]],
    ) -> None:
        """将经验向量嵌入保存为缓存文件。"""
        from evirar.knowledge import config as kb_config

        cache_data = {
            "content_hash": content_hash,
            "embedding_model": kb_config.EMBEDDING_MODEL,
            "embeddings": embeddings_map,
        }

        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
            logger.info("经验向量缓存已保存: %s", cache_path)
        except OSError:
            logger.warning("经验向量缓存保存失败: %s", cache_path)

    def _precompute_embeddings(self) -> None:
        """
        预计算经验库中所有经验的向量嵌入。

        如果缓存有效则直接加载，否则批量生成并保存缓存。
        已有 embedding 字段的经验不会被覆盖。
        """
        if self._embeddings_precomputed:
            return

        experiences = self._store.get_all_experiences()
        if not experiences:
            self._embeddings_precomputed = True
            return

        content_hash = self._compute_content_hash(experiences)

        # 尝试从缓存加载
        cached = self._load_embedding_cache(content_hash)
        if cached is not None:
            loaded_count = 0
            for exp in experiences:
                if exp.get("embedding"):
                    continue
                exp_id = exp.get("experience_id", "")
                emb = cached.get(exp_id)
                if emb:
                    exp["embedding"] = emb
                    loaded_count += 1
            if loaded_count > 0:
                logger.info("从缓存加载了 %d 条经验的向量嵌入", loaded_count)
            self._embeddings_precomputed = True
            return

        # 缓存未命中，批量生成向量嵌入
        texts_to_embed: list[tuple[int, str]] = []
        for idx, exp in enumerate(experiences):
            if exp.get("embedding"):
                continue
            text = self._build_experience_diagnostic_text(exp)
            if text:
                texts_to_embed.append((idx, text))

        if texts_to_embed:
            logger.info("正在为 %d 条经验生成向量嵌入...", len(texts_to_embed))
            try:
                batch_texts = [t for _, t in texts_to_embed]
                batch_embeddings = get_embeddings_batch(batch_texts)
                for (idx, _), emb in zip(texts_to_embed, batch_embeddings):
                    experiences[idx]["embedding"] = emb
            except Exception:
                logger.exception("批量生成经验向量嵌入失败")
                self._embeddings_precomputed = True
                return

        # 保存缓存
        embeddings_map = {}
        for exp in experiences:
            exp_id = exp.get("experience_id", "")
            emb = exp.get("embedding", [])
            if exp_id and emb:
                embeddings_map[exp_id] = emb
        self._save_embedding_cache(content_hash, embeddings_map)

        self._embeddings_precomputed = True
        logger.info("经验向量嵌入预计算完成")

    # ------------------------------------------------------------------
    # 阶段一：初始匹配 (问诊前3轮)
    # ------------------------------------------------------------------

    def stage1_initial_matching(
        self,
        chief_complaint: str,
        syndrome_tags: list[str] | None = None,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        阶段一：初始匹配。

        基于患者主诉文本的向量嵌入计算余弦相似度，
        并结合证候标签交集约束进行初步筛选。

        Args:
            chief_complaint: 患者主诉文本
            syndrome_tags: 当前患者的证候标签（可选，用于交集约束）
            top_n: 返回的候选经验数量，默认使用配置值

        Returns:
            Top-N 候选经验列表，每项附加 similarity_score
        """
        top_n = top_n or config.STAGE1_TOP_N

        # 预计算经验向量嵌入（首次调用时执行，后续使用缓存）
        self._precompute_embeddings()

        experiences = self._store.get_all_experiences()

        if not experiences:
            logger.warning("经验库为空，无法执行阶段一检索")
            return []

        logger.info(
            "[阶段一] 初始匹配 - 主诉: %s, 候选经验数: %d",
            chief_complaint[:50], len(experiences),
        )

        # 计算主诉文本的向量嵌入
        try:
            query_embedding = get_embedding(chief_complaint)
        except Exception:
            logger.exception("[阶段一] 主诉向量嵌入计算失败")
            return []

        scored_results: list[dict[str, Any]] = []

        for exp in experiences:
            # 计算向量相似度（向量嵌入已在预计算阶段生成）
            exp_embedding = exp.get("embedding", [])
            if exp_embedding:
                similarity = _cosine_similarity(query_embedding, exp_embedding)
            else:
                similarity = 0.0

            # 证候标签交集约束加分
            tag_bonus = 0.0
            if syndrome_tags:
                exp_tags = set(exp.get("syndrome_tags", []))
                query_tags = set(syndrome_tags)
                if exp_tags and query_tags:
                    intersection = exp_tags & query_tags
                    tag_bonus = len(intersection) / max(
                        len(query_tags), 1
                    ) * 0.1  # 最多加0.1

            final_score = similarity + tag_bonus

            scored_results.append({
                "experience": exp,
                "similarity_score": similarity,
                "tag_bonus": tag_bonus,
                "stage1_score": final_score,
            })

        # 按最终得分降序排序
        scored_results.sort(key=lambda x: x["stage1_score"], reverse=True)

        results = scored_results[:top_n]
        logger.info(
            "[阶段一] 返回 Top-%d 候选经验 (最高分: %.3f)",
            len(results),
            results[0]["stage1_score"] if results else 0.0,
        )

        return results

    # ------------------------------------------------------------------
    # 阶段二：策略参考 (问诊中期，第4-8轮)
    # ------------------------------------------------------------------

    def stage2_strategy_reference(
        self,
        current_features: dict[str, Any],
        current_turn: int,
        stage1_candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        阶段二：策略参考。

        在阶段一的候选经验中进一步筛选，
        参考高分经验的问诊策略（问什么、按什么顺序问）。

        Args:
            current_features: 当前已获取的临床特征集合
            current_turn: 当前对话轮次
            stage1_candidates: 阶段一返回的候选经验列表
            top_k: 返回的策略参考经验数量，默认使用配置值

        Returns:
            Top-K 策略参考经验列表
        """
        top_k = top_k or config.STAGE2_TOP_K

        if not stage1_candidates:
            logger.warning("[阶段二] 无候选经验可供策略参考")
            return []

        logger.info(
            "[阶段二] 策略参考 - 当前轮次: %d, 候选经验数: %d",
            current_turn, len(stage1_candidates),
        )

        # 提取当前临床特征关键词集合
        current_keywords = self._extract_feature_keywords(current_features)

        scored_results: list[dict[str, Any]] = []

        for candidate in stage1_candidates:
            exp = candidate["experience"]
            dialogue = exp.get("dialogue_trajectory", [])

            # Step 2: 计算当前临床特征与候选经验对应阶段特征的重叠度
            exp_early_keywords = self._extract_dialogue_keywords(
                dialogue, max_turn=current_turn
            )
            overlap_score = self._calculate_keyword_overlap(
                current_keywords, exp_early_keywords
            )

            # Step 3: 检查该经验在当前阶段之后是否获取了关键信息
            future_info_score = self._evaluate_future_info_gain(
                dialogue, current_turn
            )

            # Step 4 (新增): 评估经验的关键问诊动作质量
            action_quality_score = self._evaluate_key_inquiry_actions(
                exp, current_turn
            )

            # 综合评分 = 阶段一得分 * 0.25 + 重叠度 * 0.25 + 未来信息增益 * 0.30 + 动作质量 * 0.20
            stage1_score = candidate.get("stage1_score", 0.0)
            stage2_score = (
                stage1_score * 0.25
                + overlap_score * 0.25
                + future_info_score * 0.30
                + action_quality_score * 0.20
            )

            scored_results.append({
                "experience": exp,
                "stage1_score": stage1_score,
                "overlap_score": overlap_score,
                "future_info_score": future_info_score,
                "action_quality_score": action_quality_score,
                "stage2_score": stage2_score,
            })

        # 按阶段二得分降序排序
        scored_results.sort(key=lambda x: x["stage2_score"], reverse=True)

        results = scored_results[:top_k]
        logger.info(
            "[阶段二] 返回 Top-%d 策略参考经验 (最高分: %.3f)",
            len(results),
            results[0]["stage2_score"] if results else 0.0,
        )

        return results

    # ------------------------------------------------------------------
    # 阶段三：诊断参考 (问诊后期/出诊断时)
    # ------------------------------------------------------------------

    def stage3_diagnostic_reference(
        self,
        complete_features: dict[str, Any],
        syndrome_hypotheses: list[str] | None = None,
        stage1_candidates: list[dict[str, Any]] | None = None,
        top_m: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        阶段三：诊断参考。

        使用完整临床特征和证候假设列表进行精细化语义匹配，
        结合新颖性过滤和奖励分数加权。

        Args:
            complete_features: 完整的临床特征集合
            syndrome_hypotheses: 证候假设列表
            stage1_candidates: 阶段一的候选经验（可选，用于缩小范围）
            top_m: 返回的诊断参考经验数量，默认使用配置值

        Returns:
            Top-M 诊断参考经验列表
        """
        top_m = top_m or config.STAGE3_TOP_M

        # 确定候选经验范围
        if stage1_candidates:
            candidates = [c["experience"] for c in stage1_candidates]
        else:
            candidates = self._store.get_all_experiences()

        if not candidates:
            logger.warning("[阶段三] 无候选经验可供诊断参考")
            return []

        logger.info(
            "[阶段三] 诊断参考 - 候选经验数: %d, 证候假设: %s",
            len(candidates),
            syndrome_hypotheses or "无",
        )

        # Step 1: 精细化语义匹配
        query_text = self._build_diagnostic_query(
            complete_features, syndrome_hypotheses
        )
        try:
            query_embedding = get_embedding(query_text)
        except Exception:
            logger.exception("[阶段三] 诊断查询向量嵌入计算失败")
            return []

        scored_results: list[dict[str, Any]] = []

        for exp in candidates:
            # 语义相似度（向量嵌入已在预计算阶段生成）
            exp_embedding = exp.get("embedding", [])
            if exp_embedding:
                similarity = _cosine_similarity(query_embedding, exp_embedding)
            else:
                similarity = 0.0

            # Step 2: 新颖性过滤
            if similarity > config.NOVELTY_MAX_SIMILARITY:
                logger.debug(
                    "排除过于相似的经验 (相似度=%.3f): %s",
                    similarity, exp.get("experience_id", ""),
                )
                continue

            # Step 3: 奖励分数加权
            reward_score = exp.get("reward_scores", {}).get("total_score", 0.0)
            # 归一化奖励分数到 0~1 范围 (假设满分10分)
            normalized_reward = min(reward_score / 10.0, 1.0)

            # 综合评分 = 语义相似度 * 0.6 + 奖励分数权重 * 0.4
            final_score = similarity * 0.6 + normalized_reward * 0.4

            scored_results.append({
                "experience": exp,
                "semantic_similarity": similarity,
                "reward_score": reward_score,
                "stage3_score": final_score,
            })

        # 按最终得分降序排序
        scored_results.sort(key=lambda x: x["stage3_score"], reverse=True)

        results = scored_results[:top_m]
        logger.info(
            "[阶段三] 返回 Top-%d 诊断参考经验 (最高分: %.3f)",
            len(results),
            results[0]["stage3_score"] if results else 0.0,
        )

        return results

    # ------------------------------------------------------------------
    # 自适应检索入口：根据当前对话轮次自动选择阶段
    # ------------------------------------------------------------------

    def adaptive_retrieve(
        self,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any] | None = None,
        syndrome_hypotheses: list[str] | None = None,
        syndrome_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        自适应经验检索：根据当前对话轮次自动选择检索阶段。

        Args:
            dialogue_history: 当前对话历史
            clinical_features: 当前临床特征（可选）
            syndrome_hypotheses: 证候假设列表（可选）
            syndrome_tags: 证候标签（可选）

        Returns:
            包含检索阶段和检索结果的字典
        """
        current_turn = self._count_turns(dialogue_history)
        chief_complaint = self._extract_chief_complaint(dialogue_history)

        logger.info("自适应检索 - 当前轮次: %d", current_turn)

        # 阶段一始终执行
        stage1_results = self.stage1_initial_matching(
            chief_complaint, syndrome_tags
        )

        result: dict[str, Any] = {
            "current_turn": current_turn,
            "stage": "",
            "stage1_candidates": stage1_results,
            "stage2_references": [],
            "stage3_references": [],
            "recommended_experiences": [],
        }

        if current_turn <= config.EARLY_STAGE_END_TURN:
            # 初期阶段：返回阶段一结果
            result["stage"] = "initial"
            result["recommended_experiences"] = stage1_results

        elif current_turn <= config.MIDDLE_STAGE_END_TURN:
            # 中期阶段：执行阶段二
            features = clinical_features or {}
            stage2_results = self.stage2_strategy_reference(
                features, current_turn, stage1_results
            )
            result["stage"] = "strategy"
            result["stage2_references"] = stage2_results
            result["recommended_experiences"] = stage2_results

        else:
            # 后期阶段：执行阶段三
            features = clinical_features or {}
            stage3_results = self.stage3_diagnostic_reference(
                features, syndrome_hypotheses, stage1_results
            )
            result["stage"] = "diagnostic"
            result["stage3_references"] = stage3_results
            result["recommended_experiences"] = stage3_results

        logger.info(
            "自适应检索完成 - 阶段: %s, 推荐经验数: %d",
            result["stage"], len(result["recommended_experiences"]),
        )

        return result

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _extract_feature_keywords(
        self, features: dict[str, Any]
    ) -> set[str]:
        """从临床特征字典中提取关键词集合。"""
        keywords: set[str] = set()

        if features.get("chief_complaint") and features["chief_complaint"] != _NOT_MENTIONED:
            keywords.add(features["chief_complaint"])

        for symptom in features.get("symptoms", []):
            keywords.add(symptom)

        tongue = features.get("tongue", {})
        if isinstance(tongue, dict):
            if tongue.get("quality") and tongue["quality"] != _NOT_MENTIONED:
                keywords.add(tongue["quality"])
            if tongue.get("coating") and tongue["coating"] != _NOT_MENTIONED:
                keywords.add(tongue["coating"])
        elif isinstance(tongue, str) and tongue and tongue != _NOT_MENTIONED:
            keywords.add(tongue)

        if features.get("pulse") and features["pulse"] != _NOT_MENTIONED:
            keywords.add(features["pulse"])

        for item in features.get("inspection", []):
            keywords.add(item)

        for item in features.get("inquiry", []):
            keywords.add(item)

        return keywords

    def _extract_dialogue_keywords(
        self, dialogue: list[dict[str, Any]], max_turn: int
    ) -> set[str]:
        """从经验对话轨迹中提取指定轮次范围内的关键词。"""
        keywords: set[str] = set()

        for turn_data in dialogue:
            turn_num = turn_data.get("turn", 0)
            if turn_num > max_turn:
                break

            patient_text = turn_data.get("patient", "")
            doctor_text = turn_data.get("doctor", "")

            # 简单的关键词提取：取长度适当的文本片段
            for text in [patient_text, doctor_text]:
                if text:
                    segments = re.split(r"[，。？！、；：\s]+", text)
                    for seg in segments:
                        seg = seg.strip()
                        if 2 <= len(seg) <= 15:
                            keywords.add(seg)

        return keywords

    def _calculate_keyword_overlap(
        self, set_a: set[str], set_b: set[str]
    ) -> float:
        """计算两个关键词集合的重叠度（Jaccard相似度）。"""
        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b

        return len(intersection) / len(union) if union else 0.0

    def _evaluate_future_info_gain(
        self, dialogue: list[dict[str, Any]], current_turn: int
    ) -> float:
        """
        评估经验在当前轮次之后是否获取了关键信息。

        通过比较当前轮次前后的对话内容丰富度来估计。
        """
        future_turns = [
            t for t in dialogue if t.get("turn", 0) > current_turn
        ]

        if not future_turns:
            return 0.0

        # 统计未来轮次中医生问诊内容的丰富度
        total_content_length = 0
        for turn_data in future_turns:
            doctor_text = turn_data.get("doctor", "")
            patient_text = turn_data.get("patient", "")
            total_content_length += len(doctor_text) + len(patient_text)

        # 归一化：假设平均每轮约100字
        avg_per_turn = total_content_length / max(len(future_turns), 1)
        normalized = min(avg_per_turn / 100.0, 1.0)

        return normalized

    def _evaluate_key_inquiry_actions(
        self, experience: dict[str, Any], current_turn: int
    ) -> float:
        """
        评估经验中关键问诊动作的质量和适用性。

        利用经验记录中的 key_inquiry_actions 字段（标注的关键问诊动作），
        评估该经验在当前问诊阶段可提供的策略参考价值。

        一个高质量的策略参考经验应当：
        1. 有明确标注的关键问诊动作
        2. 这些动作在当前阶段或之后的阶段执行
        3. 动作类型覆盖多个诊法维度

        Returns:
            动作质量分数 ∈ [0, 1]
        """
        key_actions = experience.get("key_inquiry_actions", [])
        if not key_actions:
            # 没有标注关键动作的经验，给予基线分数
            return 0.3

        # 统计关键动作的质量指标
        total_actions = len(key_actions)
        future_actions = 0
        action_types: set[str] = set()

        for action in key_actions:
            action_turn = action.get("turn", 0)
            action_type = action.get("action_type", "")

            # 在当前轮次之后的动作更有策略参考价值
            if action_turn >= current_turn:
                future_actions += 1

            # 收集动作类型（用于评估覆盖度）
            if action_type:
                action_types.add(action_type)

        # 评分维度1：未来可参考动作比例
        future_ratio = future_actions / max(total_actions, 1)

        # 评分维度2：动作类型多样性（覆盖多个诊法维度）
        # 四诊对应4种诊法类型：望、闻、问、切
        _NUM_DIAGNOSIS_METHODS = 4
        diversity_score = min(len(action_types) / _NUM_DIAGNOSIS_METHODS, 1.0)

        # 评分维度3：总动作数量的合理性（太少说明信息不足，太多可能冗余）
        quantity_score = min(total_actions / 3.0, 1.0)  # 3个以上视为充分

        # 综合评分
        quality_score = (
            future_ratio * 0.5
            + diversity_score * 0.3
            + quantity_score * 0.2
        )

        return min(quality_score, 1.0)

    def _build_diagnostic_query(
        self,
        features: dict[str, Any],
        syndrome_hypotheses: list[str] | None = None,
    ) -> str:
        """构建诊断阶段的语义检索查询文本。"""
        parts = []

        if features.get("chief_complaint") and features["chief_complaint"] != _NOT_MENTIONED:
            parts.append(features["chief_complaint"])

        if features.get("symptoms"):
            parts.append("症状：" + "、".join(features["symptoms"]))

        tongue = features.get("tongue", {})
        if isinstance(tongue, dict):
            if tongue.get("quality") and tongue["quality"] != _NOT_MENTIONED:
                parts.append(f"舌质{tongue['quality']}")
            if tongue.get("coating") and tongue["coating"] != _NOT_MENTIONED:
                parts.append(f"苔{tongue['coating']}")
        elif isinstance(tongue, str) and tongue and tongue != _NOT_MENTIONED:
            parts.append(f"舌{tongue}")

        if features.get("pulse") and features["pulse"] != _NOT_MENTIONED:
            parts.append(f"脉{features['pulse']}")

        if syndrome_hypotheses:
            parts.append("证候假设：" + "、".join(syndrome_hypotheses))

        return "，".join(parts) if parts else "中医诊断参考"

    def _build_experience_diagnostic_text(
        self, experience: dict[str, Any]
    ) -> str:
        """构建经验记录的诊断相关文本（用于语义匹配）。"""
        parts = []

        profile = experience.get("patient_profile", {})
        if profile.get("chief_complaint"):
            parts.append(profile["chief_complaint"])
        if profile.get("key_symptoms"):
            parts.append("、".join(profile["key_symptoms"]))
        if profile.get("tongue"):
            parts.append(profile["tongue"])
        if profile.get("pulse"):
            parts.append(profile["pulse"])

        diagnosis = experience.get("diagnosis_result", {})
        if diagnosis.get("syndrome"):
            parts.append(diagnosis["syndrome"])
        if diagnosis.get("reasoning"):
            parts.append(diagnosis["reasoning"][:200])

        return "，".join(parts) if parts else ""

    def _count_turns(self, dialogue_history: list[dict[str, str]]) -> int:
        """计算对话历史中的轮次数。"""
        turn_count = 0
        for msg in dialogue_history:
            if msg.get("role") == "user":
                turn_count += 1
        return turn_count

    def _extract_chief_complaint(
        self, dialogue_history: list[dict[str, str]]
    ) -> str:
        """从对话历史中提取患者主诉（第一条患者消息）。"""
        for msg in dialogue_history:
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]
        return ""
