"""
经验检索流水线模块（Experience Retrieval Pipeline）

将经验库构建、存储管理和多阶段自适应检索串联为完整流程。
对应 TCM_BianZheng_RAE_Proposal.md 5.2 节。

支持两种经验来源：
- 传统 JSON 经验库（基于轨迹转换）
- MD 格式医案经验文件（每个 md 文件为一条医案经验）
"""

import logging
import os
from typing import Any

from evirar.experience import config
from evirar.experience.experience_retriever import ExperienceRetriever
from evirar.experience.experience_store import ExperienceStore
from evirar.experience.md_experience_retriever import MdExperienceRetriever
from evirar.experience.md_experience_store import MdExperienceStore
from evirar.experience.prompt_formatter import (
    format_diagnostic_prompt,
    format_experience_prompt,
    format_md_experience_prompt,
    format_md_experience_retrieval_prompt,
    format_strategy_prompt,
)
from evirar.experience.trajectory_converter import (
    convert_trajectories,
    load_raw_trajectories,
)

logger = logging.getLogger(__name__)


class ExperienceRetrievalPipeline:
    """
    高分轨迹经验库检索流水线。

    整合经验库管理和多阶段自适应检索：
    - 支持从原始轨迹数据构建经验库
    - 根据当前问诊阶段自动选择检索策略
    - 将检索结果格式化为可注入的Prompt文本
    """

    def __init__(
        self,
        library_dir: str | None = None,
        md_experience_dir: str | None = None,
    ):
        """
        初始化经验检索流水线。

        Args:
            library_dir: 经验库目录路径，默认使用配置值
            md_experience_dir: MD 经验文件目录路径，设置后启用 MD 经验检索
        """
        self._store = ExperienceStore(library_dir)
        self._retriever = ExperienceRetriever(self._store)

        # MD 经验库（可选）
        self._md_store: MdExperienceStore | None = None
        self._md_retriever: MdExperienceRetriever | None = None
        if md_experience_dir:
            self._md_store = MdExperienceStore(md_experience_dir)
            self._md_retriever = MdExperienceRetriever(self._md_store)

    @property
    def store(self) -> ExperienceStore:
        """获取经验库存储实例。"""
        return self._store

    # ------------------------------------------------------------------
    # 经验库构建
    # ------------------------------------------------------------------

    def build_from_trajectories(
        self,
        trajectory_file: str,
        score_threshold: float | None = None,
        save: bool = True,
    ) -> int:
        """
        从原始轨迹文件构建经验库。

        Args:
            trajectory_file: 原始轨迹JSON文件路径
            score_threshold: 评分阈值
            save: 是否立即保存

        Returns:
            新增的经验数量
        """
        logger.info("=" * 60)
        logger.info("开始从轨迹文件构建经验库")
        logger.info("=" * 60)

        # 1. 加载原始轨迹
        raw_data = load_raw_trajectories(trajectory_file)

        # 2. 转换为经验记录
        experiences = convert_trajectories(raw_data, score_threshold)

        # 3. 添加到经验库
        added = self._store.add_experiences(experiences)

        # 4. 保存
        if save:
            self._store.save()

        logger.info("经验库构建完成: 新增 %d 条经验", added)
        return added

    # ------------------------------------------------------------------
    # 多阶段自适应检索
    # ------------------------------------------------------------------

    def retrieve(
        self,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any] | None = None,
        syndrome_hypotheses: list[str] | None = None,
        syndrome_tags: list[str] | None = None,
    ) -> str:
        """
        执行多阶段自适应经验检索并返回格式化的Prompt文本。

        自动根据当前对话轮次选择检索阶段和策略。

        Args:
            dialogue_history: 当前对话历史
            clinical_features: 当前临床特征（可选）
            syndrome_hypotheses: 证候假设列表（可选）
            syndrome_tags: 证候标签（可选）

        Returns:
            格式化的经验参考Prompt文本
        """
        logger.info("=" * 60)
        logger.info("开始多阶段自适应经验检索")
        logger.info("=" * 60)

        result = self._retriever.adaptive_retrieve(
            dialogue_history,
            clinical_features,
            syndrome_hypotheses,
            syndrome_tags,
        )

        stage = result["stage"]
        recommended = result["recommended_experiences"]

        # 根据阶段选择格式化方式
        if stage == "initial":
            prompt = format_experience_prompt(recommended, stage="initial")
        elif stage == "strategy":
            prompt = format_strategy_prompt(recommended, clinical_features)
        elif stage == "diagnostic":
            prompt = format_diagnostic_prompt(recommended)
        else:
            prompt = format_experience_prompt(recommended)

        logger.info(
            "经验检索完成 - 阶段: %s, 推荐经验数: %d",
            stage, len(recommended),
        )
        logger.info("=" * 60)

        return prompt

    def retrieve_with_details(
        self,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any] | None = None,
        syndrome_hypotheses: list[str] | None = None,
        syndrome_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        执行检索并返回详细的中间结果。

        Args:
            dialogue_history: 当前对话历史
            clinical_features: 当前临床特征（可选）
            syndrome_hypotheses: 证候假设列表（可选）
            syndrome_tags: 证候标签（可选）

        Returns:
            包含各阶段中间结果的详细字典
        """
        result = self._retriever.adaptive_retrieve(
            dialogue_history,
            clinical_features,
            syndrome_hypotheses,
            syndrome_tags,
        )

        stage = result["stage"]
        recommended = result["recommended_experiences"]

        # 生成格式化Prompt
        if stage == "initial":
            prompt = format_experience_prompt(recommended, stage="initial")
        elif stage == "strategy":
            prompt = format_strategy_prompt(recommended, clinical_features)
        elif stage == "diagnostic":
            prompt = format_diagnostic_prompt(recommended)
        else:
            prompt = format_experience_prompt(recommended)

        return {
            "current_turn": result["current_turn"],
            "stage": stage,
            "stage1_candidates_count": len(result["stage1_candidates"]),
            "stage2_references_count": len(result["stage2_references"]),
            "stage3_references_count": len(result["stage3_references"]),
            "recommended_count": len(recommended),
            "recommended_experiences": recommended,
            "formatted_prompt": prompt,
        }

    # ------------------------------------------------------------------
    # MD 经验文件检索
    # ------------------------------------------------------------------

    def retrieve_md_experiences(
        self,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any] | None = None,
        current_turn: int = 0,
        top_k: int = 2,
    ) -> str:
        """
        从 MD 经验库中检索相关经验并返回格式化的 Prompt 文本。

        根据当前轮次自动选择问诊阶段的 Prompt 格式。

        Args:
            dialogue_history: 当前对话历史
            clinical_features: 当前临床特征（可选）
            current_turn: 当前轮次
            top_k: 返回的最大结果数

        Returns:
            格式化的 MD 经验参考 Prompt 文本
        """
        if self._md_retriever is None:
            return ""

        results = self._md_retriever.retrieve_from_dialogue(
            dialogue_history, clinical_features, top_k=top_k
        )
        if not results:
            return ""

        # 根据轮次选择展示阶段
        stage = "diagnostic" if current_turn > config.MIDDLE_STAGE_END_TURN else "inquiry"

        return format_md_experience_retrieval_prompt(results, stage=stage)

    def get_md_experience_by_id(self, exp_id: str) -> dict[str, Any] | None:
        """
        根据 ID 获取特定的 MD 经验。

        Args:
            exp_id: 经验 ID（即 md 文件名，不含扩展名）

        Returns:
            经验记录字典，未找到返回 None
        """
        if self._md_store is None:
            return None
        return self._md_store.get_experience_by_id(exp_id)

    @staticmethod
    def format_md_experience_for_stage(
        md_content: str,
        current_turn: int = 0,
    ) -> str:
        """
        将 MD 经验内容格式化为指定阶段的 Prompt。

        Args:
            md_content: MD 文件内容
            current_turn: 当前轮次

        Returns:
            格式化的 Prompt 文本
        """
        stage = "diagnostic" if current_turn > config.MIDDLE_STAGE_END_TURN else "inquiry"
        return format_md_experience_prompt(md_content, stage=stage)

    def load_single_md_experience(self, file_path: str) -> dict[str, Any] | None:
        """
        加载单个 MD 经验文件。

        Args:
            file_path: MD 文件的完整路径

        Returns:
            经验记录字典
        """
        if self._md_store is None:
            # 临时创建一个 store 来加载单个文件
            temp_store = MdExperienceStore(os.path.dirname(file_path))
            return temp_store.load_single(file_path)
        return self._md_store.load_single(file_path)
