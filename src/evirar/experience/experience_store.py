"""
经验库存储与管理模块（Experience Store）

负责经验记录的持久化存储、加载、索引和管理，
包括多样性控制和过期清理。
"""

import json
import logging
from pathlib import Path
from typing import Any

from evirar.experience import config

logger = logging.getLogger(__name__)

# 经验库默认文件名
EXPERIENCE_LIB_FILENAME = "experience_library.json"


class ExperienceStore:
    """经验库存储管理器：负责加载、保存和索引经验记录。"""

    def __init__(self, library_dir: str | None = None):
        """
        初始化经验库存储管理器。

        Args:
            library_dir: 经验库目录路径，默认使用配置值
        """
        self._library_dir = library_dir or config.EXPERIENCE_LIBRARY_DIR
        self._experiences: list[dict[str, Any]] = []
        self._loaded = False

        # 索引：按证候标签分组的经验ID映射
        self._syndrome_index: dict[str, list[int]] = {}

    @property
    def library_dir(self) -> str:
        """获取经验库目录路径。"""
        return self._library_dir

    @property
    def experiences(self) -> list[dict[str, Any]]:
        """获取所有经验记录。"""
        if not self._loaded:
            self.load()
        return self._experiences

    @property
    def size(self) -> int:
        """获取经验库中的经验数量。"""
        return len(self._experiences)

    def load(self, file_path: str | None = None) -> None:
        """
        从文件加载经验库。

        Args:
            file_path: 经验库文件路径，默认使用 library_dir 下的标准文件名
        """
        if file_path is None:
            file_path = str(
                Path(self._library_dir) / EXPERIENCE_LIB_FILENAME
            )

        path = Path(file_path)
        if not path.exists():
            logger.warning("经验库文件不存在: %s，初始化空经验库", file_path)
            self._experiences = []
            self._loaded = True
            self._build_index()
            return

        with open(path, "r", encoding="utf-8") as f:
            self._experiences = json.load(f)

        self._loaded = True
        self._build_index()
        logger.info("经验库加载完成: %s (共 %d 条经验)", file_path, len(self._experiences))

    def save(self, file_path: str | None = None) -> None:
        """
        保存经验库到文件。

        Args:
            file_path: 保存路径，默认使用 library_dir 下的标准文件名
        """
        if file_path is None:
            file_path = str(
                Path(self._library_dir) / EXPERIENCE_LIB_FILENAME
            )

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._experiences, f, ensure_ascii=False, indent=2)

        logger.info("经验库已保存: %s (共 %d 条经验)", file_path, len(self._experiences))

    def add_experiences(self, new_experiences: list[dict[str, Any]]) -> int:
        """
        批量添加经验记录到经验库。

        会自动去重（根据 experience_id）和执行多样性控制。

        Args:
            new_experiences: 新的经验记录列表

        Returns:
            实际添加的经验数量
        """
        if not self._loaded:
            self.load()

        existing_ids = {exp["experience_id"] for exp in self._experiences}
        added_count = 0

        for exp in new_experiences:
            exp_id = exp.get("experience_id", "")
            if exp_id in existing_ids:
                logger.debug("跳过重复经验: %s", exp_id)
                continue

            self._experiences.append(exp)
            existing_ids.add(exp_id)
            added_count += 1

        if added_count > 0:
            self._build_index()
            self._enforce_diversity()

        logger.info("添加了 %d 条新经验 (总计 %d 条)", added_count, len(self._experiences))
        return added_count

    def get_experiences_by_tags(self, tags: list[str]) -> list[dict[str, Any]]:
        """
        根据证候标签获取经验记录。

        Args:
            tags: 证候标签列表

        Returns:
            匹配标签的经验记录列表
        """
        if not self._loaded:
            self.load()

        matched_indices: set[int] = set()
        for tag in tags:
            if tag in self._syndrome_index:
                matched_indices.update(self._syndrome_index[tag])

        return [self._experiences[i] for i in sorted(matched_indices)]

    def get_all_experiences(self) -> list[dict[str, Any]]:
        """获取所有经验记录。"""
        if not self._loaded:
            self.load()
        return list(self._experiences)

    def _build_index(self) -> None:
        """构建证候标签索引。"""
        self._syndrome_index.clear()

        for idx, exp in enumerate(self._experiences):
            tags = exp.get("syndrome_tags", [])
            for tag in tags:
                if tag not in self._syndrome_index:
                    self._syndrome_index[tag] = []
                self._syndrome_index[tag].append(idx)

        logger.debug(
            "索引构建完成: %d 个标签, %d 条经验",
            len(self._syndrome_index), len(self._experiences),
        )

    def _enforce_diversity(self) -> None:
        """
        多样性控制：对同一证候类型限制最大经验数量。

        按评分降序保留 MAX_EXPERIENCES_PER_SYNDROME 条经验，
        删除超出限制的低分经验。
        """
        max_per_syndrome = config.MAX_EXPERIENCES_PER_SYNDROME

        # 统计每个证候标签的经验分布
        syndrome_experiences: dict[str, list[tuple[int, float]]] = {}
        for idx, exp in enumerate(self._experiences):
            label = exp.get("patient_profile", {}).get("syndrome_label", "")
            if not label:
                continue
            score = exp.get("reward_scores", {}).get("total_score", 0.0)
            if label not in syndrome_experiences:
                syndrome_experiences[label] = []
            syndrome_experiences[label].append((idx, score))

        # 找出需要删除的索引
        indices_to_remove: set[int] = set()
        for label, items in syndrome_experiences.items():
            if len(items) <= max_per_syndrome:
                continue
            # 按评分降序排序，保留前 max_per_syndrome 条
            items.sort(key=lambda x: x[1], reverse=True)
            for idx, _score in items[max_per_syndrome:]:
                indices_to_remove.add(idx)

        if indices_to_remove:
            self._experiences = [
                exp
                for i, exp in enumerate(self._experiences)
                if i not in indices_to_remove
            ]
            self._build_index()
            logger.info(
                "多样性控制: 移除了 %d 条经验", len(indices_to_remove)
            )
