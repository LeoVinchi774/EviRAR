"""
证候知识卡片解析工具（Knowledge Card Parser）

解析Markdown格式的证候知识卡片文件，提取结构化内容和元数据标签，
用于后续的结构化过滤检索。

对应 TCM_BianZheng_RAE_Proposal.md 中5.1.1节的知识卡片格式。
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeCard:
    """证候知识卡片的数据结构。"""

    syndrome_name: str = ""
    content: str = ""  # 完整的卡片文本内容
    file_path: str = ""

    # 元数据标签（用于结构化过滤）
    diagnosis_systems: list[str] = field(default_factory=list)
    organs_involved: list[str] = field(default_factory=list)
    cold_heat: str = ""
    deficiency_excess: str = ""
    exterior_interior: str = ""
    yin_yang: str = ""
    key_symptoms: list[str] = field(default_factory=list)
    differential_syndromes: list[str] = field(default_factory=list)

    # 向量嵌入（运行时填充）
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（不含embedding，用于序列化）。"""
        return {
            "syndrome_name": self.syndrome_name,
            "content": self.content,
            "file_path": self.file_path,
            "diagnosis_systems": self.diagnosis_systems,
            "organs_involved": self.organs_involved,
            "cold_heat": self.cold_heat,
            "deficiency_excess": self.deficiency_excess,
            "exterior_interior": self.exterior_interior,
            "yin_yang": self.yin_yang,
            "key_symptoms": self.key_symptoms,
            "differential_syndromes": self.differential_syndromes,
        }


class KnowledgeCardParser:
    """证候知识卡片解析器：从Markdown文件中解析结构化知识卡片。"""

    def parse_file(self, file_path: str) -> KnowledgeCard:
        """
        解析单个知识卡片Markdown文件。

        Args:
            file_path: 知识卡片文件路径

        Returns:
            KnowledgeCard: 解析后的知识卡片对象
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        card = KnowledgeCard(content=content, file_path=file_path)

        # 提取证候名称
        name_match = re.search(r"##\s*证候名称[：:]\s*([^\n]+)", content)
        if name_match:
            card.syndrome_name = name_match.group(1).strip()

        # 提取元数据标签（从JSON块中）
        metadata = self._extract_metadata_json(content)
        if metadata:
            card.diagnosis_systems = metadata.get("辨证体系", [])
            card.organs_involved = metadata.get("涉及脏腑", [])
            card.cold_heat = metadata.get("寒热属性", "")
            card.deficiency_excess = metadata.get("虚实属性", "")
            card.exterior_interior = metadata.get("表里属性", "")
            card.yin_yang = metadata.get("阴阳属性", "")
            card.key_symptoms = metadata.get("关键症状标签", [])
            card.differential_syndromes = metadata.get("鉴别证候", [])
        else:
            # 如果没有JSON元数据块，从文本中解析
            self._parse_metadata_from_text(content, card)

        return card

    def parse_directory(self, dir_path: str) -> list[KnowledgeCard]:
        """
        解析目录下的所有知识卡片文件。

        Args:
            dir_path: 知识卡片目录路径

        Returns:
            解析后的知识卡片列表
        """
        cards = []
        if not os.path.isdir(dir_path):
            logger.warning("知识库目录不存在: %s", dir_path)
            return cards

        for filename in os.listdir(dir_path):
            if filename.endswith(".md"):
                file_path = os.path.join(dir_path, filename)
                try:
                    card = self.parse_file(file_path)
                    cards.append(card)
                    logger.debug("成功解析知识卡片: %s", card.syndrome_name)
                except Exception:
                    logger.exception("解析知识卡片失败: %s", file_path)

        logger.info("共解析 %d 张知识卡片", len(cards))
        return cards

    def _extract_metadata_json(self, content: str) -> dict[str, Any] | None:
        """从卡片内容中提取元数据JSON块。"""
        # 查找 ### 元数据标签 后面的JSON块
        pattern = r"###\s*元数据标签\s*\n+```json\s*\n(.*?)\n```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("元数据JSON解析失败")
        return None

    def _parse_metadata_from_text(self, content: str, card: KnowledgeCard) -> None:
        """从卡片文本内容中解析元数据（当没有JSON块时的回退方案）。"""
        # 解析辨证体系
        system_match = re.search(r"所属体系[：:]\s*(.+)", content)
        if system_match:
            card.diagnosis_systems = [
                s.strip() for s in system_match.group(1).split("、")
            ]

        # 解析涉及脏腑
        organ_match = re.search(r"涉及脏腑[：:]\s*(.+)", content)
        if organ_match:
            card.organs_involved = [
                s.strip() for s in organ_match.group(1).split("、")
            ]

        # 解析主症（作为关键症状标签）
        main_symptoms_section = re.search(
            r"###\s*主症.*?\n(.*?)(?=###|\Z)", content, re.DOTALL
        )
        if main_symptoms_section:
            symptoms = re.findall(r"-\s*(.+)", main_symptoms_section.group(1))
            card.key_symptoms = [s.strip() for s in symptoms]
