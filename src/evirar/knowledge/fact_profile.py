"""病例事实画像模块：从已提取特征构建可审计事实，不生成辨证约束。"""

from __future__ import annotations

import re
from typing import Any


class FactProfileBuilder:
    """将结构化临床特征转换为 B-D 模块使用的事实画像。"""

    _ATTRIBUTES = {
        "cold_heat": {
            "寒": ("怕冷", "畏寒", "恶寒", "清稀"),
            "热": ("发热", "怕热", "口苦", "口渴", "黄腻", "灼热"),
        },
        "deficiency_excess": {
            "虚": ("乏力", "倦怠", "无力", "虚弱"),
            "实": ("胀", "拒按", "疼痛", "痰", "湿", "瘀"),
        },
        "exterior_interior": {
            "表": ("恶寒", "发热", "汗"),
            "里": ("腹", "胃", "便", "尿", "口苦"),
        },
    }

    def build(self, features: dict[str, Any]) -> dict[str, Any]:
        facts: list[dict[str, str]] = []
        self._add(facts, "chief_complaint", features.get("chief_complaint"))
        for value in features.get("symptoms", []) or []:
            self._add(facts, "symptom", value)

        tongue = features.get("tongue", {}) or {}
        if isinstance(tongue, dict):
            self._add(facts, "tongue", tongue.get("quality"))
            self._add(facts, "tongue", tongue.get("coating"))
        self._add(facts, "pulse", features.get("pulse"))
        self._add(facts, "course", features.get("disease_course"))

        text = " ".join(item["text"] for item in facts)
        return {
            "facts": facts,
            "attributes": {
                name: self._attribute(options, text)
                for name, options in self._ATTRIBUTES.items()
            },
        }

    @staticmethod
    def _add(facts: list[dict[str, str]], field: str, value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in {"未提及", "待定", "None"}:
            facts.append({"field": field, "text": text})

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "").lower()

    @staticmethod
    def _attribute(options: dict[str, tuple[str, ...]], text: str) -> str:
        found = [
            name for name, words in options.items()
            if any(word in text for word in words)
        ]
        return found[0] if len(found) == 1 else "待定"
