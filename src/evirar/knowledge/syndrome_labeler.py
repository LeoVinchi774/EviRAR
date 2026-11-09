"""
Step 2：辨证标签动态生成模块（Syndrome Labeler）

利用LLM为当前临床特征集合生成辨证约束标签：
- 辨证标签 L = {l₁, l₂, ..., lₘ}（如：表证、热证、风热犯表...）
- 排除标签 L_exclude = {l'₁, l'₂, ...}（与当前证据明显矛盾的方向）
- 涉及脏腑
- 寒热虚实初判

对应 TCM_BianZheng_RAE_Proposal.md 中5.1.2节 Step 2。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from evirar.knowledge.llm_client import call_llm
from evirar.knowledge.utils import format_clinical_features

logger = logging.getLogger(__name__)

SYNDROME_LABELING_PROMPT = """\
你是一位资深中医辨证分析专家。请根据以下临床特征，执行辨证分析并生成辨证约束标签。

## 临床特征
{clinical_features}

## 分析要求

请基于中医辨证论治理论，综合分析以上临床特征，输出以下内容：

1. **可能涉及的辨证标签**：列出当前症状可能指向的证候方向标签（如：肝郁脾虚证、\
气滞血瘀证、表证、里证、热证、寒证等），按可能性从高到低排列
2. **可明确排除的辨证标签**：列出与当前临床特征明显矛盾、可以排除的证候方向
3. **涉及的脏腑**：列出当前症状可能涉及的脏腑
4. **寒热虚实初判**：根据现有信息判断寒/热/虚/实的倾向

## 输出格式

请严格以如下JSON格式输出，不要输出其他内容：

```json
{{
  "inclusion_labels": ["辨证标签1", "辨证标签2", "..."],
  "exclusion_labels": ["排除标签1", "排除标签2", "..."],
  "organs_involved": ["脏腑1", "脏腑2", "..."],
  "cold_heat": "寒/热/寒热错杂/待定",
  "deficiency_excess": "虚/实/虚实夹杂/待定",
  "exterior_interior": "表/里/半表半里/待定",
  "analysis_brief": "简要辨证分析思路（不超过100字）"
}}
```

注意：
- 辨证标签应涵盖具体证候名称和辨证体系分类
- 排除标签应只列出有明确证据可以排除的方向
- 如信息不足以判断某项，标注"待定"
- 分析应基于中医理论，考虑脏腑归经、病因病机等"""


@dataclass
class SyndromeLabels:
    """辨证标签结果的数据结构。"""

    inclusion_labels: list[str] = field(default_factory=list)
    exclusion_labels: list[str] = field(default_factory=list)
    organs_involved: list[str] = field(default_factory=list)
    cold_heat: str = "待定"
    deficiency_excess: str = "待定"
    exterior_interior: str = "待定"
    analysis_brief: str = ""


class SyndromeLabeler:
    """辨证标签动态生成器：根据临床特征生成辨证约束标签。"""

    def generate_labels(self, clinical_features: dict[str, Any]) -> SyndromeLabels:
        """
        根据临床特征生成辨证约束标签。

        Args:
            clinical_features: 临床特征字典（由ClinicalFeatureExtractor输出）

        Returns:
            SyndromeLabels: 辨证标签结果
        """
        features_text = format_clinical_features(clinical_features)
        prompt = SYNDROME_LABELING_PROMPT.format(clinical_features=features_text)

        logger.info("正在生成辨证约束标签...")
        response = call_llm(prompt)
        labels = self._parse_response(response)
        logger.info(
            "辨证标签生成完成：纳入标签 %d 个，排除标签 %d 个",
            len(labels.inclusion_labels),
            len(labels.exclusion_labels),
        )
        return labels

    def _parse_response(self, response: str) -> SyndromeLabels:
        """解析LLM返回的JSON响应为SyndromeLabels。"""
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            return SyndromeLabels(
                inclusion_labels=data.get("inclusion_labels", []),
                exclusion_labels=data.get("exclusion_labels", []),
                organs_involved=data.get("organs_involved", []),
                cold_heat=data.get("cold_heat", "待定"),
                deficiency_excess=data.get("deficiency_excess", "待定"),
                exterior_interior=data.get("exterior_interior", "待定"),
                analysis_brief=data.get("analysis_brief", ""),
            )
        except json.JSONDecodeError:
            logger.warning("辨证标签结果JSON解析失败，返回空标签集")
            return SyndromeLabels()
