"""
Step 1：临床特征提取模块（Clinical Feature Extractor）

从对话历史中利用LLM提取已获取的临床特征集合：
F = {f₁, f₂, ..., fₙ}  (如：头痛、畏寒、舌红苔黄、脉浮数...)

对应 TCM_BianZheng_RAE_Proposal.md 中5.1.2节 Step 1。
"""

import json
import logging
from typing import Any

from evirar.knowledge.llm_client import call_llm

logger = logging.getLogger(__name__)

FEATURE_EXTRACTION_PROMPT = """\
你是一位经验丰富的中医诊断专家。请从以下医患对话历史中，提取所有已获取的临床特征信息。

## 对话历史
{dialogue_history}

## 提取要求

请系统性地提取以下维度的临床特征：

1. **主诉症状**：患者的主要不适和症状
2. **望诊信息**：面色、神态、形体、舌象（舌质、舌苔）等
3. **闻诊信息**：声音、气味等
4. **问诊信息**：寒热、汗出、疼痛部位与性质、饮食、睡眠、二便、情志等
5. **切诊信息**：脉象等
6. **病程信息**：发病时间、病程长短、加重/缓解因素等
7. **体质与既往史**：体质特征、既往疾病等

## 输出格式

请严格以如下JSON格式输出，不要输出其他内容：

```json
{{
  "symptoms": ["症状1", "症状2", "..."],
  "tongue": {{
    "quality": "舌质描述（如有）",
    "coating": "舌苔描述（如有）"
  }},
  "pulse": "脉象描述（如有）",
  "inspection": ["望诊发现1", "..."],
  "inquiry": ["问诊信息1", "..."],
  "disease_course": "病程描述",
  "constitution": "体质与既往史描述",
  "chief_complaint": "主诉概述"
}}
```

注意：
- 仅提取对话中明确提到的信息，不要推测
- 如某项信息未在对话中出现，对应字段填"未提及"
- 症状描述应使用中医术语"""


class ClinicalFeatureExtractor:
    """临床特征提取器：从对话历史中提取结构化的临床特征集合。"""

    def extract(self, dialogue_history: list[dict[str, str]]) -> dict[str, Any]:
        """
        从对话历史中提取临床特征。

        Args:
            dialogue_history: 对话历史列表，每个元素为
                {"role": "patient"/"doctor", "content": "..."}

        Returns:
            结构化的临床特征字典
        """
        formatted_history = self._format_dialogue(dialogue_history)
        prompt = FEATURE_EXTRACTION_PROMPT.format(dialogue_history=formatted_history)

        logger.info("正在从对话历史中提取临床特征...")
        response = call_llm(prompt, response_format={"type": "json_object"})
        features = self._parse_response(response)
        logger.info("临床特征提取完成，共提取 %d 个症状", len(features.get("symptoms", [])))
        return features

    def _format_dialogue(self, dialogue_history: list[dict[str, str]]) -> str:
        """将对话历史格式化为文本。"""
        lines = []
        for turn in dialogue_history:
            role = "患者" if turn["role"] == "patient" else "医生"
            lines.append(f"{role}：{turn['content']}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """解析LLM返回的JSON响应。"""
        # 尝试从响应中提取JSON块
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("临床特征提取结果JSON解析失败，返回原始文本")
            return {
                "symptoms": [],
                "tongue": {"quality": "未提及", "coating": "未提及"},
                "pulse": "未提及",
                "inspection": [],
                "inquiry": [],
                "disease_course": "未提及",
                "constitution": "未提及",
                "chief_complaint": response[:200],
                "_raw_response": response,
            }
