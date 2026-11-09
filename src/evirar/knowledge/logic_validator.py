"""
Step 4：LLM辨证逻辑校验模块（Logic Validator）

利用LLM作为辨证逻辑校验器，对检索到的每条知识进行深层理论一致性验证：
- 判断知识与当前辨证方向的相关性
- 验证证候与症状组合的中医理论逻辑关联
- 输出相关性评分并过滤低分结果

对应 TCM_BianZheng_RAE_Proposal.md 中5.1.2节 Step 4。
"""

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from evirar.knowledge import config
from evirar.knowledge.llm_client import call_llm
from evirar.knowledge.syndrome_labeler import SyndromeLabels
from evirar.knowledge.utils import format_clinical_features
from evirar.clients import bind_runtime, current_runtime

logger = logging.getLogger(__name__)

LOGIC_VALIDATION_PROMPT = """\
你是中医辨证逻辑校验器。请判断以下检索到的知识与当前临床特征的辨证相关性。

## 当前临床特征
{clinical_features}

## 辨证方向
{syndrome_direction}

## 检索到的知识
{retrieved_knowledge}

## 校验要求

请基于中医辨证论治理论进行判断，考虑以下维度的逻辑一致性：
- 寒热虚实的一致性
- 脏腑归经的关联性
- 病因病机的逻辑链条
- 症状组合的辨证意义

## 输出格式

请严格以如下JSON格式输出，不要输出其他内容：

```json
{{
  "relevance": "相关/部分相关/不相关",
  "relevance_score": 0.0,
  "reasoning": "判断理由（简要说明为何相关或不相关）",
  "key_connections": ["与当前症状的关键关联点1", "关联点2"]
}}
```

注意：
- relevance_score 取值范围 0-1，1表示高度相关
- "相关"对应评分 0.7-1.0
- "部分相关"对应评分 0.4-0.7
- "不相关"对应评分 0-0.4"""

LOGIC_VALIDATION_BATCH_PROMPT = """\
你是中医辨证逻辑校验器。下面给出 {card_count} 张候选知识卡，请分别判断每张卡与当前临床特征的辨证相关性。

## 当前临床特征
{clinical_features}

## 辨证方向
{syndrome_direction}

## 校验要求

请基于中医辨证论治理论逐张判断，考虑以下维度的逻辑一致性：
- 寒热虚实的一致性
- 脏腑归经的关联性
- 病因病机的逻辑链条
- 症状组合的辨证意义

## 候选知识卡

{retrieved_knowledge}

## 输出格式

请严格以如下JSON对象格式输出，不要输出其他内容。results数组必须恰好包含 {card_count} 个元素，index 与候选编号一一对应。每项只输出index和relevance_score，不要输出reasoning或key_connections：

```json
{{
  "results": [
    {{"index": 1, "relevance_score": 0.0}}
  ]
}}
```

注意：index 从 1 到 {card_count}，不得遗漏或重复；relevance_score 取值范围 0-1。只需输出判断分数，不要生成理由。"""


class LogicValidator:
    """LLM辨证逻辑校验器：验证检索结果与辨证方向的理论一致性。"""

    def __init__(self, threshold: float | None = None):
        """
        初始化逻辑校验器。

        Args:
            threshold: 相关性评分阈值，低于此值的结果将被过滤。
                默认使用配置值。
        """
        self._threshold = (
            config.LOGIC_VALIDATION_THRESHOLD if threshold is None else threshold
        )

    def validate_and_rerank(
        self,
        retrieved_results: list[dict[str, Any]],
        clinical_features: dict[str, Any],
        syndrome_labels: SyndromeLabels,
    ) -> list[dict[str, Any]]:
        """
        对检索结果进行辨证逻辑校验、过滤和重排序。

        Args:
            retrieved_results: Step 3的检索结果列表
            clinical_features: 临床特征字典
            syndrome_labels: 辨证标签

        Returns:
            经过校验、过滤和重排序的知识列表
        """
        if not retrieved_results:
            return []

        features_text = format_clinical_features(clinical_features)
        direction_text = self._format_syndrome_direction(syndrome_labels)

        workers = min(
            len(retrieved_results), config.LOGIC_VALIDATION_MAX_WORKERS
        )
        if config.LOGIC_VALIDATION_BATCH and len(retrieved_results) > 1:
            validations = self._validate_batch(
                retrieved_results, features_text, direction_text
            )
            mode_text = f"批量 {max(1, config.LOGIC_VALIDATION_BATCH_SIZE)}"
        else:
            def validate_one(result: dict[str, Any]) -> dict[str, Any]:
                with current_runtime().validation_semaphore:
                    return self._validate_single(
                        result, features_text, direction_text
                    )

            if workers == 1:
                validations = [validate_one(result) for result in retrieved_results]
            else:
                # executor.map按输入顺序返回，确保卡片与评分仍按原顺序配对。
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    validations = list(
                        executor.map(bind_runtime(validate_one), retrieved_results)
                    )
            mode_text = f"并发 {workers}"

        validated_results = []
        for result, validation in zip(retrieved_results, validations):
            result["validation"] = validation
            result["relevance_score"] = validation.get("relevance_score", 0.0)
            result["relevance"] = validation.get("relevance", "不相关")

            if result["relevance_score"] >= self._threshold:
                validated_results.append(result)
            else:
                logger.debug(
                    "过滤低相关性知识: %s (评分: %.2f)",
                    result.get("syndrome_name", "未知"),
                    result["relevance_score"],
                )

        # 按校验评分重排序
        validated_results.sort(
            key=lambda x: x["relevance_score"], reverse=True
        )

        logger.info(
            "辨证逻辑校验完成：%d / %d 条知识通过校验（阈值 %.2f，%s）",
            len(validated_results),
            len(retrieved_results),
            self._threshold,
            mode_text,
        )

        return validated_results

    def _validate_batch(
        self,
        retrieved_results: list[dict[str, Any]],
        features_text: str,
        direction_text: str,
    ) -> list[dict[str, Any]]:
        """合并校验候选卡；无法可靠解析的卡片回退到原单卡校验。"""
        batch_size = max(1, config.LOGIC_VALIDATION_BATCH_SIZE)
        validations: list[dict[str, Any] | None] = [None] * len(retrieved_results)

        for start in range(0, len(retrieved_results), batch_size):
            chunk = retrieved_results[start : start + batch_size]
            with current_runtime().validation_semaphore:
                parsed = self._validate_chunk(chunk, features_text, direction_text)
            validations[start : start + len(chunk)] = parsed

        for index, validation in enumerate(validations):
            if validation is None:
                logger.warning(
                    "批量校验缺少第 %d 张卡的有效评分，回退单卡校验",
                    index + 1,
                )
                with current_runtime().validation_semaphore:
                    validations[index] = self._validate_single(
                        retrieved_results[index], features_text, direction_text
                    )

        return [validation for validation in validations if validation is not None]

    def _validate_chunk(
        self,
        chunk: list[dict[str, Any]],
        features_text: str,
        direction_text: str,
    ) -> list[dict[str, Any] | None]:
        cards_text = "\n\n".join(
            f"### 候选 {index + 1}\n{self._truncate_content(result)}"
            for index, result in enumerate(chunk)
        )
        prompt = LOGIC_VALIDATION_BATCH_PROMPT.format(
            card_count=len(chunk),
            clinical_features=features_text,
            syndrome_direction=direction_text,
            retrieved_knowledge=cards_text,
        )
        try:
            response = call_llm(
                prompt,
                max_tokens=config.LOGIC_VALIDATION_BATCH_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("批量辨证逻辑校验调用失败")
            return [None] * len(chunk)
        return self._parse_batch_response(response, len(chunk))

    def _parse_batch_response(
        self, response: str, expected: int
    ) -> list[dict[str, Any] | None]:
        """解析批量响应并按1-based index对齐；无效或重复项回退单卡。"""
        results: list[dict[str, Any] | None] = [None] * expected
        try:
            data = json.loads(self._strip_code_fence(response))
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("批量校验结果解析失败，回退单卡校验")
            return results

        items = data.get("results") if isinstance(data, dict) else data
        if not isinstance(items, list):
            logger.warning("批量校验响应不是JSON数组，回退单卡校验")
            return results

        seen: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index")
            if isinstance(raw_index, bool):
                continue
            if isinstance(raw_index, int):
                index = raw_index - 1
            elif isinstance(raw_index, str) and raw_index.strip().isdigit():
                index = int(raw_index.strip()) - 1
            else:
                continue
            if not 0 <= index < expected:
                continue
            if index in seen:
                results[index] = None
                continue
            seen.add(index)
            results[index] = self._normalize_batch_validation(item)
        return results

    @staticmethod
    def _normalize_batch_validation(
        data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """批量响应缺分或分值非法时标为不可用，避免把错误当作有效低分。"""
        try:
            score = float(data["relevance_score"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(score):
            return None
        normalized = {key: value for key, value in data.items() if key != "index"}
        normalized["relevance_score"] = max(0.0, min(1.0, score))
        normalized.setdefault("relevance", "未知")
        normalized.setdefault("reasoning", "")
        normalized.setdefault("key_connections", [])
        return normalized

    @staticmethod
    def _strip_code_fence(response: str) -> str:
        text = response.strip()
        if "```json" in text:
            return text.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text

    @staticmethod
    def _truncate_content(result: dict[str, Any]) -> str:
        """截取知识内容用于校验，避免候选卡造成过长输入。"""
        knowledge_text = result.get("content", "")
        max_len = config.VALIDATION_CONTENT_MAX_LENGTH
        if len(knowledge_text) > max_len:
            return knowledge_text[:max_len] + "...(已截断)"
        return knowledge_text

    def _validate_single(
        self,
        result: dict[str, Any],
        features_text: str,
        direction_text: str,
    ) -> dict[str, Any]:
        """对单条检索结果进行辨证逻辑校验。"""
        prompt = LOGIC_VALIDATION_PROMPT.format(
            clinical_features=features_text,
            syndrome_direction=direction_text,
            retrieved_knowledge=self._truncate_content(result),
        )

        try:
            response = call_llm(prompt)
            return self._parse_validation_response(response)
        except Exception:
            logger.exception("辨证逻辑校验调用失败")
            return {
                "relevance": "未知",
                "relevance_score": 0.0,
                "reasoning": "校验调用失败",
                "key_connections": [],
            }

    def _parse_validation_response(self, response: str) -> dict[str, Any]:
        """解析校验响应。"""
        text = self._strip_code_fence(response)

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("校验响应不是JSON对象")
            # 确保评分在合理范围
            score = float(data.get("relevance_score", 0.0))
            if not math.isfinite(score):
                score = 0.0
            data["relevance_score"] = max(0.0, min(1.0, score))
            return data
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
            logger.warning("校验结果解析失败，默认标记为部分相关")
            return {
                "relevance": "部分相关",
                "relevance_score": 0.5,
                "reasoning": "校验结果解析失败",
                "key_connections": [],
            }

    def _format_syndrome_direction(self, labels: SyndromeLabels) -> str:
        """格式化辨证方向为文本。"""
        parts = []
        if labels.inclusion_labels:
            parts.append(f"辨证标签：{', '.join(labels.inclusion_labels)}")
        if labels.organs_involved:
            parts.append(f"涉及脏腑：{', '.join(labels.organs_involved)}")
        if labels.cold_heat != "待定":
            parts.append(f"寒热：{labels.cold_heat}")
        if labels.deficiency_excess != "待定":
            parts.append(f"虚实：{labels.deficiency_excess}")
        if labels.analysis_brief:
            parts.append(f"分析：{labels.analysis_brief}")

        return "\n".join(parts) if parts else "（辨证方向未明确）"
