"""ATCMD-Bench expert scoring for completed consultation dialogues."""
from __future__ import annotations

import json
import re
from typing import Any

from evirar.clients import Runtime


SCORE_FIELDS = ("问诊评分", "诊断依据评分", "诊断结果评分")

EXPERT_SCORE_JSON_FORMAT = """{
    "问诊评分": {
        "reason": "给出该分数的简要理由",
        "score": "评分（1-10）"
    },
    "诊断依据评分": {
        "reason": "给出该分数的简要理由",
        "score": "评分（1-10）"
    },
    "诊断结果评分": {
        "reason": "给出该分数的简要理由",
        "score": "评分（1-10）"
    }
}"""

EXPERT_PROMPT = """你是一位资深的中医临床专家。你的任务是基于「原始医案」, 严格、客观地对比分析「模型问诊对话」, 并从三个维度进行评分。

### 评估维度

1.  **问诊信息收集**:
    * **核心考察点**: 模型是否通过提问，全面覆盖了医案中的主诉、关键症状、相关病史及舌脉信息？是否存在对核心信息的重大遗漏？

2.  **诊断依据与推理**:
    * **核心考察点**: 模型的诊断推理是否符合中医理论、逻辑清晰？诊断依据是否能充分解释其收集到的症状和体征？推理过程是否与原始医案的辨证思路大体一致？

3.  **诊断结果准确性**:
    * **核心考察点**: 模型的最终诊断（病名和中医证型）与原始医案的结论是否一致或高度相似？

### 评分要求

* 为每个维度提供 **1-10分** 的评分 (1分最差, 10分最佳)。
* 为每个评分提供 **简明扼要** 的理由。
* **按以下述JSON格式输出评分：
{json_format}

**输入数据：**
1. **原始医案信息**：
{expert_full_info}
2. **模型问诊对话内容**：
{dialogue}

**请开始您的评估。**
"""

def clean_dialogue(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Remove the private system prompt, reasoning tags, and routing labels."""
    cleaned = []
    for message in messages:
        role = message.get("role")
        if role not in {"assistant", "user"}:
            continue
        content = str(message.get("content", ""))
        if role == "assistant":
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"^\s*<(?:对患者|对助理|诊断结束)>\s*", "", content)
        cleaned.append({"role": role, "content": content.strip()})
    return cleaned


def build_expert_prompt(case: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    dialogue = json.dumps(clean_dialogue(messages), ensure_ascii=False, indent=4)
    case_info = (
        f"患者个人信息：{case['患者个人信息']}\n"
        f"问诊信息：{case['问诊信息']}\n"
        f"其余信息：{case['其余信息']}\n"
        f"诊断结果：{case['诊断结果']}\n"
        f"诊断依据：\n{case['诊断依据']}"
    )
    return EXPERT_PROMPT.format(
        json_format=EXPERT_SCORE_JSON_FORMAT, expert_full_info=case_info, dialogue=dialogue,
    )


def parse_expert_score(response_text: str) -> dict[str, dict[str, Any]]:
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Expert scorer returned an empty response")
    start, end = response_text.find("{"), response_text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Expert scorer response does not contain a JSON object")
    try:
        parsed = json.loads(response_text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Expert scorer returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Expert scorer response must be a JSON object")

    result = {}
    for field in SCORE_FIELDS:
        entry = parsed.get(field)
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"Expert scorer response is missing {field}.score")
        try:
            score = float(entry["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}.score must be a number from 1 to 10") from exc
        if not 1 <= score <= 10:
            raise ValueError(f"{field}.score must be between 1 and 10")
        reason = entry.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError(f"{field}.reason must be text")
        result[field] = {"reason": reason, "score": score}
    return result


def score_dialogue(runtime: Runtime, case: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = build_expert_prompt(case, messages)
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = runtime.chat(
                [{"role": "user", "content": prompt}],
                service="simulator", label="expert_evaluator", temperature=0,
            )
            scores = parse_expert_score(response)
            raw_mean = sum(item["score"] for item in scores.values()) / len(SCORE_FIELDS)
            scores["三项均分"] = round(raw_mean * 10, 2)
            return scores
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Expert scoring failed after two attempts: {last_error}") from last_error


def build_score_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    score_records = [record for record in records if all(
        isinstance(record.get(field), dict) and "score" in record[field]
        for field in SCORE_FIELDS
    )]
    means = {
        field: round(sum(float(record[field]["score"]) for record in score_records) / len(score_records) * 10, 2)
        for field in SCORE_FIELDS
    } if score_records else {field: None for field in SCORE_FIELDS}
    means["三项均分"] = round(sum(means[field] for field in SCORE_FIELDS) / len(SCORE_FIELDS), 2) if score_records else None
    return {"cases": len(records), "scored_cases": len(score_records), "score_scale": "0-100", "means": means}
