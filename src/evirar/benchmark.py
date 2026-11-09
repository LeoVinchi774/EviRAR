"""ATCMD-Bench dataset loading and case-conditioned simulator prompts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evirar.clients import Runtime
from evirar.protocol import parse_doctor_action


REQUIRED_CASE_FIELDS = ("id", "患者个人信息", "问诊信息", "其余信息", "诊断结果", "诊断依据")


def _patient_profile_text(value: Any, case_id: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, (str, int, float))
        for key, item in value.items()
    ):
        return str(value)
    raise ValueError(f"Dataset case {case_id} has an unsupported 患者个人信息 value")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path).expanduser()
    try:
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read ATCMD-Bench JSON dataset at {dataset_path}: {exc}") from exc
    if not isinstance(cases, list) or not cases:
        raise ValueError("ATCMD-Bench dataset must be a non-empty JSON array")
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Dataset case at index {index} must be a JSON object")
        missing = [key for key in REQUIRED_CASE_FIELDS if key not in case]
        if missing:
            raise ValueError(f"Dataset case at index {index} is missing fields: {', '.join(missing)}")
        case_id = str(case["id"])
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Dataset has a missing or duplicate case id: {case_id!r}")
        seen_ids.add(case_id)
        department = case.get("科室", "未标注")
        if not isinstance(department, str):
            raise ValueError(f"Dataset case {case_id} has a non-text 科室 field")
        case["科室"] = department
        case["患者个人信息"] = _patient_profile_text(case["患者个人信息"], case_id)
        if any(
            not isinstance(case[key], str)
            for key in REQUIRED_CASE_FIELDS[1:]
            if key != "患者个人信息"
        ):
            raise ValueError(f"Dataset case {case_id} has a non-text required field")
    return cases


PATIENT_PROMPT = """你是一名患者,现在正与一位经验丰富的中医进行对话以寻求医生的意见，严格遵守以下要求:
1. 医案信息[Case]标记: 严格按照医案中记载的症状信息回答，不要添加医案中没有的症状，要把专业的症状描述转换成日常用语。
2. 注意历史对话: 遵循先前对话的语境和已提及的信息，避免前后矛盾。
3. 首次描述主要症状: 只说最困扰的1-2个症状，使用生活化的语言，简要描述。
4. 回应医生追问时: 不要出现"医案里没有提到"这样的回答。根据医案信息回答具体症状。当医生询问的信息不存在于医案时，回答"不清楚"或"没太注意"。
5. 回答原则: 不要一次性说太多症状，描述可简洁自然。
[Case]
{patient_full_info}
[/Case]
"""

ASSISTANT_PROMPT = """你是一位中医助理，已掌握患者的脉诊、舌诊以及相关实验室检查等信息。
行为规范：
(1) 基于医生的询问提供信息：当医生问及患者的脉象、舌象或具体的实验室检查结果时，直接引用你所掌握的资料中对应的内容做出回答。
(2) 若信息缺失则声明"没有相关信息"。
(3) 回答必须简洁、客观、准确，不输出多余描述或额外推断。
以下为目前掌握的患者信息：
{assistant_full_info}"""

class CaseSimulator:
    """Keep separate patient/assistant histories in each simulated speaker's view."""

    def __init__(self, runtime: Runtime, case: dict[str, Any]):
        self.runtime = runtime
        patient_full_info = (
            f"患者个人信息：{case['患者个人信息']}\n问诊信息：{case['问诊信息']}"
            f"\n其余信息：{case['其余信息']}"
        )
        assistant_full_info = f"助理所掌握的患者信息：{case['问诊信息']}\n{case['其余信息']}"
        self.patient_messages = [
            {"role": "system", "content": PATIENT_PROMPT.format(patient_full_info=patient_full_info)},
            {"role": "user", "content": "你好，请问有哪里不舒服的吗"},
        ]
        self.assistant_messages = [
            {"role": "system", "content": ASSISTANT_PROMPT.format(assistant_full_info=assistant_full_info)},
        ]
        self._opening: str | None = None

    def opening(self) -> str:
        if self._opening is None:
            self._opening = self.runtime.chat(
                self.patient_messages, service="simulator", label="patient_opening", temperature=0.6,
            )
            self.patient_messages.append({"role": "assistant", "content": self._opening})
        return self._opening

    def __call__(self, history: list[dict[str, str]]) -> str:
        if self._opening is None:
            raise RuntimeError("Generate the case opening before requesting a simulator reply")
        last_doctor = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
        )
        if not last_doctor:
            raise ValueError("Simulator requires a doctor question")
        action, payload = parse_doctor_action(last_doctor)
        if action == "<诊断结束>":
            raise ValueError("A completed diagnosis does not require a simulator reply")
        if action == "<对助理>":
            messages, label = self.assistant_messages, "assistant_simulator"
        else:
            messages, label = self.patient_messages, "patient_simulator"
        messages.append({"role": "user", "content": payload})
        reply = self.runtime.chat(messages, service="simulator", label=label, temperature=0.6)
        messages.append({"role": "assistant", "content": reply})
        return reply
