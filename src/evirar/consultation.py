"""Interactive and simulated consultation loop."""
from __future__ import annotations

from typing import Callable

from evirar.clients import Runtime
from evirar.config import Settings
from evirar.context import ContextManager
from evirar.protocol import parse_doctor_action
from evirar.prompts import BASELINE_PROMPT, EVIRAR_PROMPT


class Consultation:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.runtime = Runtime(settings)
        enhanced = settings.enhanced
        system_prompt = EVIRAR_PROMPT if enhanced else BASELINE_PROMPT
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.reflection_result = {}
        with self.runtime.activate():
            self.context = ContextManager(
                enable_kb_rag=enhanced,
                enable_experience=enhanced,
                enable_self_reflection=enhanced,
                experience_md_dir=(str(settings.resource_path("experience")) if enhanced else ""),
            )

    def run(
        self,
        initial_complaint: str,
        patient_reply: Callable[[list[dict[str, str]]], str],
        on_doctor_reply: Callable[[str], None] = print,
    ) -> str:
        if not initial_complaint.strip():
            raise ValueError("Initial complaint cannot be empty")
        if len(self.messages) != 1:
            raise RuntimeError("Create a new Consultation for each case to isolate retrieval state")
        self.messages.append({"role": "user", "content": initial_complaint.strip()})
        last_reply = ""
        with self.runtime.activate():
            for turn in range(self.settings.values["runtime"]["max_turns"]):
                doctor_messages = [m for m in self.messages if m["role"] != "system"]
                context = (
                    self.context.run_behind_the_scenes(doctor_messages, turn)
                    if self.settings.enhanced and turn >= 1 else ""
                )
                request_messages = [dict(message) for message in self.messages]
                if context:
                    for message in reversed(request_messages):
                        if message["role"] == "user":
                            message["content"] += context
                            break
                last_reply = self.runtime.chat(request_messages, label="doctor")
                self.messages.append({"role": "assistant", "content": last_reply})
                on_doctor_reply(last_reply)
                action, payload = parse_doctor_action(last_reply)
                if action == "<诊断结束>":
                    return self._finish_diagnosis(payload, on_doctor_reply)
                patient = patient_reply(self.messages)
                if not patient.strip():
                    raise ValueError("Patient/assistant reply cannot be empty")
                self.messages.append({"role": "user", "content": patient.strip()})

            # Match the benchmark runner: after the question budget, request one
            # final diagnosis rather than discarding the completed consultation.
            self.messages.append({"role": "user", "content": (
                "【系统提示】：已经达到最大轮次，请根据已有信息推断出"
                "患者可能的疾病，诊断结果包括病名和中医证型，同时给出诊断依据。"
                "请必须使用<诊断结束>前缀。"
            )})
            request_messages = [dict(message) for message in self.messages]
            final_reply = self.runtime.chat(request_messages, label="doctor_forced_diagnosis")
            on_doctor_reply(final_reply)
            action, payload = parse_doctor_action(final_reply)
            if action != "<诊断结束>" or not payload.strip():
                raise RuntimeError("Doctor did not return a final diagnosis after the turn-limit instruction")
            self.messages.append({"role": "assistant", "content": final_reply})
            return self._finish_diagnosis(payload, on_doctor_reply)

    def _finish_diagnosis(self, payload: str, on_doctor_reply: Callable[[str], None]) -> str:
        if not payload.strip():
            raise RuntimeError("Doctor returned an empty final diagnosis")
        result = self.context.refine_diagnosis(
            [m for m in self.messages if m["role"] != "system"], payload
        )
        self.reflection_result = result
        if result.get("refined"):
            payload = result["final_diagnosis"]
            self.messages[-1]["content"] = "<诊断结束> " + payload
            on_doctor_reply("自省精炼后的诊断：\n" + payload)
        return payload

    def close(self) -> None:
        self.runtime.close()


def simulated_patient(runtime: Runtime) -> Callable[[list[dict[str, str]]], str]:
    def reply(messages: list[dict[str, str]]) -> str:
        prompt = [
            {"role": "system", "content": "你扮演病例中的患者或病历助理。只依据对话和已知病史回答医生刚才的问题，不主动添加未提供的信息。简洁作答。"},
            *messages[1:],
        ]
        return runtime.chat(prompt, service="simulator", label="simulator", temperature=0.7)
    return reply
