"""Doctor action protocol shared by interactive and scripted consultations."""
import re


def extract_thinking(content: str, reason_content: str | None = None) -> tuple[str, str | None]:
    match = re.search(r"<think>(.*?)</think>\s*(.*)", content, re.DOTALL)
    return (match.group(2).strip(), match.group(1).strip()) if match else (content, reason_content)


def parse_doctor_action(response_text: str) -> tuple[str, str]:
    clean, _ = extract_thinking(response_text)
    stripped = clean.strip()
    for prefix in ("<对患者>", "<对助理>", "<诊断结束>"):
        if stripped.startswith(prefix):
            return prefix, stripped[len(prefix):].strip()
    return "<对患者>", stripped

