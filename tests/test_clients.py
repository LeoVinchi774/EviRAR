from types import SimpleNamespace

from evirar.clients import Runtime
from evirar.config import load_settings


def test_external_simulation_and_scoring_do_not_inherit_doctor_sampling(monkeypatch):
    settings = load_settings("configs/default.yaml", env_file="missing.env")
    runtime = Runtime(settings)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))], usage=None)

    monkeypatch.setattr(runtime, "client", lambda _kind: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ))
    messages = [{"role": "user", "content": "test"}]
    runtime.chat(messages, label="doctor")
    runtime.chat(messages, label="reasoning")
    runtime.chat(messages, service="simulator", label="patient_simulator", temperature=0.6)
    runtime.chat(messages, service="simulator", label="expert_evaluator", temperature=0)
    doctor, reasoning, patient, expert = calls
    assert doctor["presence_penalty"] == 1.5
    assert doctor["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert doctor["stop"] == ["<|im_end|>"]
    assert "stop" not in reasoning
    for request in (patient, expert):
        assert request["presence_penalty"] == 0
        assert request["max_tokens"] == 2048
        assert request["extra_body"] == {"enable_thinking": False}
    assert patient["temperature"] == 0.6
    assert expert["temperature"] == 0
