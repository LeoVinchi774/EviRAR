from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from evirar import consultation as module


@pytest.mark.parametrize("final_reply, forced", [
    ("<诊断结束>病名及证型，依据已知症状。", False),
    ("<诊断结束>病名及证型，依据已知症状。", True),
    ("<对患者>请继续回答。", True),
])
def test_turn_budget_requires_real_final_diagnosis(monkeypatch, final_reply, forced):
    calls = []
    patient_calls = []
    refinements = []

    class FakeRuntime:
        def __init__(self, _settings):
            pass

        @contextmanager
        def activate(self):
            yield self

        def chat(self, messages, *, label):
            calls.append((messages, label))
            return "<对患者>多久了？" if forced and len(calls) == 1 else final_reply

    class FakeContext:
        def __init__(self, **_options):
            pass

        def run_behind_the_scenes(self, _messages, _turn):
            return ""

        def refine_diagnosis(self, messages, diagnosis):
            refinements.append((messages, diagnosis))
            return {"refined": True, "final_diagnosis": "精炼后的诊断及依据。"}

    def patient_reply(history):
        patient_calls.append(list(history))
        return "两天。"

    monkeypatch.setattr(module, "Runtime", FakeRuntime)
    monkeypatch.setattr(module, "ContextManager", FakeContext)
    settings = SimpleNamespace(enhanced=False, values={"runtime": {"max_turns": 1}})
    consultation = module.Consultation(settings)
    if final_reply.startswith("<对患者>"):
        with pytest.raises(RuntimeError, match="did not return a final diagnosis"):
            consultation.run("头痛。", patient_reply, lambda _reply: None)
        assert not refinements
    else:
        assert consultation.run("头痛。", patient_reply, lambda _reply: None) == "精炼后的诊断及依据。"
        assert consultation.messages[-1]["content"] == "<诊断结束> 精炼后的诊断及依据。"
        assert len(refinements) == 1
    assert len(patient_calls) == int(forced)
    assert len(calls) == 1 + int(forced)
    if forced:
        assert calls[-1][1] == "doctor_forced_diagnosis"
        assert calls[-1][0][-1]["role"] == "user"
        assert "请必须使用<诊断结束>前缀" in calls[-1][0][-1]["content"]
        # Preserve the same explicit control instruction as the benchmark.
        assert calls[-1][0][-1] in consultation.messages


def test_enhancement_starts_after_first_doctor_question_and_never_persists_in_history(monkeypatch):
    calls, retrieval_turns = [], []

    class FakeRuntime:
        def __init__(self, _settings):
            pass

        @contextmanager
        def activate(self):
            yield self

        def chat(self, messages, *, label):
            calls.append([dict(m) for m in messages])
            return "<对患者>多久了？" if len(calls) == 1 else "<诊断结束>诊断及依据。"

    class FakeContext:
        def __init__(self, **_options):
            pass

        def run_behind_the_scenes(self, messages, turn):
            retrieval_turns.append(turn)
            return "[检索辅助]"

        def refine_diagnosis(self, messages, diagnosis):
            return {"refined": False}

    monkeypatch.setattr(module, "Runtime", FakeRuntime)
    monkeypatch.setattr(module, "ContextManager", FakeContext)
    settings = SimpleNamespace(enhanced=True, values={"runtime": {"max_turns": 3}},
                               resource_path=lambda _kind: "cards")
    consultation = module.Consultation(settings)
    consultation.run("头痛。", lambda _history: "两天。", lambda _reply: None)
    assert retrieval_turns == [1]
    assert "[检索辅助]" not in str(calls[0])
    assert calls[1][-1]["content"] == "两天。[检索辅助]"
    assert "[检索辅助]" not in str(consultation.messages)
    with pytest.raises(RuntimeError, match="new Consultation for each case"):
        consultation.run("新病例", lambda _history: "", lambda _reply: None)
