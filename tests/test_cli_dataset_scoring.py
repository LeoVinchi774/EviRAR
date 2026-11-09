import json
from contextlib import contextmanager
from types import SimpleNamespace

from evirar import cli


def test_dataset_cli_writes_dialogue_scores_events_and_summary(tmp_path, monkeypatch):
    case = {
        "id": "case-1", "科室": "内科", "患者个人信息": "女，45岁", "问诊信息": "头痛。",
        "其余信息": "舌淡红。", "诊断结果": "证型", "诊断依据": "依据",
    }

    class FakeSettings:
        values = {"mode": "baseline"}

        def validate_services(self, *, simulator=False):
            assert simulator

    class FakeRuntime:
        def __init__(self):
            self.events = []

        @contextmanager
        def activate(self):
            yield self

        def ensure_success(self, _since=0):
            return None

    consultations = []

    class FakeConsultation:
        def __init__(self, _settings):
            self.runtime = FakeRuntime()
            self.reflection_result = {}
            self.context = SimpleNamespace(_kb_gate_turns=[])
            self.closed = False
            consultations.append(self)
            self.messages = [{"role": "system", "content": "private"}]

        def run(self, _opening, _respondent, _on_doctor_reply):
            self.messages.extend([
                {"role": "user", "content": "头痛"},
                {"role": "assistant", "content": "<诊断结束>证型"},
            ])
            return "证型"

        def close(self):
            self.closed = True

    def fake_score(runtime, _case, _messages):
        runtime.events.append({"service": "simulator", "label": "expert_evaluator", "ok": True})
        return {
            "问诊评分": {"reason": "r1", "score": 7.0},
            "诊断依据评分": {"reason": "r2", "score": 8.0},
            "诊断结果评分": {"reason": "r3", "score": 6.0},
            "三项均分": 70.0,
        }

    output_path = tmp_path / "run.jsonl"
    monkeypatch.setattr(cli, "load_settings", lambda *_args, **_kwargs: FakeSettings())
    monkeypatch.setattr(cli, "load_dataset", lambda _path: [case, {**case, "id": "case-2"}])
    monkeypatch.setattr(cli, "Consultation", FakeConsultation)
    monkeypatch.setattr(cli, "CaseSimulator", lambda *_args: SimpleNamespace(opening=lambda: "头痛"))
    monkeypatch.setattr(cli, "score_dialogue", fake_score)

    assert cli.main([
        "--dataset", "cases.json", "--simulated-patient", "--output-jsonl", str(output_path),
    ]) == 0

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    record = records[0]
    assert len(consultations) == 2
    assert all(c.closed for c in consultations)
    assert all(len(r["api_events"]) == 1 for r in records)
    assert all(len(r["dialogue"]) == 2 for r in records)
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert record["dialogue"][-1]["content"] == "<诊断结束>证型"
    assert record["问诊评分"]["score"] == 7.0
    assert record["api_events"][0]["label"] == "expert_evaluator"
    assert summary["means"]["三项均分"] == 70.0
