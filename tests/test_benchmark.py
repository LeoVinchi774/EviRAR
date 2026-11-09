import json

import pytest

from evirar.benchmark import CaseSimulator, load_dataset


CASE = {
    "id": "case-1",
    "科室": "内科",
    "患者个人信息": "女，45岁",
    "问诊信息": "头痛两天，偶有恶心。",
    "其余信息": "舌淡红，苔薄白，脉弦。",
    "诊断结果": "肝阳上亢证",
    "诊断依据": "头痛，脉弦。",
}


def test_load_dataset_validates_required_fields_and_unique_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([CASE], ensure_ascii=False), encoding="utf-8")

    assert load_dataset(path) == [CASE]

    path.write_text(json.dumps([CASE, CASE], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_dataset(path)


def test_load_dataset_rejects_missing_case_fields(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "case-1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        load_dataset(path)


def test_load_dataset_preserves_native_structured_profile_representation(tmp_path):
    case = {**CASE, "患者个人信息": {"姓名": "未提供", "性别": "女", "年龄": "15岁"}}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case], ensure_ascii=False), encoding="utf-8")

    loaded = load_dataset(path)

    assert loaded[0]["患者个人信息"] == str(case["患者个人信息"])


def test_load_dataset_marks_missing_department_as_unlabeled(tmp_path):
    case = {key: value for key, value in CASE.items() if key != "科室"}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case], ensure_ascii=False), encoding="utf-8")

    assert load_dataset(path)[0]["科室"] == "未标注"


def test_case_opening_uses_case_facts_without_reference_answer():
    class FakeRuntime:
        def chat(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "我这两天头痛。"

    runtime = FakeRuntime()

    assert CaseSimulator(runtime, CASE).opening() == "我这两天头痛。"
    prompt = runtime.messages[0]["content"]
    assert "女，45岁" in prompt
    assert "头痛两天" in prompt
    assert CASE["诊断结果"] not in prompt
    assert CASE["诊断依据"] not in prompt
    assert runtime.kwargs["service"] == "simulator"


@pytest.mark.parametrize(
    ("doctor_message", "expected_label", "expected_facts"),
    [
        ("<对患者>头痛是否持续？", "patient_simulator", "女，45岁"),
        ("<对助理>舌脉如何？", "assistant_simulator", "舌淡红"),
    ],
)
def test_case_respondent_routes_patient_and_assistant_turns(
    doctor_message, expected_label, expected_facts
):
    class FakeRuntime:
        def chat(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "病例记载的回复"

    runtime = FakeRuntime()
    respondent = CaseSimulator(runtime, CASE)
    respondent.opening()

    reply = respondent([
        {"role": "system", "content": "system"},
        {"role": "user", "content": "我这两天头痛。"},
        {"role": "assistant", "content": doctor_message},
    ])

    assert reply == "病例记载的回复"
    assert expected_facts in runtime.messages[0]["content"]
    assert CASE["诊断结果"] not in runtime.messages[0]["content"]
    assert runtime.kwargs["label"] == expected_label


def test_simulator_keeps_speaker_roles_and_histories_separate():
    calls = []

    class FakeRuntime:
        def chat(self, messages, **kwargs):
            calls.append(([dict(m) for m in messages], kwargs["label"]))
            return {"patient_opening": "我头痛。", "assistant_simulator": "舌淡红。"}.get(
                kwargs["label"], "两天。"
            )

    simulator = CaseSimulator(FakeRuntime(), CASE)
    assert simulator.opening() == "我头痛。"
    assert simulator.opening() == "我头痛。"  # Does not regenerate the complaint.
    for question in ("<对助理>舌象如何？", "<对患者>多久了？", "<对患者>何时加重？"):
        simulator([{"role": "assistant", "content": question}])

    assistant_history = calls[1][0]
    assert assistant_history[1:] == [{"role": "user", "content": "舌象如何？"}]
    patient_history = calls[-1][0]
    assert [(m["role"], m["content"]) for m in patient_history[1:]] == [
        ("user", "你好，请问有哪里不舒服的吗"), ("assistant", "我头痛。"),
        ("user", "多久了？"), ("assistant", "两天。"), ("user", "何时加重？"),
    ]
    assert "舌象如何？" not in str(patient_history)
    assert CASE["其余信息"] in patient_history[0]["content"]
    assert CASE["诊断结果"] not in patient_history[0]["content"]
