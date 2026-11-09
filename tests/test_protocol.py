from evirar.protocol import extract_thinking, parse_doctor_action


def test_extracts_thinking_and_preserves_action():
    text, thinking = extract_thinking("<think>check facts</think>\n<对患者>哪里不适？")
    assert thinking == "check facts"
    assert parse_doctor_action(text) == ("<对患者>", "哪里不适？")


def test_unprefixed_reply_defaults_to_patient_action():
    assert parse_doctor_action("请继续描述") == ("<对患者>", "请继续描述")


def test_action_routing_ignores_embedded_model_thinking():
    assert parse_doctor_action("<think>先考虑诊断</think><诊断结束>病名及证型") == (
        "<诊断结束>", "病名及证型",
    )
