import json

import pytest

from evirar.scoring import (
    SCORE_FIELDS,
    build_expert_prompt,
    build_score_summary,
    parse_expert_score,
    score_dialogue,
)


CASE = {
    "患者个人信息": "女，45岁",
    "问诊信息": "头痛两天，偶有恶心。",
    "其余信息": "舌淡红，苔薄白，脉弦。",
    "诊断结果": "肝阳上亢证",
    "诊断依据": "头痛，脉弦。",
}

RESPONSE = json.dumps({
    "问诊评分": {"reason": "问询较完整", "score": "7"},
    "诊断依据评分": {"reason": "推理合理", "score": 8},
    "诊断结果评分": {"reason": "证型相符", "score": 6},
}, ensure_ascii=False)


def test_expert_prompt_uses_case_and_cleans_private_prompt_and_action_tags():
    prompt = build_expert_prompt(CASE, [
        {"role": "system", "content": "PRIVATE SYSTEM PROMPT"},
        {"role": "user", "content": "我头痛。"},
        {"role": "assistant", "content": "<think>private reasoning</think><对患者>头痛何时加重？"},
        {"role": "user", "content": "下午加重。"},
        {"role": "assistant", "content": "<诊断结束>肝阳上亢证"},
    ])

    assert "PRIVATE SYSTEM PROMPT" not in prompt
    assert "private reasoning" not in prompt
    assert "<对患者>" not in prompt
    assert "<诊断结束>" not in prompt
    assert "头痛何时加重？" in prompt
    assert CASE["诊断结果"] in prompt


def test_parse_expert_score_validates_all_dimensions_and_range():
    scores = parse_expert_score(RESPONSE)
    assert tuple(scores) == SCORE_FIELDS
    assert scores["问诊评分"] == {"reason": "问询较完整", "score": 7.0}

    with pytest.raises(ValueError, match="missing 诊断依据评分"):
        parse_expert_score(json.dumps({"问诊评分": {"score": 5}, "诊断结果评分": {"score": 5}}))
    invalid = json.loads(RESPONSE)
    invalid["诊断结果评分"]["score"] = 11
    with pytest.raises(ValueError, match="between 1 and 10"):
        parse_expert_score(json.dumps(invalid))


def test_score_dialogue_records_benchmark_scale_and_provider_event():
    class FakeRuntime:
        def chat(self, messages, **kwargs):
            self.messages, self.kwargs = messages, kwargs
            return RESPONSE

    runtime = FakeRuntime()
    result = score_dialogue(runtime, CASE, [{"role": "user", "content": "头痛"}])

    assert result["三项均分"] == 70.0
    assert runtime.kwargs == {
        "service": "simulator", "label": "expert_evaluator", "temperature": 0,
    }


def test_score_summary_converts_raw_scores_to_100_point_scale():
    record = {field: {"score": value} for field, value in zip(SCORE_FIELDS, (7, 8, 6))}
    summary = build_score_summary([record])

    assert summary == {
        "cases": 1,
        "scored_cases": 1,
        "score_scale": "0-100",
        "means": {"问诊评分": 70.0, "诊断依据评分": 80.0, "诊断结果评分": 60.0, "三项均分": 70.0},
    }
