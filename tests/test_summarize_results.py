import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summarize_results_script_writes_benchmark_scale_summary(tmp_path):
    result_path = tmp_path / "baseline_20261109.jsonl"
    records = [
        {
            "id": "case-1",
            "问诊评分": {"reason": "完整", "score": 7},
            "诊断依据评分": {"reason": "合理", "score": 8},
            "诊断结果评分": {"reason": "相符", "score": 6},
        },
        {
            "id": "case-2",
            "问诊评分": {"reason": "完整", "score": 8},
            "诊断依据评分": {"reason": "合理", "score": 7},
            "诊断结果评分": {"reason": "相符", "score": 9},
        },
    ]
    result_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_results.py"), str(result_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "baseline_20261109_summary.json").read_text(encoding="utf-8"))
    assert summary["cases"] == 2
    assert summary["scored_cases"] == 2
    assert summary["means"] == {
        "问诊评分": 75.0,
        "诊断依据评分": 75.0,
        "诊断结果评分": 75.0,
        "三项均分": 75.0,
    }


def test_summarize_results_reports_bad_jsonl_line_number(tmp_path):
    result_path = tmp_path / "bad.jsonl"
    result_path.write_text('{"id":"case-1"}\nnot-json\n', encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_results.py"), str(result_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "line 2" in completed.stderr
