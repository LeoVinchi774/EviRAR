from pathlib import Path

import pytest

from evirar.config import load_settings


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_configs_load_without_secrets():
    enhanced = load_settings(ROOT / "configs/default.yaml", env_file=ROOT / "missing.env")
    baseline = load_settings(ROOT / "configs/baseline.yaml", env_file=ROOT / "missing.env")
    assert enhanced.enhanced is True
    assert baseline.enhanced is False
    assert enhanced.resource_path("knowledge").is_dir()
    assert enhanced.resource_path("experience").is_dir()


def test_resource_card_counts():
    settings = load_settings(ROOT / "configs/default.yaml", env_file=ROOT / "missing.env")
    counts = settings.validate_resources()
    assert counts == {"knowledge": 500, "experience": 500}


def test_rejects_unknown_config_key(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_settings(config_file, env_file=tmp_path / "missing.env")


def test_rejects_dynamic_gate_without_classifier(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        "mode: evirar\nretrieval:\n  evidence_classification: false\n  dynamic_evidence: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires evidence_classification"):
        load_settings(config_file, env_file=tmp_path / "missing.env")
