"""Validated settings; secrets are read from the environment, never YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


DEFAULTS: dict[str, Any] = {
    "mode": "evirar",
    "resources": {"knowledge": "../resources/knowledge", "experience": "../resources/experience"},
    "generation": {"temperature": 1.0, "max_tokens": 2048, "top_p": 0.95,
                   "presence_penalty": 1.5, "extra_body": {}},
    "simulator_generation": {"temperature": 0.7, "max_tokens": 2048, "top_p": 0.95,
                             "presence_penalty": 0.0, "extra_body": {"enable_thinking": False}},
    "retrieval": {"top_k": 10, "validation_threshold": 0.6, "max_decision_cards": 5,
                  "metadata_prefilter": False, "evidence_classification": True,
                  "dynamic_evidence": True, "validation_batch": True,
                  "validation_batch_size": 10, "validation_workers": 10},
    "reflection": {"max_iterations": 3, "score_threshold": 7.0},
    "runtime": {"max_turns": 10, "timeout_seconds": 120.0, "max_retries": 2,
                "embedding_batch_size": 16},
}


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str = field(repr=False)

    def validate(self, label: str) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"{label}: API_BASE must be an http(s) URL")
        if not self.model or not self.api_key:
            raise ValueError(f"{label}: set MODEL and API_KEY in .env (local key can be 'local')")


@dataclass(frozen=True)
class Settings:
    path: Path
    values: dict[str, Any]
    llm: Endpoint
    embedding: Endpoint
    simulator: Endpoint

    @property
    def enhanced(self) -> bool:
        return self.values["mode"] == "evirar"

    def resource_path(self, kind: str) -> Path:
        return (self.path.parent / self.values["resources"][kind]).resolve()

    def validate_services(self, *, simulator: bool = False) -> None:
        self.llm.validate("TCM_LLM")
        if self.enhanced:
            self.embedding.validate("TCM_EMBEDDING")
        if simulator:
            self.simulator.validate("TCM_SIMULATOR")

    def validate_resources(self) -> dict[str, int]:
        if not self.enhanced:
            return {}
        from evirar.knowledge.knowledge_card_parser import KnowledgeCardParser
        counts = {}
        for kind in ("knowledge", "experience"):
            path = self.resource_path(kind)
            files = sorted(path.glob("*.md"))
            if not files:
                raise ValueError(f"No {kind} cards in {path}")
            if any(not p.read_text(encoding="utf-8").strip() for p in files):
                raise ValueError(f"Empty {kind} card in {path}")
            if kind == "knowledge":
                parser = KnowledgeCardParser()
                for file in files:
                    if not parser.parse_file(str(file)).syndrome_name:
                        raise ValueError(f"Knowledge card has no syndrome name: {file}")
            counts[kind] = len(files)
        return counts


def load_settings(path: str | Path = "configs/default.yaml", *, env_file: str | Path | None = None) -> Settings:
    import copy
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"Config not found: {config_path}. Use --config /path/to/config.yaml")
    env_path = Path(env_file).expanduser().resolve() if env_file else config_path.parent.parent / ".env"
    load_dotenv(env_path, override=False)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    values = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key not in values:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(values[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            unknown = value.keys() - values[key].keys()
            if unknown:
                raise ValueError(f"Unknown {key} settings: {sorted(unknown)}")
            values[key].update(value)
        else:
            values[key] = value
    if values["mode"] not in {"baseline", "evirar"}:
        raise ValueError("mode must be baseline or evirar")
    for section, keys in {
        "runtime": ("max_turns", "embedding_batch_size"),
        "generation": ("max_tokens",),
        "simulator_generation": ("max_tokens",),
        "retrieval": ("top_k", "max_decision_cards", "validation_batch_size", "validation_workers"),
        "reflection": ("max_iterations",),
    }.items():
        for key in keys:
            value = values[section][key]
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ("metadata_prefilter", "evidence_classification", "dynamic_evidence", "validation_batch"):
        if type(values["retrieval"][key]) is not bool:
            raise ValueError(f"retrieval.{key} must be a YAML boolean")
    if values["retrieval"]["dynamic_evidence"] and not values["retrieval"]["evidence_classification"]:
        raise ValueError("dynamic_evidence requires evidence_classification")
    if values["retrieval"]["max_decision_cards"] > 5:
        raise ValueError("max_decision_cards must be between 1 and 5")
    for section, key, low, high in (
        ("retrieval", "validation_threshold", 0, 1),
        ("reflection", "score_threshold", 1, 10),
        ("generation", "temperature", 0, 2),
        ("generation", "top_p", 0, 1),
        ("generation", "presence_penalty", -2, 2),
        ("simulator_generation", "temperature", 0, 2),
        ("simulator_generation", "top_p", 0, 1),
        ("simulator_generation", "presence_penalty", -2, 2),
    ):
        value = values[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{section}.{key} must be in [{low}, {high}]")
    if values["runtime"]["timeout_seconds"] <= 0 or type(values["runtime"]["max_retries"]) is not int or values["runtime"]["max_retries"] < 0:
        raise ValueError("timeout_seconds must be positive and max_retries a nonnegative integer")
    for section in ("generation", "simulator_generation"):
        if not isinstance(values[section]["extra_body"], dict):
            raise ValueError(f"{section}.extra_body must be a mapping")
    for kind, value in values["resources"].items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"resources.{kind} must be a nonempty path")

    def endpoint(prefix: str, fallback: Endpoint | None = None) -> Endpoint:
        return Endpoint(
            os.environ.get(prefix + "_API_BASE", "").strip() or (fallback.base_url if fallback else ""),
            os.environ.get(prefix + "_MODEL", "").strip() or (fallback.model if fallback else ""),
            os.environ.get(prefix + "_API_KEY", "").strip() or (fallback.api_key if fallback else ""),
        )
    llm = endpoint("TCM_LLM")
    return Settings(config_path, values, llm, endpoint("TCM_EMBEDDING"), endpoint("TCM_SIMULATOR", llm))

