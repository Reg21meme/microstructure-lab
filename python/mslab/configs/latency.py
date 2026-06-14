"""
latency.py
Load latency scenarios from configs/latency_scenarios.yaml.

Usage:
    from mslab.configs.latency import load_latency, LatencyConfig
    cfg = load_latency("fast")
    print(cfg.base_ms, cfg.jitter_ms)
"""

import pathlib
from dataclasses import dataclass

import yaml

ROOT         = pathlib.Path(__file__).resolve().parents[3]
LATENCY_YAML = ROOT / "configs" / "latency_scenarios.yaml"


@dataclass
class LatencyConfig:
    scenario: str
    base_ms: float
    jitter_ms: float
    description: str = ""


def load_latency(scenario: str = "fast") -> LatencyConfig:
    with open(LATENCY_YAML) as f:
        data = yaml.safe_load(f)

    scenarios = data.get("scenarios", {})
    if scenario not in scenarios:
        available = list(scenarios.keys())
        raise ValueError(
            f"Unknown latency scenario '{scenario}'. Available: {available}"
        )

    s = scenarios[scenario]
    return LatencyConfig(
        scenario=scenario,
        base_ms=float(s["base_ms"]),
        jitter_ms=float(s["jitter_ms"]),
        description=s.get("description", ""),
    )


def list_scenarios() -> list[str]:
    with open(LATENCY_YAML) as f:
        data = yaml.safe_load(f)
    return list(data.get("scenarios", {}).keys())