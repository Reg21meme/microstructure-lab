"""
fees.py
Load fee scenarios from configs/fees.yaml.

Usage:
    from mslab.configs.fees import load_fees, FeeConfig

    cfg = load_fees("binance_vip0")
    print(cfg.maker_fee, cfg.taker_fee)
"""

import pathlib
from dataclasses import dataclass

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
FEES_YAML = ROOT / "configs" / "fees.yaml"


@dataclass
class FeeConfig:
    scenario: str
    maker_fee: float
    taker_fee: float
    description: str = ""

    @property
    def is_naive(self) -> bool:
        return self.maker_fee == 0.0 and self.taker_fee == 0.0


def load_fees(scenario: str = "binance_vip0") -> FeeConfig:
    """
    Load a fee scenario from configs/fees.yaml.

    Parameters
    ----------
    scenario : one of the keys under 'scenarios' in fees.yaml
               e.g. 'naive', 'binance_vip0', 'binance_vip1', 'taker_only'

    Returns
    -------
    FeeConfig dataclass
    """
    with open(FEES_YAML) as f:
        data = yaml.safe_load(f)

    scenarios = data.get("scenarios", {})
    if scenario not in scenarios:
        available = list(scenarios.keys())
        raise ValueError(
            f"Unknown fee scenario '{scenario}'. "
            f"Available: {available}"
        )

    s = scenarios[scenario]
    return FeeConfig(
        scenario=scenario,
        maker_fee=float(s["maker_fee"]),
        taker_fee=float(s["taker_fee"]),
        description=s.get("description", ""),
    )


def list_scenarios() -> list[str]:
    """Return all scenario names defined in fees.yaml."""
    with open(FEES_YAML) as f:
        data = yaml.safe_load(f)
    return list(data.get("scenarios", {}).keys())