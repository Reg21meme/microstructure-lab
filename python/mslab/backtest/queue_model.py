"""
queue_model.py
Queue-position fill probability approximation.

Model
-----
When a limit order arrives at the book, it joins a queue behind all
existing resting orders at that price level. Fill probability is
approximated from the depth imbalance at the time of order submission:

    ask_fraction = (1 - depth_imbalance_5) / 2   # relative ask-side depth
    bid_fraction = (1 + depth_imbalance_5) / 2   # relative bid-side depth

For a BUY order, depth ahead ∝ ask_fraction (we queue behind ask-side volume)
For a SELL order, depth ahead ∝ bid_fraction (we queue behind bid-side volume)

    fill_prob = clip(1 - depth_ahead_fraction, min_fill_prob, max_fill_prob)

A uniform draw decides whether the order fills:
    fills = uniform(0, 1) < fill_prob

Reference: Gould & Bonart (2016), "Queue Imbalance as a One-Tick-Ahead
Price Predictor in a Limit Order Book"
"""

import pathlib
from dataclasses import dataclass

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
QUEUE_YAML = ROOT / "configs" / "queue_model.yaml"


@dataclass
class QueueConfig:
    scenario: str
    min_fill_prob: float
    max_fill_prob: float
    use_queue_model: bool
    random_seed: int
    description: str = ""


def load_queue_config(scenario: str = "base") -> QueueConfig:
    """Load a queue model scenario from configs/queue_model.yaml."""
    with open(QUEUE_YAML) as f:
        data = yaml.safe_load(f)

    scenarios = data.get("scenarios", {})
    if scenario not in scenarios:
        available = list(scenarios.keys())
        raise ValueError(
            f"Unknown queue scenario '{scenario}'. Available: {available}"
        )

    s = scenarios[scenario]
    return QueueConfig(
        scenario=scenario,
        min_fill_prob=float(s["min_fill_prob"]),
        max_fill_prob=float(s["max_fill_prob"]),
        use_queue_model=bool(s["use_queue_model"]),
        random_seed=int(s["random_seed"]),
        description=s.get("description", ""),
    )


class QueueModel:
    """
    Stateless fill-probability calculator.

    Parameters
    ----------
    config : QueueConfig loaded from queue_model.yaml
    """

    def __init__(self, config: QueueConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)

    def fill_probability(self, is_buy: bool, depth_imbalance_5: float) -> float:
        """
        Compute fill probability for an order given current book state.

        Parameters
        ----------
        is_buy           : True for buy orders, False for sell orders
        depth_imbalance_5: (bid_depth - ask_depth) / (bid_depth + ask_depth)
                           from the feature snapshot at order submission time

        Returns
        -------
        fill probability in [min_fill_prob, max_fill_prob]
        """
        if not self.cfg.use_queue_model:
            return 1.0

        if is_buy:
            # Buying: queue behind ask-side depth
            # Large positive imbalance = thin ask side = easier to fill
            depth_ahead_fraction = (1.0 - depth_imbalance_5) / 2.0
        else:
            # Selling: queue behind bid-side depth
            # Large negative imbalance = thin bid side = easier to fill
            depth_ahead_fraction = (1.0 + depth_imbalance_5) / 2.0

        prob = 1.0 - depth_ahead_fraction
        return float(np.clip(prob, self.cfg.min_fill_prob, self.cfg.max_fill_prob))

    def should_fill(self, is_buy: bool, depth_imbalance_5: float) -> tuple[bool, float]:
        """
        Probabilistic fill decision.

        Returns
        -------
        (filled: bool, fill_prob: float)
        """
        prob = self.fill_probability(is_buy, depth_imbalance_5)
        filled = bool(self.rng.uniform(0.0, 1.0) < prob)
        return filled, prob