from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from strategies.ma_crossover import generate_ma_positions
from strategies.rsi import generate_rsi_positions


StrategyGenerator = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    generator: StrategyGenerator
    description: str
    default_parameters: dict[str, Any]


STRATEGIES: dict[str, StrategyDefinition] = {
    "Moving Average": StrategyDefinition(
        name="Moving Average",
        generator=generate_ma_positions,
        description="Classic moving-average crossover strategy.",
        default_parameters={
            "fast_window": 20,
            "slow_window": 50,
        },
    ),
    "RSI": StrategyDefinition(
        name="RSI",
        generator=generate_rsi_positions,
        description="Relative Strength Index mean-reversion strategy.",
        default_parameters={
            "rsi_period": 14,
            "oversold": 30.0,
            "overbought": 70.0,
        },
    ),
}


def get_strategy(name: str) -> StrategyDefinition:
    try:
        return STRATEGIES[name]
    except KeyError as error:
        available = ", ".join(STRATEGIES)

        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available strategies: {available}"
        ) from error