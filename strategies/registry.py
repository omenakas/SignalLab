from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from strategies.ma_crossover import generate_ma_positions
from strategies.rsi import generate_rsi_positions
from strategies.macd import generate_macd_positions


StrategyGenerator = Callable[..., pd.DataFrame]
ParameterType = Literal["int", "float"]


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    label: str
    parameter_type: ParameterType
    default: int | float
    minimum: int | float
    maximum: int | float
    step: int | float

    optimization_minimum: int | float
    optimization_maximum: int | float
    optimization_step: int | float


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    generator: StrategyGenerator
    description: str
    parameters: tuple[ParameterDefinition, ...]
    price_overlays: dict[str, str] | None = None

    @property
    def default_parameters(self) -> dict[str, Any]:
        return {
            parameter.name: parameter.default
            for parameter in self.parameters
        }


STRATEGIES: dict[str, StrategyDefinition] = {
    "Moving Average": StrategyDefinition(
        name="Moving Average",
        generator=generate_ma_positions,
        description="Classic moving-average crossover strategy.",
        parameters=(
            ParameterDefinition(
                name="fast_window",
                label="Fast MA",
                parameter_type="int",
                default=20,
                minimum=2,
                maximum=100,
                step=1,
                optimization_minimum=5,
                optimization_maximum=40,
                optimization_step=5,
            ),
            ParameterDefinition(
                name="slow_window",
                label="Slow MA",
                parameter_type="int",
                default=50,
                minimum=3,
                maximum=250,
                step=1,
                optimization_minimum=30,
                optimization_maximum=150,
                optimization_step=10,
            ),
        ),
        price_overlays={
            "Fast MA": "fast_ma",
            "Slow MA": "slow_ma",
        },
    ),
    "RSI": StrategyDefinition(
        name="RSI",
        generator=generate_rsi_positions,
        description="Relative Strength Index mean-reversion strategy.",
        parameters=(
            ParameterDefinition(
                name="rsi_period",
                label="RSI period",
                parameter_type="int",
                default=14,
                minimum=2,
                maximum=100,
                step=1,
                optimization_minimum=7,
                optimization_maximum=21,
                optimization_step=7,
            ),
            ParameterDefinition(
                name="oversold",
                label="Oversold threshold",
                parameter_type="float",
                default=30.0,
                minimum=1.0,
                maximum=49.0,
                step=1.0,
                optimization_minimum=25.0,
                optimization_maximum=35.0,
                optimization_step=5.0,
            ),
            ParameterDefinition(
                name="overbought",
                label="Overbought threshold",
                parameter_type="float",
                default=70.0,
                minimum=51.0,
                maximum=99.0,
                step=1.0,
                optimization_minimum=65.0,
                optimization_maximum=75.0,
                optimization_step=5.0,
            ),
        ),
        price_overlays=None,
    ),
    "MACD": StrategyDefinition(
        name="MACD",
        generator=generate_macd_positions,
        description=(
            "Trend-following strategy based on MACD "
            "and signal-line crossovers."
        ),
        parameters=(
            ParameterDefinition(
                name="fast_period",
                label="Fast period",
                parameter_type="int",
                default=12,
                minimum=2,
                maximum=50,
                step=1,
                optimization_minimum=8,
                optimization_maximum=16,
                optimization_step=2,
            ),
            ParameterDefinition(
                name="slow_period",
                label="Slow period",
                parameter_type="int",
                default=26,
                minimum=3,
                maximum=100,
                step=1,
                optimization_minimum=20,
                optimization_maximum=40,
                optimization_step=5,
            ),
            ParameterDefinition(
                name="signal_period",
                label="Signal period",
                parameter_type="int",
                default=9,
                minimum=2,
                maximum=30,
                step=1,
                optimization_minimum=5,
                optimization_maximum=15,
                optimization_step=2,
            ),
        ),
        price_overlays=None,
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