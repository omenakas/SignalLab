from __future__ import annotations


def validate_strategy_parameters(
    strategy_name: str,
    parameters: dict[str, int | float],
) -> tuple[list[str], list[str]]:
    """
    Validate strategy parameters and return a list
    of validation errors.
    """

    errors: list[str] = []

    research_tips: list[str] = []   

    if strategy_name == "Moving Average":

        fast = parameters["fast_window"]

        slow = parameters["slow_window"]

        if fast >= slow:

            errors.append(
                "Fast MA must be smaller than Slow MA."
            )

        difference = slow - fast

        if difference < 10:

            research_tips.append(
                "Fast and Slow moving averages are very close together. "
                "This may increase the number of crossover signals in "
                "sideways markets."
            )

    if strategy_name == "RSI":

        oversold = parameters["oversold"]

        overbought = parameters["overbought"]

        if oversold >= overbought:

            errors.append(
                "Oversold level must be below "
                "Overbought level."
            )

        if (
            overbought - oversold
            ) < 20:

            research_tips.append(
                "The RSI thresholds are relatively close together. "
                "This may generate more frequent trading signals."
            )

    if strategy_name == "MACD":

        fast = parameters["fast_period"]

        slow = parameters["slow_period"]

        if fast >= slow:

            errors.append(
                "Fast EMA must be smaller than "
                "Slow EMA."
            )

        if (
            slow - fast
        ) < 10:

            research_tips.append(
                "Fast and Slow EMA periods are close together. "
                "The MACD may become more sensitive to short-term "
                "price fluctuations."
            )

    return errors, research_tips