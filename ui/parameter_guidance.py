from __future__ import annotations


def get_research_guidance(
    strategy_name: str,
    parameters: dict[str, int | float],
) -> list[str]:
    """
    Return educational guidance for the selected
    strategy configuration.
    """

    guidance: list[str] = []

    if strategy_name == "Moving Average":

        fast = parameters["fast_window"]
        slow = parameters["slow_window"]

        difference = slow - fast

        if difference >= 100:

            guidance.append(
                "This configuration emphasizes long-term trend "
                "following and is likely to generate relatively "
                "few crossover signals."
            )

        elif difference >= 30:

            guidance.append(
                "This configuration balances responsiveness "
                "with trend filtering and is well suited for "
                "medium-term trend analysis."
            )

        else:

            guidance.append(
                "This configuration is relatively sensitive to "
                "short-term price movements and may generate "
                "more frequent crossover signals."
            )

    if strategy_name == "RSI":

        oversold = parameters["oversold"]
        overbought = parameters["overbought"]

        width = overbought - oversold

        if width >= 60:

            guidance.append(
                "Wide RSI thresholds wait for stronger momentum "
                "extremes before generating trading signals."
            )

        elif width >= 40:

            guidance.append(
                "This RSI configuration is close to the "
                "traditional settings commonly used in "
                "technical analysis."
            )

        else:

            guidance.append(
                "Narrow RSI thresholds may produce more "
                "frequent trading signals."
            )

    if strategy_name == "MACD":

        fast = parameters["fast_period"]
        slow = parameters["slow_period"]
        signal = parameters["signal_period"]

        if (
            fast == 12
            and slow == 26
            and signal == 9
        ):

            guidance.append(
                "This matches the traditional MACD "
                "configuration."
            )

        elif slow - fast >= 20:

            guidance.append(
                "A larger separation between the EMAs "
                "places greater emphasis on long-term "
                "market trends."
            )

        else:

            guidance.append(
                "Closer EMA periods make the MACD more "
                "responsive to recent price movements."
            )

    if strategy_name == "Bollinger Bands":

        deviations = parameters["window_dev"]

        if deviations >= 3:

            guidance.append(
                "Wider Bollinger Bands require larger price "
                "movements before generating signals."
            )

        elif deviations <= 1.5:

            guidance.append(
                "Narrow Bollinger Bands respond more quickly "
                "to changing market conditions."
            )

        else:

            guidance.append(
                "This configuration is close to the "
                "traditional Bollinger Band settings."
            )

    return guidance