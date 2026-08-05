from __future__ import annotations

import pandas as pd
import streamlit as st

def _calculate_score(
    best_result: pd.Series,
) -> int:
    """
    Calculate a transparent strategy score from 0 to 10.
    """

    score = 0

    sharpe = float(
        best_result["Sharpe ratio"]
    )

    profit_factor = float(
        best_result["Profit factor"]
    )

    expectancy = float(
        best_result["Expectancy (€)"]
    )

    calmar = float(
        best_result["Calmar ratio"]
    )

    volatility = float(
        best_result["Volatility (%)"]
    )

    if sharpe >= 1.0:
        score += 2
    elif sharpe >= 0.5:
        score += 1

    if profit_factor >= 2.0:
        score += 2
    elif profit_factor >= 1.0:
        score += 1

    if expectancy > 0:
        score += 2

    if calmar >= 1.0:
        score += 2
    elif calmar >= 0.5:
        score += 1

    if volatility <= 20.0:
        score += 2
    elif volatility <= 40.0:
        score += 1

    return score


def _rating(
    score: int,
) -> str:
    if score >= 8:
        return "★★★★★"

    if score >= 6:
        return "★★★★☆"

    if score >= 4:
        return "★★★☆☆"

    if score >= 2:
        return "★★☆☆☆"

    return "★☆☆☆☆"


def render_strategy_report_card(
    best_result: pd.Series,
) -> None:
    """
    Render a rule-based report card for the
    highest-ranked strategy.
    """

    st.markdown("## 🏆 Strategy Report Card")

    st.info(
        "The report card summarizes the historical "
        "strengths and weaknesses of the selected "
        "strategy using transparent scoring rules."
    )

    print(
        "Report card strategy:",
        best_result["Strategy"],
    )

    print(
        best_result[
            [
                "Sharpe ratio",
                "Profit factor",
                "Expectancy (€)",
                "Calmar ratio",
                "Volatility (%)",
            ]
        ]
    )

    score = _calculate_score(
        best_result
    )

    st.metric(
        "Overall Rating",
        _rating(score),
    )

    sharpe = float(best_result["Sharpe ratio"])
    profit_factor = float(best_result["Profit factor"])
    expectancy = float(best_result["Expectancy (€)"])
    calmar = float(best_result["Calmar ratio"])
    volatility = float(best_result["Volatility (%)"])

    strengths = []
    weaknesses = []

    if sharpe >= 1.0:
        strengths.append(
            "✓ Strong risk-adjusted performance"
        )
    elif sharpe >= 0.0:
        strengths.append(
            "✓ Positive risk-adjusted performance"
        )
    else:
        weaknesses.append(
            "• Negative risk-adjusted performance"
        )

    if profit_factor >= 2.0:
        strengths.append(
            "✓ Strong profit factor"
        )
    elif profit_factor >= 1.0:
        strengths.append(
            "✓ Gross profits exceeded gross losses"
        )
    else:
        weaknesses.append(
            "• Gross losses exceeded gross profits"
        )

    if expectancy > 0:
        strengths.append(
            "✓ Positive trade expectancy"
        )
    elif expectancy < 0:
        weaknesses.append(
            "• Negative trade expectancy"
        )

    if calmar >= 1.0:
        strengths.append(
            "✓ Strong drawdown efficiency"
        )
    elif calmar >= 0.5:
        strengths.append(
            "✓ Moderate drawdown efficiency"
        )
    else:
        weaknesses.append(
            "• Weak drawdown efficiency"
        )

    if volatility <= 20.0:
        strengths.append(
            "✓ Relatively low annualized volatility"
        )
    elif volatility > 40.0:
        weaknesses.append(
            "• Elevated annualized volatility"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Strengths")

        if strengths:
            for strength in strengths:
                st.write(strength)
        else:
            st.write("—")

    with col2:
        st.markdown("### Weaknesses")

        if weaknesses:
            for weakness in weaknesses:
                st.write(weakness)
        else:
            st.write("—")