import pandas as pd
import streamlit as st

def _render_badge(
    title: str,
    message: str,
    level: str,
) -> None:
    """
    Render a rule-based performance badge.
    """

    text = f"**{title}**\n\n{message}"

    if level == "positive":
        st.success(text)

    elif level == "warning":
        st.warning(text)

    else:
        st.error(text)

def _risk_adjusted_badge(
    sharpe_ratio: float,
) -> tuple[str, str]:
    if sharpe_ratio >= 1.0:
        return (
            "Strong risk-adjusted performance",
            "positive",
        )

    if sharpe_ratio >= 0.0:
        return (
            "Positive but modest risk-adjusted performance",
            "warning",
        )

    return (
        "Negative risk-adjusted performance",
        "negative",
    )


def _downside_badge(
    sortino_ratio: float,
) -> tuple[str, str]:
    if sortino_ratio >= 1.5:
        return (
            "Strong return relative to downside volatility",
            "positive",
        )

    if sortino_ratio >= 0.0:
        return (
            "Limited compensation for downside volatility",
            "warning",
        )

    return (
        "Returns did not compensate for downside volatility",
        "negative",
    )


def _drawdown_badge(
    calmar_ratio: float,
) -> tuple[str, str]:
    if calmar_ratio >= 1.0:
        return (
            "Strong annualized growth relative to drawdown",
            "positive",
        )

    if calmar_ratio >= 0.0:
        return (
            "Positive growth with modest drawdown efficiency",
            "warning",
        )

    return (
        "Negative growth relative to maximum drawdown",
        "negative",
    )

def render_performance_dashboard(
    best_result: pd.Series,
) -> None:
    """
    Render a summary dashboard for the highest-ranked strategy.
    """

    st.markdown("## 📊 Performance Dashboard")

    summary1, summary2, summary3, summary4 = st.columns(4)

    summary1.metric(
        "🏆 Best strategy",
        best_result["Strategy"],
    )

    summary2.metric(
        "📈 Return",
        f"{best_result['Return (%)']:+.2f}%",
    )

    summary3.metric(
        "⚖ Sharpe",
        f"{best_result['Sharpe ratio']:.2f}",
    )

    summary4.metric(
        "📉 Calmar",
        f"{best_result['Calmar ratio']:.2f}",
    )

    detail1, detail2, detail3, detail4 = st.columns(4)

    detail1.metric(
        "📅 CAGR",
        f"{best_result['CAGR (%)']:+.2f}%",
    )

    detail2.metric(
        "🛡 Sortino",
        f"{best_result['Sortino ratio']:.2f}",
    )

    detail3.metric(
        "💹 Trades",
        int(best_result["Completed trades"]),
    )

    detail4.metric(
        "🎯 Win rate",
        f"{best_result['Win rate (%)']:.1f}%",
    )

    st.markdown("### Performance signals")

    badge1, badge2, badge3 = st.columns(3)

    risk_message, risk_level = _risk_adjusted_badge(
        float(best_result["Sharpe ratio"])
    )

    downside_message, downside_level = _downside_badge(
        float(best_result["Sortino ratio"])
    )

    drawdown_message, drawdown_level = _drawdown_badge(
        float(best_result["Calmar ratio"])
    )

    with badge1:
        _render_badge(
            title="⚖ Risk-adjusted return",
            message=risk_message,
            level=risk_level,
        )

    with badge2:
        _render_badge(
            title="🛡 Downside efficiency",
            message=downside_message,
            level=downside_level,
        )

    with badge3:
        _render_badge(
            title="🏔 Drawdown efficiency",
            message=drawdown_message,
            level=drawdown_level,
        )

    st.caption(
        "Badges are descriptive rules based on historical metrics, "
        "not forecasts or investment recommendations."
    )

    st.divider()