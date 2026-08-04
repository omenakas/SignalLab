import pandas as pd
import streamlit as st

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

    st.divider()