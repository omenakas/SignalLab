import pandas as pd
import streamlit as st

def render_performance_highlights(
    results: pd.DataFrame,
) -> None:
    """
    Render category winners from the strategy comparison.
    """

    if results.empty:
        return

    st.markdown("## 🏆 Performance Highlights")

    col1, col2 = st.columns(2)

    with col1:

        best_sharpe = results.loc[
            results["Sharpe ratio"].idxmax()
        ]

        st.success(
            (
                "🥇 **Best Risk-Adjustet Return**\n\n"
                f"{best_sharpe['Strategy']} "
                f"({best_sharpe['Sharpe ratio']:.2f})"
            )
        )

        best_cagr = results.loc[
            results["CAGR (%)"].idxmax()
        ]

        st.success(
            (
                "🚀 **Fastest Annual Growth**\n\n"
                f"{best_cagr['Strategy']} "
                f"({best_cagr['CAGR (%)']:+.2f}%)"
            )
        )

    with col2:

        lowest_drawdown = results.loc[
            results["Max drawdown (%)"].idxmin()
        ]

        st.success(
            (
                "🛡 **Lowest Drawdown**\n\n"
                f"{lowest_drawdown['Strategy']} "
                f"({lowest_drawdown['Max drawdown (%)']:.2f}%)"
            )
        )

        best_win_rate = results.loc[
            results["Win rate (%)"].idxmax()
        ]

        st.success(
            (
                "🎯 **Highest Win Rate**\n\n"
                f"{best_win_rate['Strategy']} "
                f"({best_win_rate['Win rate (%)']:.1f}%)"
            )
        )

    st.caption(
        "Highlights identify the strongest historical performer "
        "in each category for the selected comparison."
    )

    st.divider()