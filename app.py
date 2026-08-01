import streamlit as st
import plotly.express as px
from walk_forward import walk_forward_test

from backtest import run_backtest
from indicators import add_indicators
from market import get_history, get_prices
from optimizer import optimize_ma_strategy
from strategy import analyze_market




st.set_page_config(
    page_title="Crypto Trading Assistant",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Crypto Trading Assistant")
st.caption(
    "Educational market analysis and backtesting — "
    "not financial advice."
)

prices = get_prices()

if prices is None:
    st.error("Could not fetch current market prices.")
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Bitcoin",
            f"€{prices['bitcoin']['eur']:,.0f}",
        )

    with col2:
        st.metric(
            "Ethereum",
            f"€{prices['ethereum']['eur']:,.0f}",
        )

    with col3:
        st.metric(
            "Solana",
            f"€{prices['solana']['eur']:,.0f}",
        )

st.divider()

(
    analysis_tab,
    backtest_tab,
    strategy_lab_tab,
    walk_forward_tab,
) = st.tabs(
    [
        "Analysis",
        "Backtest",
        "Strategy Lab",
        "Walk-Forward",
    ]
)

history = get_history("bitcoin", days=365)

if history is None:
    st.error("Could not fetch Bitcoin history.")
    st.stop()

data = add_indicators(history)


with analysis_tab:
    st.subheader("Bitcoin market analysis")

    chart_data = (
        data.set_index("date")[["price", "MA20", "MA50"]]
        .rename(
            columns={
                "price": "Bitcoin price",
                "MA20": "20-day average",
                "MA50": "50-day average",
            }
        )
    )

    st.line_chart(chart_data)

    try:
        analysis = analyze_market(data)

    except ValueError as error:
        st.warning(str(error))
        st.stop()

    latest = data.dropna().iloc[-1]

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Signal",
            analysis.signal,
        )

    with metric2:
        st.metric(
            "Signal strength",
            f"{analysis.confidence}%",
        )

    with metric3:
        st.metric(
            "RSI",
            f"{latest['RSI']:.1f}",
        )

    with metric4:
        st.metric(
            "Model score",
            f"{analysis.score:+d}",
        )

    st.subheader("Why this signal was generated")

    for reason in analysis.reasons:
        st.write(f"• {reason}")


with backtest_tab:
    st.subheader("Bitcoin strategy backtest")

    settings_col1, settings_col2 = st.columns(2)

    with settings_col1:
        initial_capital = st.number_input(
            "Initial capital (€)",
            min_value=50.0,
            max_value=100_000.0,
            value=500.0,
            step=50.0,
        )

    with settings_col2:
        fee_percentage = st.number_input(
            "Trading fee (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05,
        )

    try:
        result = run_backtest(
            data,
            initial_capital=initial_capital,
            fee_rate=fee_percentage / 100,
        )

    except ValueError as error:
        st.warning(str(error))
        st.stop()

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Strategy final value",
            f"€{result.final_value:,.2f}",
            f"{result.strategy_return:+.1f}%",
        )

    with result_col2:
        st.metric(
            "Buy-and-hold final value",
            f"€{result.buy_hold_final_value:,.2f}",
            f"{result.buy_hold_return:+.1f}%",
        )

    with result_col3:
        difference = (
            result.strategy_return
            - result.buy_hold_return
        )

        st.metric(
            "Difference",
            f"{difference:+.1f} percentage points",
        )

    performance_chart = (
        result.history.set_index("date")[
            ["strategy_value", "buy_hold_value"]
        ]
        .rename(
            columns={
                "strategy_value": "Strategy",
                "buy_hold_value": "Buy and hold",
            }
        )
    )

    st.subheader("Portfolio performance")
    st.line_chart(performance_chart)

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric(
            "Maximum drawdown",
            f"{result.max_drawdown:.1f}%",
        )

    with stat2:
        st.metric(
            "Completed trades",
            result.completed_trades,
        )

    with stat3:
        st.metric(
            "Winning trades",
            result.winning_trades,
        )

    with stat4:
        st.metric(
            "Win rate",
            f"{result.win_rate:.1f}%",
        )

    st.subheader("Trade history")

    if result.trades.empty:
        st.info("The strategy did not make any trades.")
    else:
        display_trades = result.trades.copy()

        display_trades["date"] = (
            display_trades["date"]
            .dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            display_trades,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("How this backtest works"):
        st.write(
            """
            The simulation starts with the selected amount of euros.

            When the model changes to Bullish, it invests all available
            cash in Bitcoin. When the model changes to Bearish, it sells
            all Bitcoin. Neutral signals cause no action.

            The selected transaction fee is charged on every purchase
            and sale. Buy-and-hold buys Bitcoin at the beginning of the
            test period and holds it until the end.
            """
        )

    st.warning(
        "Historical performance does not predict future returns. "
        "This backtest does not yet model slippage, bid-ask spreads, "
        "taxes, exchange outages, or delayed execution."
    )
    

with strategy_lab_tab:
    st.subheader("Moving-average Strategy Lab")

    st.write(
        """
        Test several fast and slow moving-average combinations.
        Signals are executed on the following day, and the selected
        transaction fee is charged on every purchase and sale.
        """
    )

    settings1, settings2, settings3 = st.columns(3)

    with settings1:
        optimizer_capital = st.number_input(
            "Optimizer initial capital (€)",
            min_value=50.0,
            max_value=100_000.0,
            value=500.0,
            step=50.0,
            key="optimizer_capital",
        )

    with settings2:
        optimizer_fee = st.number_input(
            "Optimizer trading fee (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05,
            key="optimizer_fee",
        )

    with settings3:
        minimum_trades = st.number_input(
            "Minimum completed trades",
            min_value=0,
            max_value=100,
            value=1,
            step=1,
        )

    st.markdown("#### Parameter ranges")

    range1, range2 = st.columns(2)

    with range1:
        fast_range = st.slider(
            "Fast moving-average range",
            min_value=2,
            max_value=100,
            value=(5, 40),
        )

        fast_step = st.number_input(
            "Fast MA step",
            min_value=1,
            max_value=20,
            value=5,
        )

    with range2:
        slow_range = st.slider(
            "Slow moving-average range",
            min_value=10,
            max_value=250,
            value=(30, 150),
        )

        slow_step = st.number_input(
            "Slow MA step",
            min_value=1,
            max_value=30,
            value=10,
        )

    fast_values = list(
        range(
            fast_range[0],
            fast_range[1] + 1,
            int(fast_step),
        )
    )

    slow_values = list(
        range(
            slow_range[0],
            slow_range[1] + 1,
            int(slow_step),
        )
    )

    combinations = sum(
        1
        for fast in fast_values
        for slow in slow_values
        if fast < slow
    )

    st.caption(
        f"This configuration will test "
        f"{combinations} parameter combinations."
    )

    run_optimizer = st.button(
        "Run optimizer",
        type="primary",
    )

    if run_optimizer:
        with st.spinner("Testing strategies..."):
            optimizer_results = optimize_ma_strategy(
                df=history,
                fast_values=fast_values,
                slow_values=slow_values,
                initial_capital=optimizer_capital,
                fee_rate=optimizer_fee / 100,
            )

        if optimizer_results.empty:
            st.warning(
                "No valid strategies were produced."
            )

        else:
            filtered_results = optimizer_results[
                optimizer_results["completed_trades"]
                >= minimum_trades
            ].copy()

            if filtered_results.empty:
                st.warning(
                    "No strategies met the minimum-trades requirement."
                )

            else:
                best = filtered_results.iloc[0]

                st.markdown("### Best result")

                best1, best2, best3, best4 = st.columns(4)

                with best1:
                    st.metric(
                        "Fast / slow MA",
                        (
                            f"{int(best['fast_ma'])}"
                            f" / "
                            f"{int(best['slow_ma'])}"
                        ),
                    )

                with best2:
                    st.metric(
                        "Strategy return",
                        f"{best['strategy_return']:+.1f}%",
                    )

                with best3:
                    st.metric(
                        "Excess vs hold",
                        f"{best['excess_return']:+.1f} pp",
                    )

                with best4:
                    st.metric(
                        "Maximum drawdown",
                        f"{best['max_drawdown']:.1f}%",
                    )

                st.markdown("### Ranked strategies")

                display_results = filtered_results.copy()

                display_results.insert(
                    0,
                    "rank",
                    range(
                        1,
                        len(display_results) + 1,
                    ),
                )

                display_results = display_results.rename(
                    columns={
                        "rank": "Rank",
                        "fast_ma": "Fast MA",
                        "slow_ma": "Slow MA",
                        "final_value": "Final value (€)",
                        "strategy_return": "Return (%)",
                        "buy_hold_return": "Buy & hold (%)",
                        "excess_return": "Excess return (pp)",
                        "max_drawdown": "Max drawdown (%)",
                        "completed_trades": "Trades",
                        "win_rate": "Win rate (%)",
                    }
                )

                numeric_columns = [
                    "Final value (€)",
                    "Return (%)",
                    "Buy & hold (%)",
                    "Excess return (pp)",
                    "Max drawdown (%)",
                    "Win rate (%)",
                ]

                display_results[numeric_columns] = (
                    display_results[numeric_columns]
                    .round(2)
                )

                st.dataframe(
                    display_results,
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("### Return heatmap")

                heatmap_data = (
                    filtered_results
                    .pivot_table(
                        index="slow_ma",
                        columns="fast_ma",
                        values="strategy_return",
                        aggfunc="mean",
                    )
                    .sort_index(ascending=False)
                )

                figure = px.imshow(
                    heatmap_data,
                    labels={
                        "x": "Fast moving average",
                        "y": "Slow moving average",
                        "color": "Return (%)",
                    },
                    aspect="auto",
                    text_auto=".1f",
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

                st.info(
                    "The highest historical return is not automatically "
                    "the best strategy. Look for groups of nearby parameter "
                    "combinations that perform reasonably well rather than "
                    "one isolated winner."
                )

with walk_forward_tab:
    st.subheader("Out-of-sample Walk-Forward Test")

    st.write(
        """
        The optimizer searches for the best moving-average settings
        using only the earlier training period. The winning settings
        are then frozen and evaluated on the later, unseen period.
        """
    )

    wf_settings1, wf_settings2, wf_settings3 = st.columns(3)

    with wf_settings1:
        wf_capital = st.number_input(
            "Initial capital (€)",
            min_value=50.0,
            max_value=100_000.0,
            value=500.0,
            step=50.0,
            key="wf_capital",
        )

        wf_fee_percent = st.number_input(
            "Fee per transaction (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05,
            key="wf_fee_percent",
        )

    with wf_settings2:
        wf_train_percent = st.slider(
            "Training data (%)",
            min_value=50,
            max_value=90,
            value=70,
            step=5,
            key="wf_train_percent",
        )

        wf_min_trades = st.number_input(
            "Minimum training trades",
            min_value=0,
            max_value=100,
            value=1,
            step=1,
            key="wf_min_trades",
        )

    with wf_settings3:
        wf_fast_min = st.number_input(
            "Fast MA minimum",
            min_value=2,
            max_value=200,
            value=5,
            step=1,
            key="wf_fast_min",
        )

        wf_fast_max = st.number_input(
            "Fast MA maximum",
            min_value=2,
            max_value=250,
            value=40,
            step=1,
            key="wf_fast_max",
        )

        wf_fast_step = st.number_input(
            "Fast MA step",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="wf_fast_step",
        )

    wf_ranges1, wf_ranges2 = st.columns(2)

    with wf_ranges1:
        wf_slow_min = st.number_input(
            "Slow MA minimum",
            min_value=3,
            max_value=300,
            value=30,
            step=1,
            key="wf_slow_min",
        )

        wf_slow_max = st.number_input(
            "Slow MA maximum",
            min_value=5,
            max_value=500,
            value=150,
            step=5,
            key="wf_slow_max",
        )

    with wf_ranges2:
        wf_slow_step = st.number_input(
            "Slow MA step",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="wf_slow_step",
        )

    run_walk_forward = st.button(
        "Run Walk-Forward Test",
        type="primary",
        key="run_walk_forward",
    )

    if run_walk_forward:
        if wf_fast_min > wf_fast_max:
            st.error(
                "Fast MA minimum cannot be greater than its maximum."
            )

        elif wf_slow_min > wf_slow_max:
            st.error(
                "Slow MA minimum cannot be greater than its maximum."
            )

        else:
            fast_values = range(
                int(wf_fast_min),
                int(wf_fast_max) + 1,
                int(wf_fast_step),
            )

            slow_values = range(
                int(wf_slow_min),
                int(wf_slow_max) + 1,
                int(wf_slow_step),
            )

            try:
                with st.spinner(
                    "Optimizing the training period and testing "
                    "the winner on unseen data..."
                ):
                    wf_result = walk_forward_test(
                        df=history,
                        fast_values=fast_values,
                        slow_values=slow_values,
                        train_fraction=wf_train_percent / 100,
                        initial_capital=wf_capital,
                        fee_rate=wf_fee_percent / 100,
                        min_trades=int(wf_min_trades),
                    )

                st.success(
                    "Walk-forward test completed successfully."
                )

                st.markdown("### Selected training-period strategy")

                selection1, selection2, selection3 = st.columns(3)

                selection1.metric(
                    "Fast MA",
                    wf_result.fast_ma,
                )

                selection2.metric(
                    "Slow MA",
                    wf_result.slow_ma,
                )

                selection3.metric(
                    "Training return",
                    f"{wf_result.train_return_pct:.2f}%",
                )

                st.markdown("### Unseen testing-period results")

                result1, result2, result3, result4 = st.columns(4)

                result1.metric(
                    "Strategy return",
                    f"{wf_result.test_return_pct:.2f}%",
                )

                result2.metric(
                    "Buy & Hold",
                    f"{wf_result.test_buy_hold_return_pct:.2f}%",
                )

                result3.metric(
                    "Excess return",
                    f"{wf_result.test_excess_return_pct:.2f}%",
                )

                result4.metric(
                    "Maximum drawdown",
                    f"{wf_result.test_max_drawdown_pct:.2f}%",
                )

                detail1, detail2, detail3, detail4 = st.columns(4)

                detail1.metric(
                    "Final strategy value",
                    f"€{wf_result.test_final_value:,.2f}",
                )

                detail2.metric(
                    "Buy & Hold value",
                    f"€{wf_result.test_buy_hold_final_value:,.2f}",
                )

                detail3.metric(
                    "Completed trades",
                    wf_result.test_trades,
                )

                detail4.metric(
                    "Win rate",
                    f"{wf_result.test_win_rate_pct:.1f}%",
                )

                chart_data = wf_result.test_equity_curve[
                    [
                        "strategy_value",
                        "buy_hold_value",
                    ]
                ]

                st.line_chart(chart_data)

                st.caption(
                    f"Training rows: {wf_result.train_rows} · "
                    f"Testing rows: {wf_result.test_rows}"
                )

                with st.expander(
                    "Show training-period optimization results"
                ):
                    st.dataframe(
                        wf_result.optimization_results,
                        use_container_width=True,
                    )

                with st.expander("Show unseen-period trade log"):
                    if wf_result.test_trade_log.empty:
                        st.info(
                            "The selected strategy made no completed "
                            "trades during the testing period."
                        )
                    else:
                        st.dataframe(
                            wf_result.test_trade_log,
                            use_container_width=True,
                        )

                st.warning(
                    "A single holdout test is more honest than "
                    "optimizing on all available data, but it is still "
                    "only one historical experiment. It does not "
                    "demonstrate that the strategy will work in the "
                    "future."
                )

            except Exception as error:
                st.exception(error)