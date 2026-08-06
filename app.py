import streamlit as st
import plotly.express as px
import pandas as pd

from walk_forward import walk_forward_test
from backtest import run_backtest
from indicators import add_indicators
from market import get_history, get_prices
from optimizer import optimize_ma_strategy
from strategy import analyze_market
from engine.simulator import run_position_backtest
from strategies.registry import STRATEGIES, get_strategy
from optimizer import optimize_strategy

from ui.parameter_builder import build_parameter_inputs
from ui.charts import plot_price_with_trades
from ui.parameter_builder import (
    build_optimization_grid_inputs,
)
from ui.charts import (
    plot_indicator_panel,
    plot_price_with_trades,
)
from ui.performance_dashboard import (
    render_performance_dashboard,
)
from ui.performance_highlights import (
    render_performance_highlights,
)
from ui.strategy_report_card import (
    render_strategy_report_card,
)
from ui.charts import (
    plot_drawdown,
    plot_indicator_panel,
    plot_price_with_trades,
    plot_rolling_sharpe,
    plot_monthly_returns_heatmap,
)
from ui.formatting import (
    BACKTEST_FORMATS,
    TRADE_HISTORY_FORMATS,
    format_dataframe,
)

from ui.parameter_summary import (
    render_parameter_summary,
)
from ui.parameter_validation import (
    validate_strategy_parameters,
)
from analytics.performance import calculate_performance_metrics
from analytics.drawdown import (
    calculate_drawdown_series,
)
from analytics.rolling import calculate_rolling_sharpe
from analytics.monthly_returns import (
    calculate_monthly_returns,
)
from analytics.trade_metrics import (
    calculate_trade_metrics,
)



st.title("📊 SignalLab")

st.caption(
    "A modular laboratory for quantitative trading research."
)

st.markdown(
    "**Develop • Backtest • Compare • Analyze**"
)

st.divider()

prices = get_prices()

if prices is None:
    st.warning(
        "Live prices are currently unavailable. "
        "Historical analysis, backtesting, optimization, "
        "and walk-forward testing can still use cached data."
    )
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Bitcoin",
            f"€{prices['bitcoin']['eur']:,.2f}",
        )

    with col2:
        st.metric(
            "Ethereum",
            f"€{prices['ethereum']['eur']:,.2f}",
        )

    with col3:
        st.metric(
            "Solana",
            f"€{prices['solana']['eur']:,.2f}",
        )

st.divider()

(
    analysis_tab,
    backtest_tab,
    strategy_lab_tab,
    walk_forward_tab,
    comparison_tab,
) = st.tabs(
    [
        "📈 Analysis",
        "💰 Backtest",
        "🧪 Strategy Lab",
        "🔄 Walk-Forward",
        "📊 Strategy Research",
    ]
)

@st.cache_data(ttl=3600)
def load_bitcoin_history() -> pd.DataFrame | None:
    return get_history("bitcoin", days=365)

history = load_bitcoin_history()

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
        performance_metrics = calculate_performance_metrics(
            history=result.history,
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
            "Excess return",
            f"{difference:+.1f} pp",
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
            format_dataframe(
                display_trades,
                BACKTEST_FORMATS,
            ),
            hide_index=True,
            width="stretch",
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
    st.subheader("Strategy Lab")

    st.write(
        """
        Select a registered strategy and search across multiple
        parameter combinations. Every combination is evaluated with
        the same generic simulator.
        """
    )

    selected_optimizer_name = st.selectbox(
        "Strategy to optimize",
        options=list(STRATEGIES.keys()),
        key="optimizer_strategy",
    )

    selected_optimizer_strategy = get_strategy(
        selected_optimizer_name
    )

    st.caption(
        selected_optimizer_strategy.description
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
            key="optimizer_minimum_trades",
        )

    st.markdown("#### Parameter ranges")

    parameter_grid = build_optimization_grid_inputs(
        strategy=selected_optimizer_strategy,
        key_prefix="optimizer",
    )

    combinations = 1

    for values in parameter_grid.values():
        combinations *= len(values)

    st.caption(
        f"This configuration will test "
        f"{combinations:,} parameter combinations."
    )

    if combinations > 10_000:
        st.warning(
            "This is a large search and may take considerable time. "
            "Consider increasing the parameter steps or narrowing "
            "the ranges."
        )

    run_optimizer = st.button(
        "Run optimizer",
        type="primary",
        key="run_generic_optimizer",
    )

    if run_optimizer:
        with st.spinner(
            f"Optimizing {selected_optimizer_name}..."
        ):
            optimizer_results = optimize_strategy(
                df=history,
                strategy=selected_optimizer_strategy,
                parameter_grid=parameter_grid,
                initial_capital=optimizer_capital,
                fee_rate=optimizer_fee / 100,
                min_trades=int(minimum_trades),
            )

        if optimizer_results.empty:
            st.warning(
                "No valid parameter combinations were produced. "
                "Try lowering the minimum trade count or changing "
                "the parameter ranges."
            )

        else:
            best = optimizer_results.iloc[0]

            st.markdown("### Best result")

            best_parameters = {
                parameter.label: best[parameter.name]
                for parameter
                in selected_optimizer_strategy.parameters
                if parameter.name in best.index
            }

            formatted_parameters = " · ".join(
                f"{label}: {value:g}"
                if isinstance(value, float)
                else f"{label}: {value}"
                for label, value in best_parameters.items()
            )

            best1, best2, best3, best4 = st.columns(4)

            best1.metric(
                "Best parameters",
                formatted_parameters,
            )

            best2.metric(
                "Strategy return",
                f"{best['strategy_return']:+.1f}%",
            )

            best3.metric(
                "Excess vs hold",
                f"{best['excess_return']:+.1f} pp",
            )

            best4.metric(
                "Maximum drawdown",
                f"{best['max_drawdown']:.1f}%",
            )

            st.markdown("### Ranked strategies")

            display_results = optimizer_results.copy()

            display_results.insert(
                0,
                "Rank",
                range(
                    1,
                    len(display_results) + 1,
                ),
            )

            rename_columns = {
                parameter.name: parameter.label
                for parameter
                in selected_optimizer_strategy.parameters
            }

            rename_columns.update(
                {
                    "final_value": "Final value (€)",
                    "strategy_return": "Return (%)",
                    "buy_hold_return": "Buy & hold (%)",
                    "excess_return": "Excess return (pp)",
                    "max_drawdown": "Max drawdown (%)",
                    "completed_trades": "Trades",
                    "win_rate": "Win rate (%)",
                }
            )

            display_results = display_results.rename(
                columns=rename_columns
            )

            numeric_columns = [
                "Final value (€)",
                "Return (%)",
                "Buy & hold (%)",
                "Excess return (pp)",
                "Max drawdown (%)",
                "Win rate (%)",
            ]

            existing_numeric_columns = [
                column
                for column in numeric_columns
                if column in display_results.columns
            ]

            display_results[existing_numeric_columns] = (
                display_results[
                    existing_numeric_columns
                ].round(2)
            )

            st.dataframe(
                format_dataframe(
                    result.trades,
                    TRADE_HISTORY_FORMATS,
                ),
                hide_index=True,
                width="stretch",
            )

            parameter_names = [
                parameter.name
                for parameter
                in selected_optimizer_strategy.parameters
            ]

            # A two-dimensional strategy can be displayed naturally
            # as a heatmap. Strategies with three or more parameters
            # remain available in the ranked-results table.
            if len(parameter_names) == 2:
                first_parameter = parameter_names[0]
                second_parameter = parameter_names[1]

                first_label = (
                    selected_optimizer_strategy
                    .parameters[0]
                    .label
                )

                second_label = (
                    selected_optimizer_strategy
                    .parameters[1]
                    .label
                )

                st.markdown("### Return heatmap")

                heatmap_data = (
                    optimizer_results
                    .pivot_table(
                        index=second_parameter,
                        columns=first_parameter,
                        values="strategy_return",
                        aggfunc="mean",
                    )
                    .sort_index(ascending=False)
                )

                figure = px.imshow(
                    heatmap_data,
                    labels={
                        "x": first_label,
                        "y": second_label,
                        "color": "Return (%)",
                    },
                    aspect="auto",
                    text_auto=".1f",
                )

                st.plotly_chart(
                    figure,
                    width="stretch",
                )

            else:
                st.info(
                    "This strategy has more than two parameters, "
                    "so its full optimization results are shown in "
                    "the ranked table rather than a two-dimensional "
                    "heatmap."
                )

            st.info(
                "The highest historical return is not automatically "
                "the most reliable strategy. Look for stable regions "
                "of nearby parameter combinations and confirm results "
                "with walk-forward testing."
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
                        width="stretch",
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
                            width="stretch",
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

with comparison_tab:
    st.subheader("Strategy Research")

    st.write(
        """
        Compare several registered strategies over the same historical
        period using their default parameters, identical starting
        capital, and the same transaction fee.
        """
    )

    selected_strategy_names = st.multiselect(
        "Strategies to compare",
        options=list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        key="comparison_strategies",
    )

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        comparison_capital = st.number_input(
            "Initial capital (€)",
            min_value=50.0,
            max_value=100_000.0,
            value=500.0,
            step=50.0,
            key="comparison_capital",
        )

    with comparison_col2:
        comparison_fee = st.number_input(
            "Trading fee (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05,
            key="comparison_fee",
        )

    show_comparison_details = st.checkbox(
        "Show detailed charts for each strategy",
        value=False,
        key="comparison_show_details",
    )

    
    st.markdown("### Strategy parameters")

    comparison_parameters = {}

    for strategy_name in selected_strategy_names:
        strategy = get_strategy(
            strategy_name
        )

        with st.expander(
            f"{strategy_name} parameters",
            expanded=False,
        ):
            st.caption(
                strategy.description
            )
            comparison_parameters[strategy_name] = (
                build_parameter_inputs(
                    strategy=strategy,
                    key_prefix=(
                        "comparison_"
                        f"{strategy_name}"
                        .lower()
                        .replace(" ", "_")
                    ),
                )
            )
    
    render_parameter_summary(
        comparison_parameters
    )

    st.divider()

    validation_errors: list[str] = []
    research_tips: list[str] = []

    for (
        strategy_name,
        parameters,
    ) in comparison_parameters.items():

        errors, tips = (
            validate_strategy_parameters(
                strategy_name,
                parameters,
            )
        )

        validation_errors.extend(
            errors
        )

        research_tips.extend(
            tips
        )

    for error in validation_errors:
        st.error(error)

    if research_tips:
        st.markdown("### 💡 Research Tips")

        for tip in research_tips:
            st.info(tip)

    if st.button(
        "Compare strategies",
        type="primary",
        key="compare_selected_strategies",
        disabled=bool(validation_errors),
    ):
        if not selected_strategy_names:
            st.warning(
                "Select at least one strategy to compare."
            )

        else:
            generated_positions = {}
            generation_errors = {}

            with st.spinner(
                "Generating strategy positions..."
            ):
                for strategy_name in selected_strategy_names:
                    strategy = get_strategy(strategy_name)

                    try:
                        positions = strategy.generator(
                            df=history,
                            **comparison_parameters[
                                strategy_name
                            ],
                        )

                        if positions is None or positions.empty:
                            raise ValueError(
                                "The strategy produced no valid rows."
                            )

                        if "date" not in positions.columns:
                            raise ValueError(
                                "Strategy output contains no date column."
                            )

                        positions = positions.copy()

                        positions["date"] = pd.to_datetime(
                            positions["date"],
                            errors="coerce",
                        )

                        positions = (
                            positions
                            .dropna(subset=["date"])
                            .sort_values("date")
                            .reset_index(drop=True)
                        )

                        if positions.empty:
                            raise ValueError(
                                "No valid dated rows remain."
                            )

                        generated_positions[strategy_name] = (
                            positions
                        )

                    except ValueError as error:
                        generation_errors[strategy_name] = str(
                            error
                        )

            for strategy_name, error_message in (
                generation_errors.items()
            ):
                st.warning(
                    f"{strategy_name} could not be generated: "
                    f"{error_message}"
                )

            if not generated_positions:
                st.error(
                    "None of the selected strategies could be evaluated."
                )

            else:
                # Strategies have different indicator warm-up periods.
                # Use the latest valid starting date so every strategy
                # is evaluated over exactly the same historical period.
                common_start_date = max(
                    positions["date"].min()
                    for positions
                    in generated_positions.values()
                )

                comparison_results = []
                simulations = {}
                simulation_errors = {}

                with st.spinner(
                    "Running strategy comparison..."
                ):
                    for (
                        strategy_name,
                        positions,
                    ) in generated_positions.items():
                        strategy = get_strategy(
                            strategy_name
                        )

                        common_positions = positions.loc[
                            positions["date"]
                            >= common_start_date
                        ].copy()

                        try:
                            result = run_position_backtest(
                                df=common_positions,
                                initial_capital=(
                                    comparison_capital
                                ),
                                fee_rate=(
                                    comparison_fee / 100
                                ),
                            )
                            performance_metrics = calculate_performance_metrics(
                                history=result.history,
                            )

                            trade_metrics = calculate_trade_metrics(
                                trades=result.trades,
                            )
                            
                            simulations[strategy_name] = {
                                "strategy": strategy,
                                "positions": common_positions,
                                "result": result,
                            }

                            comparison_results.append(
                                {
                                    "Strategy": strategy_name,
                                    "Final value (€)": (
                                        result.final_value
                                    ),
                                    "Return (%)": (
                                        result.strategy_return
                                    ),
                                    "Buy & hold (%)": (
                                        result.buy_hold_return
                                    ),
                                    "Excess return (pp)": (
                                        result.excess_return
                                    ),
                                    **performance_metrics.as_dict(),
                                    **trade_metrics.as_dict(),
                                    "Max drawdown (%)": (
                                        result.max_drawdown
                                    ),
                                    "Completed trades": (
                                        result.completed_trades
                                    ),
                                    "Winning trades": (
                                        result.winning_trades
                                    ),
                                    "Win rate (%)": (
                                        result.win_rate
                                    ),
                                }
                            )

                        except ValueError as error:
                            simulation_errors[
                                strategy_name
                            ] = str(error)

                for strategy_name, error_message in (
                    simulation_errors.items()
                ):
                    st.warning(
                        f"{strategy_name} could not be "
                        f"simulated: {error_message}"
                    )

                if not comparison_results:
                    st.error(
                        "No strategy produced a valid simulation."
                    )

                else:
                    st.caption(
                        "Common comparison period begins "
                        f"{common_start_date:%Y-%m-%d}."
                    )

                    results_df = pd.DataFrame(
                        comparison_results
                    )

                    results_df = results_df.sort_values(
                        by=[
                            "Return (%)",
                            "Max drawdown (%)",
                        ],
                        ascending=[
                            False,
                            False,
                        ],
                    ).reset_index(drop=True)

                    results_df.insert(
                        0,
                        "Rank",
                        range(
                            1,
                            len(results_df) + 1,
                        ),
                    )

                    st.markdown("### Ranked results")

                    best_result = results_df.iloc[0]

                    render_performance_dashboard(
                        best_result=best_result,
                    )
                    render_performance_highlights(
                        results_df,
                    )
                    render_strategy_report_card(
                        best_result=best_result,
                    )

                    

                    display_results = results_df.copy()

                    numeric_columns = [
                        "Final value (€)",
                        "Return (%)",
                        "Buy & hold (%)",
                        "Excess return (pp)",
                        "Sharpe ratio",
                        "Sortino ratio",
                        "CAGR (%)",
                        "Calmar ratio",
                        "Volatility (%)",
                        "Max drawdown (%)",
                        "Win rate (%)",
                    ]

                    for column in numeric_columns:
                        if column in display_results.columns:
                            display_results[column] = (
                                pd.to_numeric(
                                    display_results[column],
                                    errors="coerce",
                                )
                                .round(2)
                            )

                    display_results["Profit factor"] = (
                        display_results["Profit factor"]
                        .map(
                            lambda value: (
                                "∞"
                                if value == float("inf")
                                else f"{float(value):.2f}"
                            )
                        )
                    )

                    st.dataframe(
                        display_results,
                        hide_index=True,
                        width="stretch",
                    )

                    st.markdown(
                        "### Portfolio performance"
                    )

                    equity_curves = {}

                    for (
                        strategy_name,
                        simulation_data,
                    ) in simulations.items():
                        result = simulation_data["result"]

                        equity_curves[strategy_name] = (
                            result.history
                            .set_index("date")[
                                "strategy_value"
                            ]
                            .rename(strategy_name)
                        )

                    # Because every simulation begins on the same
                    # date, one buy-and-hold curve is sufficient.
                    first_simulation = next(
                        iter(simulations.values())
                    )

                    first_result = (
                        first_simulation["result"]
                    )

                    equity_curves["Buy and hold"] = (
                        first_result.history
                        .set_index("date")[
                            "buy_hold_value"
                        ]
                        .rename("Buy and hold")
                    )

                    equity_df = pd.concat(
                        equity_curves.values(),
                        axis=1,
                    )

                    st.line_chart(
                        equity_df,
                        width="stretch",
                    )

                    st.markdown(
                        "### Return and drawdown comparison"
                    )

                    comparison_chart_data = (
                        results_df
                        .set_index("Strategy")[
                            [
                                "Return (%)",
                                "Max drawdown (%)",
                            ]
                        ]
                    )

                    st.bar_chart(
                        comparison_chart_data,
                        width="stretch",
                    )

                    if show_comparison_details:
                        st.markdown(
                            "### Detailed strategy results"
                        )

                        for (
                            strategy_name,
                            simulation_data,
                        ) in simulations.items():
                            strategy = simulation_data[
                                "strategy"
                            ]

                            result = simulation_data[
                                "result"
                            ]

                            with st.expander(
                                f"{strategy_name} details"
                            ):
                                st.write(
                                    strategy.description
                                )

                                st.caption(
                                    "Parameters: "
                                    f"{strategy.default_parameters}"
                                )

                                detail1, detail2, detail3 = (
                                    st.columns(3)
                                )

                                detail1.metric(
                                    "Return",
                                    (
                                        f"{result.strategy_return:.2f}%"
                                    ),
                                )

                                detail2.metric(
                                    "Trades",
                                    result.completed_trades,
                                )

                                detail3.metric(
                                    "Win rate",
                                    (
                                        f"{result.win_rate:.1f}%"
                                    ),
                                )

                                trade_figure = (
                                    plot_price_with_trades(
                                        price_history=(
                                            result.history
                                        ),
                                        trade_log=result.trades,
                                        title=(
                                            f"{strategy.name} — "
                                            "price and trades"
                                        ),
                                        overlays=(
                                            strategy.price_overlays
                                        ),
                                    )
                                )

                                st.plotly_chart(
                                    trade_figure,
                                    width="stretch",
                                )

                                for panel in (
                                    strategy.indicator_panels
                                ):
                                    indicator_figure = (
                                        plot_indicator_panel(
                                            dataframe=(
                                                result.history
                                            ),
                                            panel=panel,
                                        )
                                    )

                                    st.plotly_chart(
                                        indicator_figure,
                                        width="stretch",
                                    )
                                    drawdown_data = calculate_drawdown_series(
                                        result.history
                                    )

                                    drawdown_figure = plot_drawdown(
                                        drawdown_data=drawdown_data,
                                        title=f"{strategy.name} — drawdown",
                                    )

                                    st.plotly_chart(
                                        drawdown_figure,
                                        width="stretch",
                                    )

                                    rolling_sharpe_data = calculate_rolling_sharpe(
                                        history=result.history,
                                        window=30,
                                    )

                                    rolling_sharpe_figure = plot_rolling_sharpe(
                                        rolling_data=rolling_sharpe_data,
                                        title=(
                                            f"{strategy.name} — "
                                            "30-day rolling Sharpe ratio"
                                        ),
                                    )

                                    st.plotly_chart(
                                        rolling_sharpe_figure,
                                        width="stretch",
                                    )

                                    monthly_returns = (
                                        calculate_monthly_returns(
                                            result.history
                                        )
                                    )

                                    monthly_heatmap = (
                                        plot_monthly_returns_heatmap(
                                            monthly_returns,
                                            title=(
                                                f"{strategy.name} — "
                                                "Monthly Returns"
                                            ),
                                        )
                                    )

                                    st.plotly_chart(
                                        monthly_heatmap,
                                        width="stretch",
                                    )

                                if result.trades.empty:
                                    st.info(
                                        "This strategy made "
                                        "no trades."
                                    )

                                else:
                                    st.dataframe(
                                        format_dataframe(
                                            result.trades,
                                            BACKTEST_FORMATS,
                                        ),
                                        hide_index=True,
                                        width="stretch",
                                    )

                    st.warning(
                        "This comparison uses each strategy's default "
                        "parameters over one shared historical period. "
                        "Ranking first in this table does not establish "
                        "future profitability."
                    )
                    