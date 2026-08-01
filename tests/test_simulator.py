import pandas as pd
import pytest

from engine.simulator import run_position_backtest


def make_data(
    prices: list[float],
    positions: list[int],
) -> pd.DataFrame:
    if len(prices) != len(positions):
        raise ValueError("Prices and positions must have equal lengths.")

    return pd.DataFrame(
        {
            "date": pd.date_range(
                start="2026-01-01",
                periods=len(prices),
                freq="D",
            ),
            "price": prices,
            "position": positions,
        }
    )


def test_strategy_stays_in_cash() -> None:
    data = make_data(
        prices=[100, 110, 120],
        positions=[0, 0, 0],
    )

    result = run_position_backtest(
        data,
        initial_capital=500.0,
        fee_rate=0.001,
    )

    assert result.final_value == pytest.approx(500.0)
    assert result.strategy_return == pytest.approx(0.0)
    assert result.completed_trades == 0
    assert result.trades.empty


def test_buy_and_sell_without_fee() -> None:
    data = make_data(
        prices=[100, 100, 120, 120],
        positions=[0, 1, 1, 0],
    )

    result = run_position_backtest(
        data,
        initial_capital=500.0,
        fee_rate=0.0,
    )

    # €500 buys five units at €100.
    # Five units sold at €120 produce €600.
    assert result.final_value == pytest.approx(600.0)
    assert result.strategy_return == pytest.approx(20.0)
    assert result.completed_trades == 1
    assert result.winning_trades == 1
    assert result.win_rate == pytest.approx(100.0)

    assert list(result.trades["action"]) == [
        "BUY",
        "SELL",
    ]


def test_fee_is_charged_on_purchase_and_sale() -> None:
    data = make_data(
        prices=[100, 100, 100],
        positions=[0, 1, 0],
    )

    result = run_position_backtest(
        data,
        initial_capital=500.0,
        fee_rate=0.001,
    )

    expected_after_purchase = 500.0 * 0.999
    expected_after_sale = expected_after_purchase * 0.999

    assert result.final_value == pytest.approx(
        expected_after_sale
    )

    assert result.final_value < 500.0


def test_open_position_gets_final_sale_fee() -> None:
    data = make_data(
        prices=[100, 100, 120],
        positions=[0, 1, 1],
    )

    result = run_position_backtest(
        data,
        initial_capital=500.0,
        fee_rate=0.001,
    )

    purchased_units = (500.0 * 0.999) / 100.0
    expected_final_value = (
        purchased_units
        * 120.0
        * 0.999
    )

    assert result.final_value == pytest.approx(
        expected_final_value
    )

    # No actual SELL row occurred during the historical data.
    assert result.completed_trades == 0


def test_invalid_position_is_rejected() -> None:
    data = make_data(
        prices=[100, 110],
        positions=[0, 2],
    )

    with pytest.raises(
        ValueError,
        match="only 0 or 1",
    ):
        run_position_backtest(data)


def test_missing_required_column_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2026-01-01",
                periods=2,
            ),
            "price": [100, 110],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        run_position_backtest(data)