# 🧪 SignalLab

SignalLab is a modular quantitative research platform for testing and comparing algorithmic trading strategies using historical market data.

It is designed for research and education—not for automated real-money trading or financial advice.

## Features

- Live cryptocurrency price dashboard
- Persistent local cache for historical market data
- Technical indicators
- Generic long-only trading simulator
- Transaction-fee modelling
- Buy-and-hold benchmark
- Maximum drawdown, trade count, and win-rate metrics
- Moving-average crossover strategy
- RSI mean-reversion strategy
- Strategy registry
- Multi-strategy comparison
- Parameter optimization
- Out-of-sample walk-forward testing
- Automated tests with pytest
- Browser interface built with Streamlit

## Architecture

```text
Historical market data
        |
        v
Strategy generator
        |
        v
Positions: 0 = cash, 1 = invested
        |
        v
Generic simulator
        |
        v
Trades, equity curve and performance metrics