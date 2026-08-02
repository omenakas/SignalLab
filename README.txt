# 🧪 SignalLab

SignalLab is a modular quantitative research platform for researching, testing, and comparing algorithmic trading strategies using historical market data.

The project began as a cryptocurrency trading assistant and has evolved into a flexible research framework built around reusable strategies, a generic trading simulator, and a modular architecture.

> **Disclaimer:** SignalLab is an educational and research project. It does **not** provide financial advice, and historical performance does not guarantee future results.
PikkuApustaja@1989
---

# ✨ Features

Current capabilities include:

- 📈 Live cryptocurrency market dashboard
- 💾 Persistent local caching of historical market data
- 📊 Technical indicators
  - Moving Averages
  - Relative Strength Index (RSI)
- ⚙️ Generic trading simulator
- 💰 Transaction fee modelling
- 📉 Maximum drawdown calculation
- 📊 Buy & Hold benchmark comparison
- 🔬 Strategy optimization
- 🚶 Walk-forward validation
- 🧪 Multi-strategy comparison
- 🧩 Modular strategy registry
- ✅ Automated testing with pytest

---

# 🏗 Architecture

SignalLab is built around a generic simulation engine.

```text
Historical Market Data
          │
          ▼
 Strategy Generator
          │
          ▼
 Portfolio Positions
(0 = Cash, 1 = Invested)
          │
          ▼
 Generic Trading Simulator
          │
          ▼
Performance Metrics
```

Every strategy is responsible only for deciding **when to be invested**.

The simulator handles:

- Portfolio accounting
- Transaction fees
- Trade execution
- Equity curve generation
- Drawdown calculation
- Performance statistics

This architecture allows new strategies to be added without modifying the simulator.

---

# 📂 Project Structure

```text
SignalLab/

├── app.py
├── market.py
├── indicators.py
├── optimizer.py
├── walk_forward.py
├── version.py
│
├── engine/
│   └── simulator.py
│
├── strategies/
│   ├── ma_crossover.py
│   ├── rsi.py
│   └── registry.py
│
├── ui/
│   └── parameter_builder.py
│
├── tests/
│   └── test_simulator.py
│
├── data/
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone <repository-url>

cd SignalLab
```

Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it

macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

# ▶ Running SignalLab

```bash
python -m streamlit run app.py
```

---

# 🧪 Running Tests

```bash
python -m pytest -v
```

---

# 📈 Included Strategies

## Moving Average Crossover

A classic trend-following strategy.

Rules:

- Buy when the fast moving average rises above the slow moving average.
- Sell when the fast moving average falls below the slow moving average.

---

## Relative Strength Index (RSI)

A mean-reversion strategy.

Rules:

- Buy when RSI falls below the oversold threshold.
- Sell when RSI rises above the overbought threshold.

---

# 🔬 Research Philosophy

SignalLab is designed to answer one question:

> **"Does this trading idea actually work?"**

Instead of assuming that a strategy is profitable, SignalLab allows strategies to be:

- backtested
- optimized
- compared
- validated using walk-forward testing

before drawing conclusions.

---

# ⚠ Current Limitations

SignalLab currently does **not** model:

- Bid-ask spreads
- Slippage
- Taxes
- Partial fills
- Market impact
- Exchange outages
- Liquidity constraints
- Portfolio diversification
- Short selling

Historical performance should never be interpreted as evidence of future profitability.

---

# 🛣 Roadmap

## Version 0.7

- Automatic strategy parameter interface
- Generic strategy optimizer
- Generic walk-forward framework

## Version 0.8

- MACD strategy
- Bollinger Bands strategy
- Additional performance metrics

## Version 0.9

- AI-assisted strategies
- Portfolio optimization
- Multiple asset support

## Version 1.0

- Stable public release
- Plugin architecture
- Documentation
- Paper trading

---

# 🛠 Built With

- Python
- Streamlit
- Pandas
- Plotly
- TA
- Pytest

---

# 📄 License

A license has not yet been selected.

---

# ❤️ Acknowledgements

SignalLab has been developed as a long-term learning project exploring software engineering, quantitative finance, and algorithmic trading.

The focus of the project is not only on developing trading strategies, but also on designing clean, modular, and maintainable software.