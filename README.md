# 📊 SignalLab

> **A modular laboratory for quantitative trading research.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Modular-purple)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Version](https://img.shields.io/badge/Version-v0.9.0-blue)

**Develop • Backtest • Compare • Analyze**

![SignalLab Home](screenshots/home.png)

---

## ✨ Features

SignalLab is a modular research platform for developing, testing and evaluating quantitative trading strategies.

### 📈 Strategy Analysis
- Interactive market data analysis
- Technical indicators
- Strategy signal visualization

### 💰 Backtesting
- Generic backtesting engine
- Transaction fee simulation
- Trade history
- Portfolio equity tracking

### 🧪 Strategy Lab
- Parameter experimentation
- Modular strategy registry
- Rapid strategy evaluation

### 🔄 Walk-Forward Validation
- Out-of-sample testing
- Rolling evaluation windows
- Strategy robustness assessment

### 📊 Strategy Research
- Performance dashboard
- Strategy comparison
- Equity curve comparison
- Drawdown analysis
- Rolling Sharpe Ratio
- Monthly Returns Heatmap
- Performance highlights
- Risk metrics (Sharpe, Sortino, Calmar, CAGR)

---

# 📸 Screenshots

## Strategy Research Dashboard

![Dashboard](screenshots/strategy-research-dashboard.png)

---

## Equity Curve Comparison

![Equity Curves](screenshots/equity-curves.png)

---

## Price & Executed Trades

![Trades](screenshots/price-trades.png)

---

## Drawdown Analysis

![Drawdown](screenshots/drawdown.png)

---

## Monthly Returns Heatmap

![Heatmap](screenshots/monthly-heatmap.png)

---

# 🏗 Architecture

SignalLab follows a modular architecture where each component has a clearly defined responsibility.

```text
app.py
│
├── analytics/
│   ├── performance.py
│   ├── drawdown.py
│   ├── rolling.py
│   └── monthly_returns.py
│
├── strategies/
│
├── simulator/
│
├── ui/
│   ├── charts.py
│   ├── formatting.py
│   ├── performance_dashboard.py
│   └── performance_highlights.py
│
└── tests/
```

---

# 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/<your-username>/SignalLab.git
cd SignalLab
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run SignalLab

```bash
streamlit run app.py
```

---

# 🧪 Running Tests

Run all tests

```bash
pytest
```

Run with verbose output

```bash
pytest -v
```

---

# 🛣 Roadmap

### ✅ Completed

- Modular strategy framework
- Generic backtesting engine
- Strategy registry
- Walk-forward validation
- Performance dashboard
- Strategy comparison
- Drawdown analytics
- Rolling Sharpe Ratio
- Monthly Returns Heatmap
- Performance highlights
- Reusable UI formatting

### 🔮 Future

- Multi-asset support
- Additional strategy library
- Portfolio optimization
- Monte Carlo analysis
- Position sizing models
- Performance report export
- Interactive strategy builder

---

# 🤝 Contributing

Ideas, suggestions and pull requests are always welcome.

---

# 📄 License

MIT License