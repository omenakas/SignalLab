from dataclasses import dataclass

import pandas as pd


@dataclass
class AnalysisResult:
    signal: str
    score: int
    confidence: int
    reasons: list[str]


def calculate_signal(
    price: float,
    ma20: float,
    ma50: float,
    rsi: float,
) -> tuple[str, int, list[str]]:
    score = 0
    reasons = []

    if price > ma20:
        score += 1
        reasons.append("Price is above the 20-day moving average.")
    else:
        score -= 1
        reasons.append("Price is below the 20-day moving average.")

    if ma20 > ma50:
        score += 2
        reasons.append("The 20-day average is above the 50-day average.")
    else:
        score -= 2
        reasons.append("The 20-day average is below the 50-day average.")

    if rsi < 30:
        score += 1
        reasons.append("RSI is below 30.")
    elif rsi > 70:
        score -= 1
        reasons.append("RSI is above 70.")
    else:
        reasons.append("RSI is between 30 and 70.")

    if score >= 2:
        signal = "Bullish"
    elif score <= -2:
        signal = "Bearish"
    else:
        signal = "Neutral"

    return signal, score, reasons


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    signals = []
    scores = []

    for row in result.itertuples():
        required_values = [row.price, row.MA20, row.MA50, row.RSI]

        if any(pd.isna(value) for value in required_values):
            signals.append(None)
            scores.append(None)
            continue

        signal, score, _ = calculate_signal(
            price=float(row.price),
            ma20=float(row.MA20),
            ma50=float(row.MA50),
            rsi=float(row.RSI),
        )

        signals.append(signal)
        scores.append(score)

    result["signal"] = signals
    result["score"] = scores

    return result


def analyze_market(df: pd.DataFrame) -> AnalysisResult:
    required_columns = {"price", "MA20", "MA50", "RSI"}

    if not required_columns.issubset(df.columns):
        raise ValueError("Indicator columns are missing.")

    clean_df = df.dropna(subset=list(required_columns))

    if clean_df.empty:
        raise ValueError("Not enough historical data for analysis.")

    latest = clean_df.iloc[-1]

    signal, score, reasons = calculate_signal(
        price=float(latest["price"]),
        ma20=float(latest["MA20"]),
        ma50=float(latest["MA50"]),
        rsi=float(latest["RSI"]),
    )

    confidence = min(50 + abs(score) * 12, 98)

    return AnalysisResult(
        signal=signal,
        score=score,
        confidence=confidence,
        reasons=reasons,
    )