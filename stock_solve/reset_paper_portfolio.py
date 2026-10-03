from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_solve.backtest import BENCHMARK, STARTING_CAPITAL, download_adjusted_closes
from stock_solve.dashboard import generate_dashboard
from stock_solve.live_portfolio import (
    CURRENT_NASDAQ_100,
    PaperPortfolioConfig,
    build_paper_portfolio,
)
from stock_solve.track_paper_portfolio import (
    PENDING_REBALANCE_FILE,
    pending_from_recommendation,
    save_pending_rebalance,
)


PORTFOLIO_COLUMNS = [
    "ticker",
    "signal_date",
    "latest_adjusted_close",
    "lookback_momentum",
    "annualized_volatility",
    "drawdown_from_high",
    "ranking_score",
    "target_weight",
    "shares",
    "market_value",
    "actual_weight",
]


def reset_to_cash_with_pending_order(
    output_dir: Path,
    capital: float = STARTING_CAPITAL,
    data_start: str = "2025-01-01",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = download_adjusted_closes(CURRENT_NASDAQ_100, data_start, None, None)
    recommended, recommendation_summary = build_paper_portfolio(
        prices,
        CURRENT_NASDAQ_100,
        PaperPortfolioConfig(capital=capital, data_start=data_start),
    )

    pending = pending_from_recommendation(recommended, recommendation_summary, [])
    if pending is None:
        raise RuntimeError("Reset could not create a pending order from the current signal.")
    signal_date = pd.Timestamp(recommendation_summary["signal_date"])
    benchmark_start = (signal_date - pd.Timedelta(days=14)).date().isoformat()
    benchmark_prices = download_adjusted_closes([BENCHMARK], benchmark_start, None, None)
    benchmark_series = (
        benchmark_prices[BENCHMARK].dropna()
        if BENCHMARK in benchmark_prices
        else pd.Series(dtype=float)
    )
    benchmark_series = benchmark_series[benchmark_series.index <= signal_date]
    if not benchmark_series.empty:
        benchmark_reset_date = benchmark_series.index[-1].date().isoformat()
        benchmark_reset_price = float(benchmark_series.iloc[-1])
    else:
        benchmark_reset_date = recommendation_summary["signal_date"]
        benchmark_reset_price = None

    cash_portfolio = pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    cash_portfolio.to_csv(output_dir / "paper_portfolio.csv", index=False)
    summary = {
        "config": recommendation_summary["config"],
        "source_universe": recommendation_summary["source_universe"],
        "requested_universe_size": recommendation_summary["requested_universe_size"],
        "priced_universe_size": recommendation_summary["priced_universe_size"],
        "signal_date": recommendation_summary["signal_date"],
        "execution_date": None,
        "execution_price_type": "pending_next_open",
        "capital": capital,
        "holdings": 0,
        "cash_after_whole_share_rounding": capital,
        "cash_weight": 1.0,
        "benchmark": BENCHMARK,
        "benchmark_reset_date": benchmark_reset_date,
        "benchmark_reset_price": benchmark_reset_price,
        "benchmark_reset_value": capital,
        "portfolio_reset_value": capital,
        "max_actual_weight": 0.0,
        "total_invested": 0.0,
        "success_constraints": False,
        "status": "cash_reset_pending_next_open",
    }
    (output_dir / "paper_portfolio_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_pending_rebalance(pending, output_dir)
    tracking_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latest_price_date": recommendation_summary["signal_date"],
        "initial_signal_date": recommendation_summary["signal_date"],
        "initial_capital": capital,
        "portfolio_value": capital,
        "portfolio_return": 0.0,
        "portfolio_reset_value": capital,
        "portfolio_return_since_reset": 0.0,
        "portfolio_return_since_trade_basis": 0.0,
        "cash": capital,
        "benchmark": BENCHMARK,
        "benchmark_reset_date": benchmark_reset_date,
        "benchmark_reset_price": benchmark_reset_price,
        "benchmark_reset_value": capital,
        "holdings": 0,
        "max_current_weight": 0.0,
        "recommended_signal_date": recommendation_summary["signal_date"],
        "recommended_adds": pending["adds"],
        "recommended_drops": pending["drops"],
        "rebalance_needed": True,
        "rebalance_signal_allowed": True,
        "rebalance_deferred_until_weekly_gate": False,
        "rebalance_pending_created": True,
        "pending_rebalance_revalidated": False,
        "pending_rebalance_replaced": False,
        "pending_rebalance_cancelled": False,
        "validation_signal_date": None,
        "replacement_adds": [],
        "replacement_drops": [],
        "pending_rebalance": True,
        "pending_signal_date": pending["signal_date"],
        "pending_adds": pending["adds"],
        "pending_drops": pending["drops"],
        "rebalance_executed": False,
        "executed_signal_date": None,
        "execution_date": None,
        "execution_price_type": None,
        "open_validation_passed": False,
        "open_validation_adjusted": False,
        "open_validation_cancelled": False,
        "open_validation_signal_date": None,
        "validated_before_execution": False,
        "executed_adds": [],
        "executed_drops": [],
    }
    (output_dir / "paper_portfolio_tracking_summary.json").write_text(
        json.dumps(tracking_summary, indent=2) + "\n"
    )
    history_path = output_dir / "paper_portfolio_tracking_history.csv"
    pd.DataFrame([tracking_summary]).to_csv(history_path, index=False)

    generate_dashboard(output_dir)
    return {
        "capital": capital,
        "signal_date": pending["signal_date"],
        "pending_file": str(output_dir / PENDING_REBALANCE_FILE),
        "benchmark": BENCHMARK,
        "benchmark_reset_date": benchmark_reset_date,
        "benchmark_reset_price": benchmark_reset_price,
        "pending_adds": pending["adds"],
        "pending_drops": pending["drops"],
        "target_tickers": pending["target_tickers"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset paper portfolio to cash and stage next-open buys.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--capital", type=float, default=STARTING_CAPITAL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = reset_to_cash_with_pending_order(args.output_dir, args.capital)
    print("Reset capital:", f"${result['capital']:,.2f}")
    print("Signal date:", result["signal_date"])
    print("Pending buys:", ", ".join(result["target_tickers"]))
    print("Pending file:", result["pending_file"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
